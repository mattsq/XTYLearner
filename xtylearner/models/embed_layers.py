"""Building blocks for deterministic Gaussian embeddings.

Provides:
  - C_EmbedAttentionModule : multi-head softmax Euclidean attention with learnable keys/values
  - C_ACN                  : auto-compressing network (residual MLP)
  - C_PermutationLayer     : deterministic permutation (exact inverse)
  - C_AffineCouplingLayer  : RealNVP-style affine coupling (exact inverse)
  - C_InvertibleFlow       : composable invertible normalizing flow

Reference
---------
Adapted from ``EmbedModels.py`` in https://github.com/mvparakhin/ml-tidbits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _eps_for_dtype(dtype: torch.dtype, large: bool = False) -> float:
    """Return a small epsilon suitable for *dtype*.

    When *large* is True the returned value is ``sqrt(eps)`` -- useful as a
    variance floor where machine-epsilon itself would be too tight.
    """
    eps = torch.finfo(dtype).eps
    return math.sqrt(eps) if large else eps


# ---------------------------------------------------------------------------
# C_EmbedAttentionModule
# ---------------------------------------------------------------------------


class C_EmbedAttentionModule(nn.Module):
    """Multi-head softmax attention with *learnable* keys and values.

    The query is the input ``x`` (optionally transformed by ``q_transform``).
    Keys ``k`` and values ``v`` are stored as trainable parameter tensors of
    shape ``(n_of_heads, n_of_basis, ...)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (query) vectors.
    hidden_dim : int
        Per-head value dimensionality.
    out_dim : int
        Final output dimensionality (after ``head_combine``).
    n_of_basis : int
        Number of learnable key/value pairs per head.
    n_of_heads : int
        Number of attention heads.
    is_euclidean : bool
        If True (default) use Euclidean attention:
        ``logits = <q, k> - 0.5 * ||k||^2``.
        Otherwise use scaled dot-product attention.
    normalize_k : bool
        If True, L2-normalise keys before computing logits.
    q_transform : nn.Module | None
        Optional transform applied to queries before attention.
    head_combine : nn.Module | None
        Optional projection from concatenated head outputs to *out_dim*.
        Defaults to ``nn.Linear(hidden_dim * n_of_heads, out_dim)`` when
        ``n_of_heads > 1``, or ``nn.Identity()`` otherwise.
    affine_experts : bool
        Enable rank-1 affine expert contribution to values:
        ``v(q) = v + v_out * (q * v_in)``.
    affine_init_scale : float
        Scale factor for affine expert initialisation (smaller -> closer to
        identity at init).
    head_temperature : bool
        Enable a learnable per-head temperature multiplier on logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_of_basis: int,
        n_of_heads: int,
        *,
        is_euclidean: bool = True,
        normalize_k: bool = False,
        q_transform: nn.Module | None = None,
        head_combine: nn.Module | None = None,
        affine_experts: bool = False,
        affine_init_scale: float = 0.1,
        head_temperature: bool = False,
    ):
        super().__init__()

        self.n_of_heads = n_of_heads
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_of_basis = n_of_basis
        self.k_norm = 1.0 / math.sqrt(input_dim)
        self.is_euclidean = is_euclidean
        self.normalize_k = normalize_k

        self.affine_experts = bool(affine_experts)
        self.head_temperature = bool(head_temperature)

        # Trainable keys, values, and per-head scale
        self.k = nn.Parameter(torch.empty((n_of_heads, n_of_basis, input_dim)))
        self.v = nn.Parameter(torch.empty((n_of_heads, n_of_basis, hidden_dim)))
        self.scale = nn.Parameter(torch.zeros((n_of_heads,)))
        self.layer_norm_scale = nn.Parameter(-torch.ones((n_of_basis,)))
        nn.init.xavier_normal_(self.k)
        nn.init.xavier_normal_(self.v)

        # Per-head temperature (always allocated; only used when head_temperature=True)
        self.head_temp = nn.Parameter(torch.zeros((n_of_heads,)))

        # Rank-1 affine expert parameters
        self.v_in = nn.Parameter(torch.empty((n_of_heads, n_of_basis, input_dim)))
        self.v_out = nn.Parameter(torch.empty((n_of_heads, n_of_basis, hidden_dim)))
        if self.affine_experts:
            nn.init.xavier_normal_(self.v_in)
            nn.init.xavier_normal_(self.v_out)
            s = float(affine_init_scale)
            if s != 1.0:
                with torch.no_grad():
                    self.v_in.mul_(s)
                    self.v_out.mul_(s)

        # Optional q transformation and head combination
        self.q_transform = q_transform if q_transform is not None else nn.Identity()
        if n_of_heads > 1:
            self.head_combine = (
                head_combine
                if head_combine is not None
                else nn.Linear(hidden_dim * n_of_heads, out_dim)
            )
        else:
            self.head_combine = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        q = x.reshape(-1, orig_shape[-1])
        q = self.q_transform(q)
        k = (
            self.k
            * torch.clip(
                self.k.square().sum(dim=-1, keepdim=True), min=1.19209e-07**2
            ).rsqrt()
            if self.normalize_k
            else self.k
        )

        if self.is_euclidean:
            cross_term = torch.einsum("bi,hni->bhn", q, k)
            if not self.normalize_k:
                k_term = torch.sum(self.k.square(), dim=-1) * torch.exp(self.scale)[
                    ..., None
                ]
                logits = cross_term - 0.5 * k_term[None, ...]
            else:
                logits = cross_term
        else:
            logits = torch.einsum("bi,hni->bhn", q, k) * self.k_norm

        logits = F.layer_norm(
            logits,
            normalized_shape=(logits.size(-1),),
            weight=torch.exp(self.layer_norm_scale),
        )

        if self.head_temperature:
            logits = logits * torch.exp(self.head_temp)[None, :, None]

        attention_weights = F.softmax(logits, dim=-1)
        output = torch.einsum("bhn,hnd->bhd", attention_weights, self.v)

        if self.affine_experts:
            s = torch.einsum("bi,hni->bhn", q, self.v_in)
            output = output + torch.einsum(
                "bhn,hnd->bhd", attention_weights * s, self.v_out
            )
        output = output.reshape(-1, self.hidden_dim * self.n_of_heads)
        output = self.head_combine(output)

        new_shape = orig_shape[:-1] + (self.out_dim,)
        output = output.reshape(new_shape)
        return output


# ---------------------------------------------------------------------------
# Auto-Compressing Network
# ---------------------------------------------------------------------------


class C_ACN(nn.Module):
    """Residual MLP whose blocks are *added* (not chained) before the output
    projection, yielding a compressed-residual architecture.

    Parameters
    ----------
    in_dim : int
        Input dimensionality.
    out_dim : int
        Output dimensionality.
    hidden_dim : int
        Width of each residual block.
    n_blocks : int
        Number of residual blocks.
    """

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 512, n_blocks: int = 2
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.in_proj(x)
        res = a
        for lin in self.blocks:
            a = lin(F.elu(a))
            res = res + a
        return self.out_proj(F.elu(res))


# ---------------------------------------------------------------------------
# Deterministic permutation layer (exact inverse)
# ---------------------------------------------------------------------------


class C_PermutationLayer(nn.Module):
    """Fixed permutation of the last dimension.  ``inverse()`` applies the
    exact inverse permutation.

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    perm : torch.Tensor
        1-D permutation indices of length ``dim``.
    """

    def __init__(self, dim: int, perm: torch.Tensor):
        super().__init__()
        dim = int(dim)
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if perm.ndim != 1 or int(perm.numel()) != dim:
            raise ValueError("perm must have shape (dim,)")

        perm = perm.to(dtype=torch.int64).contiguous()
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim, dtype=torch.int64)

        self.dim = dim
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}"
            )
        return torch.index_select(x, -1, self.perm)

    @torch.jit.export
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}"
            )
        return torch.index_select(y, -1, self.inv_perm)


# ---------------------------------------------------------------------------
# Invertible flow: RealNVP-style affine coupling (exact inverse)
# ---------------------------------------------------------------------------


class C_AffineCouplingLayer(nn.Module):
    """Single affine coupling layer.

    Works for any last-dim ``D >= 1`` (including odd D).
    Input can be any shape ``(..., D)``.
    Log-scale ``s`` is bounded via ``tanh`` for numerical stability.

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    mask : torch.Tensor
        Binary mask of shape ``(dim,)`` selecting the pass-through dimensions.
    hidden_dim : int
        Hidden width of the conditioner network.
    n_blocks : int
        Number of residual blocks in the conditioner.
    s_max : float
        Tanh saturation bound for log-scale.
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        s_max: float = 2.0,
    ):
        super().__init__()
        dim = int(dim)
        if dim < 1:
            raise ValueError("dim must be >= 1")

        self.dim = dim
        self.s_max = float(s_max)

        if mask.ndim != 1 or int(mask.numel()) != dim:
            raise ValueError("mask must have shape (dim,)")

        m = mask.to(dtype=torch.float32).contiguous()
        self.register_buffer("mask", m)
        self.register_buffer("inv_mask", 1.0 - m)

        pass_idx = (
            torch.nonzero(m >= 0.5, as_tuple=False).flatten().to(dtype=torch.int64)
        )
        trans_idx = (
            torch.nonzero(m < 0.5, as_tuple=False).flatten().to(dtype=torch.int64)
        )

        self.register_buffer("pass_idx", pass_idx)
        self.register_buffer("trans_idx", trans_idx)
        self.pass_dim = int(pass_idx.numel())
        self.trans_dim = int(trans_idx.numel())

        if self.pass_dim < 1 or self.trans_dim < 1:
            self.net = None
            self._contig = True
            self._pass_first = True
            return

        # Fast path detection: contiguous split
        self._contig = False
        self._pass_first = True
        with torch.no_grad():
            a = torch.arange(self.dim, dtype=torch.int64)
            if torch.equal(pass_idx, a[: self.pass_dim]) and torch.equal(
                trans_idx, a[self.pass_dim :]
            ):
                self._contig = True
                self._pass_first = True
            elif torch.equal(trans_idx, a[: self.trans_dim]) and torch.equal(
                pass_idx, a[self.trans_dim :]
            ):
                self._contig = True
                self._pass_first = False

        # Conditioner: (pass_dim) -> (2 * trans_dim)
        self.net = C_ACN(self.pass_dim, 2 * self.trans_dim, hidden_dim, n_blocks)
        # Identity-ish init: start close to identity for stability.
        with torch.no_grad():
            nn.init.zeros_(self.net.out_proj.weight)
            nn.init.zeros_(self.net.out_proj.bias)

    def _ST(self, x_pass: torch.Tensor):
        st = self.net(x_pass)
        s_raw, t = st.chunk(2, dim=-1)
        s = torch.tanh(s_raw) * self.s_max
        return s, t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}"
            )
        if self.net is None:
            return x

        if self._contig:
            if self._pass_first:
                x_pass = x[..., : self.pass_dim]
                x_trans = x[..., self.pass_dim :]
                s, t = self._ST(x_pass)
                y_trans = x_trans * torch.exp(s) + t
                return torch.cat((x_pass, y_trans), dim=-1)
            else:
                x_trans = x[..., : self.trans_dim]
                x_pass = x[..., self.trans_dim :]
                s, t = self._ST(x_pass)
                y_trans = x_trans * torch.exp(s) + t
                return torch.cat((y_trans, x_pass), dim=-1)

        x_pass = torch.index_select(x, -1, self.pass_idx)
        x_trans = torch.index_select(x, -1, self.trans_idx)
        s, t = self._ST(x_pass)
        y_trans = x_trans * torch.exp(s) + t

        y = torch.empty_like(x)
        y.index_copy_(-1, self.pass_idx, x_pass)
        y.index_copy_(-1, self.trans_idx, y_trans)
        return y

    @torch.jit.export
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}"
            )
        if self.net is None:
            return y

        if self._contig:
            if self._pass_first:
                y_pass = y[..., : self.pass_dim]
                y_trans = y[..., self.pass_dim :]
                s, t = self._ST(y_pass)
                x_trans = (y_trans - t) * torch.exp(-s)
                return torch.cat((y_pass, x_trans), dim=-1)
            else:
                y_trans = y[..., : self.trans_dim]
                y_pass = y[..., self.trans_dim :]
                s, t = self._ST(y_pass)
                x_trans = (y_trans - t) * torch.exp(-s)
                return torch.cat((x_trans, y_pass), dim=-1)

        y_pass = torch.index_select(y, -1, self.pass_idx)
        y_trans = torch.index_select(y, -1, self.trans_idx)
        s, t = self._ST(y_pass)
        x_trans = (y_trans - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x.index_copy_(-1, self.pass_idx, y_pass)
        x.index_copy_(-1, self.trans_idx, x_trans)
        return x


# ---------------------------------------------------------------------------
# Composable invertible flow
# ---------------------------------------------------------------------------


class C_InvertibleFlow(nn.Module):
    """Stack of affine coupling layers with optional permutations.

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    n_layers : int
        Number of coupling layers.
    hidden_dim : int
        Hidden width inside each coupling conditioner network.
    n_blocks : int
        Number of residual blocks in each conditioner.
    s_max : float
        Tanh saturation bound for log-scale in coupling layers.
    mask_mode : ``"alternating"`` | ``"half"``
        How binary masks are constructed (only used when
        ``permute_mode="none"``).
    permute_mode : ``"none"`` | ``"per_layer"`` | ``"per_pair"``
        Random permutation strategy between coupling layers.
    permute_seed : int
        Seed for deterministic permutation generation.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        s_max: float = 2.0,
        mask_mode: str = "alternating",
        *,
        permute_mode: str = "per_pair",
        permute_seed: int = 1337,
    ):
        super().__init__()
        dim = int(dim)
        if dim < 1:
            raise ValueError("dim must be >= 1")

        self.dim = dim
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.n_blocks = int(n_blocks)
        self.s_max = float(s_max)

        if mask_mode not in ("alternating", "half"):
            raise ValueError("mask_mode must be 'alternating' or 'half'")
        self.mask_mode = mask_mode

        if permute_mode not in ("none", "per_layer", "per_pair"):
            raise ValueError("permute_mode must be 'none', 'per_layer', or 'per_pair'")
        self.permute_mode = permute_mode
        self.permute_seed = int(permute_seed) & 0x7FFFFFFF

        ops: list[nn.Module] = []
        if self.n_layers > 0 and self.dim >= 2:
            idx = torch.arange(self.dim, dtype=torch.int64)
            d1 = self.dim // 2

            # Base half masks in current coordinate order.
            mask0 = torch.zeros((self.dim,), dtype=torch.float32)
            mask0[:d1] = 1.0
            mask1 = 1.0 - mask0

            if self.permute_mode != "none":
                gen = torch.Generator()
                gen.manual_seed(self.permute_seed)

                for i in range(self.n_layers):
                    if (self.permute_mode == "per_layer") or (
                        self.permute_mode == "per_pair" and ((int(i) & 1) == 0)
                    ):
                        perm = torch.randperm(self.dim, generator=gen)
                        ops.append(C_PermutationLayer(self.dim, perm))

                    m = mask0 if ((int(i) & 1) == 0) else mask1
                    ops.append(
                        C_AffineCouplingLayer(
                            self.dim,
                            m,
                            hidden_dim=self.hidden_dim,
                            n_blocks=self.n_blocks,
                            s_max=self.s_max,
                        )
                    )
            else:
                for i in range(self.n_layers):
                    if self.mask_mode == "alternating":
                        m = ((idx + int(i)) & 1).to(dtype=torch.float32)
                    else:
                        m = mask0 if ((int(i) & 1) == 0) else mask1
                    ops.append(
                        C_AffineCouplingLayer(
                            self.dim,
                            m,
                            hidden_dim=self.hidden_dim,
                            n_blocks=self.n_blocks,
                            s_max=self.s_max,
                        )
                    )

        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}"
            )
        y = x
        for op in self.ops:
            y = op(y)
        return y

    @torch.jit.export
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}"
            )
        x = y
        for op in self.ops[::-1]:
            x = op.inverse(x)
        return x


__all__ = [
    "C_EmbedAttentionModule",
    "C_ACN",
    "C_PermutationLayer",
    "C_AffineCouplingLayer",
    "C_InvertibleFlow",
]
