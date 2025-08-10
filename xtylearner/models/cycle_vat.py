from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from ..training.metrics import cross_entropy_loss
from .vat import vat_loss

try:  # optional outcome likelihood head
    from .heads import LowRankDiagHead  # type: ignore
    from ..losses import nll_lowrank_diag  # type: ignore

    HAS_LIKELIHOOD = True
except Exception:  # pragma: no cover - fallback when modules missing
    LowRankDiagHead = None  # type: ignore
    nll_lowrank_diag = None  # type: ignore
    HAS_LIKELIHOOD = False


@register_model("cycle_vat")
class CycleVAT(nn.Module):
    """Shared-encoder with two posteriors and VAT regularisation.

    The model encodes covariates ``x`` to a latent representation ``h``. A
    forward classifier :math:`p_f(t|x)` predicts treatments and receives virtual
    adversarial training. An outcome head models :math:`p(y|x,t)` and an inverse
    classifier :math:`p_i(t|x,y)` infers treatments from covariates and observed
    outcomes. Rows lacking treatment labels contribute via an expected outcome
    likelihood under a posterior mixing of ``p_f`` and ``p_i``. Optional Kendall
    uncertainty weighting balances the different loss terms.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = 0.1,
        norm_layer: callable | None = None,
        residual: bool = False,
        # outcome head
        outcome_likelihood: bool = True,
        outcome_rank: int = 8,
        # y encoder for inverse classifier
        y_embed_dims: tuple[int, ...] | list[int] | None = (64,),
        # posterior mixing
        alpha: float = 0.5,
        warmup_steps: int = 500,
        q_conf_start: float = 0.9,
        q_conf_end: float = 0.6,
        q_conf_anneal_steps: int = 3000,
        # VAT parameters (applied to x->p_f)
        eps: float = 2.0,
        xi: float = 1e-6,
        n_power: int = 1,
        # loss weighting
        use_uncertainty_weighting: bool = True,
        # agreement loss base weight
        lambda_agree: float = 1.0,
    ) -> None:
        super().__init__()

        self.k = k
        self.alpha = alpha
        self.lambda_agree = lambda_agree
        self.outcome_likelihood = outcome_likelihood and HAS_LIKELIHOOD
        self.warmup_steps = warmup_steps
        self.q_conf_start = q_conf_start
        self.q_conf_end = q_conf_end
        self.q_conf_anneal_steps = q_conf_anneal_steps

        # ----- shared encoder φ(x)
        self.encoder = make_mlp(
            [d_x, *hidden_dims],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # ----- forward classifier p_f(t|x)
        self.f_head = nn.Linear(hidden_dims[-1], k)

        # small wrapper so VAT sees a module x->logits
        class FOnX(nn.Module):
            def __init__(self, encoder: nn.Module, f_head: nn.Module) -> None:
                super().__init__()
                # store callables only to avoid recursive module references
                self._enc = encoder.forward
                self._head = f_head.forward

            def forward(
                self, x: torch.Tensor
            ) -> torch.Tensor:  # pragma: no cover - simple wrapper
                h = self._enc(x)
                return self._head(h)

        self._f_on_x = FOnX(self.encoder, self.f_head)

        # ----- outcome head p(y|x,t)
        if self.outcome_likelihood:
            assert LowRankDiagHead is not None and nll_lowrank_diag is not None
            self.outcome = LowRankDiagHead(hidden_dims[-1] + k, d_y, outcome_rank)
        else:
            self.outcome = make_mlp(
                [hidden_dims[-1] + k, *hidden_dims, d_y],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
                residual=residual,
            )

        # ----- y encoder ψ(y) for inverse classifier
        if y_embed_dims is None or len(y_embed_dims) == 0:
            self.y_enc = nn.Identity()
            y_repr = d_y
        else:
            self.y_enc = make_mlp(
                [d_y, *y_embed_dims],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
                residual=False,
            )
            y_repr = y_embed_dims[-1]

        # ----- inverse classifier p_i(t|x,y)
        self.i_head = make_mlp(
            [hidden_dims[-1] + y_repr, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # ----- Kendall-style uncertainty weighting
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if self.use_uncertainty_weighting:
            # log variances for: sup-t, outcome, inverse, agreement, vat
            self.log_vars = nn.Parameter(torch.zeros(5))

        # VAT parameters
        self.eps, self.xi, self.n_power = eps, xi, n_power
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        h = self.encoder(x)
        t1 = F.one_hot(t.to(torch.long), self.k).float()
        if self.outcome_likelihood:
            mu, _, _ = self.outcome(torch.cat([h, t1], dim=-1))
            return mu
        return self.outcome(torch.cat([h, t1], dim=-1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor | None = None, use: str = "auto"
    ) -> torch.Tensor:
        h = self.encoder(x)
        if use == "x" or (use == "auto" and y is None):
            return F.softmax(self.f_head(h), dim=-1)
        assert y is not None
        z = self.y_enc(y)
        logits = self.i_head(torch.cat([h, z], dim=-1))
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    def _pf_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.f_head(h)

    def _posterior_mix(self, pf: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        # pf, pi are probabilities (softmaxed); q ∝ pf^α * pi^(1-α)
        a = self.alpha
        log_q = a * (pf + 1e-12).log() + (1 - a) * (pi + 1e-12).log()
        return F.softmax(log_q, dim=-1)

    def _expected_outcome_loss(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return x.new_tensor(0.0)
        hm = h[mask]
        xm = x[mask]  # noqa: F841 - kept for potential extensions
        ym = y[mask]
        Bm = hm.size(0)
        losses = []
        for j in range(self.k):
            tj = torch.full((Bm,), j, dtype=torch.long, device=x.device)
            t1 = F.one_hot(tj, self.k).float()
            inp = torch.cat([hm, t1], dim=-1)
            if self.outcome_likelihood:
                mu, Fmat, sigma2 = self.outcome(inp)
                losses.append(nll_lowrank_diag(ym, mu, Fmat, sigma2))
            else:
                yhat = self.outcome(inp)
                losses.append(((yhat - ym) ** 2).mean(dim=-1))
        L = torch.stack(losses, dim=-1)  # [Bm, k]
        q_m = q[mask]
        return (q_m * L).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """Predict outcome mean for a given treatment."""

        return self.predict_outcome(x, t)

    # ------------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        t_obs: torch.Tensor | None,
        *,
        lambda_base: dict[str, float] | None = None,
    ) -> torch.Tensor:
        device = x.device
        self._step += 1
        step = int(self._step.item())
        h = self.encoder(x)
        B = x.size(0)
        zero = x.new_tensor(0.0)

        # masks
        if t_obs is None:
            has_t = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            has_t = t_obs >= 0

        has_y = y is not None
        if has_y:
            if y.ndim == 1:
                ymask = torch.isfinite(y)
            else:
                ymask = torch.isfinite(y).all(dim=-1)
        else:
            ymask = torch.zeros(B, dtype=torch.bool, device=device)

        # forward classifier on all rows
        logits_f = self._pf_logits(h)

        # supervised treatment loss
        L_sup_t = (
            cross_entropy_loss(logits_f[has_t], t_obs[has_t].to(torch.long))
            if has_t.any()
            else zero
        )

        # inverse path using real y
        if has_y and ymask.any():
            z = self.y_enc(y)
            logits_i = self.i_head(torch.cat([h, z], dim=-1))
            L_inv = (
                cross_entropy_loss(
                    logits_i[has_t & ymask], t_obs[has_t & ymask].to(torch.long)
                )
                if (has_t & ymask).any()
                else zero
            )
            pf = F.softmax(logits_f[ymask], dim=-1)
            pi = F.softmax(logits_i[ymask], dim=-1)
            if step > self.warmup_steps:
                L_agree = F.kl_div(
                    (pf + 1e-12).log(), pi.detach(), reduction="batchmean"
                )
            else:
                L_agree = zero
            q_y = self._posterior_mix(pf, pi)
            q = x.new_zeros(B, self.k)
            q[ymask] = q_y
        else:
            L_inv, L_agree = zero, zero
            q = None

        # outcome losses
        if has_y and ymask.any():
            if has_t.any():
                idx = has_t & ymask
                if idx.any():
                    t1 = F.one_hot(t_obs[idx].to(torch.long), self.k).float()
                    inp = torch.cat([h[idx], t1], dim=-1)
                    if self.outcome_likelihood:
                        mu, Fmat, sigma2 = self.outcome(inp)
                        L_outcome_obs = nll_lowrank_diag(
                            y[idx], mu, Fmat, sigma2
                        ).mean()
                    else:
                        yhat = self.outcome(inp)
                        L_outcome_obs = F.mse_loss(yhat, y[idx])
                else:
                    L_outcome_obs = zero
            else:
                L_outcome_obs = zero

            if q is not None and (step > self.warmup_steps):
                idx_miss_full = ((~has_t) & ymask) if (t_obs is not None) else ymask
                if idx_miss_full.any():
                    s = min(
                        1.0,
                        (step - self.warmup_steps) / max(1, self.q_conf_anneal_steps),
                    )
                    tau = self.q_conf_end + 0.5 * (
                        self.q_conf_start - self.q_conf_end
                    ) * (1 + math.cos((1 - s) * math.pi))
                    q_miss = q[idx_miss_full]
                    conf = q_miss.max(dim=-1).values
                    gate = conf >= tau
                    if gate.any():
                        idx2 = idx_miss_full.clone()
                        idx2[idx_miss_full] = gate
                        L_outcome_miss = self._expected_outcome_loss(h, x, y, q, idx2)
                    else:
                        L_outcome_miss = zero
                else:
                    L_outcome_miss = zero
            else:
                L_outcome_miss = zero
        else:
            L_outcome_obs = zero
            L_outcome_miss = zero

        # VAT on p_f
        if self.training and torch.is_grad_enabled() and (step > self.warmup_steps):
            L_vat = vat_loss(
                self._f_on_x, x, xi=self.xi, eps=self.eps, n_power=self.n_power
            )
        else:
            L_vat = zero

        terms = {
            "sup_t": L_sup_t,
            "outcome": L_outcome_obs + L_outcome_miss,
            "inverse": L_inv,
            "agree": self.lambda_agree * L_agree,
            "vat": L_vat,
        }

        if lambda_base is None:
            lambda_base = {
                "sup_t": 1.0,
                "outcome": 1.0,
                "inverse": 1.0,
                "agree": 1.0,
                "vat": 1.0,
            }

        if self.use_uncertainty_weighting:
            losses = []
            for i, key in enumerate(["sup_t", "outcome", "inverse", "agree", "vat"]):
                Li = terms[key]
                if Li is zero:
                    continue
                with torch.no_grad():
                    if step <= self.warmup_steps:
                        self.log_vars.data.clamp_(-2.0, 2.0)
                s = self.log_vars[i].clamp(-4.0, 4.0)
                losses.append(torch.exp(-s) * lambda_base[key] * Li + s)
            return torch.stack(losses).sum() if len(losses) else zero

        total = zero
        for key, Li in terms.items():
            total = total + lambda_base.get(key, 1.0) * Li
        return total


__all__ = ["CycleVAT"]
