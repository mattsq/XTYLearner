from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .gnn_scm import notears_acyclicity


# ----------------------------------------------------------------------
def cosine_alpha(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule used for diffusion."""
    return torch.cos((t + s) / (1 + s) * math.pi / 2)


def forward_diffuse(
    z0: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = cosine_alpha(t)
    sigma = torch.sqrt(1 - alpha**2)
    noise = torch.randn_like(z0)
    zt = alpha * z0 + sigma * noise
    return zt, noise


def gaussian_nll(
    x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor
) -> torch.Tensor:
    return 0.5 * ((x - mu) / log_sigma.exp()) ** 2 + log_sigma


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        emb = t * self.freq
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MaskedGraphConv(nn.Module):
    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:  # noqa: D401
        msg = torch.einsum("ij,bjk->bik", A, h)
        return h + msg / h.size(1)


class GraphScoreNet(nn.Module):
    def __init__(
        self, d_in: int, d_nodes: int, d_y: int, time_dim: int, hidden: int
    ) -> None:
        super().__init__()
        self.d_nodes = d_nodes
        self.d_in = d_in
        self.time = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.fc_in = nn.Linear(d_in, hidden)
        self.conv1 = MaskedGraphConv()
        self.conv2 = MaskedGraphConv()
        self.fc_out = nn.Linear(hidden, d_y)

    def forward(
        self, z: torch.Tensor, x: torch.Tensor, t_emb: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat([x, z], dim=-1)  # (b, d_in)
        h = self.fc_in(h)
        h = h.unsqueeze(1).repeat(1, self.d_nodes, 1)
        time_emb = self.time(t_emb)
        h = h + time_emb.unsqueeze(1)
        h = F.silu(self.conv1(h, A))
        h = F.silu(self.conv2(h, A))
        out = self.fc_out(h[:, -1])  # take Y node
        return out

    def root(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(x.size(0), self.d_in - x.size(1), device=x.device)
        h = self.fc_in(torch.cat([x, zeros], dim=-1))
        h = h.unsqueeze(1).repeat(1, self.d_nodes, 1)
        h = F.silu(self.conv1(h, A))
        h = F.silu(self.conv2(h, A))
        return h[:, -2]  # T node embedding


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


@register_model("diffusion_gnn_scm")
class DiffusionGNN_SCM(nn.Module):
    def __init__(
        self,
        d_x: int,
        k: int,
        d_y: int,
        time_emb_dim: int = 128,
        hidden: int = 256,
        lambda_acyc: float = 5.0,
        gamma_l1: float = 1e-2,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        self.d_nodes = d_x + 2
        mask = torch.ones(self.d_nodes, self.d_nodes)
        mask[-1, :d_x] = 0
        mask.fill_diagonal_(0)
        self.register_buffer("mask", mask)
        self.B = nn.Parameter(torch.full_like(mask, -10.0))

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.score_net = GraphScoreNet(
            d_x + k + d_y, self.d_nodes, d_y, time_emb_dim, hidden
        )
        self.head_T = MLP(d_x + hidden, k)
        self.head_Y = MLP(d_x + k + hidden, d_y * 2)
        self.lambda_acyc = lambda_acyc
        self.gamma_l1 = gamma_l1

    # ------------------------------------------------------------------
    def _A(self) -> torch.Tensor:
        return torch.sigmoid(self.B) * self.mask

    # ------------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        A = self._A()
        z_real = torch.cat([F.one_hot(t_obs.clamp_min(0), self.k).float(), y], dim=-1)
        tau = torch.rand(z_real.size(0), 1, device=x.device)
        z_noisy, noise = forward_diffuse(z_real, tau)
        t_emb = self.time_emb(tau)
        score_pred = self.score_net(z_noisy, x, t_emb, A)
        L_dm = ((score_pred + noise[:, -self.d_y :]) ** 2).mean()

        L_t = torch.tensor(0.0, device=x.device)
        if t_obs is not None:
            root_emb = self.score_net.root(x, A)
            logits_t = self.head_T(torch.cat([x, root_emb], dim=-1))
            L_t = F.cross_entropy(logits_t, t_obs.clamp_min(0), reduction="mean")

        root_emb = self.score_net.root(x, A)
        ty_in = torch.cat(
            [F.one_hot(t_obs.clamp_min(0), self.k).float(), root_emb], dim=-1
        )
        mu_y, log_sigma_y = self.head_Y(torch.cat([x, ty_in], dim=-1)).chunk(2, -1)
        L_y = gaussian_nll(y, mu_y, log_sigma_y).mean()

        L_acyc = notears_acyclicity(A)
        L_l1 = (A.abs() * self.mask).sum()
        return L_dm + L_y + L_t + self.lambda_acyc * L_acyc + self.gamma_l1 * L_l1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Predict outcome mean for treatment ``t``."""
        A = self._A()
        root_emb = self.score_net.root(x, A)
        t_in = F.one_hot(t, self.k).float()
        ty_in = torch.cat([t_in, root_emb], dim=-1)
        mu_y, _ = self.head_Y(torch.cat([x, ty_in], dim=-1)).chunk(2, -1)
        return mu_y.squeeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``p(t|x,y)`` using the learned root embedding."""
        A = self._A()
        root_emb = self.score_net.root(x, A)
        logits = self.head_T(torch.cat([x, root_emb], dim=-1))
        return logits.softmax(dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Predict outcome mean for a fixed treatment value."""
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        return self.forward(x, t)


__all__ = ["DiffusionGNN_SCM"]
