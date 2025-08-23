from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from nflows.transforms import RandomPermutation
from itertools import chain


def sigma_schedule(
    tau: torch.Tensor, sigma_min: float, sigma_max: float
) -> torch.Tensor:
    r"""Noise schedule :math:`\sigma(\tau)` used for the diffusion prior."""
    return sigma_min * (sigma_max / sigma_min) ** tau


def _one_hot(t: torch.Tensor, k: int) -> torch.Tensor:
    """Convenience wrapper around ``F.one_hot`` returning ``float``."""
    return F.one_hot(t, k).float()


class Coupling(nn.Module):
    """
    Conditional coupling layer modelling p(y | x, z, t).
    x is passed as context and never transformed.
    """

    def __init__(self, d_x: int, d_y: int, d_z: int, k: int, hidden: int) -> None:
        super().__init__()
        self.d_y = d_y
        self.k = k
        self.SCALE_MAX = 0.5  # limit after Tanh to stabilise training
        self.scale = nn.Sequential(
            nn.Linear(d_x + d_z + k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_y),
            nn.Tanh(),
        )
        self.shift = nn.Sequential(
            nn.Linear(d_x + d_z + k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_y),
        )
        # Start close to the identity transform to avoid exploding outputs
        nn.init.zeros_(self.scale[2].weight)
        nn.init.zeros_(self.scale[2].bias)
        nn.init.zeros_(self.shift[2].weight)
        nn.init.zeros_(self.shift[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the coupling transformation.

        Parameters
        ----------
        x : Tensor of shape (B, d_x)
        y : Tensor of shape (B, d_y)
        z : Tensor of shape (B, d_z)
        t : Tensor of shape (B,)
        reverse : bool, optional
            Whether to apply the inverse transformation.

        Returns
        -------
        Tuple containing transformed ``y`` and per-sample log-determinant.
        """
        ctx = torch.cat([x, z, _one_hot(t, self.k)], dim=-1)
        # diagonal scale is bounded in [-SCALE_MAX, SCALE_MAX]
        s = self.SCALE_MAX * self.scale(ctx)
        t_shift = self.shift(ctx)
        if reverse:
            y_new = (y - t_shift) * torch.exp(-s)
            logdet = (-s).sum(-1)
        else:
            y_new = y * torch.exp(s) + t_shift
            # only the diagonal scaling contributes to the log-Jacobian
            logdet = s.sum(-1)
        return y_new, logdet


class CondFlow(nn.Module):
    """y-flow: p(y | x,z,t)"""

    def __init__(
        self, d_x: int, d_y: int, d_z: int, k: int, hidden: int, n_layers: int = 6
    ):
        super().__init__()
        layers = (
            (RandomPermutation(d_y), Coupling(d_x, d_y, d_z, k, hidden))
            for _ in range(n_layers)
        )
        self.transforms = nn.ModuleList(list(chain.from_iterable(layers)))
        self.register_buffer("base_mu", torch.zeros(d_y, requires_grad=False))

    def forward(
        self, y: torch.Tensor, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform ``y`` to latent space.

        Parameters
        ----------
        y : Tensor of shape (B, d_y)
        x : Tensor of shape (B, d_x)
        z : Tensor of shape (B, d_z)
        t : Tensor of shape (B,)

        Returns
        -------
        Tuple of latent representation and log-determinant.
        """
        logdet = 0.0
        h = y - self.base_mu
        for tr in self.transforms:
            if isinstance(tr, RandomPermutation):
                h, ld = tr(h)
            else:
                h, ld = tr(x, h, z, t, reverse=False)
            logdet = logdet + ld.view(h.size(0))
        return h, logdet

    def inverse(
        self, u: torch.Tensor, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform from latent ``u`` to output space."""
        logdet = 0.0
        h = u
        for tr in reversed(self.transforms):
            if isinstance(tr, RandomPermutation):
                h, ld = tr.inverse(h)
            else:
                h, ld = tr(x, h, z, t, reverse=True)
            logdet = logdet + ld.view(h.size(0))
        h = h + self.base_mu
        return h, logdet


class Encoder(nn.Module):
    """Gaussian encoder ``q(z|x,y)``."""

    def __init__(
        self, dim_xy: int, d_z: int, hidden: int, logv_clip: float = 5.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xy, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, d_z)
        self.logv = nn.Linear(hidden, d_z)
        self.logv_clip = logv_clip

    def forward(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log-variance for ``q(z|x,y)``.

        Parameters
        ----------
        xy : Tensor of shape (B, d_x + d_y)

        Returns
        -------
        (mu, logv) tensors of shape (B, d_z).
        """
        h = self.net(xy)
        return self.mu(h), self.logv(h).clamp(-self.logv_clip, self.logv_clip)


class ScoreNet(nn.Module):
    r"""Score network predicting ``\nabla_z log p(z|t)``."""

    def __init__(
        self, d_z: int, k: int, hidden: int = 128, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.t_emb = nn.Embedding(k, embed_dim)
        self.time = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d_z + embed_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_z),
        )

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate score at noise level ``tau``.

        Parameters
        ----------
        z : Tensor of shape (B, d_z)
        t : Tensor of shape (B,)
        tau : Tensor of shape (B, 1)
        """
        h = torch.cat([z, self.t_emb(t), self.time(tau)], dim=-1)
        return self.trunk(h)


class Classifier(nn.Module):
    """Classifier head estimating ``p(t|x,y)``."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return logits for ``p(t|x,y)``."""
        return self.net(torch.cat([x, y], dim=-1))


@register_model("lt_flow_diff")
class LTFlowDiff(nn.Module):
    """Latent-Treatment Flow Diffusion model."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        d_z: int = 4,
        hidden: int = 128,
        timesteps: int = 1000,
        sigma_min: float = 0.02,
        sigma_max: float = 1.0,
        k: int = 2,
        lambda_score: float = 0.1,
        logv_clip: float = 2.0,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.k = k
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.lambda_score = lambda_score
        self.warmup_steps = warmup_steps
        self.current_step = 0
        dim_xy = d_x + d_y
        self.encoder = Encoder(dim_xy, d_z, hidden, logv_clip)
        self.flow = CondFlow(d_x, d_y, d_z, k, hidden)
        self.score = ScoreNet(d_z, k, hidden)
        self.classifier = Classifier(d_x, d_y, k, hidden)

    # ----- utilities -----
    def _sigma(self, tau: torch.Tensor) -> torch.Tensor:
        return sigma_schedule(tau, self.sigma_min, self.sigma_max)

    def _sigma_with_warmup(self, tau: torch.Tensor) -> torch.Tensor:
        """Noise schedule with warmup to prevent early explosion."""
        base_sigma = sigma_schedule(tau, self.sigma_min, self.sigma_max)
        if self.current_step < self.warmup_steps:
            warmup_progress = self.current_step / self.warmup_steps
            warmup_sigma_min = (
                0.1 * (1 - warmup_progress) + self.sigma_min * warmup_progress
            )
            base_sigma = sigma_schedule(tau, warmup_sigma_min, self.sigma_max)
        return base_sigma

    # ----- training objective -----
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        step: int | None = None,
    ) -> torch.Tensor:
        """Training objective with clipped score targets and warmup noise schedule.

        Parameters
        ----------
        x : Tensor of shape (B, d_x)
        y : Tensor of shape (B, d_y)
        t_obs : LongTensor of shape (B,) with ``-1`` for missing treatments
        """
        assert t_obs.dtype == torch.long
        if step is not None:
            self.current_step = step
        device = x.device
        xy = torch.cat([x, y], dim=-1)
        mu, logv = self.encoder(xy)
        std = (0.5 * logv).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps

        obs_mask_orig = t_obs != -1
        t_used = t_obs.clone()
        idx_u = torch.tensor([], dtype=torch.long, device=device)
        probs_u = None
        if (~obs_mask_orig).any():
            idx_u = (~obs_mask_orig).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                pseudo_logits = self.classifier(x[idx_u], y[idx_u])
                probs_u = pseudo_logits.softmax(-1)
                probs_u = (probs_u**2) / probs_u.sum(dim=-1, keepdim=True)
            t_used[idx_u] = probs_u.argmax(dim=-1)
        u, logdet = self.flow(y, x, z, t_used.clamp_min(0))
        log_pxy = (
            -0.5 * u.pow(2).sum(-1) - 0.5 * u.size(1) * math.log(2 * math.pi) + logdet
        )

        b = x.size(0)
        t_idx = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = t_idx.float() / self.timesteps
        sig = self._sigma_with_warmup(tau).unsqueeze(-1)
        noise = torch.randn_like(z)
        z_tau = z + sig * noise

        mse_all = []
        for t_val in range(self.k):
            s_t = self.score(z_tau, torch.full_like(t_obs, t_val), tau.unsqueeze(-1))
            weight = sig.pow(2).squeeze()
            score_target = noise / sig
            score_target = torch.clamp(score_target, min=-10.0, max=10.0)
            mse_all.append(weight * ((s_t + score_target) ** 2).mean(dim=-1))
        mse_all = torch.stack(mse_all, dim=1)

        w = torch.full_like(mse_all, 1 / self.k)
        if obs_mask_orig.any():
            w[obs_mask_orig] = _one_hot(t_obs[obs_mask_orig], self.k)
        if idx_u.numel() > 0:
            w[idx_u] = probs_u
        score_loss = (w * mse_all).sum(dim=1).mean()

        ce_loss = torch.tensor(0.0, device=device)
        logits = self.classifier(x, y)
        if obs_mask_orig.any():
            ce_loss = F.cross_entropy(
                logits[obs_mask_orig], t_obs[obs_mask_orig].clamp_min(0)
            )
        if idx_u.numel() > 0:
            log_probs = F.log_softmax(logits[idx_u], dim=-1)
            # heuristic agreement penalty with pseudo-label distribution
            ce_loss = ce_loss + 0.5 * F.kl_div(
                log_probs, probs_u, reduction="batchmean"
            )

        recon = log_pxy.mean()
        # simple prior matching term
        reg = 1e-3 * (mu.pow(2) + (logv.exp() - 1).pow(2)).mean()
        return -recon + self.lambda_score * score_loss + ce_loss + reg

    # ----- sampler -----
    @torch.no_grad()
    def sample_z(
        self,
        t_val: int,
        batch_size: int,
        n_steps: int = 30,
        step_size: float = 0.02,
    ) -> torch.Tensor:
        z = torch.randn(batch_size, self.d_z, device=self.score.t_emb.weight.device)
        tau_start = 0.9 + 0.1 * torch.rand(batch_size, 1, device=z.device)
        base_schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=z.device)[
            :-1
        ].view(1, n_steps, 1)
        taus = tau_start.unsqueeze(1) * base_schedule
        noise_scale = math.sqrt(2 * step_size)
        t_long = torch.full((batch_size,), t_val, device=z.device, dtype=torch.long)
        for i in range(n_steps):
            tau_i = taus[:, i]
            grad = self.score(z, t_long, tau_i)
            z = z + 0.5 * step_size * grad + noise_scale * torch.randn_like(z)
        return z

    @torch.no_grad()
    def paired_sample(
        self,
        x: torch.Tensor,
        n_steps: int = 30,
        step_size: float = 0.02,
    ) -> tuple[torch.Tensor, ...]:
        b = x.size(0)
        y_all = []
        for t_val in range(self.k):
            z = self.sample_z(t_val, b, n_steps=n_steps, step_size=step_size)
            eps = torch.randn(b, self.d_y, device=x.device)
            y_t, _ = self.flow.inverse(
                eps, x, z, torch.full((b,), t_val, device=x.device)
            )
            y_all.append(y_t)
        return tuple(y_all)

    # ----- posterior utility -----
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` estimated by the classifier."""

        logits = self.classifier(x, y)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_outcome(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        n_steps: int = 30,
        step_size: float = 0.02,
    ) -> torch.Tensor:
        """Generate outcome predictions using the flow-based sampler."""

        y_all = self.paired_sample(x, n_steps=n_steps, step_size=step_size)
        y_stack = torch.stack(y_all, dim=1)
        t_long = t.view(-1, 1, 1).long()
        return y_stack.gather(1, t_long.expand(-1, 1, self.d_y)).squeeze(1)


__all__ = ["LTFlowDiff"]
