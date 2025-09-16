from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ScoreBridge(nn.Module):
    """Score network predicting ``\nabla_y log q(y_\tau | x,t)``."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        hidden: int = 256,
        embed_dim: int = 64,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.t_embed = nn.Embedding(k, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.x_proj = nn.Linear(d_x, embed_dim)
        layers = []
        in_dim = embed_dim * 3 + d_y
        for _ in range(n_blocks):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.score_head = nn.Linear(hidden, d_y)

    def forward(
        self, y_noisy: torch.Tensor, x: torch.Tensor, t: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.t_embed(t)
        # Ensure tau is always 2D [batch_size, 1] for consistent processing
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        elif tau.dim() > 2:
            tau = tau.view(-1, 1)
        tau_emb = self.time_mlp(tau)
        h = torch.cat([y_noisy, self.x_proj(x), t_emb, tau_emb], dim=-1)
        h = self.trunk(h)
        return self.score_head(h)


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
        return self.net(torch.cat([x, y], dim=-1))


@register_model("bridge_diff")
class BridgeDiff(nn.Module):
    """Bridge-Diff Counterfactual model implementing a semi-supervised score loss."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        hidden: int = 256,
        timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float | None = None,
        k: int = 2,
        embed_dim: int = 64,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.d_y = d_y
        self.k = k
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        if sigma_max is None:
            # use the same default scale as other diffusion models
            sigma_max = 1.0
        self.sigma_max = sigma_max
        self.score_net = ScoreBridge(d_x, d_y, k, hidden, embed_dim, n_blocks)
        self.classifier = Classifier(d_x, d_y, k)

    # ----- utilities -----
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        # Clamp t to prevent extreme values and ensure numerical stability
        t_clamped = torch.clamp(t, 0.0, 1.0)
        ratio = self.sigma_max / self.sigma_min
        # Use log-space computation for better numerical stability
        log_sigma = math.log(self.sigma_min) + t_clamped * math.log(ratio)
        return torch.clamp(log_sigma.exp(), self.sigma_min, self.sigma_max)

    # ----- training objective -----
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        warmup: int = 10,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        u = torch.rand(b, device=device)
        tau = (
            u * (math.log(self.sigma_max) - math.log(self.sigma_min))
            + math.log(self.sigma_min)
        ).exp()
        sig = tau.unsqueeze(-1)
        eps = torch.randn_like(y)
        y_noisy = y + sig * eps

        logits = self.classifier(x, y)
        p_post = F.softmax(logits.detach(), dim=-1)

        obs_mask = t_obs != -1
        loss_obs = torch.tensor(0.0, device=device, requires_grad=True)
        if obs_mask.any():
            s_obs = self.score_net(
                y_noisy[obs_mask],
                x[obs_mask],
                t_obs[obs_mask].clamp_min(0),
                tau[obs_mask],
            )
            # Add small epsilon for numerical stability
            eps_val = 1e-8
            sig_obs = sig[obs_mask] + eps_val
            mse = (s_obs + eps[obs_mask] / sig_obs) ** 2
            loss_obs = (sig_obs ** 2 * mse).mean()

        unobs_mask = obs_mask.logical_not()
        loss_unobs = torch.tensor(0.0, device=device, requires_grad=True)
        if unobs_mask.any():
            # Add small epsilon for numerical stability
            eps_val = 1e-8
            sig_unobs = sig[unobs_mask] + eps_val
            mse_all = []
            for t_val in range(self.k):
                s_t = self.score_net(
                    y_noisy[unobs_mask],
                    x[unobs_mask],
                    torch.full_like(t_obs[unobs_mask], t_val),
                    tau[unobs_mask],
                )
                mse = ((s_t + eps[unobs_mask] / sig_unobs) ** 2).mean(dim=-1)
                mse_all.append(mse)
            mse_all = torch.stack(mse_all, dim=1)
            loss_unobs = (
                (p_post[unobs_mask] * (sig_unobs**2) * mse_all).sum(dim=1).mean()
            )

        # Apply warmup: disable unobserved loss during initial training epochs
        if current_epoch < warmup:
            loss_unobs = torch.tensor(0.0, device=device, requires_grad=True)

        ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if obs_mask.any():
            ce_loss = F.cross_entropy(logits[obs_mask], t_obs[obs_mask].clamp_min(0))

        return loss_obs + loss_unobs + ce_loss

    # ----- sampler -----
    @torch.no_grad()
    def paired_sample(
        self, x: torch.Tensor, n_steps: int = 50
    ) -> tuple[torch.Tensor, ...]:
        """Generate one draw ``y_t`` for each treatment ``t``."""
        b = x.size(0)
        device = x.device
        y = torch.randn(b, self.k, self.d_y, device=device) * self.sigma_max

        for step_idx in reversed(range(1, n_steps + 1)):
            tau = torch.full((b, 1), step_idx / n_steps, device=device)
            sig = self._sigma(tau)
            for t_val in range(self.k):
                score = self.score_net(
                    y[:, t_val],
                    x,
                    torch.full((b,), t_val, dtype=torch.long, device=device),
                    tau,
                )
                y[:, t_val] = y[:, t_val] + (sig**2) * score
            if step_idx > 1:
                prev = tau - 1 / n_steps
                sig_prev = self._sigma(prev)
                # Ensure noise scale is non-negative for numerical stability
                noise_scale = torch.clamp(sig**2 - sig_prev.pow(2), min=1e-8).sqrt()
                noise = torch.randn_like(y[:, 0])
                y = y + noise_scale.unsqueeze(-1) * noise.unsqueeze(1)
        return tuple(y[:, t] for t in range(self.k))

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
        n_steps: int = 50,
        n_samples: int = 32,
        return_mean: bool = True,
    ) -> torch.Tensor:
        """Generate ``n_samples`` draws and optionally average them."""

        draws = []
        for _ in range(n_samples):
            y_all = self.paired_sample(x, n_steps)
            draws.append(torch.stack(y_all, dim=1))  # [B, k, d_y]
        y_stack = torch.stack(draws, dim=0)  # [n_samples, B, k, d_y]
        t_long = t.view(-1, 1, 1).long()

        if return_mean:
            y_mean = y_stack.mean(dim=0)  # [B, k, d_y]
            return y_mean.gather(1, t_long.expand(-1, 1, self.d_y)).squeeze(1)

        # return all draws, selecting the desired treatment index
        y_stack = y_stack.transpose(0, 1)  # [B, n_samples, k, d_y]
        gathered = y_stack.gather(
            2, t_long.unsqueeze(1).expand(-1, n_samples, 1, self.d_y)
        )
        return gathered.squeeze(2)


__all__ = ["BridgeDiff"]
