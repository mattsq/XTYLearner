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
        if k is None:
            raise ValueError(
                "BridgeDiff requires discrete treatments (k must be an integer >= 2). "
                "For continuous treatments, use a different model such as cycle_dual or mean_teacher."
            )
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
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
    def _sigma(self, tau: torch.Tensor) -> torch.Tensor:
        """Noise scale for a diffusion time ``tau`` in ``[0, 1]``."""

        tau = torch.clamp(tau, 0.0, 1.0)
        ratio = self.sigma_max / self.sigma_min
        log_sigma = math.log(self.sigma_min) + tau * math.log(ratio)
        sigma = log_sigma.exp()
        return torch.clamp(sigma, self.sigma_min, self.sigma_max)

    # ----- training objective -----
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        warmup: int = 0,
        current_epoch: int | None = None,
    ) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        tau = torch.rand(b, device=device)
        sigma = self._sigma(tau).unsqueeze(-1)
        eps = torch.randn_like(y)
        y_noisy = y + sigma * eps

        logits = self.classifier(x, y)
        p_post = F.softmax(logits, dim=-1)

        obs_mask = t_obs != -1
        loss_obs = y.new_zeros(())
        eps_stab = 1e-6

        if obs_mask.any():
            s_obs = self.score_net(
                y_noisy[obs_mask],
                x[obs_mask],
                t_obs[obs_mask].clamp_min(0),
                tau[obs_mask],
            )
            sig_obs = sigma[obs_mask].clamp_min(eps_stab)
            inv_sig = sig_obs.reciprocal()
            mse = (s_obs + eps[obs_mask] * inv_sig) ** 2
            loss_obs = (sig_obs**2 * mse).mean()

        unobs_mask = obs_mask.logical_not()
        loss_unobs = y.new_zeros(())
        if unobs_mask.any():
            sig_unobs = sigma[unobs_mask].clamp_min(eps_stab)
            eps_unobs = eps[unobs_mask]
            tau_unobs = tau[unobs_mask]
            x_unobs = x[unobs_mask]
            y_unobs = y_noisy[unobs_mask]

            x_rep = x_unobs.repeat_interleave(self.k, dim=0)
            y_rep = y_unobs.repeat_interleave(self.k, dim=0)
            tau_rep = tau_unobs.repeat_interleave(self.k, dim=0)
            t_vals = torch.arange(self.k, device=device).repeat(x_unobs.size(0))

            scores = self.score_net(y_rep, x_rep, t_vals, tau_rep)
            scores = scores.view(x_unobs.size(0), self.k, self.d_y)

            inv_sig = sig_unobs.reciprocal().unsqueeze(1)
            eps_term = eps_unobs.unsqueeze(1) * inv_sig
            mse = (scores + eps_term) ** 2
            mse = mse.mean(dim=-1)

            weight = (sig_unobs.squeeze(-1).unsqueeze(1) ** 2)
            loss_unobs = (p_post[unobs_mask] * weight * mse).sum(dim=1).mean()

        # Apply warmup: disable unobserved loss during initial training epochs
        if current_epoch is not None and current_epoch < warmup:
            loss_unobs = loss_unobs.detach()
            loss_unobs = y.new_zeros(())

        ce_loss = y.new_zeros(())
        if obs_mask.any():
            ce_loss = F.cross_entropy(logits[obs_mask], t_obs[obs_mask].clamp_min(0))

        return loss_obs + loss_unobs + ce_loss

    # ----- sampler -----
    @torch.no_grad()
    def paired_sample(
        self,
        x: torch.Tensor,
        n_steps: int | None = None,
        *,
        n_samples: int = 1,
        return_tensor: bool = False,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Generate one or more draws ``y_t`` for each treatment ``t``."""

        b = x.size(0)
        device = x.device
        if n_steps is None:
            n_steps = min(self.timesteps, 50)
        n_draws = max(1, n_samples)

        y = torch.randn(n_draws, b, self.k, self.d_y, device=device) * self.sigma_max

        x_rep = x.repeat_interleave(self.k, dim=0)
        t_vals = torch.arange(self.k, device=device).repeat(b)
        if n_draws > 1:
            x_rep = x_rep.repeat(n_draws, 1)
            t_vals = t_vals.repeat(n_draws)

        for step_idx in reversed(range(1, n_steps + 1)):
            tau_val = step_idx / n_steps
            tau_tensor = torch.full(
                (y.shape[0] * b * self.k, 1), tau_val, device=device
            )

            y_flat = y.view(-1, self.d_y)
            score = self.score_net(y_flat, x_rep, t_vals, tau_tensor)
            score = score.view_as(y)

            sigma = self._sigma(torch.tensor([tau_val], device=device))[0]
            y = y + (sigma**2) * score

            if step_idx > 1:
                prev_tau = (step_idx - 1) / n_steps
                sig_prev = self._sigma(torch.tensor([prev_tau], device=device))[0]
                noise_scale = math.sqrt(max((sigma**2 - sig_prev**2).item(), 1e-8))
                noise = torch.randn_like(y)
                y = y + noise_scale * noise

        if return_tensor or n_draws > 1:
            return y if n_draws > 1 else y.squeeze(0)

        y_single = y.squeeze(0)
        return tuple(y_single[:, t] for t in range(self.k))

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
        n_steps: int | None = None,
        n_samples: int = 16,
        return_mean: bool = True,
    ) -> torch.Tensor:
        """Generate ``n_samples`` draws and optionally average them."""

        samples = self.paired_sample(
            x, n_steps=n_steps, n_samples=n_samples, return_tensor=True
        )
        if samples.dim() == 3:
            y_stack = samples.unsqueeze(0)
        else:
            y_stack = samples
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
