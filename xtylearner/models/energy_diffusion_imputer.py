"""Energy-guided Discrete Diffusion Imputer (EG-DDI)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


class ScoreNet(nn.Module):
    """Score network predicting logits for the reverse diffusion."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.t_embed = nn.Embedding(k, hidden)
        self.x_proj = nn.Linear(d_x, hidden)
        self.y_proj = nn.Linear(d_y, hidden)
        self.time_fc = make_mlp([1, hidden, hidden], activation=nn.SiLU)
        self.trunk = make_mlp([hidden * 4, hidden, k], activation=nn.SiLU)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_corrupt: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat(
            [
                self.x_proj(x),
                self.y_proj(y),
                self.t_embed(t_corrupt),
                self.time_fc(tau),
            ],
            dim=-1,
        )
        return self.trunk(h)


class EnergyNet(nn.Module):
    """Simple energy model over ``(x,y,t)``."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_y, hidden, hidden, k],
            activation=nn.ReLU,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


@register_model("eg_ddi")
class EnergyDiffusionImputer(nn.Module):
    """Energy-guided Discrete Diffusion Imputer."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        timesteps: int = 1000,
        hidden: int = 128,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.timesteps = timesteps
        self.score_net = ScoreNet(d_x, d_y, k, hidden)
        self.energy_net = EnergyNet(d_x, d_y, k, hidden)

    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        steps: int = 20,
        lr: float = 0.1,
        reg: float = 0.01,
        noise: float = 0.0,
        score_weight: float = 1.0,
    ) -> torch.Tensor:
        """Approximate ``p(y|x,t)`` via gradient-based energy minimisation.

        ``score_weight`` controls how strongly the diffusion score network guides
        the optimisation.  ``noise`` can be used to perform Langevin-style
        updates.
        """

        y = torch.zeros(x.size(0), self.d_y, device=x.device, requires_grad=True)
        target_t = t.view(-1, 1).long()
        # Clamp for gather operation, but preserve original t for validation
        target_t_clamped = target_t.clamp_min(0)
        with torch.enable_grad():
            for _ in range(steps):
                e_all = self.energy_net(x, y)
                e = e_all.gather(1, target_t_clamped)
                reg_term = reg * (y ** 2).sum(dim=1, keepdim=True)
                total = e + reg_term

                if score_weight > 0.0:
                    valid = t >= 0
                    if valid.any():
                        tau = target_t_clamped.to(dtype=y.dtype) / self.timesteps
                        tau = tau.clamp_min(1e-6)  # More stable minimum bound
                        logits = self.score_net(x, y, target_t_clamped.squeeze(1), tau)
                        per_elem = F.cross_entropy(
                            logits, target_t_clamped.squeeze(1), reduction="none"
                        ).unsqueeze(1)
                        mask = valid.view(-1, 1).to(per_elem.dtype)
                        total = total + score_weight * (per_elem * mask)

                grad = torch.autograd.grad(total.sum(), y, create_graph=False)[0]
                if grad.norm() < 1e-3:
                    break
                y = (y - lr * grad).detach()
                if noise > 0.0:
                    y += noise * torch.randn_like(y)
                y.requires_grad_(True)
        return y.detach()

    # --------------------------------------------------------------
    def predict_outcome(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        steps: int = 20,
        lr: float = 0.1,
        score_weight: float = 1.0,
    ) -> torch.Tensor:
        """Convenience wrapper around :meth:`forward`.

        This method matches the interface expected by ``BaseTrainer`` for
        computing outcome metrics during training.
        """

        t = t.to(torch.long)
        return self.forward(
            x, t, steps=steps, lr=lr, score_weight=score_weight
        )

    # ----- diffusion utilities -----
    def _gamma(self, k: torch.Tensor) -> torch.Tensor:
        return k.float() / self.timesteps

    def _q_sample(self, t0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Corrupt ``t0`` at step ``k`` by randomly replacing with random labels."""
        gam = self._gamma(k).clamp(0.0, 1.0)
        flip = torch.bernoulli(gam).to(torch.bool)
        rand = torch.randint(0, self.k, t0.shape, device=t0.device)
        return torch.where(flip, rand, t0)

    # ----- training objective -----
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        *,
        w_obs: float = 1.0,
        w_unobs: float = 1.0,
        w_align: float = 1.0,
        w_energy_obs: float = 1.0,
        w_energy_recon: float = 1.0,
        contrastive_noise_scale: float = 0.1,
    ) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        t_idx = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = t_idx.float().unsqueeze(-1) / self.timesteps

        t_clean = t_obs.clone()
        missing = t_obs == -1
        if missing.any():
            t_clean[missing] = torch.randint(0, self.k, (missing.sum(),), device=device)
        t_corrupt = self._q_sample(t_clean, t_idx)

        logits = self.score_net(x, y, t_corrupt, tau)

        obs_mask = t_obs != -1
        loss_obs = torch.tensor(0.0, device=device)
        if obs_mask.any():
            loss_obs = F.cross_entropy(logits[obs_mask], t_obs[obs_mask])

        loss_unobs = torch.tensor(0.0, device=device)
        if missing.any():
            logits_m = logits[missing]
            log_probs = F.log_softmax(logits_m, dim=-1)
            E = self.energy_net(x[missing], y[missing])
            w = F.softmax(-E, dim=-1)
            loss_unobs = (w * (-log_probs)).sum(dim=-1).mean()

        loss_energy_obs = torch.tensor(0.0, device=device)
        loss_energy_recon = torch.tensor(0.0, device=device)
        if obs_mask.any():
            x_obs = x[obs_mask]
            y_obs = y[obs_mask]
            t_obs_idx = t_obs[obs_mask]

            energy_obs = self.energy_net(x_obs, y_obs)
            loss_energy_obs = F.cross_entropy(-energy_obs, t_obs_idx)

            # Jitter observed outcomes to construct contrastive negatives for the
            # energy network.
            noise = torch.randn_like(y_obs)
            # Scale noise relative to outcome standard deviation for better contrastives
            y_std = y_obs.std(dim=0, keepdim=True).clamp_min(1e-6)
            y_neg = y_obs + contrastive_noise_scale * y_std * noise
            energy_pos = energy_obs.gather(1, t_obs_idx.view(-1, 1))
            energy_neg_all = self.energy_net(x_obs, y_neg)
            energy_neg = energy_neg_all.gather(1, t_obs_idx.view(-1, 1))
            loss_energy_recon = F.softplus(energy_pos - energy_neg).mean()

        E_all = self.energy_net(x, y)
        log_p_score = F.log_softmax(logits, dim=-1)
        log_p_energy = F.log_softmax(-E_all, dim=-1)
        # KL(energy || score) to align score network with energy guidance
        kl_div = F.kl_div(log_p_energy, log_p_score.exp(), reduction="batchmean")

        total = (
            w_obs * loss_obs
            + w_unobs * loss_unobs
            + w_align * kl_div
            + w_energy_obs * loss_energy_obs
            + w_energy_recon * loss_energy_recon
        )
        return total

    # ----- simple sampler -----
    @torch.no_grad()
    def sample_T(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int = 30,
        lambda_energy: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw labels ``t`` and return their final probabilities.

        ``lambda_energy`` controls the influence of the energy guidance during
        sampling.
        """

        b = x.size(0)
        t = torch.randint(0, self.k, (b,), device=x.device)
        probs = None
        tau_schedule = torch.linspace(
            float(self.timesteps), 1.0, steps, device=x.device
        )
        tau_schedule = tau_schedule / self.timesteps
        for tau_scalar in tau_schedule:
            tau = tau_scalar.view(1, 1).repeat(b, 1)
            logits = self.score_net(x, y, t, tau)
            energy = self.energy_net(x, y)
            guided = logits - lambda_energy * energy
            probs = F.softmax(guided, dim=-1)
            t = torch.multinomial(probs, 1).squeeze(-1)
        return t, probs

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor, steps: int = 30
    ) -> torch.Tensor:
        """Posterior ``p(t|x,y)`` estimated by the sampler."""

        _, probs = self.sample_T(x, y, steps)
        return probs


__all__ = ["EnergyDiffusionImputer"]
