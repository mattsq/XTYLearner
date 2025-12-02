"""Adversarial flow models for conditional outcome generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


@dataclass
class OTSchedule:
    """Linear schedule for the OT penalty weight."""

    start: float = 1.0
    end: float = 0.1
    steps: int = 1_000

    def weight(self, step: int) -> float:
        if self.steps <= 0:
            return self.end
        alpha = min(max(step, 0), self.steps) / float(self.steps)
        return (1 - alpha) * self.start + alpha * self.end


class GradientScaler:
    """EMA-based gradient normalisation helper used by AFM."""

    def __init__(self, n_dim: int, beta2: float = 0.999) -> None:
        self.beta2 = beta2
        self.ema_sq: torch.Tensor | None = None
        self.n_dim = n_dim

    def scale(self, grad: torch.Tensor) -> torch.Tensor:
        g2 = grad.pow(2).mean()
        if self.ema_sq is None:
            self.ema_sq = g2.detach()
        else:
            self.ema_sq = self.beta2 * self.ema_sq + (1 - self.beta2) * g2.detach()
        scale = (1.0 / (self.ema_sq.sqrt() + 1e-8)) * (1.0 / (self.n_dim**0.5))
        return grad * scale


@register_model("af_outcome")
class AFOutcomeModel(nn.Module):
    """Conditional adversarial flow model for :math:`p(Y|X,T)`.

    The generator transports Gaussian noise for the outcome block while the
    discriminator distinguishes real vs. generated outcomes conditioned on
    covariates and treatments.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        k: int | None = None,
        hidden_dims: Iterable[int] = (256, 256),
        lambda_gp: float = 1.0,
        lambda_cp: float = 1e-3,
        lambda_ot: float = 1.0,
        ot_schedule: OTSchedule | None = None,
        multi_step: bool = False,
    ) -> None:
        super().__init__()
        if multi_step:
            raise NotImplementedError("Multi-step AFM training is not implemented yet.")
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.lambda_gp = lambda_gp
        self.lambda_cp = lambda_cp
        self.lambda_ot = lambda_ot
        self.ot_schedule = ot_schedule
        self.grad_scaler = GradientScaler(d_y)
        self.step_count = 0

        self.t_dim = 1 if k is None else k
        gen_in = d_y + d_x + self.t_dim
        disc_in = d_x + self.t_dim + d_y

        dims_g = [gen_in, *hidden_dims, d_y]
        dims_d = [disc_in, *hidden_dims, 1]
        self.generator = make_mlp(dims_g, residual=True)
        self.discriminator = make_mlp(dims_d, residual=True)

    # --------------------------------------------------------------
    def _embed_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.k is None:
            if t.dim() == 1:
                return t.unsqueeze(-1)
            return t
        t_int = t.to(torch.long)
        mask = t_int < 0
        t_onehot = F.one_hot(t_int.clamp_min(0), self.k).float()
        t_onehot[mask] = 0.0
        return t_onehot

    def _concat_features(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        t_feat = self._embed_t(t)
        return torch.cat([x, t_feat, y], dim=-1)

    def _current_lambda_ot(self) -> float:
        if self.ot_schedule is None:
            return self.lambda_ot
        weight = self.ot_schedule.weight(self.step_count)
        return float(weight)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.size(0), self.d_y, device=x.device, dtype=x.dtype)
        return self._generate(z, x, t)

    def _generate(self, z: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_feat = self._embed_t(t)
        gen_in = torch.cat([z, x, t_feat], dim=-1)
        return self.generator(gen_in)

    # --------------------------------------------------------------
    def _relativistic_hinge(
        self, real_scores: torch.Tensor, fake_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diff_real = real_scores - fake_scores
        diff_fake = fake_scores - real_scores
        loss_d = F.relu(1.0 - diff_real).mean() + F.relu(1.0 + diff_fake).mean()
        loss_g = F.relu(1.0 + diff_real).mean() + F.relu(1.0 - diff_fake).mean()
        return loss_g, loss_d

    def _gradient_penalty(self, scores: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad.pow(2).view(inputs.size(0), -1).sum(dim=1).mean()

    # --------------------------------------------------------------
    def loss_G(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.step_count += 1
        z = torch.randn(x.size(0), self.d_y, device=x.device, dtype=x.dtype)
        y_fake = self._generate(z, x, t_obs)
        if not y_fake.requires_grad:
            y_fake = y_fake.requires_grad_(True)
        y_fake_adv = y_fake.clone()
        y_fake_adv.register_hook(self.grad_scaler.scale)

        disc_fake = self.discriminator(self._concat_features(x, t_obs, y_fake_adv))
        with torch.no_grad():
            disc_real = self.discriminator(self._concat_features(x, t_obs, y))
        loss_g_adv, _ = self._relativistic_hinge(disc_real, disc_fake)

        lambda_ot = self._current_lambda_ot()
        loss_ot = (y_fake - z).pow(2).mean() / float(self.d_y)
        loss_total = loss_g_adv + lambda_ot * loss_ot
        return {
            "loss_G": loss_total,
            "loss_G_adv": loss_g_adv.detach(),
            "loss_ot": loss_ot.detach(),
            "lambda_ot": torch.tensor(lambda_ot, device=x.device),
        }

    def loss_D(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z = torch.randn(x.size(0), self.d_y, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            y_fake = self._generate(z, x, t_obs)

        real_in = self._concat_features(x, t_obs, y).detach().requires_grad_(True)
        fake_in = self._concat_features(x, t_obs, y_fake).detach().requires_grad_(True)

        disc_real = self.discriminator(real_in)
        disc_fake = self.discriminator(fake_in)
        _, loss_d_adv = self._relativistic_hinge(disc_real, disc_fake)

        loss_r1 = self._gradient_penalty(disc_real, real_in)
        loss_r2 = self._gradient_penalty(disc_fake, fake_in)
        logits_mean = torch.cat([disc_real, disc_fake], dim=0).mean()
        loss_cp = logits_mean.pow(2)

        loss_d = (
            loss_d_adv
            + self.lambda_gp * (loss_r1 + loss_r2)
            + self.lambda_cp * loss_cp
        )
        return {
            "loss_D": loss_d,
            "loss_D_adv": loss_d_adv.detach(),
            "loss_r1": loss_r1.detach(),
            "loss_r2": loss_r2.detach(),
            "loss_cp": loss_cp.detach(),
        }

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        if n_samples <= 1:
            z = torch.zeros(x.size(0), self.d_y, device=x.device, dtype=x.dtype)
            return self._generate(z, x, t)
        preds = []
        for _ in range(n_samples):
            z = torch.randn(x.size(0), self.d_y, device=x.device, dtype=x.dtype)
            preds.append(self._generate(z, x, t))
        return torch.stack(preds, dim=0).mean(dim=0)

    def generator_parameters(self):
        yield from self.generator.parameters()

    def discriminator_parameters(self):
        yield from self.discriminator.parameters()

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, *args) -> torch.Tensor:
        """Return a simple categorical prior over treatments.

        This model does not learn a treatment classifier, but the training
        pipelines expect a ``p(t\mid x,y)`` interface. We therefore provide a
        uniform distribution across the configured ``k`` classes to satisfy
        evaluation hooks without raising, regardless of whether inputs are
        provided as separate ``(x, y)`` tensors or a concatenated ``[x, y]``
        array.
        """

        if len(args) == 1:
            xy = torch.as_tensor(args[0])
            x = xy[..., : self.d_x]
            y = xy[..., -self.d_y :]
        elif len(args) == 2:
            x, y = (torch.as_tensor(a) for a in args)
        else:
            raise ValueError("predict_treatment_proba expects (x, y) or (xy,)")

        batch = x.shape[0]
        num_classes = self.k if self.k is not None else 1
        probs = torch.full((batch, num_classes), 1.0 / num_classes, device=x.device)
        return probs


__all__ = ["AFOutcomeModel", "OTSchedule", "GradientScaler"]
