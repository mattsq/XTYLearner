"""Conditional Random Field models for treatment/outcome prediction."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


def _to_sequence(hidden: Sequence[int] | int | None) -> list[int]:
    if hidden is None:
        return []
    if isinstance(hidden, int):
        return [hidden]
    return list(hidden)


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _as_tensor(
    value: torch.Tensor | Iterable | float | int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


@register_model("crf")
class CRFModel(nn.Module):
    """Conditional Gaussian CRF with exact marginalisation over missing treatments."""

    def __init__(
        self,
        d_x: int,
        d_y: int = 1,
        k: int = 2,
        *,
        hidden_dims: Sequence[int] | int | None = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        eps: float = 1e-4,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if k is None or k < 2:
            raise ValueError("crf requires a discrete treatment with k>=2")
        if d_y < 1:
            raise ValueError("crf requires d_y >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")

        hidden = _to_sequence(hidden_dims)
        dims = [d_x, *hidden]
        self.g_T = make_mlp([*dims, k], activation=activation)
        self.g_prec = make_mlp([*dims, d_y], activation=activation)
        self.g_beta0 = make_mlp([*dims, d_y], activation=activation)
        self.g_W = make_mlp([*dims, d_y * k], activation=activation)

        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.eps = eps
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = _ensure_2d(x)
        logits_t = self.g_T(x) / self.temperature
        raw_prec = self.g_prec(x)
        prec = F.softplus(raw_prec) + self.eps
        beta0 = self.g_beta0(x)
        W = self.g_W(x).view(-1, self.k, self.d_y)
        return {
            "logits_t": logits_t,
            "prec": prec,
            "beta0": beta0,
            "W": W,
        }

    # ------------------------------------------------------------------
    def _mu_and_var(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        prec = outputs["prec"]
        var = 1.0 / prec
        beta0 = outputs["beta0"].unsqueeze(1)
        W = outputs["W"]
        mu_t = (beta0 + W) * var.unsqueeze(1)
        return mu_t, var

    # ------------------------------------------------------------------
    @staticmethod
    def _log_normal(
        y: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if mu.dim() == 2:
            diff2 = ((y - mu) ** 2 / var).sum(-1)
            logdet = torch.log(var).sum(-1)
        else:
            diff2 = ((y.unsqueeze(1) - mu) ** 2 / var.unsqueeze(1)).sum(-1)
            logdet = torch.log(var).sum(-1).unsqueeze(1)
        const = y.size(-1) * math.log(2 * math.pi)
        return -0.5 * (diff2 + logdet + const)

    # ------------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        outputs = self.forward(x)
        mu_t, var = self._mu_and_var(outputs)
        logits_t = outputs["logits_t"]
        log_probs_t = F.log_softmax(logits_t, dim=-1)

        labelled = t_obs >= 0
        unlabelled = ~labelled
        total = torch.tensor(0.0, device=device)
        n = x.size(0)

        if labelled.any():
            idx = labelled.nonzero(as_tuple=True)[0]
            t_lab = t_obs[idx].long()
            y_lab = y[idx]
            mu_lab = mu_t[idx, t_lab, :]
            var_lab = var[idx, :]
            log_cond = self._log_normal(y_lab, mu_lab, var_lab)
            if self.label_smoothing > 0:
                target = F.one_hot(t_lab, self.k).float()
                target = target * (1.0 - self.label_smoothing) + self.label_smoothing / self.k
                log_pt = (target * log_probs_t[idx]).sum(-1)
            else:
                log_pt = log_probs_t[idx, t_lab]
            total = total - (log_pt + log_cond).sum()

        if unlabelled.any():
            idx = unlabelled.nonzero(as_tuple=True)[0]
            y_unlab = y[idx]
            log_py = self._log_normal(y_unlab, mu_t[idx], var[idx])
            log_mix = torch.logsumexp(log_probs_t[idx] + log_py, dim=-1)
            total = total - log_mix.sum()

        if n == 0:
            return torch.tensor(0.0, device=device)
        return total / n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor | Iterable,
        t: torch.Tensor | Iterable | int | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        if t is None:
            return self.predict_y(x_tensor)
        return self.predict_outcome(x_tensor, t)

    # ------------------------------------------------------------------
    def _parse_treatment(
        self,
        t: torch.Tensor | Iterable | int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            t = t.to(device)
            if t.dim() == 0:
                return t.long().expand(batch)
            if t.dim() == 1:
                return t.long()
            if t.dim() == 2 and t.size(1) == self.k:
                return t.argmax(dim=1).long()
        elif isinstance(t, int):
            return torch.full((batch,), t, dtype=torch.long, device=device)
        elif isinstance(t, float):
            return torch.full((batch,), int(t), dtype=torch.long, device=device)
        else:
            t_tensor = torch.as_tensor(t, device=device)
            if t_tensor.dim() == 0:
                return t_tensor.long().expand(batch)
            if t_tensor.dim() == 1:
                return t_tensor.long()
            if t_tensor.dim() == 2 and t_tensor.size(1) == self.k:
                return t_tensor.argmax(dim=1).long()
        raise ValueError("Unsupported treatment format")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(
        self,
        x: torch.Tensor | Iterable,
        t: torch.Tensor | Iterable | int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        batch = x_tensor.size(0)
        t_idx = self._parse_treatment(t, batch, device)
        outputs = self.forward(x_tensor)
        mu_t, _ = self._mu_and_var(outputs)
        gathered = mu_t.gather(
            1, t_idx.view(-1, 1, 1).expand(-1, 1, self.d_y)
        ).squeeze(1)
        return gathered

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_y(self, x: torch.Tensor | Iterable) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        outputs = self.forward(x_tensor)
        mu_t, _ = self._mu_and_var(outputs)
        probs = F.softmax(outputs["logits_t"], dim=-1).unsqueeze(-1)
        return (probs * mu_t).sum(dim=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(
        self,
        x: torch.Tensor | Iterable,
        y: torch.Tensor | Iterable | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        y_tensor: torch.Tensor | None = None
        if y is None and x_tensor.size(1) == self.d_x + self.d_y:
            y_tensor = x_tensor[:, self.d_x :]
            x_tensor = x_tensor[:, : self.d_x]
        elif y is not None:
            y_tensor = _as_tensor(y, device=device, dtype=torch.float32)
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(-1)
        outputs = self.forward(x_tensor)
        log_probs = F.log_softmax(outputs["logits_t"], dim=-1)
        if y_tensor is None:
            return log_probs.exp()
        mu_t, var = self._mu_and_var(outputs)
        log_py = self._log_normal(y_tensor, mu_t, var)
        post = torch.softmax(log_probs + log_py, dim=-1)
        return post

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_t_given_y(
        self,
        x: torch.Tensor | Iterable,
        y: torch.Tensor | Iterable,
    ) -> torch.Tensor:
        return self.predict_treatment_proba(x, y)


@register_model("crf_discrete")
class CRFDiscreteModel(nn.Module):
    """Discrete CRF over ``(T, Y_b)`` with exact normalisation."""

    def __init__(
        self,
        d_x: int,
        d_y: int = 1,
        k: int = 2,
        *,
        hidden_dims: Sequence[int] | int | None = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        n_bins: int = 8,
        bin_edges: Sequence[float] | torch.Tensor | None = None,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        binning: str = "quantile",
    ) -> None:
        super().__init__()
        if k is None or k < 2:
            raise ValueError("crf_discrete requires k>=2")
        if n_bins < 2:
            raise ValueError("crf_discrete requires at least two bins")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0,1)")

        hidden = _to_sequence(hidden_dims)
        dims = [d_x, *hidden]
        self.node_t = make_mlp([*dims, k], activation=activation)
        self.node_y = make_mlp([*dims, n_bins], activation=activation)
        self.node_ty = make_mlp([*dims, k * n_bins], activation=activation)

        self.d_x = d_x
        self.k = k
        self.n_bins = n_bins
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.binning = binning

        edges_tensor: torch.Tensor
        values_tensor: torch.Tensor
        if bin_edges is not None:
            edges_tensor = torch.as_tensor(bin_edges, dtype=torch.float32)
            if edges_tensor.numel() != n_bins + 1:
                raise ValueError("bin_edges must contain n_bins + 1 values")
            values_tensor = 0.5 * (edges_tensor[:-1] + edges_tensor[1:])
        else:
            edges_tensor = torch.empty(0)
            values_tensor = torch.empty(0)
        self.register_buffer("_bin_edges", edges_tensor)
        self.register_buffer("_bin_values", values_tensor)

    # ------------------------------------------------------------------
    def _ensure_bin_edges(self, y: torch.Tensor) -> None:
        if self._bin_edges.numel() == self.n_bins + 1:
            return
        y_flat = y.detach().flatten().float()
        if self.binning == "quantile" and y_flat.numel() >= self.n_bins + 1:
            quantiles = torch.linspace(0, 1, self.n_bins + 1, device=y_flat.device)
            edges = torch.quantile(y_flat, quantiles)
        else:
            y_min = y_flat.min()
            y_max = y_flat.max()
            if y_min == y_max:
                y_min = y_min - 0.5
                y_max = y_max + 0.5
            edges = torch.linspace(y_min, y_max, self.n_bins + 1, device=y_flat.device)
        edges = edges.to(y.device)
        edges[0] = edges[0] - 1e-6
        edges[-1] = edges[-1] + 1e-6
        for i in range(1, edges.numel()):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        values = 0.5 * (edges[:-1] + edges[1:])
        self._bin_edges = edges
        self._bin_values = values

    # ------------------------------------------------------------------
    def _bin_outcomes(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if y.size(-1) != 1:
            raise ValueError("crf_discrete currently supports scalar outcomes")
        self._ensure_bin_edges(y)
        boundaries = self._bin_edges[1:-1]
        bins = torch.bucketize(y.flatten(), boundaries).view(y.shape[:-1])
        return bins.long()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = _ensure_2d(x)
        logits_t = self.node_t(x) / self.temperature
        phi_y = self.node_y(x)
        phi_ty = self.node_ty(x).view(-1, self.k, self.n_bins)
        return {
            "logits_t": logits_t,
            "phi_y": phi_y,
            "phi_ty": phi_ty,
        }

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        device = x.device
        outputs = self.forward(x)
        logits_t = outputs["logits_t"]
        phi_y = outputs["phi_y"]
        phi_ty = outputs["phi_ty"]

        log_probs_t = F.log_softmax(logits_t, dim=-1)
        score = logits_t.unsqueeze(2) + phi_y.unsqueeze(1) + phi_ty
        logZ = torch.logsumexp(score, dim=(1, 2))

        y_bins = self._bin_outcomes(y)
        if y_bins.dim() == 1:
            y_bins = y_bins.unsqueeze(-1)

        labelled = t_obs >= 0
        unlabelled = ~labelled
        total = torch.tensor(0.0, device=device)
        n = x.size(0)

        if labelled.any():
            idx = labelled.nonzero(as_tuple=True)[0]
            t_lab = t_obs[idx].long()
            y_lab = y_bins[idx, 0]
            joint_scores = score[idx, t_lab, y_lab] - logZ[idx]
            log_probs_lab = log_probs_t[idx, t_lab]
            log_cond = joint_scores - log_probs_lab
            if self.label_smoothing > 0:
                target = F.one_hot(t_lab, self.k).float()
                target = target * (1.0 - self.label_smoothing) + self.label_smoothing / self.k
                log_pt = (target * log_probs_t[idx]).sum(-1)
            else:
                log_pt = log_probs_lab
            total = total - (log_pt + log_cond).sum()

        if unlabelled.any():
            idx = unlabelled.nonzero(as_tuple=True)[0]
            y_unlab = y_bins[idx, 0]
            joint_scores = score[idx, :, :].gather(
                2, y_unlab.view(-1, 1, 1).expand(-1, self.k, 1)
            ).squeeze(-1)
            log_joint = joint_scores - logZ[idx].unsqueeze(1)
            log_mix = torch.logsumexp(log_joint, dim=1)
            total = total - log_mix.sum()

        if n == 0:
            return torch.tensor(0.0, device=device)
        return total / n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor | Iterable,
        t: torch.Tensor | Iterable | int | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        if t is None:
            return self.predict_y(x_tensor)
        return self.predict_outcome(x_tensor, t)

    # ------------------------------------------------------------------
    def _parse_treatment(
        self,
        t: torch.Tensor | Iterable | int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            t = t.to(device)
            if t.dim() == 0:
                return t.long().expand(batch)
            if t.dim() == 1:
                return t.long()
            if t.dim() == 2 and t.size(1) == self.k:
                return t.argmax(dim=1).long()
        elif isinstance(t, int):
            return torch.full((batch,), t, dtype=torch.long, device=device)
        elif isinstance(t, float):
            return torch.full((batch,), int(t), dtype=torch.long, device=device)
        else:
            t_tensor = torch.as_tensor(t, device=device)
            if t_tensor.dim() == 0:
                return t_tensor.long().expand(batch)
            if t_tensor.dim() == 1:
                return t_tensor.long()
            if t_tensor.dim() == 2 and t_tensor.size(1) == self.k:
                return t_tensor.argmax(dim=1).long()
        raise ValueError("Unsupported treatment format")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _p_y_given_t(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        phi_y = outputs["phi_y"].unsqueeze(1)
        phi_ty = outputs["phi_ty"]
        logits = phi_y + phi_ty
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(
        self,
        x: torch.Tensor | Iterable,
        t: torch.Tensor | Iterable | int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        outputs = self.forward(x_tensor)
        probs_y = self._p_y_given_t(outputs)
        batch = x_tensor.size(0)
        t_idx = self._parse_treatment(t, batch, device)
        preds = probs_y.gather(1, t_idx.view(-1, 1, 1).expand(-1, 1, self.n_bins)).squeeze(1)
        if self._bin_values.numel() == self.n_bins:
            values = self._bin_values.to(device)
        else:
            values = torch.arange(self.n_bins, device=device, dtype=preds.dtype)
        return (preds * values).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_y(self, x: torch.Tensor | Iterable) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        outputs = self.forward(x_tensor)
        probs_t = torch.softmax(outputs["logits_t"], dim=-1).unsqueeze(-1)
        probs_y = self._p_y_given_t(outputs)
        mixture = (probs_t * probs_y).sum(dim=1)
        if self._bin_values.numel() == self.n_bins:
            values = self._bin_values.to(device)
        else:
            values = torch.arange(self.n_bins, device=device, dtype=mixture.dtype)
        return (mixture * values).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(
        self,
        x: torch.Tensor | Iterable,
        y: torch.Tensor | Iterable | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x_tensor = _as_tensor(x, device=device, dtype=torch.float32)
        x_tensor = _ensure_2d(x_tensor)
        y_tensor: torch.Tensor | None = None
        if y is None and x_tensor.size(1) == self.d_x + 1:
            y_tensor = x_tensor[:, self.d_x :]
            x_tensor = x_tensor[:, : self.d_x]
        elif y is not None:
            y_tensor = _as_tensor(y, device=device, dtype=torch.float32)
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(-1)
        outputs = self.forward(x_tensor)
        log_probs_t = F.log_softmax(outputs["logits_t"], dim=-1)
        if y_tensor is None:
            return log_probs_t.exp()
        y_bins = self._bin_outcomes(y_tensor)
        score = outputs["logits_t"].unsqueeze(2) + outputs["phi_y"].unsqueeze(1) + outputs["phi_ty"]
        joint_scores = score.gather(
            2, y_bins.view(-1, 1, 1).expand(-1, self.k, 1)
        ).squeeze(-1)
        logZ = torch.logsumexp(score, dim=(1, 2))
        log_joint = joint_scores - logZ.unsqueeze(1)
        return torch.softmax(log_joint, dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_t_given_y(
        self,
        x: torch.Tensor | Iterable,
        y: torch.Tensor | Iterable,
    ) -> torch.Tensor:
        return self.predict_treatment_proba(x, y)


__all__ = ["CRFModel", "CRFDiscreteModel"]

