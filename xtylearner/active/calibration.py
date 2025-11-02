"""Utilities for conformal calibration of potential outcome models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import predict_outcome

logger = logging.getLogger(__name__)


@dataclass
class ConformalCalibrator:
    """Hold conformal residual quantiles for binary treatment arms."""

    q_lo_t0: torch.Tensor
    q_hi_t0: torch.Tensor
    q_lo_t1: torch.Tensor
    q_hi_t1: torch.Tensor

    def interval_for_outcome(
        self, y_pred: torch.Tensor, t_arm: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a calibrated interval for ``y_pred`` under treatment ``t_arm``."""

        if t_arm not in (0, 1):
            raise ValueError(f"ConformalCalibrator only supports binary treatments; got {t_arm}.")

        q_lo, q_hi = (
            (self.q_lo_t0, self.q_hi_t0) if t_arm == 0 else (self.q_lo_t1, self.q_hi_t1)
        )

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.as_tensor(y_pred, dtype=torch.float32)

        device = y_pred.device
        q_lo = q_lo.to(device=device, dtype=y_pred.dtype)
        q_hi = q_hi.to(device=device, dtype=y_pred.dtype)

        return y_pred + q_lo, y_pred + q_hi


def _as_dataloader(
    data: TensorDataset | DataLoader | Sequence[torch.Tensor],
    batch_size: int,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    if isinstance(data, TensorDataset):
        return DataLoader(data, batch_size=batch_size, shuffle=False)
    if isinstance(data, Sequence) and len(data) == 3:
        dataset = TensorDataset(*data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    raise TypeError("Calibration data must be a TensorDataset, DataLoader or tuple of tensors")


def _prepare_treatment(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.to(torch.long)
    if t.dim() == 2 and t.size(-1) > 1:
        return t.argmax(dim=-1)
    return t.view(-1).to(torch.long)


def build_conformal_calibrator(
    model: torch.nn.Module,
    data: TensorDataset | DataLoader | Sequence[torch.Tensor],
    *,
    coverage: float = 0.9,
    batch_size: int = 256,
    min_count: int = 3,
) -> ConformalCalibrator | None:
    """Estimate conformal residual quantiles for each treatment arm."""

    if coverage <= 0 or coverage >= 1:
        raise ValueError("coverage must be between 0 and 1")

    loader = _as_dataloader(data, batch_size)
    if len(loader) == 0:
        return None

    lower_q = (1 - coverage) / 2
    upper_q = 1 - lower_q

    residuals: dict[int, list[torch.Tensor]] = {0: [], 1: []}

    model_state = model.training
    model.eval()
    params_iter = iter(model.parameters())
    first_param = next(params_iter, None)
    model_device = first_param.device if first_param is not None else None
    try:
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x, y, t = batch[:3]
            else:
                raise ValueError("Calibration batches must provide (x, y, t)")

            device = model_device or x.device
            x = x.to(device)
            y = y.to(device)
            t = _prepare_treatment(t.to(device))

            with torch.no_grad():
                preds = predict_outcome(model, x, t)

            if preds.dim() > 1:
                preds = preds.view(len(preds), -1).mean(dim=1)
            if y.dim() > 1:
                y = y.view(len(y), -1).mean(dim=1)

            res = y - preds
            for arm in (0, 1):
                mask = (t == arm)
                if mask.any():
                    residuals[arm].append(res[mask].detach().cpu())
    except Exception:
        logger.warning("Failed to build conformal calibrator; falling back to None.", exc_info=True)
        return None
    finally:
        model.train(model_state)

    collected = {arm: (torch.cat(values) if values else torch.empty(0)) for arm, values in residuals.items()}
    if all(len(res) == 0 for res in collected.values()):
        return None

    all_residuals = torch.cat([res for res in collected.values() if len(res) > 0])
    global_max = all_residuals.abs().max() if len(all_residuals) > 0 else torch.tensor(0.0)

    def _quantiles(res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(res) >= min_count:
            return (
                torch.quantile(res, lower_q),
                torch.quantile(res, upper_q),
            )
        if len(res) > 0:
            width = torch.max(global_max, res.abs().max())
        else:
            width = global_max
        if width.dim() == 0:
            width = width.unsqueeze(0)
        value = width.max()
        if value == 0:
            value = torch.tensor(0.0)
        return -value, value

    q_lo_t0, q_hi_t0 = _quantiles(collected[0])
    q_lo_t1, q_hi_t1 = _quantiles(collected[1])

    return ConformalCalibrator(q_lo_t0, q_hi_t0, q_lo_t1, q_hi_t1)


__all__ = ["ConformalCalibrator", "build_conformal_calibrator"]
