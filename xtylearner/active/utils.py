"""Utility helpers for active learning strategies."""

from __future__ import annotations

import torch
import torch.nn as nn


def predict_outcome(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Return ``y`` predictions for covariates ``x`` and treatment ``t``.

    The helper first checks for a ``predict_outcome`` implementation and
    otherwise tries a combination of ``head_Y`` and ``h`` attributes. When no
    specialised head is found the model is called directly. This mirrors the
    logic previously embedded in :mod:`xtylearner.active.strategies`.
    """

    if hasattr(model, "predict_outcome"):
        return model.predict_outcome(x, int(t[0].item()) if t.numel() == 1 else t)

    k = getattr(model, "k", None)
    if k is not None:
        t_in = torch.nn.functional.one_hot(t.to(torch.long), k).float()
    else:
        t_in = t.unsqueeze(-1) if t.dim() == 1 else t

    if hasattr(model, "head_Y"):
        if hasattr(model, "h"):
            h = model.h(x)
            return model.head_Y(torch.cat([h, t_in], dim=-1))
        return model.head_Y(torch.cat([x, t_in], dim=-1))

    return model(x, t)


__all__ = ["predict_outcome"]
