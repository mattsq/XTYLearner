from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Mapping

import torch
import torch.nn.functional as F

from .metrics import accuracy


from .logger import TrainerLogger


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        device: str = "cpu",
        logger: Optional[TrainerLogger] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.scheduler = scheduler

    @abstractmethod
    def fit(self, num_epochs: int) -> None:
        """Train the model for ``num_epochs`` epochs."""

    @abstractmethod
    def evaluate(self, data_loader: Iterable) -> float:
        """Return a scalar loss/metric evaluated on ``data_loader``."""

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Return model predictions for the supplied inputs."""

    def _metrics_from_loss(
        self, loss: torch.Tensor | Mapping[str, float]
    ) -> Mapping[str, float]:
        if isinstance(loss, torch.Tensor):
            return {"loss": float(loss.item())}
        if isinstance(loss, Mapping):
            return {k: float(v) for k, v in loss.items()}
        return {"loss": float(loss)}

    # --------------------------------------------------------------
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return treatment class probabilities ``p(t|x,y)``."""

        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)

            if hasattr(self.model, "predict_treatment_proba"):
                return self.model.predict_treatment_proba(x, y)

            logits = None
            if hasattr(self.model, "cls_t"):
                logits = self.model.cls_t(x, y)
            elif hasattr(self.model, "head_T"):
                logits = self.model.head_T(torch.cat([x, y], dim=-1))
            elif hasattr(self.model, "C"):
                logits = self.model.C(torch.cat([x, y], dim=-1))
            if logits is None:
                raise ValueError(
                    "Model does not support treatment probability prediction"
                )
            return logits.softmax(dim=-1)

    # --------------------------------------------------------------
    def _extract_batch(
        self, batch: Iterable[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(X, Y, T_obs)`` tensors from ``batch``."""

        inputs = [b.to(self.device) for b in batch]
        if len(inputs) == 2:
            x, y = inputs
            t_obs = torch.full((x.size(0),), -1, dtype=torch.long, device=self.device)
        else:
            x, y, t_obs = inputs
        return x, y, t_obs

    def _treatment_metrics(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> Mapping[str, float]:
        """Negative log-likelihood and accuracy for ``p(t|x,y)``."""

        if not hasattr(self.model, "predict_treatment_proba") and not any(
            hasattr(self.model, attr) for attr in ["cls_t", "head_T", "C"]
        ):
            return {}

        mask = t_obs >= 0
        if not mask.any():
            return {}

        try:
            probs = self.predict_treatment_proba(x, y)
        except Exception:
            return {}

        log_probs = probs.clamp_min(1e-12).log()
        nll = F.nll_loss(log_probs[mask], t_obs[mask])
        acc = accuracy(log_probs[mask], t_obs[mask])
        return {"nll": float(nll.item()), "accuracy": float(acc.item())}
