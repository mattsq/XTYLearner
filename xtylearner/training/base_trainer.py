from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Mapping
from collections import defaultdict

import torch
import torch.nn.functional as F

import numpy as np

from .metrics import accuracy, rmse_loss


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
        scheduler: (
            torch.optim.lr_scheduler._LRScheduler
            | tuple[
                torch.optim.lr_scheduler._LRScheduler,
                torch.optim.lr_scheduler._LRScheduler,
            ]
            | None
        ) = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        # Some simple models (e.g. probabilistic circuits) do not inherit from
        # ``torch.nn.Module`` and therefore lack a ``to`` method.  In that case
        # we simply keep the model instance as-is.
        if hasattr(model, "to"):
            self.model = model.to(device)
        else:
            self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

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
        """Convert a loss tensor or dictionary into a metrics mapping."""

        if isinstance(loss, torch.Tensor):
            return {"loss": float(loss.item())}
        if isinstance(loss, Mapping):
            metrics: dict[str, float] = {}
            for k, v in loss.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics[k] = float(v.item())
                elif not isinstance(v, torch.Tensor):
                    try:
                        metrics[k] = float(v)
                    except Exception:
                        pass
            return metrics
        return {"loss": float(loss)}

    # --------------------------------------------------------------
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute treatment probabilities ``p(t|x,y)`` for the current model."""

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
        """Move a training batch to the target device and split its parts."""

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
        """Return NLL and accuracy of ``p(t|x,y)`` for observed labels."""

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

        if isinstance(probs, (list, tuple)):
            if len(probs) == 0:
                return {}
            probs = probs[0]

        log_probs = probs.clamp_min(1e-12).log()
        if t_obs.dim() > 1 and t_obs.size(-1) == 1:
            t_obs = t_obs.squeeze(-1)
        nll = F.nll_loss(log_probs[mask], t_obs[mask])
        acc = accuracy(log_probs[mask], t_obs[mask])
        return {"nll": float(nll.item()), "accuracy": float(acc.item())}

    # --------------------------------------------------------------
    def _predict_outcome(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor | None:
        """Internal helper used to guess outcomes for RMSE computation."""

        if hasattr(self.model, "predict_outcome"):
            try:
                out = self.model.predict_outcome(x, t)
            except Exception:
                try:
                    out = self.model.predict_outcome(x.cpu().numpy(), t.cpu().numpy())
                except Exception:
                    return None
            if isinstance(out, np.ndarray):
                return torch.from_numpy(out).to(self.device)
            return out.to(self.device)

        if hasattr(self.model, "head_Y"):
            try:
                k = getattr(self.model, "k", None)
                if k is None:
                    return None
                t1h = torch.nn.functional.one_hot(t.to(torch.long), k).float()
                if hasattr(self.model, "h"):
                    h = self.model.h(x)
                    return self.model.head_Y(torch.cat([h, t1h], dim=-1))
                return self.model.head_Y(torch.cat([x, t1h], dim=-1))
            except Exception:
                return None

        if hasattr(self.model, "G_Y"):
            try:
                k = getattr(self.model, "k", None)
                if k is None:
                    return None
                t1h = torch.nn.functional.one_hot(t.to(torch.long), k).float()
                return self.model.G_Y(torch.cat([x, t1h], dim=-1))
            except Exception:
                return None

        if hasattr(self, "predict"):
            try:
                # handle vector t by batching unique values
                if t.dim() == 0 or t.numel() == 1:
                    return self.predict(x, int(t.item()))
                preds = torch.zeros(x.size(0), y.size(-1), device=self.device)
                for val in t.unique():
                    idx = t == val
                    preds[idx] = self.predict(x[idx], int(val.item()))
                return preds
            except Exception:
                return None

        return None

    def _outcome_metrics(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> Mapping[str, float]:
        """Compute RMSE for observed outcomes if the model supports it."""

        mask = t_obs >= 0
        if not mask.any():
            return {}

        with torch.no_grad():
            preds = self._predict_outcome(x[mask], t_obs[mask], y[mask])
        if preds is None:
            return {}

        rmse = rmse_loss(preds, y[mask])
        return {"rmse": float(rmse.item())}

    # --------------------------------------------------------------
    def _clip_grads(self) -> None:
        """Clip gradients of model parameters if ``grad_clip_norm`` is set."""

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

    # --------------------------------------------------------------
    def _eval_metrics(self, data_loader: Iterable) -> Mapping[str, float]:
        """Evaluate the model on ``data_loader`` and return averaged metrics."""

        self.model.eval()
        running: defaultdict[str, float] = defaultdict(float)
        n = 0
        with torch.no_grad():
            for batch in data_loader:
                X, Y, T_obs = self._extract_batch(batch)
                loss = self.step(batch)  # type: ignore[attr-defined]
                metrics = dict(self._metrics_from_loss(loss))
                metrics.update(self._treatment_metrics(X, Y, T_obs))
                metrics.update(self._outcome_metrics(X, Y, T_obs))
                for k, v in metrics.items():
                    running[k] += float(v)
                n += 1
        return {k: v / max(n, 1) for k, v in running.items()}
