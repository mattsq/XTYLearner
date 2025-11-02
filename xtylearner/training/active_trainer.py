from __future__ import annotations

import logging
from typing import Iterable, Optional, Mapping, Iterator, Dict, Any, Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from .trainer import Trainer
from .logger import TrainerLogger, ConsoleLogger
from ..active import (
    QueryStrategy,
    CATEUncertainty,
    ConformalCATEIntervalStrategy,
)
from ..active.calibration import build_conformal_calibrator
from ..active.label_propensity import train_label_propensity_model


logger = logging.getLogger(__name__)


class ActiveTrainer:
    """Simple active learning loop over a semi-supervised dataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        strategy: QueryStrategy,
        budget: int,
        batch: int,
        val_loader: Optional[Iterable] = None,
        device: str = "cpu",
        trainer_logger: Optional["TrainerLogger"] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self._trainer = Trainer(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            trainer_logger,
            scheduler,
            grad_clip_norm,
        )
        self.strategy = strategy
        self.budget = budget
        self.batch = batch
        self.queries = 0
        self._calibrator = None
        self.label_propensity_model = None
        self.current_round = 0

        if hasattr(self.strategy, "set_trainer_context"):
            try:
                self.strategy.set_trainer_context(self)
            except Exception:
                logger.warning(
                    "Strategy %s rejected trainer context; continuing without MNAR features.",
                    strategy.__class__.__name__,
                    exc_info=True,
                )

    # --------------------------------------------------------------
    def _log_status(
        self, L: TensorDataset, U: TensorDataset, action: str | None = None
    ) -> None:
        """Print current dataset sizes and budget usage when using ``ConsoleLogger``."""
        if isinstance(self._trainer.logger, ConsoleLogger):
            msg = f"labelled={len(L)} unlabelled={len(U)} budget={self.queries}/{self.budget}"
            if action:
                msg = f"{action}: " + msg
            print(msg)

    # --------------------------------------------------------------
    def _budget_left(self) -> bool:
        return self.queries < self.budget

    # --------------------------------------------------------------
    def fit(self, num_epochs: int, round_callback: Callable[[Dict[str, Any]], None] | None = None) -> None:
        for state in self.iterate_rounds(num_epochs):
            if round_callback is not None:
                round_callback(state)

    def iterate_rounds(self, num_epochs: int) -> Iterator[Dict[str, Any]]:
        dataset = self._trainer.train_loader.dataset  # type: ignore[attr-defined]
        if not hasattr(dataset, "tensors"):
            raise ValueError("ActiveTrainer requires a TensorDataset")
        X, Y, T = dataset.tensors
        if T.dim() == 1:
            labelled = T >= 0
        else:
            labelled = torch.isfinite(T).all(-1)
        L = TensorDataset(X[labelled], Y[labelled], T[labelled])
        U = TensorDataset(X[~labelled], Y[~labelled], T[~labelled])

        self._log_status(L, U, action="start")
        round_idx = 0
        self.current_round = 0

        while self._budget_left() and len(U) > 0:
            round_idx += 1
            loader_L = DataLoader(
                L, batch_size=self._trainer.train_loader.batch_size, shuffle=True
            )
            self._trainer.train_loader = loader_L
            self._trainer.fit(num_epochs)

            if hasattr(self.strategy, "update_labeled"):
                self.strategy.update_labeled(L.tensors[0])

            coverage = getattr(self.strategy, "coverage", 0.9)
            try:
                calibrator = build_conformal_calibrator(
                    self._trainer.model,
                    L,
                    coverage=coverage,
                )
            except Exception:
                logger.warning(
                    "Failed to build conformal calibrator; proceeding without calibration.",
                    exc_info=True,
                )
                calibrator = None
            self._calibrator = calibrator
            if hasattr(self.strategy, "update_calibrator"):
                try:
                    self.strategy.update_calibrator(calibrator)
                except Exception:
                    logger.warning(
                        "Strategy %s failed to accept calibrator.",
                        self.strategy.__class__.__name__,
                        exc_info=True,
                    )

            try:
                self.update_label_propensity_model(L, U)
            except Exception:
                logger.warning(
                    "Failed to update label propensity model; falling back to defaults.",
                    exc_info=True,
                )

            rep_fn = getattr(self._trainer.model, "encoder", None)
            state: Dict[str, Any] = {
                "round": round_idx,
                "labels_used": len(L),
                "labeled_dataset": L,
                "unlabeled_dataset": U,
            }
            self.current_round = round_idx
            yield state

            scores = self.strategy(
                self._trainer.model, U.tensors[0], rep_fn, self.batch
            )
            topk = torch.topk(scores, min(self.batch, len(U)), largest=True).indices

            new_X = U.tensors[0][topk]
            new_Y = U.tensors[1][topk]
            new_T = U.tensors[2][topk]
            L = TensorDataset(
                torch.cat([L.tensors[0], new_X]),
                torch.cat([L.tensors[1], new_Y]),
                torch.cat([L.tensors[2], new_T]),
            )
            mask = torch.ones(len(U), dtype=torch.bool)
            mask[topk] = False
            U = TensorDataset(
                U.tensors[0][mask],
                U.tensors[1][mask],
                U.tensors[2][mask],
            )
            self.queries += len(topk)
            self._log_status(L, U, action="query")

        self._trainer.train_loader = DataLoader(
            L, batch_size=self._trainer.train_loader.batch_size, shuffle=True
        )
        self._log_status(L, U, action="final")
        self._trainer.fit(num_epochs)
        self.current_round = round_idx

    # --------------------------------------------------------------
    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        return self._trainer.evaluate(data_loader)

    # --------------------------------------------------------------
    def predict(self, *args, **kwargs):
        return self._trainer.predict(*args, **kwargs)

    # --------------------------------------------------------------
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._trainer.predict_treatment_proba(x, y)

    # --------------------------------------------------------------
    def get_calibrator(self):
        """Return the most recent conformal calibrator if available."""

        return self._calibrator

    # --------------------------------------------------------------
    def score_cate_need(self, X_pool: torch.Tensor) -> torch.Tensor:
        """Return an uncertainty score for the conditional treatment effect."""

        device = getattr(self._trainer, "device", X_pool.device)
        model = self._trainer.model
        rep_fn = getattr(model, "encoder", None)
        calibrator = self._calibrator
        coverage = getattr(self.strategy, "coverage", 0.9)
        X_device = X_pool.to(device)

        scores: torch.Tensor | None = None

        with torch.no_grad():
            if calibrator is not None:
                base = ConformalCATEIntervalStrategy(coverage=coverage)
                base.update_calibrator(calibrator)
                try:
                    scores = base(model, X_device, rep_fn, self.batch)
                except Exception:
                    logger.warning(
                        "Conformal interval scoring failed; reverting to CATE uncertainty.",
                        exc_info=True,
                    )
                    scores = None

            if scores is None:
                try:
                    base_unc = CATEUncertainty()
                    scores = base_unc(model, X_device, rep_fn, self.batch)
                except Exception:
                    logger.warning(
                        "CATE uncertainty scoring failed; falling back to |tau| heuristic.",
                        exc_info=True,
                    )
                    scores = None

            if scores is None and hasattr(model, "predict_cate"):
                try:
                    tau = model.predict_cate(X_device)
                except Exception:
                    pass
                else:
                    if isinstance(tau, tuple):
                        tau = tau[0]
                    if tau is not None:
                        tau_tensor = torch.as_tensor(tau, device=device, dtype=torch.float32)
                        if tau_tensor.dim() > 1:
                            tau_tensor = tau_tensor.view(len(X_pool), -1).mean(dim=1)
                        scores = tau_tensor.abs()

        if scores is None:
            scores = torch.zeros(len(X_pool), device=device)

        return torch.as_tensor(scores, device=device, dtype=torch.float32).to(X_pool.device)

    # --------------------------------------------------------------
    def predict_label_propensity(self, X: torch.Tensor) -> torch.Tensor:
        r"""Predict :math:`P(L=1\mid X)` using the auxiliary propensity model."""

        model = self.label_propensity_model
        if model is None:
            return torch.full((len(X),), 0.5, device=X.device, dtype=torch.float32)

        try:
            next_param = next(model.parameters(), None)
        except Exception:
            next_param = None

        device = next_param.device if next_param is not None else getattr(self._trainer, "device", X.device)

        model.eval()
        with torch.no_grad():
            preds = model(X.to(device)).view(-1)
        return preds.clamp(0.0, 1.0).to(X.device)

    # --------------------------------------------------------------
    def update_label_propensity_model(
        self,
        labeled: TensorDataset,
        unlabeled: TensorDataset,
    ) -> None:
        """(Re)fit the auxiliary label propensity model after each round."""

        device = getattr(self._trainer, "device", "cpu")
        model = train_label_propensity_model(labeled, unlabeled, device=device)
        self.label_propensity_model = model

    # --------------------------------------------------------------
    def __getattr__(self, name):
        return getattr(self._trainer, name)


__all__ = ["ActiveTrainer"]
