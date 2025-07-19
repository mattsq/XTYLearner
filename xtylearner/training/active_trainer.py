from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .supervised import SupervisedTrainer
from .logger import TrainerLogger
from ..active import QueryStrategy


class ActiveTrainer(SupervisedTrainer):
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
        logger: Optional["TrainerLogger"] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            logger,
            scheduler,
            grad_clip_norm,
        )
        self.strategy = strategy
        self.budget = budget
        self.batch = batch
        self.queries = 0

    # --------------------------------------------------------------
    def _budget_left(self) -> bool:
        return self.queries < self.budget

    # --------------------------------------------------------------
    def fit(self, num_epochs: int) -> None:
        dataset = self.train_loader.dataset  # type: ignore[attr-defined]
        if not hasattr(dataset, "tensors"):
            raise ValueError("ActiveTrainer requires a TensorDataset")
        X, Y, T = dataset.tensors
        labelled = T >= 0
        L = TensorDataset(X[labelled], Y[labelled], T[labelled])
        U = TensorDataset(X[~labelled], Y[~labelled], T[~labelled])

        while self._budget_left() and len(U) > 0:
            loader_L = DataLoader(
                L, batch_size=self.train_loader.batch_size, shuffle=True
            )
            self.train_loader = loader_L
            super().fit(num_epochs)

            if hasattr(self.strategy, "update_labeled"):
                self.strategy.update_labeled(L.tensors[0])

            rep_fn = getattr(self.model, "encoder", None)
            scores = self.strategy(self.model, U.tensors[0], rep_fn, self.batch)
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

        self.train_loader = DataLoader(
            L, batch_size=self.train_loader.batch_size, shuffle=True
        )
        super().fit(num_epochs)
