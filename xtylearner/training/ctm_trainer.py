from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
import optuna

from .base_trainer import BaseTrainer
from ..noise_schedules import add_noise


class CTMTrainer(BaseTrainer):
    """Trainer for the CTMT model."""

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader=None,
        device="cpu",
        logger=None,
        scheduler=None,
        grad_clip_norm=None,
        cfg=None,
        optuna_trial: None | optuna.Trial = None,
    ):
        """Create a trainer for a ``CTMT`` model.

        Parameters
        ----------
        model : nn.Module
            Model instance to train.
        optimizer : torch.optim.Optimizer
            Optimiser used for parameter updates.
        train_loader : Iterable
            Loader providing training batches ``(X, Y, T)``.
        val_loader : Iterable, optional
            Optional validation set loader.
        device : str, default ``"cpu"``
            Device on which to run training.
        logger : TrainerLogger, optional
            Optional training logger.
        scheduler : torch.optim.lr_scheduler._LRScheduler or tuple, optional
            Learning rate scheduler(s).
        grad_clip_norm : float, optional
            Gradient clipping norm.
        cfg : dict, optional
            Extra configuration parameters controlling pseudo labelling.
        """
        super().__init__(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            logger,
            scheduler,
            grad_clip_norm,
            optuna_trial,
        )
        self.cfg = cfg or {}

    def step(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        """Compute a single optimisation step on ``batch``."""

        x, y, t_obs = self._extract_batch(batch)
        x0 = torch.cat([x, y, t_obs.clamp_min(0).unsqueeze(-1).float()], dim=-1)

        b = x0.size(0)
        s = torch.rand(b, 1, device=self.device) * 0.9 + 0.1
        t_small = torch.rand(b, 1, device=self.device) * 0.0
        x_s = add_noise(x0, s)
        x_t = add_noise(x0, t_small)

        x_hat, t_logits = self.model(x_t, t_small, s - t_small)

        loss_ctm = F.mse_loss(x_hat, x_s)
        mask = t_obs >= 0
        loss_prop = torch.tensor(0.0, device=self.device)
        if mask.any():
            loss_prop = F.cross_entropy(t_logits[mask], t_obs[mask])
        if self.cfg.get("pseudo", False):
            conf = t_logits.softmax(-1).max(-1).values
            keep = (~mask) & (conf > self.cfg.get("tau", 0.8))
            if keep.any():
                pseudo = t_logits[keep].argmax(-1)
                loss_prop = loss_prop + F.cross_entropy(t_logits[keep], pseudo.detach())
        return loss_ctm + self.cfg.get("lmb_t", 1.0) * loss_prop

    def fit(self, num_epochs: int) -> None:
        """Run the training loop for ``num_epochs`` epochs."""

        for epoch in range(num_epochs):
            self.model.train()
            num_batches = len(self.train_loader)
            if self.logger:
                self.logger.start_epoch(epoch + 1, num_batches)
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self._clip_grads()
                self.optimizer.step()
                if self.logger:
                    X, Y, T = self._extract_batch(batch)
                    metrics = dict(self._metrics_from_loss(loss))
                    metrics.update(self._treatment_metrics(X, Y, T))
                    metrics.update(self._outcome_metrics(X, Y, T))
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._eval_metrics(self.val_loader)
                if self.logger:
                    self.logger.log_validation(epoch + 1, val_metrics)

            if self.logger:
                self.logger.end_epoch(epoch + 1)

            if self.optuna_trial is not None and val_metrics is not None:
                metric = val_metrics.get("loss", next(iter(val_metrics.values()), 0.0))
                self.optuna_trial.report(metric, step=epoch + 1)
                if self.optuna_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        """Return evaluation metrics averaged over ``data_loader``."""

        metrics = self._eval_metrics(data_loader)
        loss_val = metrics.get("loss", next(iter(metrics.values()), 0.0))
        return {
            "loss": float(loss_val),
            "treatment accuracy": float(metrics.get("accuracy", 0.0)),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
            "outcome rmse labelled": float(metrics.get("rmse_labelled", 0.0)),
            "outcome rmse unlabelled": float(metrics.get("rmse_unlabelled", 0.0)),
        }

    def predict(self, *args):
        """Predict outcomes for covariates ``x`` under treatment ``t``."""

        self.model.eval()
        with torch.no_grad():
            if len(args) == 1 and isinstance(args[0], int):
                raise ValueError("CTMT does not support unconditional sampling")
            if len(args) != 2:
                raise TypeError("predict() expects `(x, t)` arguments")
            x, t_val = args
            x = x.to(self.device)
            t_tensor = (
                torch.full((x.size(0),), t_val, dtype=torch.long, device=self.device)
                if isinstance(t_val, int)
                else t_val.to(self.device)
            )
            x0 = torch.cat(
                [
                    x,
                    torch.zeros(x.size(0), 1, device=self.device),
                    t_tensor.unsqueeze(-1).float(),
                ],
                dim=-1,
            )
            out, _ = self.model(
                x0,
                torch.zeros_like(t_tensor, dtype=torch.float32).unsqueeze(-1),
                torch.zeros_like(t_tensor, dtype=torch.float32).unsqueeze(-1),
            )
            y_hat = out[:, x.size(1) : x.size(1) + 1]
            return y_hat

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``p(t\mid x,y)`` estimated by the model."""

        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        zeros = torch.zeros(x.size(0), 1, device=self.device)
        x0 = torch.cat([x, y, zeros], dim=-1)
        _, logits = self.model(x0, zeros, zeros)
        return logits.softmax(dim=-1)


__all__ = ["CTMTrainer"]
