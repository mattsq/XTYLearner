from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
import optuna

from .base_trainer import BaseTrainer


class CoTrainTrainer(BaseTrainer):
    """Trainer implementing the SemiITE co-training loop."""

    def step(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        """Compute the loss on a single batch."""

        x, y, t_obs = self._extract_batch(batch)
        z = self.model.encode(x)
        logits = self.model.prop(z)
        logits_xy = self.model.prop_y(torch.cat([z, y], dim=-1))
        k = self.model.k

        labelled = t_obs >= 0
        loss = torch.tensor(0.0, device=self.device)

        if labelled.any():
            t_lab = t_obs[labelled].to(torch.long)
            t1h_lab = F.one_hot(t_lab, k).float()
            z_lab = z[labelled]
            for head in self.model.outcome:
                pred = head(torch.cat([z_lab, t1h_lab], dim=-1))
                loss = loss + F.mse_loss(pred, y[labelled])
            loss = loss + F.cross_entropy(logits[labelled], t_lab)
            loss = loss + F.cross_entropy(logits_xy[labelled], t_lab)
            if self.model.mmd_beta > 0:
                loss = loss + self.model.mmd_beta * self.model.compute_mmd(z_lab, t_lab)

        unlabelled = ~labelled
        if unlabelled.any():
            kl = F.kl_div(
                F.log_softmax(logits[unlabelled], dim=-1),
                F.softmax(logits_xy[unlabelled].detach(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + self.model.lambda_kl * kl

        if unlabelled.any() and self.model.lambda_u > 0 and self.model.q_pseudo > 0:
            z_u = z[unlabelled]
            logits_u = logits[unlabelled]
            t_hat = logits_u.argmax(dim=-1)
            t1h_hat = F.one_hot(t_hat, k).float()

            t_all = [
                F.one_hot(
                    torch.full((len(z_u),), j, dtype=torch.long, device=self.device),
                    k,
                ).float()
                for j in range(k)
            ]
            preds = []
            disagree = []
            for head in self.model.outcome:
                y_all = [head(torch.cat([z_u, tj], dim=-1)) for tj in t_all]
                y_all = torch.stack(y_all, dim=1)
                preds.append(y_all)
                disagree.append(y_all.var(dim=1).mean(dim=-1))

            q = min(self.model.q_pseudo, z_u.size(0))
            if q > 0:
                unsup = torch.tensor(0.0, device=self.device)
                for i, head in enumerate(self.model.outcome):
                    idx = torch.topk(disagree[i], q, largest=False).indices
                    if idx.numel() == 0:
                        continue
                    teacher = (i + 1) % 3
                    y_teacher = preds[teacher][idx, t_hat[idx]]
                    y_student = head(torch.cat([z_u[idx], t1h_hat[idx]], dim=-1))
                    unsup = unsup + F.mse_loss(y_student, y_teacher.detach())
                loss = loss + self.model.lambda_u * unsup

        return loss

    def fit(self, num_epochs: int) -> None:
        """Train the model for ``num_epochs`` epochs."""

        for epoch in range(num_epochs):
            self.model.train()
            num_batches = len(self.train_loader)
            if self.logger:
                self.logger.start_epoch(epoch + 1, num_batches)
            for batch_idx, batch in enumerate(self.train_loader):
                X, Y, T_obs = self._extract_batch(batch)
                loss = self.step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self._clip_grads()
                self.optimizer.step()
                if self.logger:
                    metrics = dict(self._metrics_from_loss(loss))
                    metrics.update(self._treatment_metrics(X, Y, T_obs))
                    metrics.update(self._outcome_metrics(X, Y, T_obs))
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.logger and self.val_loader is not None:
                val_metrics = self._eval_metrics(self.val_loader)
                self.logger.log_validation(epoch + 1, val_metrics)
            if self.logger:
                self.logger.end_epoch(epoch + 1)
            if self.optuna_trial is not None:
                self.trial.report(val_metrics)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()                 

    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        """Return evaluation metrics on ``data_loader``."""

        metrics = self._eval_metrics(data_loader)
        loss_val = metrics.get("loss", next(iter(metrics.values()), 0.0))
        return {
            "loss": float(loss_val),
            "treatment accuracy": float(metrics.get("accuracy", 0.0)),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
            "outcome rmse labelled": float(metrics.get("rmse_labelled", 0.0)),
            "outcome rmse unlabelled": float(metrics.get("rmse_unlabelled", 0.0)),
        }

    def predict(self, *inputs: torch.Tensor):
        """Return model predictions for ``inputs``."""

        self.model.eval()
        with torch.no_grad():
            inputs = [i.to(self.device) for i in inputs]
            if hasattr(self.model, "predict"):
                return self.model.predict(*inputs)
            return self.model(*inputs)


__all__ = ["CoTrainTrainer"]
