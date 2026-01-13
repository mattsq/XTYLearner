"""Specialized trainer for ordinal classification models."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch
import optuna

from .supervised import SupervisedTrainer


class OrdinalTrainer(SupervisedTrainer):
    """Specialized trainer for ordinal classification models.

    This trainer extends :class:`SupervisedTrainer` with ordinal-specific
    evaluation metrics and provides hooks for advanced ordinal training
    techniques.

    Parameters
    ----------
    model
        The ordinal classification model to train.
    optimizer
        PyTorch optimizer for model parameters.
    train_loader
        DataLoader for training batches.
    val_loader
        Optional DataLoader for validation batches.
    device
        Device to use for training (e.g., "cpu", "cuda").
    logger
        Optional logger for tracking training progress.
    scheduler
        Optional learning rate scheduler(s).
    grad_clip_norm
        Optional gradient clipping norm.
    optuna_trial
        Optional Optuna trial for hyperparameter optimization.
    ordinal_weight
        Weight for ordinal loss component (for future extensions).
        Currently unused but reserved for future ordinal-specific losses.
    unimodal_reg
        Regularization strength for enforcing unimodal distributions.
        Currently unused but reserved for future enhancements.

    Examples
    --------
    >>> from xtylearner.models import create_model
    >>> from xtylearner.training import OrdinalTrainer
    >>> import torch
    >>> model = create_model("dragon_net", d_x=10, d_y=1, k=5, ordinal=True)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> trainer = OrdinalTrainer(model, optimizer, train_loader)
    >>> trainer.fit(num_epochs=100)
    >>> metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        device: str = "cpu",
        logger: Optional["TrainerLogger"] = None,
        scheduler: (
            torch.optim.lr_scheduler._LRScheduler
            | tuple[
                torch.optim.lr_scheduler._LRScheduler,
                torch.optim.lr_scheduler._LRScheduler,
            ]
            | None
        ) = None,
        grad_clip_norm: float | None = None,
        optuna_trial: Optional[optuna.Trial] = None,
        ordinal_weight: float = 1.0,
        unimodal_reg: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=logger,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            optuna_trial=optuna_trial,
        )
        self.ordinal_weight = ordinal_weight
        self.unimodal_reg = unimodal_reg

        # Validate that the model is in ordinal mode
        if not getattr(model, "ordinal", False):
            import warnings

            warnings.warn(
                "OrdinalTrainer is being used with a model that does not have "
                "ordinal=True. Ordinal metrics will not be computed. "
                "Consider setting ordinal=True when creating the model.",
                UserWarning,
            )

    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        """Return averaged metrics on ``data_loader``.

        For ordinal models, includes ordinal-specific metrics in addition
        to standard loss and RMSE metrics.

        Parameters
        ----------
        data_loader
            Iterable yielding evaluation batches.

        Returns
        -------
        Mapping[str, float]
            Dictionary with loss, treatment metrics, and outcome RMSE.
            For ordinal models, includes:
            - ``treatment_mae``: Mean Absolute Error on class indices
            - ``treatment_qwk``: Quadratic Weighted Kappa
            - ``treatment_adjacent_acc``: Adjacent-class accuracy
            - ``treatment_accuracy``: Exact match accuracy
        """
        metrics = self._eval_metrics(data_loader)
        loss_val = metrics.get("loss", next(iter(metrics.values()), 0.0))

        is_ordinal = getattr(self.model, "ordinal", False)

        result = {
            "loss": float(loss_val),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
            "outcome rmse labelled": float(metrics.get("rmse_labelled", 0.0)),
            "outcome rmse unlabelled": float(metrics.get("rmse_unlabelled", 0.0)),
        }

        if is_ordinal:
            # Include ordinal-specific metrics
            result["treatment mae"] = float(metrics.get("treatment_mae", 0.0))
            result["treatment qwk"] = float(metrics.get("treatment_qwk", 0.0))
            result["treatment adjacent acc"] = float(
                metrics.get("treatment_adjacent_acc", 0.0)
            )
            result["treatment accuracy"] = float(metrics.get("treatment_accuracy", 0.0))
        else:
            # Standard accuracy for non-ordinal models
            result["treatment accuracy"] = float(metrics.get("accuracy", 0.0))

        return result


__all__ = ["OrdinalTrainer"]
