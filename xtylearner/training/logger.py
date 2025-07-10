from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Mapping


class TrainerLogger(ABC):
    """Abstract base logger for training progress."""

    def __init__(self, print_every: int = 10) -> None:
        self.print_every = print_every
        self._running: defaultdict[str, float] = defaultdict(float)
        self._count = 0
        self.num_batches = 0

    def start_epoch(self, epoch: int, num_batches: int) -> None:
        """Begin a new epoch and reset internal counters."""

        self._running.clear()
        self._count = 0
        self.num_batches = num_batches
        self.epoch = epoch

    def update(self, metrics: Mapping[str, float]) -> None:
        """Accumulate metrics from a training step."""

        self._count += 1
        for k, v in metrics.items():
            self._running[k] += float(v)

    def averages(self) -> dict[str, float]:
        """Return average of all tracked metrics in the current epoch."""

        return {k: v / max(self._count, 1) for k, v in self._running.items()}

    @abstractmethod
    def log_step(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        metrics: Mapping[str, float],
    ) -> None:
        """Log metrics produced by a single optimisation step."""

    @abstractmethod
    def log_validation(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Report validation metrics once an epoch finishes."""

    def end_epoch(self, epoch: int) -> None:
        """Print averaged metrics collected during the epoch."""

        avg = self.averages()
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in avg.items())
        print(f"Epoch {epoch} finished: {metric_str}")


class ConsoleLogger(TrainerLogger):
    """Simple ``TrainerLogger`` that prints to stdout."""

    def log_step(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        metrics: Mapping[str, float],
    ) -> None:
        """Print metrics for the current batch."""

        self.update(metrics)
        if self._count % self.print_every == 0 or batch_idx == num_batches - 1:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            end = "\n" if batch_idx == num_batches - 1 else "\r"
            print(
                f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] {metric_str}",
                end=end,
                flush=True,
            )

    def log_validation(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Print aggregated validation metrics."""

        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch} validation: {metric_str}")
