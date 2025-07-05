from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class TrainerLogger(ABC):
    """Abstract interface for trainer logging utilities."""

    def start_epoch(self, epoch: int, num_epochs: int, num_batches: int) -> None:
        """Called at the start of each epoch."""

    def log_batch(self, batch_idx: int, num_batches: int, loss: Any) -> None:
        """Log progress for an individual batch."""

    def end_epoch(self, epoch: int, avg_loss: float) -> None:
        """Called at the end of each epoch."""

    @abstractmethod
    def format_loss(self, loss: Any) -> str:
        """Return a formatted string representation of ``loss``."""


class ConsoleLogger(TrainerLogger):
    """Simple console logger printing progress and losses."""

    def __init__(self, print_every: int = 1) -> None:
        self.print_every = print_every
        self.num_batches = 0
        self.num_epochs = 0

    def start_epoch(self, epoch: int, num_epochs: int, num_batches: int) -> None:
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        print(f"Epoch {epoch + 1}/{num_epochs}")

    def log_batch(self, batch_idx: int, num_batches: int, loss: Any) -> None:
        if self.print_every and ((batch_idx + 1) % self.print_every == 0):
            loss_str = self.format_loss(loss)
            total_batches = num_batches if num_batches > 0 else "?"
            print(f"  Batch {batch_idx + 1}/{total_batches}: {loss_str}")

    def end_epoch(self, epoch: int, avg_loss: float) -> None:
        print(f"End epoch {epoch + 1}: avg {self.format_loss(avg_loss)}")

    def format_loss(self, loss: Any) -> str:
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(loss, dict):
            parts = []
            for k, v in loss.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                parts.append(f"{k}={float(v):.4f}")
            return ", ".join(parts)
        if isinstance(loss, (float, int)):
            return f"loss={float(loss):.4f}"
        return str(loss)


__all__ = ["TrainerLogger", "ConsoleLogger"]
