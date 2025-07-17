import torch


def add_noise(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise with per-sample standard deviation ``scale``."""
    while scale.dim() < x.dim():
        scale = scale.unsqueeze(-1)
    noise = torch.randn_like(x)
    return x + scale * noise

__all__ = ["add_noise"]
