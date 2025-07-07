from __future__ import annotations

import math


def ramp_up_sigmoid(epoch: int, ramp: int, max_val: float = 1.0) -> float:
    """Sigmoid ramp-up used for VAT and Mean Teacher baselines."""
    t = min(epoch / ramp, 1.0)
    return max_val * math.exp(-5 * (1 - t) ** 2)


__all__ = ["ramp_up_sigmoid"]
