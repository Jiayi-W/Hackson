from __future__ import annotations

import numpy as np


def best_so_far_trace(start: float, finish: float, evaluations: int, shape: str) -> np.ndarray:
    x = np.linspace(0.0, 1.0, evaluations)
    if shape == "fast":
        progress = 1.0 - np.exp(-4.2 * x)
    elif shape == "slow":
        progress = x ** 1.8
    elif shape == "reset":
        progress = np.where(x < 0.45, 0.28 * x, 0.12 + 0.95 * (x - 0.45))
        progress = np.clip(progress, 0.0, 1.0)
    else:
        progress = x

    values = start + (finish - start) * progress
    values = np.minimum.accumulate(values)
    return values

