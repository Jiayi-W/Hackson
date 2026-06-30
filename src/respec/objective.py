from __future__ import annotations

import numpy as np


def interference_cost(allocation: np.ndarray, weights: np.ndarray, eps: float = 1e-9) -> float:
    total_weight = float(np.triu(weights, k=1).sum())
    if total_weight <= eps:
        return 0.0

    value = 0.0
    n_users = allocation.shape[0]
    for u in range(n_users):
        for v in range(u + 1, n_users):
            if allocation[u] == allocation[v]:
                value += weights[u, v]
    return float(value / (total_weight + eps))


def switching_ratio(allocation: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None:
        return 0.0
    return float(np.mean(allocation != previous))


def classical_objective(
    allocation: np.ndarray,
    weights: np.ndarray,
    previous: np.ndarray | None,
    lambda_switch: float,
) -> float:
    return interference_cost(allocation, weights) + lambda_switch * switching_ratio(allocation, previous)

