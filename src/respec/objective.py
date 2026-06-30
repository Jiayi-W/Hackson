from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .sampling import decode_onehot_bitstring


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


def summarize_counts_objective(
    counts: Mapping[str, int],
    weights: np.ndarray,
    previous: np.ndarray | None,
    lambda_switch: float,
    n_users: int,
    n_channels: int,
) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one sample.")

    expected_cost = 0.0
    feasible_total = 0
    best_cost = float("inf")
    best_allocation: np.ndarray | None = None
    best_allocation_count = -1
    best_cost_mass = 0

    for bitstring, count in counts.items():
        count = int(count)
        try:
            allocation = decode_onehot_bitstring(bitstring, n_users=n_users, n_channels=n_channels)
        except ValueError:
            continue

        feasible_total += count
        cost = classical_objective(allocation, weights, previous, lambda_switch)
        expected_cost += count * cost

        allocation_tuple = tuple(int(value) for value in allocation.tolist())
        best_tuple = None if best_allocation is None else tuple(int(value) for value in best_allocation.tolist())
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_allocation = allocation.copy()
            best_allocation_count = count
            best_cost_mass = count
        elif abs(cost - best_cost) <= 1e-12:
            best_cost_mass += count
            if (
                count > best_allocation_count
                or (count == best_allocation_count and (best_tuple is None or allocation_tuple < best_tuple))
            ):
                best_allocation = allocation.copy()
                best_allocation_count = count

    if feasible_total == 0 or best_allocation is None:
        raise RuntimeError("No feasible sample was observed in the provided counts.")

    return {
        "allocation": best_allocation,
        "expected_cost": float(expected_cost / total),
        "best_sample_cost": float(best_cost),
        "feasible_fraction": float(feasible_total / total),
        "success_probability": float(best_cost_mass / total),
    }
