from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from .objective import classical_objective
from .types import DynamicSequence


@lru_cache(maxsize=None)
def all_allocations(n_users: int, n_channels: int) -> tuple[tuple[int, ...], ...]:
    return tuple(product(range(n_channels), repeat=n_users))


def ranked_feasible_allocations(
    weights: np.ndarray,
    previous: np.ndarray | None,
    n_channels: int,
    lambda_switch: float,
    limit: int = 8,
) -> list[np.ndarray]:
    n_users = weights.shape[0]
    scored: list[tuple[float, tuple[int, ...]]] = []
    for allocation in all_allocations(n_users, n_channels):
        alloc_arr = np.array(allocation, dtype=int)
        score = classical_objective(alloc_arr, weights, previous, lambda_switch)
        scored.append((score, allocation))

    scored.sort(key=lambda item: (item[0], item[1]))
    unique: list[np.ndarray] = []
    for _, allocation in scored:
        alloc_arr = np.array(allocation, dtype=int)
        if not unique or not np.array_equal(unique[-1], alloc_arr):
            unique.append(alloc_arr)
        if len(unique) >= limit:
            break
    return unique


def solve_exact_step(
    weights: np.ndarray,
    previous: np.ndarray | None,
    n_channels: int,
    lambda_switch: float,
) -> tuple[float, list[np.ndarray]]:
    ranked = ranked_feasible_allocations(weights, previous, n_channels, lambda_switch, limit=8)
    best_value = classical_objective(ranked[0], weights, previous, lambda_switch)
    best_allocations = [
        alloc
        for alloc in ranked
        if abs(classical_objective(alloc, weights, previous, lambda_switch) - best_value) < 1e-10
    ]
    return best_value, best_allocations


def solve_offline_dp(
    sequence: DynamicSequence,
    n_channels: int,
    lambda_switch: float,
) -> tuple[float, list[np.ndarray]]:
    n_users = sequence.snapshots[0].weights.shape[0]
    states = [np.array(a, dtype=int) for a in all_allocations(n_users, n_channels)]
    keys = [tuple(state.tolist()) for state in states]

    costs: dict[tuple[int, ...], float] = {}
    parents: list[dict[tuple[int, ...], tuple[int, ...] | None]] = []

    for t, snapshot in enumerate(sequence.snapshots):
        layer_parent: dict[tuple[int, ...], tuple[int, ...] | None] = {}
        next_costs: dict[tuple[int, ...], float] = {}

        for state, key in zip(states, keys, strict=True):
            if t == 0:
                next_costs[key] = classical_objective(state, snapshot.weights, None, lambda_switch)
                layer_parent[key] = None
                continue

            best_prev: tuple[int, ...] | None = None
            best_value = float("inf")
            for prev_state, prev_key in zip(states, keys, strict=True):
                candidate = costs[prev_key] + classical_objective(
                    state,
                    snapshot.weights,
                    prev_state,
                    lambda_switch,
                )
                if candidate < best_value:
                    best_value = candidate
                    best_prev = prev_key

            next_costs[key] = best_value
            layer_parent[key] = best_prev

        parents.append(layer_parent)
        costs = next_costs

    best_final_key = min(costs, key=costs.get)
    path: list[np.ndarray] = []
    current = best_final_key
    for t in range(len(sequence.snapshots) - 1, -1, -1):
        path.append(np.array(current, dtype=int))
        current = parents[t][current]
        if current is None:
            break

    path.reverse()
    return costs[best_final_key], path

