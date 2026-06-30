from __future__ import annotations

import numpy as np

from .objective import classical_objective
from .types import DynamicSequence, StrategyRollout


def solve_greedy_step(
    weights: np.ndarray,
    previous: np.ndarray | None,
    n_channels: int,
    lambda_switch: float,
) -> np.ndarray:
    n_users = weights.shape[0]
    allocation = np.zeros(n_users, dtype=int)
    order = np.argsort(weights.sum(axis=1))[::-1]

    for user in order:
        best_channel = 0
        best_score = float("inf")
        for channel in range(n_channels):
            trial = allocation.copy()
            trial[user] = channel
            score = classical_objective(trial, weights, previous, lambda_switch)
            if score < best_score:
                best_score = score
                best_channel = channel
        allocation[user] = best_channel

    return allocation


def solve_local_search_step(
    weights: np.ndarray,
    previous: np.ndarray | None,
    n_channels: int,
    lambda_switch: float,
) -> np.ndarray:
    current = solve_greedy_step(weights, previous, n_channels, lambda_switch)
    current_score = classical_objective(current, weights, previous, lambda_switch)

    improved = True
    while improved:
        improved = False
        for user in range(current.shape[0]):
            for channel in range(n_channels):
                if channel == current[user]:
                    continue
                trial = current.copy()
                trial[user] = channel
                score = classical_objective(trial, weights, previous, lambda_switch)
                if score + 1e-12 < current_score:
                    current = trial
                    current_score = score
                    improved = True
    return current


def rollout_heuristic(
    sequence: DynamicSequence,
    method: str,
    n_channels: int,
    lambda_switch: float,
) -> StrategyRollout:
    allocations: list[np.ndarray] = []
    step_costs: list[float] = []
    interference_terms: list[float] = []
    switch_terms: list[float] = []
    previous: np.ndarray | None = None

    for snapshot in sequence.snapshots:
        if method == "Greedy":
            allocation = solve_greedy_step(snapshot.weights, previous, n_channels, lambda_switch)
        elif method == "Local Search":
            allocation = solve_local_search_step(snapshot.weights, previous, n_channels, lambda_switch)
        else:
            raise ValueError(f"Unknown heuristic method: {method}")

        interference = classical_objective(allocation, snapshot.weights, None, 0.0)
        switches = 0.0 if previous is None else float(np.mean(allocation != previous))
        total = interference + lambda_switch * switches

        allocations.append(allocation)
        step_costs.append(total)
        interference_terms.append(interference)
        switch_terms.append(switches)
        previous = allocation

    return StrategyRollout(
        method=method,
        allocations=allocations,
        step_costs=np.array(step_costs),
        cumulative_costs=np.cumsum(step_costs),
        cumulative_interference=np.cumsum(interference_terms),
        cumulative_switches=np.cumsum(switch_terms),
    )

