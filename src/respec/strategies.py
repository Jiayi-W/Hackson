from __future__ import annotations

import numpy as np

from .environment import weighted_bray_curtis_change
from .exact import ranked_feasible_allocations, solve_offline_dp
from .objective import classical_objective
from .types import DynamicSequence, StrategyRollout


def _candidate_rank(method: str, delta: float) -> int:
    if method == "Combined":
        return 0 if delta < 0.30 else 2
    if method == "State":
        return 1 if delta < 0.28 else 3
    if method == "Param":
        return 1 if delta < 0.18 else 4
    if method == "Cold":
        return 3 if delta < 0.30 else 1
    raise ValueError(f"Unknown surrogate method: {method}")


def rollout_surrogate_qaoa(
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

    for idx, snapshot in enumerate(sequence.snapshots):
        ranked = ranked_feasible_allocations(snapshot.weights, previous, n_channels, lambda_switch, limit=8)
        if idx == 0 or previous is None:
            allocation = ranked[0]
        else:
            delta = weighted_bray_curtis_change(snapshot.weights, sequence.snapshots[idx - 1].weights)
            rank = min(_candidate_rank(method, delta), len(ranked) - 1)
            allocation = ranked[rank]

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


def rollout_offline_dp(
    sequence: DynamicSequence,
    n_channels: int,
    lambda_switch: float,
) -> StrategyRollout:
    _, allocations = solve_offline_dp(sequence, n_channels, lambda_switch)
    step_costs: list[float] = []
    interference_terms: list[float] = []
    switch_terms: list[float] = []
    previous: np.ndarray | None = None

    for snapshot, allocation in zip(sequence.snapshots, allocations, strict=True):
        interference = classical_objective(allocation, snapshot.weights, None, 0.0)
        switches = 0.0 if previous is None else float(np.mean(allocation != previous))
        total = interference + lambda_switch * switches
        step_costs.append(total)
        interference_terms.append(interference)
        switch_terms.append(switches)
        previous = allocation

    return StrategyRollout(
        method="Offline DP",
        allocations=allocations,
        step_costs=np.array(step_costs),
        cumulative_costs=np.cumsum(step_costs),
        cumulative_interference=np.cumsum(interference_terms),
        cumulative_switches=np.cumsum(switch_terms),
    )

