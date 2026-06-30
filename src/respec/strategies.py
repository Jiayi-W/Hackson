from __future__ import annotations

import numpy as np

from .environment import weighted_bray_curtis_change
from .exact import ranked_feasible_allocations, solve_offline_dp
from .objective import classical_objective, interference_cost, switching_ratio
from .optimizer import draw_random_initial_parameters, optimize_snapshot_qaoa
from .types import DynamicSequence, OptimizationTrace, SnapshotResult, StrategyRollout


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


def draw_random_feasible_allocation(
    n_users: int,
    n_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.integers(0, n_channels, size=n_users, dtype=int)


def _strategy_initial_allocation(
    method: str,
    previous_allocation: np.ndarray | None,
    fresh_allocation: np.ndarray,
) -> np.ndarray:
    if previous_allocation is not None and method in {"State", "Combined"}:
        return previous_allocation.copy()
    return fresh_allocation.copy()


def _strategy_initial_parameters(
    method: str,
    previous_parameters: np.ndarray | None,
    fresh_parameters: np.ndarray,
) -> np.ndarray:
    if previous_parameters is not None and method in {"Param", "Combined"}:
        return previous_parameters.copy()
    return fresh_parameters.copy()


def run_quantum_strategy_suite(
    sequence: DynamicSequence,
    n_channels: int,
    lambda_switch: float,
    methods: tuple[str, ...] = ("Cold", "Param", "State", "Combined"),
    optimization_shots: int = 256,
    final_shots: int = 2048,
    evaluations: int = 24,
    seed: int = 11,
) -> dict[str, StrategyRollout]:
    rng = np.random.default_rng(seed)
    n_users = sequence.snapshots[0].weights.shape[0]
    time_steps = len(sequence.snapshots)

    fresh_allocations = [
        draw_random_feasible_allocation(n_users, n_channels, rng) for _ in range(time_steps)
    ]
    fresh_parameters = [draw_random_initial_parameters(rng) for _ in range(time_steps)]

    shared_t0_result, shared_t0_trace = optimize_snapshot_qaoa(
        weights=sequence.snapshots[0].weights,
        previous=None,
        initial_allocation=fresh_allocations[0],
        initial_parameters=fresh_parameters[0],
        lambda_switch=lambda_switch,
        n_channels=n_channels,
        optimization_shots=optimization_shots,
        final_shots=final_shots,
        evaluations=evaluations,
        seed=seed,
        seed_transpiler=seed,
    )

    suite: dict[str, StrategyRollout] = {}
    for method in methods:
        snapshot_results: list[SnapshotResult] = [shared_t0_result]
        traces: list[OptimizationTrace] = [shared_t0_trace]
        previous_allocation = shared_t0_result.allocation.copy()
        previous_parameters = shared_t0_result.parameters.copy()
        step_costs = [float(shared_t0_result.best_sample_cost)]
        interference_terms = [interference_cost(shared_t0_result.allocation, sequence.snapshots[0].weights)]
        switch_terms = [0.0]
        allocations = [shared_t0_result.allocation.copy()]

        for t in range(1, time_steps):
            initial_allocation = _strategy_initial_allocation(
                method,
                previous_allocation=previous_allocation,
                fresh_allocation=fresh_allocations[t],
            )
            initial_parameters = _strategy_initial_parameters(
                method,
                previous_parameters=previous_parameters,
                fresh_parameters=fresh_parameters[t],
            )
            result, trace = optimize_snapshot_qaoa(
                weights=sequence.snapshots[t].weights,
                previous=previous_allocation,
                initial_allocation=initial_allocation,
                initial_parameters=initial_parameters,
                lambda_switch=lambda_switch,
                n_channels=n_channels,
                optimization_shots=optimization_shots,
                final_shots=final_shots,
                evaluations=evaluations,
                seed=seed + 1_000 * t + sum(ord(char) for char in method),
                seed_transpiler=seed,
            )
            snapshot_results.append(result)
            traces.append(trace)
            allocations.append(result.allocation.copy())
            step_costs.append(float(result.best_sample_cost))
            interference_terms.append(interference_cost(result.allocation, sequence.snapshots[t].weights))
            switch_terms.append(switching_ratio(result.allocation, previous_allocation))
            previous_allocation = result.allocation.copy()
            previous_parameters = result.parameters.copy()

        suite[method] = StrategyRollout(
            method=method,
            allocations=allocations,
            step_costs=np.array(step_costs, dtype=float),
            cumulative_costs=np.cumsum(step_costs),
            cumulative_interference=np.cumsum(interference_terms),
            cumulative_switches=np.cumsum(switch_terms),
            snapshot_results=snapshot_results,
            optimization_traces=traces,
        )
    return suite
