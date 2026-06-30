from __future__ import annotations

import numpy as np

from .environment import generate_sequence, weighted_bray_curtis_change
from .heuristics import rollout_heuristic
from .strategies import rollout_offline_dp, rollout_surrogate_qaoa
from .types import DynamicSequence, StrategyRollout


def build_core_rollouts(
    regime: str = "sudden",
    seed: int = 11,
    lambda_switch: float = 0.30,
) -> tuple[DynamicSequence, dict[str, StrategyRollout]]:
    sequence = generate_sequence(n_users=5, time_steps=10, regime=regime, seed=seed)
    methods = {
        name: rollout_surrogate_qaoa(sequence, name, n_channels=3, lambda_switch=lambda_switch)
        for name in ("Cold", "Param", "State", "Combined")
    }
    methods["Greedy"] = rollout_heuristic(sequence, "Greedy", n_channels=3, lambda_switch=lambda_switch)
    methods["Local Search"] = rollout_heuristic(sequence, "Local Search", n_channels=3, lambda_switch=lambda_switch)
    methods["Offline DP"] = rollout_offline_dp(sequence, n_channels=3, lambda_switch=lambda_switch)
    return sequence, methods


def build_demo_rollouts(seed: int = 11, lambda_switch: float = 0.30):
    return build_core_rollouts(regime="sudden", seed=seed, lambda_switch=lambda_switch)


def build_final_summary_records(methods: dict[str, StrategyRollout]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method, rollout in methods.items():
        rows.append(
            {
                "method": method,
                "time_steps": int(len(rollout.step_costs)),
                "final_cost": float(rollout.cumulative_costs[-1]),
                "final_interference": float(rollout.cumulative_interference[-1]),
                "final_switches": float(rollout.cumulative_switches[-1]),
            }
        )
    return rows


def build_step_metric_records(
    sequence: DynamicSequence,
    methods: dict[str, StrategyRollout],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    previous_weights = None

    for t, snapshot in enumerate(sequence.snapshots):
        delta = 0.0 if previous_weights is None else weighted_bray_curtis_change(snapshot.weights, previous_weights)
        for method, rollout in methods.items():
            step_interference = float(
                rollout.cumulative_interference[t]
                - (rollout.cumulative_interference[t - 1] if t > 0 else 0.0)
            )
            step_switches = float(
                rollout.cumulative_switches[t]
                - (rollout.cumulative_switches[t - 1] if t > 0 else 0.0)
            )
            rows.append(
                {
                    "method": method,
                    "snapshot_t": int(t),
                    "graph_change": float(delta),
                    "step_cost": float(rollout.step_costs[t]),
                    "step_interference": step_interference,
                    "step_switches": step_switches,
                    "cumulative_cost": float(rollout.cumulative_costs[t]),
                    "cumulative_interference": float(rollout.cumulative_interference[t]),
                    "cumulative_switches": float(rollout.cumulative_switches[t]),
                }
            )
        previous_weights = snapshot.weights
    return rows


def build_allocation_records(methods: dict[str, StrategyRollout]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for method, rollout in methods.items():
        for t, allocation in enumerate(rollout.allocations):
            for user, channel in enumerate(allocation.tolist()):
                rows.append(
                    {
                        "method": method,
                        "snapshot_t": int(t),
                        "user": int(user),
                        "channel": int(channel),
                    }
                )
    return rows


def build_snapshot_position_records(sequence: DynamicSequence) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for snapshot in sequence.snapshots:
        for user, (x, y) in enumerate(snapshot.positions):
            rows.append(
                {
                    "regime": sequence.regime,
                    "seed": int(sequence.seed),
                    "snapshot_t": int(snapshot.t),
                    "user": int(user),
                    "x": float(x),
                    "y": float(y),
                }
            )
    return rows


def build_snapshot_edge_records(sequence: DynamicSequence) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for snapshot in sequence.snapshots:
        n_users = snapshot.weights.shape[0]
        for u in range(n_users):
            for v in range(u + 1, n_users):
                rows.append(
                    {
                        "regime": sequence.regime,
                        "seed": int(sequence.seed),
                        "snapshot_t": int(snapshot.t),
                        "u": int(u),
                        "v": int(v),
                        "weight": float(snapshot.weights[u, v]),
                        "active": int(snapshot.weights[u, v] > 0.0),
                    }
                )
    return rows
