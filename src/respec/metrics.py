from __future__ import annotations

import numpy as np

from .environment import generate_sequence, weighted_bray_curtis_change
from .heuristics import rollout_heuristic
from .strategies import rollout_surrogate_qaoa


def build_transfer_gain_records(lambda_switch: float = 0.30) -> dict[str, np.ndarray]:
    deltas: list[float] = []
    gains: list[float] = []
    regimes: list[str] = []

    seeds_by_regime = {
        "stationary": [3, 5, 7],
        "gradual": [11, 13, 17],
        "sudden": [19, 23, 29],
    }

    for regime, seeds in seeds_by_regime.items():
        for seed in seeds:
            sequence = generate_sequence(5, 10, regime=regime, seed=seed)
            cold = rollout_surrogate_qaoa(sequence, "Cold", n_channels=3, lambda_switch=lambda_switch)
            combined = rollout_surrogate_qaoa(sequence, "Combined", n_channels=3, lambda_switch=lambda_switch)
            for t in range(1, len(sequence.snapshots)):
                delta = weighted_bray_curtis_change(
                    sequence.snapshots[t].weights,
                    sequence.snapshots[t - 1].weights,
                )
                deltas.append(delta)
                gains.append(cold.step_costs[t] - combined.step_costs[t])
                regimes.append(regime)

    return {
        "delta": np.array(deltas),
        "gain": np.array(gains),
        "regime": np.array(regimes),
    }


def build_tradeoff_table(lambda_values: list[float]) -> dict[str, dict[str, tuple[float, float]]]:
    sequence = generate_sequence(5, 10, regime="sudden", seed=11)
    table: dict[str, dict[str, tuple[float, float]]] = {"Combined": {}, "Greedy": {}}

    for value in lambda_values:
        combined = rollout_surrogate_qaoa(sequence, "Combined", n_channels=3, lambda_switch=value)
        greedy = rollout_heuristic(sequence, "Greedy", n_channels=3, lambda_switch=value)
        table["Combined"][f"{value:.2f}"] = (
            float(combined.cumulative_switches[-1]),
            float(combined.cumulative_interference[-1]),
        )
        table["Greedy"][f"{value:.2f}"] = (
            float(greedy.cumulative_switches[-1]),
            float(greedy.cumulative_interference[-1]),
        )
    return table

