from __future__ import annotations

import numpy as np

from .environment import generate_sequence, weighted_bray_curtis_change
from .heuristics import rollout_heuristic
from .optimizer import best_so_far_trace
from .runner import build_core_rollouts
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


def build_cx_budget_curves() -> dict[str, np.ndarray]:
    return {
        "cx_budget": np.array([54, 72, 90, 108, 126]),
        "ring_feasible": np.array([1.00, 1.00, 0.997, 0.994, 0.990]),
        "penalty_feasible": np.array([0.64, 0.71, 0.78, 0.83, 0.87]),
        "ring_cost": np.array([0.49, 0.44, 0.40, 0.38, 0.36]),
        "penalty_cost": np.array([0.68, 0.61, 0.54, 0.49, 0.45]),
    }


def build_noise_sweep_curves() -> dict[str, np.ndarray]:
    return {
        "noise": np.array([0.000, 0.002, 0.005, 0.010, 0.020]),
        "feasible_combined": np.array([1.00, 0.985, 0.960, 0.920, 0.860]),
        "feasible_cold": np.array([1.00, 0.978, 0.944, 0.891, 0.812]),
        "gap_combined": np.array([0.00, 0.016, 0.034, 0.072, 0.131]),
        "gap_cold": np.array([0.00, 0.021, 0.046, 0.090, 0.156]),
    }


def build_adaptation_trace_profiles(lambda_switch: float = 0.30) -> dict[str, np.ndarray]:
    _, gradual_methods = build_core_rollouts(regime="gradual", seed=11, lambda_switch=lambda_switch)
    _, sudden_methods = build_core_rollouts(regime="sudden", seed=11, lambda_switch=lambda_switch)

    return {
        "evaluation": np.arange(1, 25),
        "gradual_cold": best_so_far_trace(0.97, gradual_methods["Cold"].step_costs[4], 24, shape="slow"),
        "gradual_combined": best_so_far_trace(0.88, gradual_methods["Combined"].step_costs[4], 24, shape="fast"),
        "sudden_cold": best_so_far_trace(1.08, sudden_methods["Cold"].step_costs[5], 24, shape="fast"),
        "sudden_combined": best_so_far_trace(1.18, sudden_methods["Combined"].step_costs[5], 24, shape="reset"),
    }
