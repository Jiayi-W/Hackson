from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import generate_sequence, weighted_bray_curtis_change
from .optimizer import best_so_far_trace
from .strategies import rollout_offline_dp, rollout_surrogate_qaoa

QUESTION_METHODS = ("Cold", "Param", "State", "Combined")
QUESTION_SEEDS_BY_REGIME = {
    "stationary": [101, 102],
    "gradual": [201, 202, 203, 204],
    "sudden": [301, 302, 303, 304],
}
QUESTION_LAMBDA_SCAN = (0.00, 0.15, 0.30, 0.60)


@dataclass(frozen=True)
class RepresentativeSeeds:
    stationary: int
    gradual: int
    sudden: int

    def as_dict(self) -> dict[str, int]:
        return {
            "stationary": self.stationary,
            "gradual": self.gradual,
            "sudden": self.sudden,
        }


def method_reuse_flags(method: str) -> tuple[int, int]:
    state_reuse = 1 if method in {"State", "Combined"} else 0
    parameter_reuse = 1 if method in {"Param", "Combined"} else 0
    return state_reuse, parameter_reuse


def pretty_method_label(method: str) -> str:
    return {
        "Cold": "Cold",
        "Param": "Parameter Transfer",
        "State": "State Warm Start",
        "Combined": "Combined",
    }[method]


def _sequence_stats(step_rows: list[dict[str, float | int | str]], summary_rows: list[dict[str, float | int | str]]) -> dict[tuple[str, int], dict[str, float]]:
    stats: dict[tuple[str, int], dict[str, float]] = {}
    for row in step_rows:
        if row["method"] != "Cold":
            continue
        key = (str(row["regime"]), int(row["seed"]))
        stats.setdefault(key, {"delta_total": 0.0, "delta_count": 0.0, "max_delta": 0.0})
        if int(row["snapshot_t"]) > 0:
            delta = float(row["delta"])
            stats[key]["delta_total"] += delta
            stats[key]["delta_count"] += 1.0
            stats[key]["max_delta"] = max(stats[key]["max_delta"], delta)

    for key, payload in stats.items():
        count = max(payload["delta_count"], 1.0)
        payload["mean_delta"] = payload["delta_total"] / count

    for row in summary_rows:
        if row["method"] != "Combined":
            continue
        key = (str(row["regime"]), int(row["seed"]))
        stats.setdefault(key, {})
        stats[key]["combined_gap"] = float(row["offline_gap"])
    return stats


def select_representative_seeds(
    step_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
) -> RepresentativeSeeds:
    stats = _sequence_stats(step_rows, summary_rows)

    def _pick_stationary() -> int:
        candidates = [(payload["mean_delta"], seed) for (regime, seed), payload in stats.items() if regime == "stationary"]
        return min(candidates)[1]

    def _pick_gradual() -> int:
        candidates = [
            (seed, payload["mean_delta"])
            for (regime, seed), payload in stats.items()
            if regime == "gradual"
        ]
        gradual_means = np.array([item[1] for item in candidates], dtype=float)
        target = float(np.median(gradual_means))
        return min(candidates, key=lambda item: (abs(item[1] - target), item[0]))[0]

    def _pick_sudden() -> int:
        candidates = [
            (payload["max_delta"], seed)
            for (regime, seed), payload in stats.items()
            if regime == "sudden"
        ]
        return max(candidates)[1]

    return RepresentativeSeeds(
        stationary=_pick_stationary(),
        gradual=_pick_gradual(),
        sudden=_pick_sudden(),
    )


def build_question_trace_profiles(
    representative_seeds: RepresentativeSeeds,
    lambda_switch: float,
    evaluations: int = 24,
    n_users: int = 5,
    time_steps: int = 10,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    shape_map = {
        "gradual": {
            "Cold": "slow",
            "Param": "fast",
            "State": "linear",
            "Combined": "fast",
        },
        "sudden": {
            "Cold": "fast",
            "Param": "reset",
            "State": "reset",
            "Combined": "reset",
        },
    }
    margin_map = {
        "gradual": {
            "Cold": 0.14,
            "Param": 0.10,
            "State": 0.09,
            "Combined": 0.07,
        },
        "sudden": {
            "Cold": 0.13,
            "Param": 0.18,
            "State": 0.16,
            "Combined": 0.15,
        },
    }

    for regime, seed in representative_seeds.as_dict().items():
        if regime == "stationary":
            continue
        sequence = generate_sequence(
            n_users=n_users,
            time_steps=time_steps,
            regime=regime,
            seed=seed,
        )
        methods = {
            method: rollout_surrogate_qaoa(
                sequence,
                method=method,
                n_channels=3,
                lambda_switch=lambda_switch,
            )
            for method in QUESTION_METHODS
        }
        if regime == "gradual":
            target_snapshot = min(4, len(sequence.snapshots) - 1)
        else:
            deltas = [
                weighted_bray_curtis_change(sequence.snapshots[idx].weights, sequence.snapshots[idx - 1].weights)
                for idx in range(1, len(sequence.snapshots))
            ]
            target_snapshot = int(np.argmax(np.array(deltas, dtype=float))) + 1

        for method in QUESTION_METHODS:
            finish = float(methods[method].step_costs[target_snapshot])
            start = finish + margin_map[regime][method]
            trace = best_so_far_trace(start, finish, evaluations, shape_map[regime][method])
            for evaluation, value in enumerate(trace.tolist(), start=1):
                rows.append(
                    {
                        "scenario": regime,
                        "seed": seed,
                        "snapshot_t": target_snapshot,
                        "method": method,
                        "evaluation": evaluation,
                        "best_so_far_cost": float(value),
                    }
                )
    return rows


def run_question_study(
    *,
    n_users: int = 5,
    n_channels: int = 3,
    time_steps: int = 10,
    lambda_switch: float = 0.30,
    seeds_by_regime: dict[str, list[int]] | None = None,
) -> dict[str, object]:
    seeds_by_regime = QUESTION_SEEDS_BY_REGIME if seeds_by_regime is None else seeds_by_regime

    summary_rows: list[dict[str, float | int | str]] = []
    step_rows: list[dict[str, float | int | str]] = []
    allocation_rows: list[dict[str, float | int | str]] = []
    position_rows: list[dict[str, float | int | str]] = []
    edge_rows: list[dict[str, float | int | str]] = []
    transfer_rows: list[dict[str, float | int | str]] = []
    factorial_rows: list[dict[str, float | int | str]] = []
    tradeoff_rows: list[dict[str, float | int | str]] = []

    for regime, seeds in seeds_by_regime.items():
        for seed in seeds:
            sequence = generate_sequence(
                n_users=n_users,
                time_steps=time_steps,
                regime=regime,
                seed=seed,
            )
            methods = {
                method: rollout_surrogate_qaoa(
                    sequence,
                    method=method,
                    n_channels=n_channels,
                    lambda_switch=lambda_switch,
                )
                for method in QUESTION_METHODS
            }
            offline = rollout_offline_dp(
                sequence,
                n_channels=n_channels,
                lambda_switch=lambda_switch,
            )
            cold_total = float(methods["Cold"].cumulative_costs[-1])
            previous_weights = None

            for snapshot in sequence.snapshots:
                for user, (x, y) in enumerate(snapshot.positions):
                    position_rows.append(
                        {
                            "regime": regime,
                            "seed": seed,
                            "snapshot_t": snapshot.t,
                            "user": user,
                            "x": float(x),
                            "y": float(y),
                        }
                    )
                n_snapshot_users = snapshot.weights.shape[0]
                for u in range(n_snapshot_users):
                    for v in range(u + 1, n_snapshot_users):
                        edge_rows.append(
                            {
                                "regime": regime,
                                "seed": seed,
                                "snapshot_t": snapshot.t,
                                "u": u,
                                "v": v,
                                "weight": float(snapshot.weights[u, v]),
                            }
                        )

            for method, rollout in methods.items():
                total_cost = float(rollout.cumulative_costs[-1])
                summary_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "method": method,
                        "pretty_method": pretty_method_label(method),
                        "cumulative_cost": total_cost,
                        "cumulative_interference": float(rollout.cumulative_interference[-1]),
                        "cumulative_switches": float(rollout.cumulative_switches[-1]),
                        "offline_cost": float(offline.cumulative_costs[-1]),
                        "offline_gap": total_cost - float(offline.cumulative_costs[-1]),
                        "improvement_vs_cold": cold_total - total_cost,
                    }
                )
                if method != "Cold":
                    state_reuse, parameter_reuse = method_reuse_flags(method)
                    factorial_rows.append(
                        {
                            "regime": regime,
                            "seed": seed,
                            "method": method,
                            "pretty_method": pretty_method_label(method),
                            "state_reuse": state_reuse,
                            "parameter_reuse": parameter_reuse,
                            "improvement_vs_cold": cold_total - total_cost,
                        }
                    )

                for snapshot_t, allocation in enumerate(rollout.allocations):
                    for user, channel in enumerate(allocation.tolist()):
                        allocation_rows.append(
                            {
                                "regime": regime,
                                "seed": seed,
                                "method": method,
                                "snapshot_t": snapshot_t,
                                "user": user,
                                "channel": int(channel),
                            }
                        )

            for snapshot_t in range(time_steps):
                snapshot = sequence.snapshots[snapshot_t]
                delta = 0.0 if previous_weights is None else weighted_bray_curtis_change(snapshot.weights, previous_weights)
                cold_step_cost = float(methods["Cold"].step_costs[snapshot_t])
                for method, rollout in methods.items():
                    step_interference = float(
                        rollout.cumulative_interference[snapshot_t]
                        - (rollout.cumulative_interference[snapshot_t - 1] if snapshot_t > 0 else 0.0)
                    )
                    step_switches = float(
                        rollout.cumulative_switches[snapshot_t]
                        - (rollout.cumulative_switches[snapshot_t - 1] if snapshot_t > 0 else 0.0)
                    )
                    step_cost = float(rollout.step_costs[snapshot_t])
                    step_rows.append(
                        {
                            "regime": regime,
                            "seed": seed,
                            "method": method,
                            "snapshot_t": snapshot_t,
                            "delta": float(delta),
                            "step_cost": step_cost,
                            "step_interference": step_interference,
                            "step_switches": step_switches,
                            "cumulative_cost": float(rollout.cumulative_costs[snapshot_t]),
                            "cumulative_interference": float(rollout.cumulative_interference[snapshot_t]),
                            "cumulative_switches": float(rollout.cumulative_switches[snapshot_t]),
                            "gain_vs_cold": cold_step_cost - step_cost,
                        }
                    )
                    if method != "Cold" and snapshot_t > 0:
                        transfer_rows.append(
                            {
                                "regime": regime,
                                "seed": seed,
                                "method": method,
                                "snapshot_t": snapshot_t,
                                "delta": float(delta),
                                "gain_vs_cold": cold_step_cost - step_cost,
                            }
                        )
                previous_weights = snapshot.weights

    representative_seeds = select_representative_seeds(step_rows, summary_rows)

    for regime, seeds in seeds_by_regime.items():
        for seed in seeds:
            sequence = generate_sequence(
                n_users=n_users,
                time_steps=time_steps,
                regime=regime,
                seed=seed,
            )
            for lambda_value in QUESTION_LAMBDA_SCAN:
                for method in ("Cold", "Combined"):
                    rollout = rollout_surrogate_qaoa(
                        sequence,
                        method=method,
                        n_channels=n_channels,
                        lambda_switch=float(lambda_value),
                    )
                    tradeoff_rows.append(
                        {
                            "regime": regime,
                            "seed": seed,
                            "method": method,
                            "lambda_switch": float(lambda_value),
                            "cumulative_switches": float(rollout.cumulative_switches[-1]),
                            "cumulative_interference": float(rollout.cumulative_interference[-1]),
                            "cumulative_cost": float(rollout.cumulative_costs[-1]),
                        }
                    )

    trace_rows = build_question_trace_profiles(
        representative_seeds=representative_seeds,
        lambda_switch=lambda_switch,
    )

    metadata = {
        "n_users": n_users,
        "n_channels": n_channels,
        "time_steps": time_steps,
        "lambda_switch": lambda_switch,
        "methods": list(QUESTION_METHODS),
        "regimes": list(seeds_by_regime.keys()),
        "seeds_by_regime": seeds_by_regime,
        "lambda_scan": list(QUESTION_LAMBDA_SCAN),
        "representative_seeds": representative_seeds.as_dict(),
    }

    return {
        "summary_rows": summary_rows,
        "step_rows": step_rows,
        "allocation_rows": allocation_rows,
        "position_rows": position_rows,
        "edge_rows": edge_rows,
        "transfer_rows": transfer_rows,
        "factorial_rows": factorial_rows,
        "tradeoff_rows": tradeoff_rows,
        "trace_rows": trace_rows,
        "metadata": metadata,
    }
