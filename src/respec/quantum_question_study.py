from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np

from .environment import generate_sequence, weighted_bray_curtis_change
from .question_study import (
    QUESTION_LAMBDA_SCAN,
    QUESTION_METHODS,
    QUESTION_SEEDS_BY_REGIME,
    method_reuse_flags,
    pretty_method_label,
)
from .strategies import rollout_offline_dp, run_quantum_strategy_suite
from .types import DynamicSequence, StrategyRollout

QUANTUM_SMOKE_SEEDS_BY_REGIME = {
    "stationary": [101],
    "gradual": [201],
    "sudden": [304],
}

QUANTUM_CONTINUOUS_SMOKE_SEEDS_BY_REGIME = {
    **QUANTUM_SMOKE_SEEDS_BY_REGIME,
    "continuous_sudden": [401],
}

FULL_QUANTUM_SEEDS_BY_REGIME = {
    **QUESTION_SEEDS_BY_REGIME,
    "continuous_sudden": [401, 402, 403, 404],
}


def _snapshot_metric(result, key: str) -> float | int:
    return getattr(result, key)


def _mean_snapshot_metric(rollout: StrategyRollout, key: str) -> float:
    if not rollout.snapshot_results:
        return 0.0
    values = np.array([float(_snapshot_metric(result, key)) for result in rollout.snapshot_results], dtype=float)
    return float(values.mean())


def _sum_snapshot_metric(rollout: StrategyRollout, key: str) -> float:
    if not rollout.snapshot_results:
        return 0.0
    values = np.array([float(_snapshot_metric(result, key)) for result in rollout.snapshot_results], dtype=float)
    return float(values.sum())


def _trace_snapshot_t(sequence: DynamicSequence) -> int:
    if sequence.regime == "gradual":
        return min(4, len(sequence.snapshots) - 1)
    deltas = [
        weighted_bray_curtis_change(sequence.snapshots[idx].weights, sequence.snapshots[idx - 1].weights)
        for idx in range(1, len(sequence.snapshots))
    ]
    return int(np.argmax(np.array(deltas, dtype=float))) + 1


def _build_trace_rows(
    *,
    regime: str,
    seed: int,
    suite: dict[str, StrategyRollout],
    target_snapshot: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method in QUESTION_METHODS:
        traces = suite[method].optimization_traces
        if traces is None:
            continue
        trace = traces[target_snapshot]
        for evaluation, (objective_value, best_value) in enumerate(
            zip(trace.objective_values.tolist(), trace.best_so_far.tolist(), strict=True),
            start=1,
        ):
            rows.append(
                {
                    "scenario": regime,
                    "seed": seed,
                    "snapshot_t": target_snapshot,
                    "method": method,
                    "evaluation": evaluation,
                    "objective_value": float(objective_value),
                    "best_so_far_cost": float(best_value),
                }
            )
    return rows


def _sequence_rows_payload(
    *,
    regime: str,
    seed: int,
    n_users: int,
    n_channels: int,
    time_steps: int,
    lambda_switch: float,
    lambda_scan: tuple[float, ...],
    optimization_shots: int,
    final_shots: int,
    evaluations: int,
) -> dict[str, object]:
    sequence = generate_sequence(
        n_users=n_users,
        time_steps=time_steps,
        regime=regime,
        seed=seed,
    )
    suite = run_quantum_strategy_suite(
        sequence,
        n_channels=n_channels,
        lambda_switch=lambda_switch,
        methods=QUESTION_METHODS,
        optimization_shots=optimization_shots,
        final_shots=final_shots,
        evaluations=evaluations,
        seed=seed,
    )
    offline = rollout_offline_dp(
        sequence,
        n_channels=n_channels,
        lambda_switch=lambda_switch,
    )
    cold_total = float(suite["Cold"].cumulative_costs[-1])
    previous_weights = None

    summary_rows: list[dict[str, float | int | str]] = []
    step_rows: list[dict[str, float | int | str]] = []
    allocation_rows: list[dict[str, float | int | str]] = []
    position_rows: list[dict[str, float | int | str]] = []
    edge_rows: list[dict[str, float | int | str]] = []
    transfer_rows: list[dict[str, float | int | str]] = []
    factorial_rows: list[dict[str, float | int | str]] = []
    tradeoff_rows: list[dict[str, float | int | str]] = []

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

    for method, rollout in suite.items():
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
                "mean_feasible_fraction": _mean_snapshot_metric(rollout, "feasible_fraction"),
                "mean_success_probability": _mean_snapshot_metric(rollout, "success_probability"),
                "mean_expected_cost": _mean_snapshot_metric(rollout, "expected_cost"),
                "mean_cx_count": _mean_snapshot_metric(rollout, "cx_count"),
                "mean_circuit_depth": _mean_snapshot_metric(rollout, "circuit_depth"),
                "total_wall_seconds": _sum_snapshot_metric(rollout, "wall_seconds"),
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
        cold_step_cost = float(suite["Cold"].step_costs[snapshot_t])
        for method, rollout in suite.items():
            result = None if rollout.snapshot_results is None else rollout.snapshot_results[snapshot_t]
            step_interference = float(
                rollout.cumulative_interference[snapshot_t]
                - (rollout.cumulative_interference[snapshot_t - 1] if snapshot_t > 0 else 0.0)
            )
            step_switches = float(
                rollout.cumulative_switches[snapshot_t]
                - (rollout.cumulative_switches[snapshot_t - 1] if snapshot_t > 0 else 0.0)
            )
            step_cost = float(rollout.step_costs[snapshot_t])
            row: dict[str, float | int | str] = {
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
            if result is not None:
                row.update(
                    {
                        "expected_cost": float(result.expected_cost),
                        "best_sample_cost": float(result.best_sample_cost),
                        "feasible_fraction": float(result.feasible_fraction),
                        "success_probability": float(result.success_probability),
                        "evaluations": int(result.evaluations),
                        "shots": int(result.shots),
                        "cx_count": int(result.cx_count),
                        "circuit_depth": int(result.circuit_depth),
                        "wall_seconds": float(result.wall_seconds),
                    }
                )
            step_rows.append(row)
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

    for lambda_value in lambda_scan:
        if math.isclose(float(lambda_value), float(lambda_switch), rel_tol=0.0, abs_tol=1e-12):
            lambda_suite = suite
        else:
            lambda_suite = run_quantum_strategy_suite(
                sequence,
                n_channels=n_channels,
                lambda_switch=float(lambda_value),
                methods=("Cold", "Combined"),
                optimization_shots=optimization_shots,
                final_shots=final_shots,
                evaluations=evaluations,
                seed=seed,
            )
        for method in ("Cold", "Combined"):
            rollout = lambda_suite[method]
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

    return {
        "summary_rows": summary_rows,
        "step_rows": step_rows,
        "allocation_rows": allocation_rows,
        "position_rows": position_rows,
        "edge_rows": edge_rows,
        "transfer_rows": transfer_rows,
        "factorial_rows": factorial_rows,
        "tradeoff_rows": tradeoff_rows,
    }


def _sequence_rows_payload_from_task(task: dict[str, object]) -> dict[str, object]:
    return _sequence_rows_payload(**task)


def _trace_rows_payload(
    *,
    regime: str,
    seed: int,
    n_users: int,
    n_channels: int,
    time_steps: int,
    lambda_switch: float,
    optimization_shots: int,
    final_shots: int,
    evaluations: int,
) -> list[dict[str, float | int | str]]:
    sequence = generate_sequence(
        n_users=n_users,
        time_steps=time_steps,
        regime=regime,
        seed=seed,
    )
    suite = run_quantum_strategy_suite(
        sequence,
        n_channels=n_channels,
        lambda_switch=lambda_switch,
        methods=QUESTION_METHODS,
        optimization_shots=optimization_shots,
        final_shots=final_shots,
        evaluations=evaluations,
        seed=seed,
    )
    return _build_trace_rows(
        regime=regime,
        seed=seed,
        suite=suite,
        target_snapshot=_trace_snapshot_t(sequence),
    )


def _trace_rows_payload_from_task(task: dict[str, object]) -> list[dict[str, float | int | str]]:
    return _trace_rows_payload(**task)


def _sequence_stats(
    step_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
) -> dict[tuple[str, int], dict[str, float]]:
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


def _select_representative_seeds(
    step_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
    regimes: list[str],
) -> dict[str, int]:
    stats = _sequence_stats(step_rows, summary_rows)
    picks: dict[str, int] = {}

    for regime in regimes:
        candidates = [
            (seed, payload)
            for (candidate_regime, seed), payload in stats.items()
            if candidate_regime == regime
        ]
        if not candidates:
            continue
        if regime == "stationary":
            picks[regime] = min(candidates, key=lambda item: (item[1].get("mean_delta", 0.0), item[0]))[0]
        elif regime == "gradual":
            means = np.array([item[1].get("mean_delta", 0.0) for item in candidates], dtype=float)
            target = float(np.median(means))
            picks[regime] = min(candidates, key=lambda item: (abs(item[1].get("mean_delta", 0.0) - target), item[0]))[0]
        elif regime == "sudden":
            picks[regime] = max(candidates, key=lambda item: (item[1].get("max_delta", 0.0), -item[0]))[0]
        else:
            picks[regime] = max(
                candidates,
                key=lambda item: (item[1].get("mean_delta", 0.0), item[1].get("max_delta", 0.0), -item[0]),
            )[0]
    return picks


def run_quantum_question_study(
    *,
    n_users: int = 5,
    n_channels: int = 3,
    time_steps: int = 12,
    lambda_switch: float = 0.30,
    seeds_by_regime: dict[str, list[int]] | None = None,
    lambda_scan: tuple[float, ...] = QUESTION_LAMBDA_SCAN,
    optimization_shots: int = 64,
    final_shots: int = 256,
    evaluations: int = 6,
    n_jobs: int = 1,
) -> dict[str, object]:
    default_seed_bank = QUANTUM_SMOKE_SEEDS_BY_REGIME
    if time_steps < 6 and "sudden" in (seeds_by_regime or default_seed_bank):
        raise ValueError("time_steps must be at least 6 to include the sudden-jump event.")

    seeds_by_regime = default_seed_bank if seeds_by_regime is None else seeds_by_regime

    summary_rows: list[dict[str, float | int | str]] = []
    step_rows: list[dict[str, float | int | str]] = []
    allocation_rows: list[dict[str, float | int | str]] = []
    position_rows: list[dict[str, float | int | str]] = []
    edge_rows: list[dict[str, float | int | str]] = []
    transfer_rows: list[dict[str, float | int | str]] = []
    factorial_rows: list[dict[str, float | int | str]] = []
    tradeoff_rows: list[dict[str, float | int | str]] = []

    tasks = [
        {
            "regime": regime,
            "seed": seed,
            "n_users": n_users,
            "n_channels": n_channels,
            "time_steps": time_steps,
            "lambda_switch": lambda_switch,
            "lambda_scan": lambda_scan,
            "optimization_shots": optimization_shots,
            "final_shots": final_shots,
            "evaluations": evaluations,
        }
        for regime, seeds in seeds_by_regime.items()
        for seed in seeds
    ]

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            payloads = list(executor.map(_sequence_rows_payload_from_task, tasks))
    else:
        payloads = [_sequence_rows_payload(**task) for task in tasks]

    for payload in payloads:
        summary_rows.extend(payload["summary_rows"])
        step_rows.extend(payload["step_rows"])
        allocation_rows.extend(payload["allocation_rows"])
        position_rows.extend(payload["position_rows"])
        edge_rows.extend(payload["edge_rows"])
        transfer_rows.extend(payload["transfer_rows"])
        factorial_rows.extend(payload["factorial_rows"])
        tradeoff_rows.extend(payload["tradeoff_rows"])

    regimes = list(seeds_by_regime.keys())
    representative_seeds = _select_representative_seeds(step_rows, summary_rows, regimes)

    trace_rows: list[dict[str, float | int | str]] = []
    trace_tasks = [
        {
            "regime": regime,
            "seed": seed,
            "n_users": n_users,
            "n_channels": n_channels,
            "time_steps": time_steps,
            "lambda_switch": lambda_switch,
            "optimization_shots": optimization_shots,
            "final_shots": final_shots,
            "evaluations": evaluations,
        }
        for regime, seed in representative_seeds.items()
        if regime != "stationary"
    ]
    if n_jobs > 1 and trace_tasks:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(trace_tasks))) as executor:
            trace_payloads = list(executor.map(_trace_rows_payload_from_task, trace_tasks))
    else:
        trace_payloads = [_trace_rows_payload_from_task(task) for task in trace_tasks]
    for payload in trace_payloads:
        trace_rows.extend(payload)

    metadata = {
        "study_type": "quantum_shot_based",
        "n_users": n_users,
        "n_channels": n_channels,
        "time_steps": time_steps,
        "lambda_switch": lambda_switch,
        "lambda_scan": list(lambda_scan),
        "methods": list(QUESTION_METHODS),
        "regimes": regimes,
        "seeds_by_regime": seeds_by_regime,
        "representative_seeds": representative_seeds,
        "optimization_shots": optimization_shots,
        "final_shots": final_shots,
        "evaluations": evaluations,
        "n_jobs": n_jobs,
        "full_seed_bank": FULL_QUANTUM_SEEDS_BY_REGIME,
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
