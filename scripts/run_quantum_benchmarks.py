from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import write_csv_rows, write_json
from respec.environment import generate_sequence, weighted_bray_curtis_change
from respec.strategies import run_quantum_strategy_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-QAOA benchmark suites for figure data.")
    parser.add_argument("--time-steps", type=int, default=5)
    parser.add_argument("--evaluations", type=int, default=6)
    parser.add_argument("--optimization-shots", type=int, default=64)
    parser.add_argument("--final-shots", type=int, default=256)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.0, 0.15, 0.30, 0.60])
    parser.add_argument("--output-prefix", default="quantum_benchmarks_smoke")
    parser.add_argument("--full-budget", action="store_true")
    args = parser.parse_args()

    if args.full_budget:
        if args.time_steps == 5:
            args.time_steps = 10
        if args.evaluations == 6:
            args.evaluations = 24
        if args.optimization_shots == 64:
            args.optimization_shots = 256
        if args.final_shots == 256:
            args.final_shots = 2048
        if args.output_prefix == "quantum_benchmarks_smoke":
            args.output_prefix = "quantum_benchmarks_full"

    return args


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "artifacts" / "raw_results"
    methods = ("Cold", "Combined")
    regime_seeds = {
        "stationary": [3],
        "gradual": [11],
        "sudden": [19],
    }
    if args.full_budget:
        regime_seeds = {
            "stationary": [3, 5],
            "gradual": [11, 13],
            "sudden": [19, 23],
        }

    tradeoff_rows: list[dict[str, float | int | str]] = []
    tradeoff_sequence = generate_sequence(n_users=5, time_steps=args.time_steps, regime="sudden", seed=11)
    for lambda_switch in args.lambda_values:
        suite = run_quantum_strategy_suite(
            tradeoff_sequence,
            n_channels=3,
            lambda_switch=lambda_switch,
            methods=methods,
            optimization_shots=args.optimization_shots,
            final_shots=args.final_shots,
            evaluations=args.evaluations,
            seed=11,
        )
        for method in methods:
            rollout = suite[method]
            tradeoff_rows.append(
                {
                    "regime": tradeoff_sequence.regime,
                    "seed": tradeoff_sequence.seed,
                    "method": method,
                    "lambda_switch": float(lambda_switch),
                    "cumulative_switches": float(rollout.cumulative_switches[-1]),
                    "cumulative_interference": float(rollout.cumulative_interference[-1]),
                    "final_cost": float(rollout.cumulative_costs[-1]),
                }
            )

    transfer_rows: list[dict[str, float | int | str]] = []
    for regime, seeds in regime_seeds.items():
        for seed in seeds:
            sequence = generate_sequence(n_users=5, time_steps=args.time_steps, regime=regime, seed=seed)
            suite = run_quantum_strategy_suite(
                sequence,
                n_channels=3,
                lambda_switch=0.30,
                methods=methods,
                optimization_shots=args.optimization_shots,
                final_shots=args.final_shots,
                evaluations=args.evaluations,
                seed=seed,
            )
            cold = suite["Cold"]
            combined = suite["Combined"]
            for snapshot_t in range(1, len(sequence.snapshots)):
                delta = weighted_bray_curtis_change(
                    sequence.snapshots[snapshot_t].weights,
                    sequence.snapshots[snapshot_t - 1].weights,
                )
                cold_cost = float(cold.step_costs[snapshot_t])
                combined_cost = float(combined.step_costs[snapshot_t])
                transfer_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "snapshot_t": snapshot_t,
                        "delta": float(delta),
                        "cold_cost": cold_cost,
                        "combined_cost": combined_cost,
                        "transfer_gain": cold_cost - combined_cost,
                    }
                )

    write_csv_rows(output_dir / f"{args.output_prefix}_tradeoff_points.csv", tradeoff_rows)
    write_csv_rows(output_dir / f"{args.output_prefix}_transfer_points.csv", transfer_rows)
    write_json(
        output_dir / f"{args.output_prefix}_metadata.json",
        {
            "time_steps": args.time_steps,
            "evaluations": args.evaluations,
            "optimization_shots": args.optimization_shots,
            "final_shots": args.final_shots,
            "lambda_values": args.lambda_values,
            "methods": list(methods),
            "tradeoff_regime": tradeoff_sequence.regime,
            "tradeoff_seed": tradeoff_sequence.seed,
            "transfer_regime_seeds": regime_seeds,
            "output_prefix": args.output_prefix,
        },
    )

    print(f"wrote quantum benchmark exports to {output_dir} with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()
