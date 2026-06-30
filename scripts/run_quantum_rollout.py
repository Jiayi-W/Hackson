from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import write_csv_rows, write_json
from respec.environment import generate_sequence
from respec.strategies import run_quantum_strategy_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qiskit-backed dynamic QAOA rollout.")
    parser.add_argument("--regime", default="gradual", choices=["stationary", "gradual", "sudden", "continuous_sudden"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--time-steps", type=int, default=5)
    parser.add_argument("--evaluations", type=int, default=8)
    parser.add_argument("--optimization-shots", type=int, default=128)
    parser.add_argument("--final-shots", type=int, default=512)
    parser.add_argument("--lambda-switch", type=float, default=0.30)
    parser.add_argument("--output-prefix", default="quantum_suite_smoke")
    parser.add_argument("--full-budget", action="store_true")
    args = parser.parse_args()

    if args.full_budget:
        if args.time_steps == 5:
            args.time_steps = 10
        if args.evaluations == 8:
            args.evaluations = 24
        if args.optimization_shots == 128:
            args.optimization_shots = 256
        if args.final_shots == 512:
            args.final_shots = 2048
        if args.output_prefix == "quantum_suite_smoke":
            args.output_prefix = "quantum_suite_full"

    return args


def main() -> None:
    args = parse_args()
    sequence = generate_sequence(n_users=5, time_steps=args.time_steps, regime=args.regime, seed=args.seed)
    suite = run_quantum_strategy_suite(
        sequence,
        n_channels=3,
        lambda_switch=args.lambda_switch,
        optimization_shots=args.optimization_shots,
        final_shots=args.final_shots,
        evaluations=args.evaluations,
        seed=args.seed,
    )

    output_dir = ROOT / "artifacts" / "raw_results"
    summary_rows = []
    step_rows = []
    trace_rows = []

    for method, rollout in suite.items():
        summary_rows.append(
            {
                "method": method,
                "final_cost": float(rollout.cumulative_costs[-1]),
                "final_interference": float(rollout.cumulative_interference[-1]),
                "final_switches": float(rollout.cumulative_switches[-1]),
            }
        )
        for t, result in enumerate(rollout.snapshot_results or []):
            step_rows.append(
                {
                    "method": method,
                    "snapshot_t": t,
                    "allocation": "-".join(str(value) for value in result.allocation.tolist()),
                    "best_sample_cost": float(result.best_sample_cost),
                    "expected_cost": float(result.expected_cost),
                    "feasible_fraction": float(result.feasible_fraction),
                    "success_probability": float(result.success_probability),
                    "evaluations": int(result.evaluations),
                    "shots": int(result.shots),
                    "cx_count": int(result.cx_count),
                    "circuit_depth": int(result.circuit_depth),
                    "wall_seconds": float(result.wall_seconds),
                }
            )
        for t, trace in enumerate(rollout.optimization_traces or []):
            for evaluation, (value, best) in enumerate(
                zip(trace.objective_values.tolist(), trace.best_so_far.tolist(), strict=True),
                start=1,
            ):
                trace_rows.append(
                    {
                        "method": method,
                        "snapshot_t": t,
                        "evaluation": evaluation,
                        "objective_value": float(value),
                        "best_so_far": float(best),
                    }
                )

    write_csv_rows(output_dir / f"{args.output_prefix}_summary.csv", summary_rows)
    write_csv_rows(output_dir / f"{args.output_prefix}_step_results.csv", step_rows)
    write_csv_rows(output_dir / f"{args.output_prefix}_optimization_traces.csv", trace_rows)
    write_json(
        output_dir / f"{args.output_prefix}_metadata.json",
        {
            "regime": sequence.regime,
            "seed": sequence.seed,
            "lambda_switch": args.lambda_switch,
            "n_users": 5,
            "n_channels": 3,
            "optimization_shots": args.optimization_shots,
            "final_shots": args.final_shots,
            "evaluations": args.evaluations,
            "time_steps": args.time_steps,
            "output_prefix": args.output_prefix,
            "methods": list(suite.keys()),
        },
    )

    for row in summary_rows:
        print(
            f"{row['method']:8s} final_cost={row['final_cost']:.3f} "
            f"interference={row['final_interference']:.3f} switches={row['final_switches']:.3f}"
        )
    print(
        f"wrote quantum rollout exports to {output_dir} "
        f"with prefix {args.output_prefix}"
    )


if __name__ == "__main__":
    main()
