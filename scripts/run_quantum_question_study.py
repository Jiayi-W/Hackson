from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import write_csv_rows, write_json
from respec.quantum_question_study import (
    FULL_QUANTUM_SEEDS_BY_REGIME,
    QUANTUM_CONTINUOUS_SMOKE_SEEDS_BY_REGIME,
    QUANTUM_SMOKE_SEEDS_BY_REGIME,
    run_quantum_question_study,
)
from respec.question_study import QUESTION_LAMBDA_SCAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export shot-based Qiskit research-question study tables.")
    parser.add_argument("--output-prefix", default="quantum_question_study_smoke")
    parser.add_argument("--lambda-switch", type=float, default=0.30)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=list(QUESTION_LAMBDA_SCAN))
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--evaluations", type=int, default=6)
    parser.add_argument("--optimization-shots", type=int, default=64)
    parser.add_argument("--final-shots", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--full-budget", action="store_true")
    parser.add_argument("--include-continuous-sudden", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds_by_regime = QUANTUM_CONTINUOUS_SMOKE_SEEDS_BY_REGIME if args.include_continuous_sudden else QUANTUM_SMOKE_SEEDS_BY_REGIME

    if args.include_continuous_sudden and args.output_prefix == "quantum_question_study_smoke":
        args.output_prefix = "quantum_question_study_continuous_smoke"

    if args.full_budget:
        if args.include_continuous_sudden:
            seeds_by_regime = FULL_QUANTUM_SEEDS_BY_REGIME
        else:
            seeds_by_regime = {key: value for key, value in FULL_QUANTUM_SEEDS_BY_REGIME.items() if key != "continuous_sudden"}
        if args.evaluations == 6:
            args.evaluations = 24
        if args.optimization_shots == 64:
            args.optimization_shots = 256
        if args.final_shots == 256:
            args.final_shots = 2048
        if args.output_prefix == "quantum_question_study_smoke":
            args.output_prefix = "quantum_question_study_full"
        if args.output_prefix == "quantum_question_study_continuous_smoke":
            args.output_prefix = "quantum_question_study_continuous_full"

    payload = run_quantum_question_study(
        time_steps=args.time_steps,
        lambda_switch=args.lambda_switch,
        seeds_by_regime=seeds_by_regime,
        lambda_scan=tuple(float(value) for value in args.lambda_values),
        optimization_shots=args.optimization_shots,
        final_shots=args.final_shots,
        evaluations=args.evaluations,
        n_jobs=args.n_jobs,
    )
    output_dir = ROOT / "artifacts" / "raw_results"
    prefix = args.output_prefix

    write_csv_rows(output_dir / f"{prefix}_summary.csv", payload["summary_rows"])
    write_csv_rows(output_dir / f"{prefix}_step_metrics.csv", payload["step_rows"])
    write_csv_rows(output_dir / f"{prefix}_allocations.csv", payload["allocation_rows"])
    write_csv_rows(output_dir / f"{prefix}_positions.csv", payload["position_rows"])
    write_csv_rows(output_dir / f"{prefix}_edges.csv", payload["edge_rows"])
    write_csv_rows(output_dir / f"{prefix}_transfer_gain.csv", payload["transfer_rows"])
    write_csv_rows(output_dir / f"{prefix}_factorial_effects.csv", payload["factorial_rows"])
    write_csv_rows(output_dir / f"{prefix}_tradeoff_scan.csv", payload["tradeoff_rows"])
    write_csv_rows(output_dir / f"{prefix}_trace_profiles.csv", payload["trace_rows"])
    write_json(output_dir / f"{prefix}_metadata.json", payload["metadata"])

    print(f"wrote quantum research-question study tables to {output_dir} with prefix {prefix}")


if __name__ == "__main__":
    main()
