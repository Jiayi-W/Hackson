from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import write_csv_rows, write_json
from respec.question_study import run_question_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export research-question study tables for ReSpec-QAOA.")
    parser.add_argument("--output-prefix", default="question_study")
    parser.add_argument("--lambda-switch", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_question_study(lambda_switch=args.lambda_switch)
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

    print(f"wrote research-question study tables to {output_dir} with prefix {prefix}")


if __name__ == "__main__":
    main()
