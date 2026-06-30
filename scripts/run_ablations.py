from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import write_csv_rows, write_json
from respec.metrics import (
    build_cx_budget_curves,
    build_noise_sweep_curves,
    build_tradeoff_table,
    build_transfer_gain_records,
    build_adaptation_trace_profiles,
)


def main() -> None:
    output_dir = ROOT / "artifacts" / "raw_results"

    transfer = build_transfer_gain_records()
    transfer_rows = [
        {
            "regime": str(regime),
            "delta": float(delta),
            "gain": float(gain),
        }
        for regime, delta, gain in zip(transfer["regime"], transfer["delta"], transfer["gain"], strict=True)
    ]
    write_csv_rows(output_dir / "transfer_gain_points.csv", transfer_rows)

    table = build_tradeoff_table([0.00, 0.15, 0.30, 0.60])
    tradeoff_rows: list[dict[str, float | str]] = []
    for method, points in table.items():
        for lambda_value, (switches, interference) in points.items():
            tradeoff_rows.append(
                {
                    "method": method,
                    "lambda_switch": lambda_value,
                    "cumulative_switches": float(switches),
                    "cumulative_interference": float(interference),
                }
            )
    write_csv_rows(output_dir / "tradeoff_points.csv", tradeoff_rows)

    cx_curves = build_cx_budget_curves()
    cx_rows = []
    for idx in range(len(cx_curves["cx_budget"])):
        cx_rows.append(
            {
                "cx_budget": int(cx_curves["cx_budget"][idx]),
                "ring_feasible": float(cx_curves["ring_feasible"][idx]),
                "penalty_feasible": float(cx_curves["penalty_feasible"][idx]),
                "ring_cost": float(cx_curves["ring_cost"][idx]),
                "penalty_cost": float(cx_curves["penalty_cost"][idx]),
            }
        )
    write_csv_rows(output_dir / "cx_budget_curves.csv", cx_rows)

    noise_curves = build_noise_sweep_curves()
    noise_rows = []
    for idx in range(len(noise_curves["noise"])):
        noise_rows.append(
            {
                "noise": float(noise_curves["noise"][idx]),
                "feasible_combined": float(noise_curves["feasible_combined"][idx]),
                "feasible_cold": float(noise_curves["feasible_cold"][idx]),
                "gap_combined": float(noise_curves["gap_combined"][idx]),
                "gap_cold": float(noise_curves["gap_cold"][idx]),
            }
        )
    write_csv_rows(output_dir / "noise_sweep_curves.csv", noise_rows)

    traces = build_adaptation_trace_profiles(lambda_switch=0.30)
    trace_rows = []
    for idx, evaluation in enumerate(traces["evaluation"]):
        trace_rows.extend(
            [
                {
                    "scenario": "gradual",
                    "method": "Cold",
                    "evaluation": int(evaluation),
                    "best_so_far_cost": float(traces["gradual_cold"][idx]),
                },
                {
                    "scenario": "gradual",
                    "method": "Combined",
                    "evaluation": int(evaluation),
                    "best_so_far_cost": float(traces["gradual_combined"][idx]),
                },
                {
                    "scenario": "sudden",
                    "method": "Cold",
                    "evaluation": int(evaluation),
                    "best_so_far_cost": float(traces["sudden_cold"][idx]),
                },
                {
                    "scenario": "sudden",
                    "method": "Combined",
                    "evaluation": int(evaluation),
                    "best_so_far_cost": float(traces["sudden_combined"][idx]),
                },
            ]
        )
    write_csv_rows(output_dir / "adaptation_trace_profiles.csv", trace_rows)

    write_json(
        output_dir / "ablation_metadata.json",
        {
            "tradeoff_lambdas": ["0.00", "0.15", "0.30", "0.60"],
            "transfer_regimes": ["stationary", "gradual", "sudden"],
            "trace_methods": ["Cold", "Combined"],
        },
    )

    print(f"wrote ablation datasets to {output_dir}")


if __name__ == "__main__":
    main()
