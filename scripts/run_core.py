from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import write_csv_rows, write_json
from respec.runner import (
    build_allocation_records,
    build_core_rollouts,
    build_final_summary_records,
    build_snapshot_edge_records,
    build_snapshot_position_records,
    build_step_metric_records,
)


def main() -> None:
    regime = "sudden"
    seed = 11
    lambda_switch = 0.30
    sequence, methods = build_core_rollouts(regime=regime, seed=seed, lambda_switch=lambda_switch)

    output_dir = ROOT / "artifacts" / "raw_results"
    final_summary_rows = build_final_summary_records(methods)
    step_metric_rows = build_step_metric_records(sequence, methods)
    allocation_rows = build_allocation_records(methods)
    position_rows = build_snapshot_position_records(sequence)
    edge_rows = build_snapshot_edge_records(sequence)

    write_csv_rows(output_dir / "core_final_summary.csv", final_summary_rows)
    write_csv_rows(output_dir / "core_step_metrics.csv", step_metric_rows)
    write_csv_rows(output_dir / "core_allocations.csv", allocation_rows)
    write_csv_rows(output_dir / "core_snapshot_positions.csv", position_rows)
    write_csv_rows(output_dir / "core_edge_weights.csv", edge_rows)
    write_json(
        output_dir / "core_metadata.json",
        {
            "regime": regime,
            "seed": seed,
            "lambda_switch": lambda_switch,
            "n_users": 5,
            "n_channels": 3,
            "time_steps": 10,
            "methods": list(methods.keys()),
        },
    )

    for name, rollout in methods.items():
        print(
            f"{name:12s} final_cost={rollout.cumulative_costs[-1]:.3f} "
            f"interference={rollout.cumulative_interference[-1]:.3f} "
            f"switches={rollout.cumulative_switches[-1]:.3f}"
        )
    print(f"wrote datasets to {output_dir}")


if __name__ == "__main__":
    main()
