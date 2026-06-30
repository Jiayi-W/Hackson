from respec.runner import (
    build_allocation_records,
    build_core_rollouts,
    build_final_summary_records,
    build_snapshot_edge_records,
    build_snapshot_position_records,
    build_step_metric_records,
)


def test_runner_export_shapes() -> None:
    sequence, methods = build_core_rollouts(regime="sudden", seed=11, lambda_switch=0.30)

    final_rows = build_final_summary_records(methods)
    step_rows = build_step_metric_records(sequence, methods)
    allocation_rows = build_allocation_records(methods)
    position_rows = build_snapshot_position_records(sequence)
    edge_rows = build_snapshot_edge_records(sequence)

    assert len(final_rows) == 7
    assert len(step_rows) == 70
    assert len(allocation_rows) == 350
    assert len(position_rows) == 50
    assert len(edge_rows) == 100
