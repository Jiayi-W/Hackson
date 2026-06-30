from respec.quantum_question_study import QUANTUM_SMOKE_SEEDS_BY_REGIME, run_quantum_question_study


def test_quantum_question_study_row_counts_and_real_metrics() -> None:
    seeds_by_regime = {
        regime: [seeds[0]]
        for regime, seeds in QUANTUM_SMOKE_SEEDS_BY_REGIME.items()
    }
    payload = run_quantum_question_study(
        seeds_by_regime=seeds_by_regime,
        time_steps=6,
        evaluations=5,
        optimization_shots=16,
        final_shots=32,
        lambda_scan=(),
    )

    total_sequences = sum(len(seeds) for seeds in seeds_by_regime.values())
    assert len(payload["summary_rows"]) == total_sequences * 4
    assert len(payload["step_rows"]) == total_sequences * 6 * 4
    assert len(payload["allocation_rows"]) == total_sequences * 6 * 5 * 4
    assert len(payload["position_rows"]) == total_sequences * 6 * 5
    assert len(payload["edge_rows"]) == total_sequences * 6 * 10
    assert len(payload["transfer_rows"]) == total_sequences * 5 * 3
    assert len(payload["factorial_rows"]) == total_sequences * 3
    assert len(payload["tradeoff_rows"]) == 0

    trace_pairs = {(row["scenario"], row["method"]) for row in payload["trace_rows"]}
    assert trace_pairs == {
        ("gradual", "Cold"),
        ("gradual", "Param"),
        ("gradual", "State"),
        ("gradual", "Combined"),
        ("sudden", "Cold"),
        ("sudden", "Param"),
        ("sudden", "State"),
        ("sudden", "Combined"),
    }

    combined_step = next(
        row
        for row in payload["step_rows"]
        if row["regime"] == "gradual" and row["method"] == "Combined" and row["snapshot_t"] == 1
    )
    assert 0.0 <= float(combined_step["feasible_fraction"]) <= 1.0
    assert 0.0 <= float(combined_step["success_probability"]) <= 1.0
    assert int(combined_step["evaluations"]) >= 1

    metadata = payload["metadata"]
    assert metadata["study_type"] == "quantum_shot_based"
    assert metadata["time_steps"] == 6
    assert metadata["optimization_shots"] == 16
    assert metadata["final_shots"] == 32
    assert metadata["evaluations"] == 5
