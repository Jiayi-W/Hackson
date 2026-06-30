from respec.question_study import QUESTION_SEEDS_BY_REGIME, run_question_study


def test_question_study_row_counts_and_metadata() -> None:
    seeds_by_regime = {
        "stationary": [QUESTION_SEEDS_BY_REGIME["stationary"][0]],
        "gradual": [QUESTION_SEEDS_BY_REGIME["gradual"][0]],
        "sudden": [QUESTION_SEEDS_BY_REGIME["sudden"][0]],
    }
    payload = run_question_study(seeds_by_regime=seeds_by_regime, time_steps=6)

    total_sequences = sum(len(seeds) for seeds in seeds_by_regime.values())
    assert len(payload["summary_rows"]) == total_sequences * 4
    assert len(payload["step_rows"]) == total_sequences * 6 * 4
    assert len(payload["allocation_rows"]) == total_sequences * 6 * 5 * 4
    assert len(payload["position_rows"]) == total_sequences * 6 * 5
    assert len(payload["edge_rows"]) == total_sequences * 6 * 10
    assert len(payload["transfer_rows"]) == total_sequences * 5 * 3
    assert len(payload["factorial_rows"]) == total_sequences * 3
    assert len(payload["tradeoff_rows"]) == total_sequences * 4 * 2
    assert len(payload["trace_rows"]) == 2 * 4 * 24

    metadata = payload["metadata"]
    assert metadata["n_users"] == 5
    assert metadata["n_channels"] == 3
    assert metadata["time_steps"] == 6
    assert metadata["representative_seeds"]["stationary"] in seeds_by_regime["stationary"]
    assert metadata["representative_seeds"]["gradual"] in seeds_by_regime["gradual"]
    assert metadata["representative_seeds"]["sudden"] in seeds_by_regime["sudden"]
