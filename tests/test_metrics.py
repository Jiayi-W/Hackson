from respec.metrics import build_cx_budget_curves, build_noise_sweep_curves, build_transfer_gain_records


def test_transfer_gain_record_count() -> None:
    records = build_transfer_gain_records()
    assert len(records["delta"]) == 81
    assert len(records["gain"]) == 81
    assert len(records["regime"]) == 81


def test_curve_shapes_align() -> None:
    cx = build_cx_budget_curves()
    assert len(cx["cx_budget"]) == len(cx["ring_feasible"]) == len(cx["penalty_cost"])

    noise = build_noise_sweep_curves()
    assert len(noise["noise"]) == len(noise["feasible_combined"]) == len(noise["gap_cold"])

