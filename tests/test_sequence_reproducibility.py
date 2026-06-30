import numpy as np

from respec.environment import generate_sequence, weighted_bray_curtis_change


def test_sequence_reproducibility() -> None:
    a = generate_sequence(5, 10, regime="sudden", seed=19)
    b = generate_sequence(5, 10, regime="sudden", seed=19)
    for snap_a, snap_b in zip(a.snapshots, b.snapshots, strict=True):
        assert np.allclose(snap_a.positions, snap_b.positions)
        assert np.allclose(snap_a.weights, snap_b.weights)


def test_continuous_sudden_is_reproducible_and_more_dynamic_than_gradual() -> None:
    gradual = generate_sequence(5, 10, regime="gradual", seed=41)
    a = generate_sequence(5, 10, regime="continuous_sudden", seed=41)
    b = generate_sequence(5, 10, regime="continuous_sudden", seed=41)

    for snap_a, snap_b in zip(a.snapshots, b.snapshots, strict=True):
        assert np.allclose(snap_a.positions, snap_b.positions)
        assert np.allclose(snap_a.weights, snap_b.weights)

    gradual_mean_delta = np.mean(
        [
            weighted_bray_curtis_change(gradual.snapshots[idx].weights, gradual.snapshots[idx - 1].weights)
            for idx in range(1, len(gradual.snapshots))
        ]
    )
    continuous_mean_delta = np.mean(
        [
            weighted_bray_curtis_change(a.snapshots[idx].weights, a.snapshots[idx - 1].weights)
            for idx in range(1, len(a.snapshots))
        ]
    )
    assert continuous_mean_delta > gradual_mean_delta
