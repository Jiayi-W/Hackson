import numpy as np

from respec.environment import generate_sequence


def test_sequence_reproducibility() -> None:
    a = generate_sequence(5, 10, regime="sudden", seed=19)
    b = generate_sequence(5, 10, regime="sudden", seed=19)
    for snap_a, snap_b in zip(a.snapshots, b.snapshots, strict=True):
        assert np.allclose(snap_a.positions, snap_b.positions)
        assert np.allclose(snap_a.weights, snap_b.weights)
