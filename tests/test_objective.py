import numpy as np
import pytest

from respec.objective import classical_objective, interference_cost, switching_ratio


def test_interference_cost() -> None:
    allocation = np.array([0, 0, 1])
    weights = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.8, 0.0, 0.4],
            [0.2, 0.4, 0.0],
        ]
    )
    assert round(interference_cost(allocation, weights), 6) == round(0.8 / 1.4, 6)


def test_switching_ratio_and_total() -> None:
    previous = np.array([0, 1, 2])
    current = np.array([0, 2, 2])
    weights = np.zeros((3, 3))
    assert switching_ratio(current, previous) == 1.0 / 3.0
    assert classical_objective(current, weights, previous, 0.3) == pytest.approx(0.1)
