from itertools import product

import numpy as np
import pytest

from respec.hamiltonian import allocation_cost_hamiltonian_energy
from respec.objective import classical_objective


def test_hamiltonian_energy_matches_classical_objective() -> None:
    weights = np.array(
        [
            [0.0, 0.8],
            [0.8, 0.0],
        ]
    )
    previous = np.array([0, 1])
    n_channels = 3
    lambda_switch = 0.30

    for allocation_tuple in product(range(n_channels), repeat=2):
        allocation = np.array(allocation_tuple, dtype=int)
        expected = classical_objective(allocation, weights, previous, lambda_switch)
        observed = allocation_cost_hamiltonian_energy(
            allocation,
            weights=weights,
            previous=previous,
            lambda_switch=lambda_switch,
            n_channels=n_channels,
        )
        assert observed == pytest.approx(expected)

