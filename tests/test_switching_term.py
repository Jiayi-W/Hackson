import numpy as np
import pytest

from respec.hamiltonian import allocation_cost_hamiltonian_energy


def test_switching_term_penalizes_channel_changes() -> None:
    weights = np.zeros((1, 1))
    previous = np.array([2])
    n_channels = 3
    lambda_switch = 0.30

    stay = allocation_cost_hamiltonian_energy(
        np.array([2]),
        weights=weights,
        previous=previous,
        lambda_switch=lambda_switch,
        n_channels=n_channels,
    )
    switch = allocation_cost_hamiltonian_energy(
        np.array([1]),
        weights=weights,
        previous=previous,
        lambda_switch=lambda_switch,
        n_channels=n_channels,
    )

    assert stay == pytest.approx(0.0)
    assert switch == pytest.approx(lambda_switch)

