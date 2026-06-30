from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .hamiltonian import append_cost_layer
from .mixers import append_ring_xy_layer


def prepare_allocation_basis(
    circuit: QuantumCircuit,
    allocation: np.ndarray,
    n_channels: int,
) -> None:
    allocation = np.asarray(allocation, dtype=int)
    for user, channel in enumerate(allocation.tolist()):
        if channel < 0 or channel >= n_channels:
            raise ValueError("allocation contains an out-of-range channel index.")
        circuit.x(user * n_channels + channel)


def _resolve_n_channels(
    initial_allocation: np.ndarray,
    previous: np.ndarray | None,
    n_channels: int | None,
) -> int:
    if n_channels is not None:
        return n_channels

    max_channel = int(np.max(initial_allocation))
    if previous is not None:
        max_channel = max(max_channel, int(np.max(previous)))
    return max_channel + 1


def build_trimmed_p2_ansatz(
    weights: np.ndarray,
    previous: np.ndarray | None,
    initial_allocation: np.ndarray,
    lambda_switch: float,
    n_channels: int | None = None,
) -> tuple[QuantumCircuit, ParameterVector]:
    initial_allocation = np.asarray(initial_allocation, dtype=int)
    if previous is not None:
        previous = np.asarray(previous, dtype=int)
    n_channels = _resolve_n_channels(initial_allocation, previous, n_channels)
    n_users = weights.shape[0]

    if len(initial_allocation) != n_users:
        raise ValueError("initial_allocation length must match the number of users.")
    if previous is not None and len(previous) != n_users:
        raise ValueError("previous allocation length must match the number of users.")

    circuit = QuantumCircuit(n_users * n_channels)
    parameters = ParameterVector("theta", 3)
    circuit.metadata = {
        "parameter_order": ["beta1", "gamma2", "beta2"],
        "ansatz": "trimmed_basis_seeded_p2",
        "n_users": n_users,
        "n_channels": n_channels,
    }

    prepare_allocation_basis(circuit, initial_allocation, n_channels)
    append_ring_xy_layer(circuit, parameters[0], n_users=n_users, n_channels=n_channels, reverse_order=False)
    append_cost_layer(circuit, parameters[1], weights=weights, previous=previous, lambda_switch=lambda_switch)
    append_ring_xy_layer(circuit, parameters[2], n_users=n_users, n_channels=n_channels, reverse_order=True)
    return circuit, parameters
