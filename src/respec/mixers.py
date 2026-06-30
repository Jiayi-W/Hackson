from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import XXPlusYYGate

from .indexing import qubit_index


def ring_xy_pairs(n_channels: int, reverse_order: bool = False) -> list[tuple[int, int]]:
    if n_channels < 2:
        return []

    if n_channels == 2:
        pairs = [(0, 1)]
    else:
        pairs = [(channel, channel + 1) for channel in range(n_channels - 1)]
        pairs.append((n_channels - 1, 0))

    if reverse_order:
        pairs = list(reversed(pairs))
    return pairs


def append_ring_xy_layer(
    circuit: QuantumCircuit,
    beta: float | ParameterExpression,
    n_users: int,
    n_channels: int,
    reverse_order: bool,
) -> None:
    gate = XXPlusYYGate(2.0 * beta, beta=0.0)
    for user in range(n_users):
        for channel_a, channel_b in ring_xy_pairs(n_channels, reverse_order=reverse_order):
            qubit_a = qubit_index(user, channel_a, n_channels)
            qubit_b = qubit_index(user, channel_b, n_channels)
            circuit.append(gate, [qubit_a, qubit_b])
