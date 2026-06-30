from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


def _normalized_bitstring(bitstring: str) -> str:
    return bitstring.replace(" ", "")


def decode_onehot_bitstring(bitstring: str, n_users: int, n_channels: int) -> np.ndarray:
    bits = _normalized_bitstring(bitstring)[::-1]
    if len(bits) != n_users * n_channels:
        raise ValueError("bitstring length does not match the requested one-hot layout.")

    allocation = np.full(n_users, fill_value=-1, dtype=int)
    for user in range(n_users):
        block = bits[user * n_channels : (user + 1) * n_channels]
        if block.count("1") != 1:
            raise ValueError("bitstring is not one-hot feasible.")
        allocation[user] = block.index("1")
    return allocation


def is_onehot_feasible(bitstring: str, n_users: int, n_channels: int) -> bool:
    try:
        decode_onehot_bitstring(bitstring, n_users=n_users, n_channels=n_channels)
        return True
    except ValueError:
        return False


def feasible_fraction(counts: Mapping[str, int], n_users: int, n_channels: int) -> float:
    total = sum(int(count) for count in counts.values())
    if total == 0:
        return 0.0

    feasible = sum(
        int(count)
        for bitstring, count in counts.items()
        if is_onehot_feasible(bitstring, n_users=n_users, n_channels=n_channels)
    )
    return feasible / total


def sample_counts(
    circuit: QuantumCircuit,
    parameter_values: Sequence[float] | Mapping[Parameter, float] | None = None,
    shots: int = 2048,
    seed: int | None = None,
) -> dict[str, int]:
    if any(instruction.operation.name == "measure" for instruction in circuit.data):
        raise ValueError("sample_counts expects an unmeasured circuit.")

    bound = circuit.assign_parameters(parameter_values, inplace=False) if parameter_values is not None else circuit
    state = Statevector.from_instruction(bound)
    if seed is not None:
        state.seed(seed)
    sampled = state.sample_counts(shots=shots)
    return {str(bitstring): int(count) for bitstring, count in sampled.items()}
