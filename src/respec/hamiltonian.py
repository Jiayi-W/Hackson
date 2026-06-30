from __future__ import annotations

from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression

from .indexing import qubit_index


def _infer_n_channels(weights: np.ndarray, n_qubits: int) -> int:
    n_users = weights.shape[0]
    if n_users == 0 or n_qubits % n_users != 0:
        raise ValueError("The circuit qubit count must be divisible by the number of users.")
    return n_qubits // n_users


def cost_hamiltonian_terms(
    weights: np.ndarray,
    previous: np.ndarray | None,
    lambda_switch: float,
    n_channels: int,
    eps: float = 1e-9,
) -> tuple[float, dict[int, float], dict[tuple[int, int], float]]:
    n_users = weights.shape[0]
    if weights.shape != (n_users, n_users):
        raise ValueError("weights must be a square matrix.")
    if previous is not None and len(previous) != n_users:
        raise ValueError("previous allocation length must match the number of users.")

    constant = 0.0
    z_terms: defaultdict[int, float] = defaultdict(float)
    zz_terms: defaultdict[tuple[int, int], float] = defaultdict(float)

    normalization = float(np.triu(weights, k=1).sum()) + eps
    for u in range(n_users):
        for v in range(u + 1, n_users):
            if weights[u, v] == 0.0:
                continue
            base = float(weights[u, v]) / (4.0 * normalization)
            for channel in range(n_channels):
                qu = qubit_index(u, channel, n_channels)
                qv = qubit_index(v, channel, n_channels)
                constant += base
                z_terms[qu] -= base
                z_terms[qv] -= base
                zz_terms[(min(qu, qv), max(qu, qv))] += base

    if previous is not None:
        z_coeff = float(lambda_switch) / (2.0 * n_users)
        for user, channel in enumerate(previous.tolist()):
            qubit = qubit_index(user, int(channel), n_channels)
            constant += z_coeff
            z_terms[qubit] += z_coeff

    return constant, dict(z_terms), dict(zz_terms)


def allocation_cost_hamiltonian_energy(
    allocation: np.ndarray,
    weights: np.ndarray,
    previous: np.ndarray | None,
    lambda_switch: float,
    n_channels: int,
) -> float:
    constant, z_terms, zz_terms = cost_hamiltonian_terms(
        weights=weights,
        previous=previous,
        lambda_switch=lambda_switch,
        n_channels=n_channels,
    )
    z_values = {}
    for user, channel in enumerate(allocation.tolist()):
        for current_channel in range(n_channels):
            qubit = qubit_index(user, current_channel, n_channels)
            z_values[qubit] = -1.0 if current_channel == int(channel) else 1.0

    energy = constant
    for qubit, coeff in z_terms.items():
        energy += coeff * z_values[qubit]
    for (qubit_a, qubit_b), coeff in zz_terms.items():
        energy += coeff * z_values[qubit_a] * z_values[qubit_b]
    return float(energy)


def append_cost_layer(
    circuit: QuantumCircuit,
    gamma: float | ParameterExpression,
    weights: np.ndarray,
    previous: np.ndarray | None,
    lambda_switch: float,
) -> None:
    n_channels = _infer_n_channels(weights, circuit.num_qubits)
    _, z_terms, zz_terms = cost_hamiltonian_terms(
        weights=weights,
        previous=previous,
        lambda_switch=lambda_switch,
        n_channels=n_channels,
    )

    for qubit, coeff in sorted(z_terms.items()):
        if abs(coeff) > 1e-15:
            circuit.rz(2.0 * gamma * coeff, qubit)

    for (qubit_a, qubit_b), coeff in sorted(zz_terms.items()):
        if abs(coeff) > 1e-15:
            circuit.rzz(2.0 * gamma * coeff, qubit_a, qubit_b)
