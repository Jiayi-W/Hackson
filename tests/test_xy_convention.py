import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector


def test_xxplusyy_gate_matches_exchange_convention() -> None:
    beta = np.pi / 7.0
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.append(XXPlusYYGate(2.0 * beta, beta=0.0), [0, 1])

    amplitudes = Statevector.from_instruction(circuit).to_dict()
    assert amplitudes["01"] == pytest.approx(np.cos(beta))
    assert amplitudes["10"] == pytest.approx(-1j * np.sin(beta))

