import numpy as np
from qiskit.quantum_info import Statevector

from respec.ansatz import build_trimmed_p2_ansatz
from respec.sampling import is_onehot_feasible


def test_trimmed_ansatz_preserves_onehot_feasibility() -> None:
    weights = np.array(
        [
            [0.0, 0.6],
            [0.6, 0.0],
        ]
    )
    previous = np.array([0, 1])
    initial_allocation = np.array([0, 1])

    circuit, parameters = build_trimmed_p2_ansatz(
        weights=weights,
        previous=previous,
        initial_allocation=initial_allocation,
        lambda_switch=0.30,
        n_channels=3,
    )
    bound = circuit.assign_parameters([0.31, 0.22, 0.17], inplace=False)
    state = Statevector.from_instruction(bound)

    for bitstring, amplitude in state.to_dict().items():
        if abs(amplitude) > 1e-10:
            assert is_onehot_feasible(bitstring, n_users=2, n_channels=3)

