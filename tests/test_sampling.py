from qiskit import QuantumCircuit

from respec.sampling import sample_counts


def test_sample_counts_is_seeded_and_shot_exact() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)

    counts_a = sample_counts(circuit, shots=64, seed=123)
    counts_b = sample_counts(circuit, shots=64, seed=123)

    assert counts_a == counts_b
    assert sum(counts_a.values()) == 64

