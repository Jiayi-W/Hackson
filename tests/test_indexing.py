from respec.indexing import qubit_index


def test_qubit_index() -> None:
    assert qubit_index(0, 0, 3) == 0
    assert qubit_index(2, 1, 3) == 7
    assert qubit_index(4, 2, 3) == 14

