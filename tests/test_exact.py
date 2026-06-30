import numpy as np

from respec.exact import solve_exact_step


def test_solve_exact_step_prefers_non_conflicting_channels() -> None:
    weights = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    best_value, best_allocations = solve_exact_step(weights, previous=None, n_channels=2, lambda_switch=0.3)
    assert best_value == 0.0
    assert {tuple(allocation.tolist()) for allocation in best_allocations} == {(0, 1), (1, 0)}

