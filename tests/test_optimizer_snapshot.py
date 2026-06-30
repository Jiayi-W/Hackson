import numpy as np

from respec.optimizer import optimize_snapshot_qaoa


def test_optimize_snapshot_qaoa_runs_and_returns_feasible_result() -> None:
    weights = np.array(
        [
            [0.0, 0.7],
            [0.7, 0.0],
        ]
    )
    previous = np.array([0, 1])
    initial_allocation = np.array([0, 1])
    initial_parameters = np.array([0.2, 0.1, -0.2])

    result, trace = optimize_snapshot_qaoa(
        weights=weights,
        previous=previous,
        initial_allocation=initial_allocation,
        initial_parameters=initial_parameters,
        lambda_switch=0.30,
        n_channels=3,
        optimization_shots=64,
        final_shots=128,
        evaluations=6,
        seed=5,
        seed_transpiler=5,
    )

    assert result.allocation.shape == (2,)
    assert result.feasible_fraction == 1.0
    assert result.evaluations >= 1
    assert result.shots == 128
    assert len(trace.parameters) == result.evaluations
    assert len(trace.objective_values) == result.evaluations
    assert trace.best_so_far[-1] <= trace.objective_values[0] + 1e-12

