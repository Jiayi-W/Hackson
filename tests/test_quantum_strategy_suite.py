from respec.environment import generate_sequence
from respec.strategies import run_quantum_strategy_suite


def test_quantum_strategy_suite_returns_all_methods() -> None:
    sequence = generate_sequence(n_users=2, time_steps=3, regime="gradual", seed=3)
    suite = run_quantum_strategy_suite(
        sequence,
        n_channels=3,
        lambda_switch=0.30,
        optimization_shots=64,
        final_shots=128,
        evaluations=5,
        seed=3,
    )

    assert set(suite.keys()) == {"Cold", "Param", "State", "Combined"}
    for rollout in suite.values():
        assert len(rollout.allocations) == 3
        assert len(rollout.snapshot_results or []) == 3
        assert len(rollout.optimization_traces or []) == 3
        assert all(result.feasible_fraction == 1.0 for result in rollout.snapshot_results or [])
