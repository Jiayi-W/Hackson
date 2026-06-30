from __future__ import annotations

from .environment import generate_sequence
from .heuristics import rollout_heuristic
from .strategies import rollout_offline_dp, rollout_surrogate_qaoa


def build_demo_rollouts(seed: int = 11, lambda_switch: float = 0.30):
    sequence = generate_sequence(n_users=5, time_steps=10, regime="sudden", seed=seed)
    methods = {
        name: rollout_surrogate_qaoa(sequence, name, n_channels=3, lambda_switch=lambda_switch)
        for name in ("Cold", "Param", "State", "Combined")
    }
    methods["Greedy"] = rollout_heuristic(sequence, "Greedy", n_channels=3, lambda_switch=lambda_switch)
    methods["Local Search"] = rollout_heuristic(sequence, "Local Search", n_channels=3, lambda_switch=lambda_switch)
    methods["Offline DP"] = rollout_offline_dp(sequence, n_channels=3, lambda_switch=lambda_switch)
    return sequence, methods

