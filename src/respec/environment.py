from __future__ import annotations

from typing import Literal

import numpy as np

from .types import DynamicSequence, Snapshot

BASE_POSITIONS = np.array(
    [
        [0.14, 0.78],
        [0.30, 0.30],
        [0.47, 0.62],
        [0.66, 0.33],
        [0.82, 0.77],
    ],
    dtype=float,
)


def _reflect(points: np.ndarray) -> np.ndarray:
    reflected = points.copy()
    reflected = np.where(reflected < 0.0, -reflected, reflected)
    reflected = np.where(reflected > 1.0, 2.0 - reflected, reflected)
    return np.clip(reflected, 0.02, 0.98)


def compute_weight_matrix(positions: np.ndarray, radius: float = 0.55) -> np.ndarray:
    n_users = positions.shape[0]
    weights = np.zeros((n_users, n_users), dtype=float)
    for u in range(n_users):
        for v in range(u + 1, n_users):
            distance = float(np.linalg.norm(positions[u] - positions[v]))
            if distance < radius:
                weight = 0.25 + 0.75 * (1.0 - distance / radius)
            else:
                weight = 0.0
            weights[u, v] = weights[v, u] = weight
    return weights


def weighted_bray_curtis_change(current: np.ndarray, previous: np.ndarray, eps: float = 1e-9) -> float:
    upper = np.triu_indices_from(current, k=1)
    numerator = np.abs(current[upper] - previous[upper]).sum()
    denominator = (current[upper] + previous[upper]).sum() + eps
    return float(numerator / denominator)


def generate_sequence(
    n_users: int,
    time_steps: int,
    regime: Literal["stationary", "gradual", "sudden"],
    seed: int,
) -> DynamicSequence:
    rng = np.random.default_rng(seed)
    if n_users <= len(BASE_POSITIONS):
        positions = BASE_POSITIONS[:n_users].copy()
    else:
        positions = rng.uniform(0.15, 0.85, size=(n_users, 2))

    snapshots: list[Snapshot] = []
    moved_user = min(n_users - 1, 4)

    for t in range(time_steps):
        if t > 0 and regime != "stationary":
            step = rng.normal(0.0, 0.03, size=positions.shape)
            positions = _reflect(positions + step)

            if regime == "sudden" and t == 5:
                centroid = positions.mean(axis=0)
                jump_target = np.clip(0.92 * centroid + np.array([-0.04, 0.02]), 0.06, 0.94)
                positions[moved_user] = jump_target

        weights = compute_weight_matrix(positions)
        snapshots.append(Snapshot(t=t, positions=positions.copy(), weights=weights.copy()))

    return DynamicSequence(regime=regime, seed=seed, snapshots=tuple(snapshots))

