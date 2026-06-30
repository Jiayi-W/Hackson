from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    t: int
    positions: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class DynamicSequence:
    regime: Literal["stationary", "gradual", "sudden"]
    seed: int
    snapshots: tuple[Snapshot, ...]


@dataclass
class SnapshotResult:
    allocation: np.ndarray
    parameters: np.ndarray
    expected_cost: float
    best_sample_cost: float
    feasible_fraction: float
    success_probability: float
    evaluations: int
    shots: int
    cx_count: int
    circuit_depth: int
    wall_seconds: float


@dataclass
class OptimizationTrace:
    parameters: list[np.ndarray]
    objective_values: np.ndarray
    best_so_far: np.ndarray


@dataclass
class StrategyRollout:
    method: str
    allocations: list[np.ndarray]
    step_costs: np.ndarray
    cumulative_costs: np.ndarray
    cumulative_interference: np.ndarray
    cumulative_switches: np.ndarray
    snapshot_results: list[SnapshotResult] | None = None
    optimization_traces: list[OptimizationTrace] | None = None
