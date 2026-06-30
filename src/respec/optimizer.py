from __future__ import annotations

from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

from .ansatz import build_trimmed_p2_ansatz
from .objective import summarize_counts_objective
from .sampling import sample_counts
from .types import OptimizationTrace, SnapshotResult


def best_so_far_trace(start: float, finish: float, evaluations: int, shape: str) -> np.ndarray:
    x = np.linspace(0.0, 1.0, evaluations)
    if shape == "fast":
        progress = 1.0 - np.exp(-4.2 * x)
    elif shape == "slow":
        progress = x ** 1.8
    elif shape == "reset":
        progress = np.where(x < 0.45, 0.28 * x, 0.12 + 0.95 * (x - 0.45))
        progress = np.clip(progress, 0.0, 1.0)
    else:
        progress = x

    values = start + (finish - start) * progress
    values = np.minimum.accumulate(values)
    return values


def draw_random_initial_parameters(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(-0.8, 0.8),
            rng.uniform(-1.2, 1.2),
            rng.uniform(-0.8, 0.8),
        ],
        dtype=float,
    )


def estimate_circuit_resources(circuit: QuantumCircuit, seed_transpiler: int | None = None) -> tuple[int, int]:
    compiled = transpile(
        circuit,
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=0,
        seed_transpiler=seed_transpiler,
    )
    cx_count = int(compiled.count_ops().get("cx", 0))
    depth = int(compiled.depth() or 0)
    return cx_count, depth


def optimize_snapshot_qaoa(
    weights: np.ndarray,
    previous: np.ndarray | None,
    initial_allocation: np.ndarray,
    initial_parameters: np.ndarray,
    lambda_switch: float,
    n_channels: int,
    optimization_shots: int = 256,
    final_shots: int = 2048,
    evaluations: int = 24,
    seed: int | None = None,
    seed_transpiler: int | None = None,
) -> tuple[SnapshotResult, OptimizationTrace]:
    if evaluations < 1:
        raise ValueError("evaluations must be positive.")

    start = perf_counter()
    circuit, parameters = build_trimmed_p2_ansatz(
        weights=weights,
        previous=previous,
        initial_allocation=initial_allocation,
        lambda_switch=lambda_switch,
        n_channels=n_channels,
    )
    cx_count, depth = estimate_circuit_resources(circuit, seed_transpiler=seed_transpiler)

    parameter_history: list[np.ndarray] = []
    objective_values: list[float] = []

    def _binding(theta: np.ndarray) -> dict:
        return {parameters[idx]: float(theta[idx]) for idx in range(len(parameters))}

    def objective(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        eval_seed = None if seed is None else seed + len(objective_values)
        counts = sample_counts(circuit, parameter_values=_binding(theta), shots=optimization_shots, seed=eval_seed)
        summary = summarize_counts_objective(
            counts,
            weights=weights,
            previous=previous,
            lambda_switch=lambda_switch,
            n_users=weights.shape[0],
            n_channels=n_channels,
        )
        value = float(summary["expected_cost"])
        parameter_history.append(theta.copy())
        objective_values.append(value)
        return value

    x0 = np.asarray(initial_parameters, dtype=float)
    minimize(
        objective,
        x0=x0,
        method="COBYLA",
        options={"maxiter": evaluations, "rhobeg": 0.4, "tol": 1e-4},
    )

    objective_array = np.array(objective_values, dtype=float)
    best_index = int(np.argmin(objective_array))
    best_parameters = parameter_history[best_index]
    final_counts = sample_counts(
        circuit,
        parameter_values=_binding(best_parameters),
        shots=final_shots,
        seed=None if seed is None else seed + 100_000,
    )
    summary = summarize_counts_objective(
        final_counts,
        weights=weights,
        previous=previous,
        lambda_switch=lambda_switch,
        n_users=weights.shape[0],
        n_channels=n_channels,
    )

    result = SnapshotResult(
        allocation=np.asarray(summary["allocation"], dtype=int),
        parameters=best_parameters.copy(),
        expected_cost=float(summary["expected_cost"]),
        best_sample_cost=float(summary["best_sample_cost"]),
        feasible_fraction=float(summary["feasible_fraction"]),
        success_probability=float(summary["success_probability"]),
        evaluations=len(objective_values),
        shots=final_shots,
        cx_count=cx_count,
        circuit_depth=depth,
        wall_seconds=perf_counter() - start,
    )
    trace = OptimizationTrace(
        parameters=[theta.copy() for theta in parameter_history],
        objective_values=objective_array,
        best_so_far=np.minimum.accumulate(objective_array),
    )
    return result, trace
