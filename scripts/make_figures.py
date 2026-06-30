from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from respec.environment import generate_sequence, weighted_bray_curtis_change
from respec.metrics import (
    build_adaptation_trace_profiles,
    build_cx_budget_curves,
    build_noise_sweep_curves,
    build_tradeoff_table,
    build_transfer_gain_records,
)
from respec.objective import classical_objective
from respec.runner import build_demo_rollouts
from respec.visualization import (
    CHANNEL_COLORS,
    METHOD_COLORS,
    REGIME_COLORS,
    apply_style,
    polish_axes,
    save_figure,
)


def _channel_cmap():
    return apply_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the ReSpec-QAOA figure set.")
    parser.add_argument(
        "--rollout-source",
        choices=["surrogate", "quantum"],
        default="surrogate",
        help="Which data source to use for F2/F3/F6.",
    )
    parser.add_argument(
        "--quantum-prefix",
        default="quantum_suite_smoke",
        help="Prefix under artifacts/raw_results for quantum rollout exports.",
    )
    return parser.parse_args()


def load_quantum_rollout_dataset(prefix: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    raw_dir = ROOT / "artifacts" / "raw_results"
    metadata_path = raw_dir / f"{prefix}_metadata.json"
    step_path = raw_dir / f"{prefix}_step_results.csv"
    trace_path = raw_dir / f"{prefix}_optimization_traces.csv"

    import json

    if not metadata_path.exists() or not step_path.exists() or not trace_path.exists():
        raise FileNotFoundError(
            f"Quantum rollout files for prefix '{prefix}' are missing under {raw_dir}."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lambda_switch = float(metadata["lambda_switch"])

    methods: dict[str, dict[str, list]] = {}
    with step_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            methods.setdefault(
                method,
                {
                    "snapshot_t": [],
                    "allocations": [],
                    "step_costs": [],
                    "cumulative_interference": [],
                    "cumulative_switches": [],
                },
            )
            methods[method]["snapshot_t"].append(int(row["snapshot_t"]))
            methods[method]["allocations"].append(
                np.array([int(value) for value in row["allocation"].split("-")], dtype=int)
            )
            methods[method]["step_costs"].append(float(row["best_sample_cost"]))
            methods[method]["cumulative_interference"].append(float(row["best_sample_cost"]))
            methods[method]["cumulative_switches"].append(float(row["snapshot_t"] != "0"))

    loaded_methods: dict[str, object] = {}
    for method, payload in methods.items():
        ordering = np.argsort(payload["snapshot_t"])
        allocations = [payload["allocations"][idx] for idx in ordering]
        step_costs = np.array([payload["step_costs"][idx] for idx in ordering], dtype=float)
        step_switches = [0.0]
        step_interference = [float(step_costs[0])]
        for allocation_index in range(1, len(allocations)):
            switches = float(np.mean(allocations[allocation_index] != allocations[allocation_index - 1]))
            step_switches.append(switches)
            step_interference.append(float(step_costs[allocation_index] - lambda_switch * switches))
        loaded_methods[method] = SimpleNamespace(
            method=method,
            allocations=allocations,
            step_costs=step_costs,
            cumulative_costs=np.cumsum(step_costs),
            cumulative_interference=np.cumsum(np.array(step_interference, dtype=float)),
            cumulative_switches=np.cumsum(np.array(step_switches, dtype=float)),
        )

    traces: dict[str, dict[int, dict[str, list[float]]]] = {}
    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            snapshot_t = int(row["snapshot_t"])
            traces.setdefault(method, {})
            traces[method].setdefault(snapshot_t, {"evaluation": [], "best_so_far": []})
            traces[method][snapshot_t]["evaluation"].append(int(row["evaluation"]))
            traces[method][snapshot_t]["best_so_far"].append(float(row["best_so_far"]))

    return metadata, loaded_methods, traces


def make_f1(sequence, combined_allocations, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    snapshots = [0, 4, 5, 9]
    moved_user = 4

    for ax, t in zip(axes, snapshots, strict=True):
        snapshot = sequence.snapshots[t]
        positions = snapshot.positions
        weights = snapshot.weights

        for u in range(weights.shape[0]):
            for v in range(u + 1, weights.shape[0]):
                if weights[u, v] <= 0:
                    continue
                xs = [positions[u, 0], positions[v, 0]]
                ys = [positions[u, 1], positions[v, 1]]
                ax.plot(
                    xs,
                    ys,
                    color="#9C6644",
                    alpha=0.18 + 0.72 * weights[u, v],
                    linewidth=1.2 + 3.3 * weights[u, v],
                    zorder=1,
                )

        for user, (x, y) in enumerate(positions):
            color = CHANNEL_COLORS[int(combined_allocations[t][user])]
            halo = "#F94144" if t == 5 and user == moved_user else "#FCFBF8"
            ax.scatter(x, y, s=410, color=halo, edgecolor="none", zorder=2)
            ax.scatter(x, y, s=245, color=color, edgecolor="#1F1F1F", linewidth=1.0, zorder=3)
            ax.text(x, y, f"U{user}", ha="center", va="center", color="white", weight="bold", fontsize=9)

        if t == 5:
            x, y = positions[moved_user]
            ax.text(x + 0.03, y + 0.07, "sudden jump", color="#C1121F", weight="bold", fontsize=9)

        ax.set_title(f"t = {t}")
        ax.set_xlim(0.03, 0.97)
        ax.set_ylim(0.03, 0.97)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("F1. Network Evolution", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F1_network_evolution.png")


def make_f2(methods, output_dir: Path, quantum_metadata: dict[str, object] | None = None) -> None:
    cmap = _channel_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    is_sudden = True if quantum_metadata is None else str(quantum_metadata["regime"]) == "sudden"

    for ax, method in zip(axes, ("Cold", "Combined"), strict=True):
        data = np.vstack(methods[method].allocations).T
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_title(method)
        ax.set_xlabel("Snapshot t")
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([f"User {idx}" for idx in range(data.shape[0])])
        if is_sudden and data.shape[1] >= 6:
            ax.axvline(4.5, color="#C1121F", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(4.65, -0.85, "jump", color="#C1121F", fontsize=9, weight="bold")

    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Ch 0", "Ch 1", "Ch 2"])
    fig.suptitle("F2. Allocation Timeline Heatmap", fontsize=15, y=1.02, weight="bold")
    save_figure(fig, output_dir / "F2_allocation_timeline_heatmap.png")


def make_f3(methods, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    available_methods = tuple(
        method
        for method in ("Cold", "Param", "State", "Combined", "Greedy", "Local Search", "Offline DP")
        if method in methods
    )
    for method in available_methods:
        ax.plot(
            range(len(methods[method].cumulative_costs)),
            methods[method].cumulative_costs,
            label=method,
            linewidth=2.4 if method in {"Combined", "Offline DP"} else 1.9,
            color=METHOD_COLORS[method],
        )
    ax.set_xlabel("Snapshot t")
    ax.set_ylabel("Cumulative cost")
    ax.set_title("F3. Cumulative Total Cost")
    polish_axes(ax)
    ax.legend(ncol=4 if len(available_methods) > 4 else 2, fontsize=8)
    save_figure(fig, output_dir / "F3_cumulative_total_cost.png")


def make_f4(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.0))
    tradeoff = build_tradeoff_table([0.00, 0.15, 0.30, 0.60])
    ordered = ["0.00", "0.15", "0.30", "0.60"]

    for method in ("Combined", "Greedy"):
        xs = [tradeoff[method][key][0] for key in ordered]
        ys = [tradeoff[method][key][1] for key in ordered]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2.2, label=method, color=METHOD_COLORS[method])
        for x, y, label in zip(xs, ys, ordered, strict=True):
            ax.annotate(f"λ={label}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)

    ax.set_xlabel("Cumulative switches")
    ax.set_ylabel("Cumulative interference")
    ax.set_title("F4. Interference-Switching Tradeoff")
    polish_axes(ax)
    ax.legend()
    save_figure(fig, output_dir / "F4_interference_switching_tradeoff.png")


def make_f5(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.9, 5.1))
    records = build_transfer_gain_records()
    for regime in ("stationary", "gradual", "sudden"):
        mask = records["regime"] == regime
        ax.scatter(
            records["delta"][mask],
            records["gain"][mask],
            s=56,
            alpha=0.88,
            color=REGIME_COLORS[regime],
            label=regime.capitalize(),
            edgecolor="white",
            linewidth=0.6,
        )
    ax.axvline(0.35, color="#C1121F", linestyle="--", linewidth=1.3, alpha=0.9)
    ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    ax.text(0.355, ax.get_ylim()[1] * 0.84, "reset threshold", color="#C1121F", fontsize=9, weight="bold")
    ax.set_xlabel("Graph change magnitude Δt")
    ax.set_ylabel("Transfer gain JCold - JCombined")
    ax.set_title("F5. Transfer Gain vs Graph Change")
    polish_axes(ax)
    ax.legend()
    save_figure(fig, output_dir / "F5_transfer_gain_vs_graph_change.png")


def make_f6(methods, output_dir: Path, quantum_metadata: dict[str, object] | None = None, quantum_traces: dict[str, object] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    if quantum_metadata is None or quantum_traces is None:
        traces = build_adaptation_trace_profiles(lambda_switch=0.30)
        evaluations = traces["evaluation"]
        panels = [
            ("Gradual change", evaluations, traces["gradual_cold"], traces["gradual_combined"]),
            ("Sudden change", evaluations, traces["sudden_cold"], traces["sudden_combined"]),
        ]
    else:
        regime = str(quantum_metadata["regime"]).capitalize()
        time_steps = int(quantum_metadata["time_steps"])
        selected_snapshots = [1, max(1, time_steps - 1)]
        panels = []
        for snapshot_t in selected_snapshots:
            cold = quantum_traces["Cold"][snapshot_t]
            combined = quantum_traces["Combined"][snapshot_t]
            evaluations = np.array(cold["evaluation"], dtype=int)
            panels.append(
                (
                    f"{regime} change (t={snapshot_t})",
                    evaluations,
                    np.array(cold["best_so_far"], dtype=float),
                    np.array(combined["best_so_far"], dtype=float),
                )
            )

    for ax, (title, evaluations, cold_trace, combined_trace) in zip(axes, panels, strict=True):
        ax.step(evaluations, cold_trace, where="post", color=METHOD_COLORS["Cold"], linewidth=2.1, label="Cold")
        ax.step(
            evaluations,
            combined_trace,
            where="post",
            color=METHOD_COLORS["Combined"],
            linewidth=2.4,
            label="Combined",
        )
        ax.set_title(title)
        ax.set_xlabel("Evaluation")
        polish_axes(ax)

    axes[0].set_ylabel("Best-so-far cost")
    axes[0].legend()
    fig.suptitle("F6. Optimization Adaptation Traces", fontsize=15, y=1.02, weight="bold")
    save_figure(fig, output_dir / "F6_optimization_adaptation_traces.png")


def make_f7(output_dir: Path) -> None:
    curves = build_cx_budget_curves()
    cx = curves["cx_budget"]
    ring_feasible = curves["ring_feasible"]
    penalty_feasible = curves["penalty_feasible"]
    ring_cost = curves["ring_cost"]
    penalty_cost = curves["penalty_cost"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))

    axes[0].plot(cx, ring_feasible, marker="o", linewidth=2.3, color="#1B9C73", label="Ring-XY")
    axes[0].plot(cx, penalty_feasible, marker="o", linewidth=2.3, color="#C1121F", label="Penalty-X")
    axes[0].set_title("Feasible fraction")
    axes[0].set_xlabel("CX budget")
    axes[0].set_ylabel("Feasible fraction")
    axes[0].set_ylim(0.58, 1.02)
    polish_axes(axes[0])
    axes[0].legend()

    axes[1].plot(cx, ring_cost, marker="o", linewidth=2.3, color="#1B9C73", label="Ring-XY")
    axes[1].plot(cx, penalty_cost, marker="o", linewidth=2.3, color="#C1121F", label="Penalty-X")
    axes[1].set_title("Best sample cost")
    axes[1].set_xlabel("CX budget")
    axes[1].set_ylabel("Cost")
    polish_axes(axes[1])

    fig.suptitle("F7. Feasibility vs CX Budget", fontsize=15, y=1.02, weight="bold")
    save_figure(fig, output_dir / "F7_feasibility_vs_cx_budget.png")


def make_f8(output_dir: Path) -> None:
    curves = build_noise_sweep_curves()
    noise = curves["noise"]
    feasible_combined = curves["feasible_combined"]
    feasible_cold = curves["feasible_cold"]
    gap_combined = curves["gap_combined"]
    gap_cold = curves["gap_cold"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))

    axes[0].plot(noise, feasible_combined, marker="o", linewidth=2.3, color=METHOD_COLORS["Combined"], label="Combined")
    axes[0].plot(noise, feasible_cold, marker="o", linewidth=2.3, color=METHOD_COLORS["Cold"], label="Cold")
    axes[0].set_title("Feasible fraction under noise")
    axes[0].set_xlabel("Depolarizing rate")
    axes[0].set_ylabel("Feasible fraction")
    axes[0].set_ylim(0.78, 1.02)
    polish_axes(axes[0])
    axes[0].legend()

    axes[1].plot(noise, gap_combined, marker="o", linewidth=2.3, color=METHOD_COLORS["Combined"], label="Combined")
    axes[1].plot(noise, gap_cold, marker="o", linewidth=2.3, color=METHOD_COLORS["Cold"], label="Cold")
    axes[1].set_title("Cumulative optimality-gap degradation")
    axes[1].set_xlabel("Depolarizing rate")
    axes[1].set_ylabel("Cumulative gap")
    polish_axes(axes[1])

    fig.suptitle("F8. Ideal vs Noisy", fontsize=15, y=1.02, weight="bold")
    save_figure(fig, output_dir / "F8_ideal_vs_noisy.png")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "artifacts" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _channel_cmap()

    sequence, base_methods = build_demo_rollouts()
    methods = dict(base_methods)
    quantum_metadata = None
    quantum_traces = None

    if args.rollout_source == "quantum":
        quantum_metadata, quantum_methods, quantum_traces = load_quantum_rollout_dataset(args.quantum_prefix)
        methods = {**methods, **quantum_methods}

    combined_allocations = base_methods["Combined"].allocations

    make_f1(sequence, combined_allocations, output_dir)
    make_f2(methods, output_dir, quantum_metadata=quantum_metadata)
    make_f3(methods, output_dir)
    make_f4(output_dir)
    make_f5(output_dir)
    make_f6(methods, output_dir, quantum_metadata=quantum_metadata, quantum_traces=quantum_traces)
    make_f7(output_dir)
    make_f8(output_dir)

    for path in sorted(output_dir.glob("F*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
