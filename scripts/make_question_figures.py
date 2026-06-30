from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from respec.question_study import QUESTION_METHODS, pretty_method_label
from respec.visualization import (
    CHANNEL_COLORS,
    METHOD_COLORS,
    REGIME_COLORS,
    apply_style,
    polish_axes,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render research-question figures for ReSpec-QAOA.")
    parser.add_argument("--input-prefix", default="question_study")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "question_figures"))
    parser.add_argument("--fps", type=int, default=3)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_question_dataset(prefix: str) -> dict[str, object]:
    raw_dir = ROOT / "artifacts" / "raw_results"
    metadata = json.loads((raw_dir / f"{prefix}_metadata.json").read_text(encoding="utf-8"))

    def _parse_rows(name: str, int_fields: set[str], float_fields: set[str]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in _read_csv_rows(raw_dir / f"{prefix}_{name}.csv"):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in int_fields:
                    parsed[key] = int(value)
                elif key in float_fields:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
        return rows

    return {
        "metadata": metadata,
        "summary_rows": _parse_rows(
            "summary",
            {"seed"},
            {"cumulative_cost", "cumulative_interference", "cumulative_switches", "offline_cost", "offline_gap", "improvement_vs_cold"},
        ),
        "step_rows": _parse_rows(
            "step_metrics",
            {"seed", "snapshot_t"},
            {"delta", "step_cost", "step_interference", "step_switches", "cumulative_cost", "cumulative_interference", "cumulative_switches", "gain_vs_cold"},
        ),
        "allocation_rows": _parse_rows(
            "allocations",
            {"seed", "snapshot_t", "user", "channel"},
            set(),
        ),
        "position_rows": _parse_rows(
            "positions",
            {"seed", "snapshot_t", "user"},
            {"x", "y"},
        ),
        "edge_rows": _parse_rows(
            "edges",
            {"seed", "snapshot_t", "u", "v"},
            {"weight"},
        ),
        "transfer_rows": _parse_rows(
            "transfer_gain",
            {"seed", "snapshot_t"},
            {"delta", "gain_vs_cold"},
        ),
        "factorial_rows": _parse_rows(
            "factorial_effects",
            {"seed", "state_reuse", "parameter_reuse"},
            {"improvement_vs_cold"},
        ),
        "tradeoff_rows": _parse_rows(
            "tradeoff_scan",
            {"seed"},
            {"lambda_switch", "cumulative_switches", "cumulative_interference", "cumulative_cost"},
        ),
        "trace_rows": _parse_rows(
            "trace_profiles",
            {"seed", "snapshot_t", "evaluation"},
            {"best_so_far_cost"},
        ),
    }


def _method_pretty(method: str) -> str:
    return pretty_method_label(method)


def _representative_seed(metadata: dict[str, object], regime: str) -> int:
    return int(metadata["representative_seeds"][regime])


def _allocation_matrix(
    allocation_rows: list[dict[str, object]],
    *,
    regime: str,
    seed: int,
    method: str,
    n_users: int,
    time_steps: int,
) -> np.ndarray:
    data = np.zeros((n_users, time_steps), dtype=int)
    for row in allocation_rows:
        if row["regime"] == regime and row["seed"] == seed and row["method"] == method:
            data[int(row["user"]), int(row["snapshot_t"])] = int(row["channel"])
    return data


def _jump_snapshot(step_rows: list[dict[str, object]], regime: str, seed: int) -> int | None:
    candidates = [row for row in step_rows if row["regime"] == regime and row["seed"] == seed and row["method"] == "Cold" and int(row["snapshot_t"]) > 0]
    if not candidates:
        return None
    return int(max(candidates, key=lambda row: float(row["delta"]))["snapshot_t"])


def _edge_matrix(edge_rows: list[dict[str, object]], regime: str, seed: int, snapshot_t: int, n_users: int) -> np.ndarray:
    weights = np.zeros((n_users, n_users), dtype=float)
    for row in edge_rows:
        if row["regime"] == regime and row["seed"] == seed and row["snapshot_t"] == snapshot_t:
            u = int(row["u"])
            v = int(row["v"])
            weights[u, v] = float(row["weight"])
            weights[v, u] = float(row["weight"])
    return weights


def _positions(position_rows: list[dict[str, object]], regime: str, seed: int, snapshot_t: int, n_users: int) -> np.ndarray:
    positions = np.zeros((n_users, 2), dtype=float)
    for row in position_rows:
        if row["regime"] == regime and row["seed"] == seed and row["snapshot_t"] == snapshot_t:
            positions[int(row["user"]), 0] = float(row["x"])
            positions[int(row["user"]), 1] = float(row["y"])
    return positions


def make_f1_main_result(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    methods = QUESTION_METHODS
    regimes = ("stationary", "gradual", "sudden")

    for ax, regime in zip(axes, regimes, strict=True):
        data = [
            [float(row["offline_gap"]) for row in summary_rows if row["regime"] == regime and row["method"] == method]
            for method in methods
        ]
        positions = np.arange(1, len(methods) + 1)
        box = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6, showfliers=False)
        for patch, method in zip(box["boxes"], methods, strict=True):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.25)
            patch.set_edgecolor(METHOD_COLORS[method])
            patch.set_linewidth(1.4)
        for median in box["medians"]:
            median.set_color("#1F1F1F")
            median.set_linewidth(1.5)
        for whisker in box["whiskers"]:
            whisker.set_color("#666666")
        for cap in box["caps"]:
            cap.set_color("#666666")

        rng = np.random.default_rng(abs(hash(regime)) % (2 ** 32))
        for idx, method in enumerate(methods, start=1):
            values = np.array(
                [float(row["offline_gap"]) for row in summary_rows if row["regime"] == regime and row["method"] == method],
                dtype=float,
            )
            jitter = rng.normal(0.0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), idx, dtype=float) + jitter,
                values,
                s=38,
                color=METHOD_COLORS[method],
                alpha=0.75,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

        combined = np.array(
            [float(row["improvement_vs_cold"]) for row in summary_rows if row["regime"] == regime and row["method"] == "Combined"],
            dtype=float,
        )
        ax.text(
            0.03,
            0.94,
            f"Median Combined gain = {np.median(combined):+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=REGIME_COLORS[regime],
            weight="bold",
        )
        ax.set_title(regime.capitalize())
        ax.set_xticks(positions)
        ax.set_xticklabels(["Cold", "Param", "State", "Combined"], rotation=18)
        ax.set_xlabel("Dynamic strategy")
        polish_axes(ax)

    axes[0].set_ylabel("Cumulative gap above offline DP")
    fig.suptitle("F1. Warm-Start Benefit by Network Regime", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F1_main_result_by_regime.png")


def make_f2_factorial_heatmaps(factorial_rows: list[dict[str, object]], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    regimes = ("stationary", "gradual", "sudden")

    medians: list[float] = [0.0]
    for row in factorial_rows:
        medians.append(float(row["improvement_vs_cold"]))
    bound = max(abs(min(medians)), abs(max(medians)), 1e-6)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    cell_to_method = {
        (0, 0): "Cold",
        (0, 1): "Param",
        (1, 0): "State",
        (1, 1): "Combined",
    }

    for ax, regime in zip(axes, regimes, strict=True):
        matrix = np.zeros((2, 2), dtype=float)
        labels: dict[tuple[int, int], str] = {}
        for row_index in range(2):
            for col_index in range(2):
                method = cell_to_method[(row_index, col_index)]
                if method == "Cold":
                    value = 0.0
                else:
                    matches = [
                        float(row["improvement_vs_cold"])
                        for row in factorial_rows
                        if row["regime"] == regime and row["method"] == method
                    ]
                    value = float(np.median(np.array(matches, dtype=float)))
                matrix[row_index, col_index] = value
                labels[(row_index, col_index)] = method

        im = ax.imshow(matrix, cmap="RdYlGn", norm=norm)
        ax.set_title(regime.capitalize())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Param", "Param"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No State", "State"])
        ax.set_xlabel("Parameter reuse")
        for row_index in range(2):
            for col_index in range(2):
                method = labels[(row_index, col_index)]
                ax.text(
                    col_index,
                    row_index,
                    f"{method}\n{matrix[row_index, col_index]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    color="#1F1F1F",
                )
        ax.set_xticks(np.arange(-0.5, 2.0, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, 2.0, 1.0), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="#E9E3D8", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    axes[0].set_ylabel("State reuse")
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("Median improvement vs Cold")
    fig.suptitle("F2. State and Parameter Reuse Decomposition", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F2_factorial_reuse_heatmaps.png")


def _plot_binned_trend(ax, xs: np.ndarray, ys: np.ndarray, color: str) -> None:
    if len(xs) < 2:
        return
    bins = np.linspace(float(xs.min()), float(xs.max()) + 1e-9, num=5)
    centers: list[float] = []
    medians: list[float] = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        mask = (xs >= left) & (xs <= right if right == bins[-1] else xs < right)
        if np.count_nonzero(mask) == 0:
            continue
        centers.append(float((left + right) / 2.0))
        medians.append(float(np.median(ys[mask])))
    if centers:
        ax.plot(centers, medians, color=color, linewidth=2.0, alpha=0.95)


def make_f3_transfer_gain(transfer_rows: list[dict[str, object]], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    methods = ("Param", "State", "Combined")
    regimes = ("stationary", "gradual", "sudden")

    for ax, method in zip(axes, methods, strict=True):
        for regime in regimes:
            rows = [row for row in transfer_rows if row["method"] == method and row["regime"] == regime]
            xs = np.array([float(row["delta"]) for row in rows], dtype=float)
            ys = np.array([float(row["gain_vs_cold"]) for row in rows], dtype=float)
            ax.scatter(
                xs,
                ys,
                s=34,
                alpha=0.78,
                color=REGIME_COLORS[regime],
                label=regime.capitalize(),
                edgecolor="white",
                linewidth=0.5,
            )
            _plot_binned_trend(ax, xs, ys, REGIME_COLORS[regime])
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
        ax.set_title(_method_pretty(method))
        ax.set_xlabel("Graph change magnitude Δt")
        polish_axes(ax)

    axes[0].set_ylabel("Step gain vs Cold")
    axes[0].legend(ncol=1, fontsize=8)
    fig.suptitle("F3. Transfer Gain vs Graph Change", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F3_transfer_gain_vs_graph_change.png")


def make_f4_adaptation_traces(trace_rows: list[dict[str, object]], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    scenarios = ("gradual", "sudden")

    for ax, scenario in zip(axes, scenarios, strict=True):
        for method in QUESTION_METHODS:
            rows = [
                row for row in trace_rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            rows.sort(key=lambda row: int(row["evaluation"]))
            evaluations = np.array([int(row["evaluation"]) for row in rows], dtype=int)
            values = np.array([float(row["best_so_far_cost"]) for row in rows], dtype=float)
            ax.step(
                evaluations,
                values,
                where="post",
                linewidth=2.15 if method == "Combined" else 1.85,
                color=METHOD_COLORS[method],
                label=method,
            )
        ax.set_title(f"{scenario.capitalize()} change")
        ax.set_xlabel("Evaluation")
        polish_axes(ax)

    axes[0].set_ylabel("Best-so-far cost")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("F4. Adaptation Traces Under Fixed Budget", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F4_adaptation_traces.png")


def make_f5_allocation_timelines(
    allocation_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    apply_style()
    n_users = int(metadata["n_users"])
    time_steps = int(metadata["time_steps"])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.0), sharex=True, sharey=True)
    regimes = ("gradual", "sudden")
    methods = ("Cold", "Combined")
    cmap = apply_style()

    for row_index, regime in enumerate(regimes):
        seed = _representative_seed(metadata, regime)
        jump_snapshot = _jump_snapshot(step_rows, regime, seed)
        for col_index, method in enumerate(methods):
            ax = axes[row_index, col_index]
            data = _allocation_matrix(
                allocation_rows,
                regime=regime,
                seed=seed,
                method=method,
                n_users=n_users,
                time_steps=time_steps,
            )
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=2)
            if regime == "sudden" and jump_snapshot is not None:
                ax.axvline(jump_snapshot - 0.5, color="#C1121F", linestyle="--", linewidth=1.2, alpha=0.85)
            ax.set_title(f"{regime.capitalize()} | {method}")
            ax.set_xticks(range(time_steps))
            ax.set_yticks(range(n_users))
            ax.set_yticklabels([f"User {idx}" for idx in range(n_users)])
            ax.set_xlabel("Snapshot t")

    axes[0, 0].set_ylabel("User")
    axes[1, 0].set_ylabel("User")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Ch 0", "Ch 1", "Ch 2"])
    fig.suptitle("F5. Allocation Trajectories in Gradual and Sudden Regimes", fontsize=15, y=1.02, weight="bold")
    save_figure(fig, output_dir / "F5_allocation_timeline_heatmaps.png")


def make_f6_tradeoff(tradeoff_rows: list[dict[str, object]], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    regimes = ("gradual", "sudden")

    for ax, regime in zip(axes, regimes, strict=True):
        regime_rows = [row for row in tradeoff_rows if row["regime"] == regime]
        for method in ("Cold", "Combined"):
            method_rows = [row for row in regime_rows if row["method"] == method]
            lambdas = sorted({float(row["lambda_switch"]) for row in method_rows})
            xs: list[float] = []
            ys: list[float] = []
            for lambda_value in lambdas:
                matches = [row for row in method_rows if abs(float(row["lambda_switch"]) - lambda_value) < 1e-9]
                xs.append(float(np.median([float(row["cumulative_switches"]) for row in matches])))
                ys.append(float(np.median([float(row["cumulative_interference"]) for row in matches])))
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.2,
                markersize=7,
                color=METHOD_COLORS[method],
                label=method,
            )
            grouped_labels: dict[tuple[float, float], list[float]] = {}
            for x, y, lambda_value in zip(xs, ys, lambdas, strict=True):
                grouped_labels.setdefault((round(x, 6), round(y, 6)), []).append(lambda_value)
            for (x, y), lambda_values in grouped_labels.items():
                label = "/".join(f"{value:.2f}" for value in lambda_values)
                if method == "Cold":
                    label_offset = (6, 5)
                elif y < 0.08 and x < 0.2:
                    label_offset = (6, -18)
                elif y < 0.08:
                    label_offset = (6, -4)
                else:
                    label_offset = (6, -14)
                ax.annotate(
                    f"λ={label}",
                    (x, y),
                    textcoords="offset points",
                    xytext=label_offset,
                    fontsize=8,
                )
        ax.set_title(f"{regime.capitalize()} sequences")
        ax.set_xlabel("Cumulative switches")
        polish_axes(ax)

    axes[0].set_ylabel("Cumulative interference")
    axes[0].legend()
    fig.suptitle("F6. Interference-Switching Tradeoff Under λ Sweep", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F6_interference_switching_tradeoff.png")


def make_f7_network_animation(
    position_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    allocation_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
    fps: int,
) -> None:
    apply_style()
    regime = "sudden"
    seed = _representative_seed(metadata, regime)
    n_users = int(metadata["n_users"])
    time_steps = int(metadata["time_steps"])
    jump_snapshot = _jump_snapshot(step_rows, regime, seed)

    deltas = {
        int(row["snapshot_t"]): float(row["delta"])
        for row in step_rows
        if row["regime"] == regime and row["seed"] == seed and row["method"] == "Cold"
    }
    combined_costs = {
        int(row["snapshot_t"]): float(row["cumulative_cost"])
        for row in step_rows
        if row["regime"] == regime and row["seed"] == seed and row["method"] == "Combined"
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.8))

    def _draw_frame(snapshot_t: int) -> None:
        ax.clear()
        positions = _positions(position_rows, regime, seed, snapshot_t, n_users)
        weights = _edge_matrix(edge_rows, regime, seed, snapshot_t, n_users)
        allocation = _allocation_matrix(
            allocation_rows,
            regime=regime,
            seed=seed,
            method="Combined",
            n_users=n_users,
            time_steps=time_steps,
        )[:, snapshot_t]

        for u in range(n_users):
            for v in range(u + 1, n_users):
                if weights[u, v] <= 0:
                    continue
                ax.plot(
                    [positions[u, 0], positions[v, 0]],
                    [positions[u, 1], positions[v, 1]],
                    color="#9C6644",
                    alpha=0.18 + 0.72 * weights[u, v],
                    linewidth=1.1 + 3.0 * weights[u, v],
                    zorder=1,
                )

        moved_user = None
        if jump_snapshot is not None and snapshot_t == jump_snapshot and snapshot_t > 0:
            prev = _positions(position_rows, regime, seed, snapshot_t - 1, n_users)
            displacement = np.linalg.norm(positions - prev, axis=1)
            moved_user = int(np.argmax(displacement))

        for user, (x, y) in enumerate(positions):
            halo = "#F94144" if moved_user == user else "#FCFBF8"
            ax.scatter(x, y, s=360, color=halo, edgecolor="none", zorder=2)
            ax.scatter(
                x,
                y,
                s=215,
                color=CHANNEL_COLORS[int(allocation[user])],
                edgecolor="#1F1F1F",
                linewidth=1.0,
                zorder=3,
            )
            ax.text(x, y, f"U{user}", ha="center", va="center", color="white", weight="bold", fontsize=9)

        ax.set_title(f"Sudden sequence | Combined | t={snapshot_t}")
        ax.text(
            0.02,
            0.98,
            f"Δt = {deltas.get(snapshot_t, 0.0):.3f}\nCumulative cost = {combined_costs.get(snapshot_t, 0.0):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FCFBF8", "edgecolor": "#D9D3C7"},
        )
        if jump_snapshot is not None and snapshot_t == jump_snapshot and moved_user is not None:
            ax.text(
                positions[moved_user, 0] + 0.03,
                positions[moved_user, 1] + 0.05,
                "jump",
                color="#C1121F",
                fontsize=9,
                weight="bold",
            )
        ax.set_xlim(0.03, 0.97)
        ax.set_ylim(0.03, 0.97)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    animation = FuncAnimation(fig, _draw_frame, frames=range(time_steps), interval=max(200, 1000 // max(fps, 1)))
    output_path = output_dir / "F7_network_evolution.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_f8_optimization_race(
    trace_rows: list[dict[str, object]],
    output_dir: Path,
    fps: int,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    scenarios = ("gradual", "sudden")
    max_evaluation = max(int(row["evaluation"]) for row in trace_rows)

    def _draw_frame(frame: int) -> None:
        for ax, scenario in zip(axes, scenarios, strict=True):
            ax.clear()
            for method in QUESTION_METHODS:
                rows = [
                    row for row in trace_rows
                    if row["scenario"] == scenario and row["method"] == method and int(row["evaluation"]) <= frame
                ]
                rows.sort(key=lambda row: int(row["evaluation"]))
                evaluations = np.array([int(row["evaluation"]) for row in rows], dtype=int)
                values = np.array([float(row["best_so_far_cost"]) for row in rows], dtype=float)
                if len(evaluations) == 0:
                    continue
                ax.step(
                    evaluations,
                    values,
                    where="post",
                    linewidth=2.2 if method == "Combined" else 1.9,
                    color=METHOD_COLORS[method],
                    label=method,
                )
            ax.set_title(f"{scenario.capitalize()} change")
            ax.set_xlabel("Evaluation")
            ax.set_xlim(1, max_evaluation)
            polish_axes(ax)
        axes[0].set_ylabel("Best-so-far cost")
        axes[0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"F8. Optimization Race Under Fixed Budget (eval {frame}/{max_evaluation})", fontsize=15, y=1.03, weight="bold")

    animation = FuncAnimation(fig, _draw_frame, frames=range(1, max_evaluation + 1), interval=max(180, 1000 // max(fps, 1)))
    output_path = output_dir / "F8_optimization_race.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_question_dataset(args.input_prefix)
    metadata = dataset["metadata"]

    make_f1_main_result(dataset["summary_rows"], output_dir)
    make_f2_factorial_heatmaps(dataset["factorial_rows"], output_dir)
    make_f3_transfer_gain(dataset["transfer_rows"], output_dir)
    make_f4_adaptation_traces(dataset["trace_rows"], output_dir)
    make_f5_allocation_timelines(dataset["allocation_rows"], dataset["step_rows"], metadata, output_dir)
    make_f6_tradeoff(dataset["tradeoff_rows"], output_dir)
    make_f7_network_animation(
        dataset["position_rows"],
        dataset["edge_rows"],
        dataset["allocation_rows"],
        dataset["step_rows"],
        metadata,
        output_dir,
        args.fps,
    )
    make_f8_optimization_race(dataset["trace_rows"], output_dir, args.fps)

    for path in sorted(output_dir.iterdir()):
        if path.suffix.lower() in {".png", ".gif"}:
            print(path.name)


if __name__ == "__main__":
    main()
