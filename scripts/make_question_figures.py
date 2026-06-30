from __future__ import annotations

import argparse
import csv
import json
import math
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

from respec.environment import generate_sequence
from respec.heuristics import rollout_heuristic
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
            {"objective_value", "best_so_far_cost"},
        ),
    }


def _method_pretty(method: str) -> str:
    return pretty_method_label(method)


def _regime_label(regime: str) -> str:
    return regime.replace("_", " ").title()


def _study_regimes(metadata: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(regime) for regime in metadata.get("regimes", ("stationary", "gradual", "sudden")))


def _dynamic_regimes(metadata: dict[str, object]) -> tuple[str, ...]:
    return tuple(regime for regime in _study_regimes(metadata) if regime != "stationary")


def _focus_animation_regime(metadata: dict[str, object]) -> str:
    dynamic_regimes = _dynamic_regimes(metadata)
    if "continuous_sudden" in dynamic_regimes:
        return "continuous_sudden"
    if "sudden" in dynamic_regimes:
        return "sudden"
    return dynamic_regimes[-1]


def _question_main_regimes(metadata: dict[str, object]) -> tuple[str, ...]:
    preferred = ("stationary", "gradual", "continuous_sudden")
    available = set(_study_regimes(metadata))
    return tuple(regime for regime in preferred if regime in available)


def _axes_grid(
    count: int,
    *,
    sharey: bool = False,
    panel_width: float = 4.6,
    panel_height: float = 4.5,
) -> tuple[plt.Figure, np.ndarray]:
    if count <= 3:
        rows = 1
        cols = count
    elif count == 4:
        rows = 2
        cols = 2
    else:
        cols = 3
        rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(panel_width * cols, panel_height * rows), sharey=sharey, squeeze=False)
    flat = axes.ravel()
    for extra in flat[count:]:
        fig.delaxes(extra)
    return fig, flat[:count]


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


def _peak_change_snapshot(step_rows: list[dict[str, object]], regime: str, seed: int) -> int | None:
    if regime == "stationary":
        return None
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


def _summary_rows_for_regime(
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
    regime: str,
) -> list[dict[str, object]]:
    rows = [row.copy() for row in summary_rows if row["regime"] == regime]
    rows.extend(row.copy() for row in _extension_summary_rows_for_regime(regime))

    cold_cost_by_seed = {
        int(row["seed"]): float(row["cumulative_cost"])
        for row in rows
        if row["method"] == "Cold"
    }
    for row in rows:
        seed = int(row["seed"])
        cumulative_cost = float(row["cumulative_cost"])
        if row["method"] == "Cold":
            row["improvement_vs_cold"] = 0.0
        else:
            row["improvement_vs_cold"] = cold_cost_by_seed[seed] - cumulative_cost
    return rows


def make_f1_main_result(summary_rows: list[dict[str, object]], metadata: dict[str, object], output_dir: Path) -> None:
    apply_style()
    regimes = _question_main_regimes(metadata)
    fig, axes = _axes_grid(len(regimes), sharey=True, panel_width=4.7, panel_height=4.7)
    methods = QUESTION_METHODS

    for ax, regime in zip(axes, regimes, strict=True):
        regime_rows = _summary_rows_for_regime(summary_rows, metadata, regime)
        data = [
            [float(row["cumulative_cost"]) for row in regime_rows if row["method"] == method]
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
                [float(row["cumulative_cost"]) for row in regime_rows if row["method"] == method],
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
            [float(row["improvement_vs_cold"]) for row in regime_rows if row["method"] == "Combined"],
            dtype=float,
        )
        seed_count = len({int(row["seed"]) for row in regime_rows})
        ax.text(
            0.03,
            0.94,
            f"Median Combined gain = {np.median(combined):+.3f}\nSeeds = {seed_count}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=REGIME_COLORS[regime],
            weight="bold",
            bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FCFBF8", "edgecolor": "#D9D3C7"},
        )
        ax.set_title(_regime_label(regime))
        ax.set_xticks(positions)
        ax.set_xticklabels(["Cold", "Parameter\nTransfer", "State\nWarm Start", "Combined"], rotation=0)
        ax.set_xlabel("Dynamic strategy")
        polish_axes(ax)

    axes[0].set_ylabel("Cumulative total cost")
    fig.suptitle("Warm-Start Benefit by Network Regime", fontsize=15, y=1.02, weight="bold")
    save_figure(fig, output_dir / "F1_main_result_by_regime.png")


def make_f2_factorial_heatmaps(summary_rows: list[dict[str, object]], metadata: dict[str, object], output_dir: Path) -> None:
    apply_style()
    regimes = _question_main_regimes(metadata)
    fig, axes = _axes_grid(len(regimes), sharey=True, panel_width=4.6, panel_height=4.5)

    medians: list[float] = [0.0]
    for regime in regimes:
        regime_rows = _summary_rows_for_regime(summary_rows, metadata, regime)
        for method in QUESTION_METHODS:
            if method == "Cold":
                medians.append(0.0)
                continue
            method_values = [float(row["improvement_vs_cold"]) for row in regime_rows if row["method"] == method]
            if method_values:
                medians.append(float(np.median(np.array(method_values, dtype=float))))
    bound = max(abs(min(medians)), abs(max(medians)), 1e-6)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    cell_to_method = {
        (0, 0): "Cold",
        (0, 1): "Param",
        (1, 0): "State",
        (1, 1): "Combined",
    }

    for ax, regime in zip(axes, regimes, strict=True):
        regime_rows = _summary_rows_for_regime(summary_rows, metadata, regime)
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
                        for row in regime_rows
                        if row["method"] == method
                    ]
                    value = float(np.median(np.array(matches, dtype=float)))
                matrix[row_index, col_index] = value
                labels[(row_index, col_index)] = method

        im = ax.imshow(matrix, cmap="RdYlGn", norm=norm)
        ax.set_title(_regime_label(regime))
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
        seed_count = len({int(row["seed"]) for row in regime_rows})
        ax.text(
            0.03,
            0.97,
            f"Seeds = {seed_count}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=REGIME_COLORS[regime],
            weight="bold",
        )

    axes[0].set_ylabel("State reuse")
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("Median improvement vs Cold")
    fig.suptitle("State and Parameter Reuse Decomposition", fontsize=15, y=1.02, weight="bold")
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


def make_f3_transfer_gain(transfer_rows: list[dict[str, object]], metadata: dict[str, object], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    methods = ("Param", "State", "Combined")
    regimes = _study_regimes(metadata)

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
                label=_regime_label(regime),
                edgecolor="white",
                linewidth=0.5,
            )
            _plot_binned_trend(ax, xs, ys, REGIME_COLORS[regime])
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
        ax.set_title(_method_pretty(method))
        ax.set_xlabel("Graph change magnitude Δt")
        polish_axes(ax)

    axes[0].set_ylabel("Step gain vs Cold")
    axes[0].legend(ncol=1 if len(regimes) <= 3 else 2, fontsize=8)
    fig.suptitle("F3. Transfer Gain vs Graph Change", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F3_transfer_gain_vs_graph_change.png")


def make_f4_adaptation_traces(trace_rows: list[dict[str, object]], metadata: dict[str, object], output_dir: Path) -> None:
    apply_style()
    scenarios = _dynamic_regimes(metadata)
    fig, axes = _axes_grid(len(scenarios), sharey=True, panel_width=4.4, panel_height=4.7)

    for ax, scenario in zip(axes, scenarios, strict=True):
        for method in QUESTION_METHODS:
            rows = [row for row in trace_rows if row["scenario"] == scenario and row["method"] == method]
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
        ax.set_title(f"{_regime_label(scenario)} change")
        ax.set_xlabel("Evaluation")
        polish_axes(ax)

    axes[0].set_ylabel("Best-so-far cost")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("F4. Adaptation Traces Under Fixed Budget", fontsize=15, y=1.02, weight="bold")
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
    regimes = _dynamic_regimes(metadata)
    methods = ("Cold", "Combined")
    cmap = apply_style()
    fig, axes = plt.subplots(len(regimes), len(methods), figsize=(12.8, 3.0 * len(regimes) + 1.2), sharex=True, sharey=True, squeeze=False)

    for row_index, regime in enumerate(regimes):
        seed = _representative_seed(metadata, regime)
        peak_snapshot = _peak_change_snapshot(step_rows, regime, seed)
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
            if peak_snapshot is not None and regime in {"sudden", "continuous_sudden"}:
                ax.axvline(peak_snapshot - 0.5, color="#C1121F", linestyle="--", linewidth=1.2, alpha=0.85)
            ax.set_title(f"{_regime_label(regime)} | {method}")
            ax.set_xticks(range(time_steps))
            ax.set_yticks(range(n_users))
            ax.set_yticklabels([f"User {idx}" for idx in range(n_users)])
            ax.set_xlabel("Snapshot t")

    for row_index in range(len(regimes)):
        axes[row_index, 0].set_ylabel("User")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Ch 0", "Ch 1", "Ch 2"])
    fig.suptitle("F5. Allocation Trajectories Across Dynamic Regimes", fontsize=15, y=1.01, weight="bold")
    save_figure(fig, output_dir / "F5_allocation_timeline_heatmaps.png")


def make_f6_tradeoff(tradeoff_rows: list[dict[str, object]], metadata: dict[str, object], output_dir: Path) -> None:
    apply_style()
    regimes = _dynamic_regimes(metadata)
    fig, axes = _axes_grid(len(regimes), sharey=True, panel_width=4.4, panel_height=4.7)

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
        ax.set_title(f"{_regime_label(regime)} sequences")
        ax.set_xlabel("Cumulative switches")
        polish_axes(ax)

    axes[0].set_ylabel("Cumulative interference")
    axes[0].legend()
    fig.suptitle("F6. Interference-Switching Tradeoff Under λ Sweep", fontsize=15, y=1.02, weight="bold")
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
    regime = _focus_animation_regime(metadata)
    seed = _representative_seed(metadata, regime)
    n_users = int(metadata["n_users"])
    time_steps = int(metadata["time_steps"])
    peak_snapshot = _peak_change_snapshot(step_rows, regime, seed)

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
        if peak_snapshot is not None and snapshot_t == peak_snapshot and snapshot_t > 0:
            prev = _positions(position_rows, regime, seed, snapshot_t - 1, n_users)
            displacement = np.linalg.norm(positions - prev, axis=1)
            moved_user = int(np.argmax(displacement))

        for user, (x, y) in enumerate(positions):
            halo = REGIME_COLORS[regime] if moved_user == user else "#FCFBF8"
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

        ax.set_title(f"{_regime_label(regime)} sequence | Combined | t={snapshot_t}")
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
        if peak_snapshot is not None and snapshot_t == peak_snapshot and moved_user is not None:
            marker = "jump" if regime == "sudden" else "peak drift"
            ax.text(
                positions[moved_user, 0] + 0.03,
                positions[moved_user, 1] + 0.05,
                marker,
                color=REGIME_COLORS[regime],
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
    metadata: dict[str, object],
    output_dir: Path,
    fps: int,
) -> None:
    apply_style()
    scenarios = _dynamic_regimes(metadata)
    fig, axes = _axes_grid(len(scenarios), sharey=True, panel_width=4.3, panel_height=4.6)
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
            ax.set_title(f"{_regime_label(scenario)} change")
            ax.set_xlabel("Evaluation")
            ax.set_xlim(1, max_evaluation)
            polish_axes(ax)
        axes[0].set_ylabel("Best-so-far cost")
        axes[0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"F8. Optimization Race Under Fixed Budget (eval {frame}/{max_evaluation})", fontsize=15, y=1.02, weight="bold")

    animation = FuncAnimation(fig, _draw_frame, frames=range(1, max_evaluation + 1), interval=max(180, 1000 // max(fps, 1)))
    output_path = output_dir / "F8_optimization_race.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_f9_combined_vs_state_seed_gaps(
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    apply_style()
    regimes = _study_regimes(metadata)
    fig, axes = _axes_grid(len(regimes), sharey=True, panel_width=4.5, panel_height=4.7)

    regime_values: dict[str, tuple[list[int], np.ndarray]] = {}
    all_deltas: list[float] = []
    for regime in regimes:
        seeds = sorted({int(row["seed"]) for row in summary_rows if row["regime"] == regime})
        deltas: list[float] = []
        for seed in seeds:
            state = next(
                float(row["cumulative_cost"])
                for row in summary_rows
                if row["regime"] == regime and row["seed"] == seed and row["method"] == "State"
            )
            combined = next(
                float(row["cumulative_cost"])
                for row in summary_rows
                if row["regime"] == regime and row["seed"] == seed and row["method"] == "Combined"
            )
            delta = combined - state
            deltas.append(delta)
            all_deltas.append(delta)
        regime_values[regime] = (seeds, np.array(deltas, dtype=float))

    bound = max(max(abs(value) for value in all_deltas), 0.2)

    for ax, regime in zip(axes, regimes, strict=True):
        seeds, deltas = regime_values[regime]
        positions = np.arange(len(seeds), dtype=float)
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2)
        ax.axhspan(-bound, 0.0, color=METHOD_COLORS["Combined"], alpha=0.08)
        ax.axhspan(0.0, bound, color=METHOD_COLORS["State"], alpha=0.08)

        for x, seed, delta in zip(positions, seeds, deltas.tolist(), strict=True):
            color = METHOD_COLORS["Combined"] if delta < -1e-9 else METHOD_COLORS["State"] if delta > 1e-9 else "#6C757D"
            ax.vlines(x, 0.0, delta, color=color, linewidth=2.1, alpha=0.78, zorder=2)
            ax.scatter(x, delta, s=105, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(x, delta + 0.04 * bound * (1 if delta >= 0 else -1), str(seed), ha="center", va="center", fontsize=8)

        median = float(np.median(deltas))
        wins = int(np.count_nonzero(deltas < -1e-9))
        ties = int(np.count_nonzero(np.abs(deltas) <= 1e-9))
        ax.scatter(
            np.mean(positions) if len(positions) > 0 else 0.0,
            median,
            marker="D",
            s=80,
            color=REGIME_COLORS[regime],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        ax.text(
            0.03,
            0.95,
            f"Combined wins = {wins}/{len(deltas)}\nMedian Δ = {median:+.3f}\nTies = {ties}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=REGIME_COLORS[regime],
            weight="bold",
            bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FCFBF8", "edgecolor": "#D9D3C7"},
        )
        ax.set_title(_regime_label(regime))
        ax.set_xticks(positions)
        ax.set_xticklabels([f"seed {seed}" for seed in seeds], rotation=18 if len(seeds) > 2 else 0)
        ax.set_xlabel("Sequence seed")
        ax.set_ylim(-1.08 * bound, 1.08 * bound)
        polish_axes(ax)

    axes[0].set_ylabel("Cumulative cost gap\n(Combined - State)")
    fig.text(0.5, -0.01, "Negative values mean Combined is better; positive values mean State is better.", ha="center", fontsize=10)
    fig.suptitle("F9. Combined vs State Depends on Dynamic Regime", fontsize=15, y=1.03, weight="bold")
    save_figure(fig, output_dir / "F9_combined_vs_state_seed_gaps.png")


def _extension_summary_rows_for_regime(regime: str) -> list[dict[str, object]]:
    raw_dir = ROOT / "artifacts" / "raw_results"
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob(f"{regime}_extension_*_summary.csv")):
        for row in _read_csv_rows(path):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key == "seed":
                    parsed[key] = int(value)
                elif key in {
                    "cumulative_cost",
                    "cumulative_interference",
                    "cumulative_switches",
                    "offline_cost",
                    "offline_gap",
                    "improvement_vs_cold",
                    "mean_feasible_fraction",
                    "mean_success_probability",
                    "mean_expected_cost",
                    "mean_cx_count",
                    "mean_circuit_depth",
                    "total_wall_seconds",
                }:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def make_regime_quantum_vs_classical(
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
    *,
    regime: str,
    output_name: str,
) -> None:
    if regime not in _study_regimes(metadata):
        return

    apply_style()
    regime_summary_rows = [row for row in summary_rows if row["regime"] == regime]
    regime_summary_rows.extend(_extension_summary_rows_for_regime(regime))
    seeds = sorted({int(row["seed"]) for row in regime_summary_rows})
    if not seeds:
        return
    methods = ("Greedy", "Local Search", "Cold", "Param", "State", "Combined")
    n_users = int(metadata["n_users"])
    n_channels = int(metadata["n_channels"])
    time_steps = int(metadata["time_steps"])
    lambda_switch = float(metadata["lambda_switch"])

    values: dict[str, list[float]] = {method: [] for method in methods}
    for seed in seeds:
        sequence = generate_sequence(
            n_users=n_users,
            time_steps=time_steps,
            regime=regime,
            seed=seed,
        )
        greedy = rollout_heuristic(sequence, "Greedy", n_channels=n_channels, lambda_switch=lambda_switch)
        local = rollout_heuristic(sequence, "Local Search", n_channels=n_channels, lambda_switch=lambda_switch)
        values["Greedy"].append(float(greedy.cumulative_costs[-1]))
        values["Local Search"].append(float(local.cumulative_costs[-1]))
        values["Cold"].append(
            next(
                float(row["cumulative_cost"])
                for row in regime_summary_rows
                if row["seed"] == seed and row["method"] == "Cold"
            )
        )
        values["State"].append(
            next(
                float(row["cumulative_cost"])
                for row in regime_summary_rows
                if row["seed"] == seed and row["method"] == "State"
            )
        )
        values["Combined"].append(
            next(
                float(row["cumulative_cost"])
                for row in regime_summary_rows
                if row["seed"] == seed and row["method"] == "Combined"
            )
        )
        values["Param"].append(
            next(
                float(row["cumulative_cost"])
                for row in regime_summary_rows
                if row["seed"] == seed and row["method"] == "Param"
            )
        )

    seed_panel_width = max(7.9, 1.38 * len(seeds) + 2.8)
    fig, (ax_seed, ax_mean) = plt.subplots(
        1,
        2,
        figsize=(seed_panel_width + 4.7, 5.2),
        gridspec_kw={"width_ratios": [1.95, 1.0]},
    )
    fig.subplots_adjust(top=0.79)

    x = np.arange(len(seeds), dtype=float)
    width = 0.12
    offsets = (np.arange(len(methods), dtype=float) - 0.5 * (len(methods) - 1)) * width
    y_max = max(max(series) for series in values.values()) * 1.18

    for offset, method in zip(offsets, methods, strict=True):
        bars = ax_seed.bar(
            x + offset,
            values[method],
            width=width,
            color=METHOD_COLORS[method],
            alpha=0.9,
            label=method,
        )
        for bar in bars:
            height = float(bar.get_height())
            ax_seed.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02 * y_max,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    for idx, seed in enumerate(seeds):
        seed_costs = {method: values[method][idx] for method in methods}
        best_value = min(seed_costs.values())
        star_y = min(max(seed_costs.values()) + 0.12 * y_max, 0.93 * y_max)
        for method in methods:
            if abs(seed_costs[method] - best_value) > 1e-9:
                continue
            best_x = x[idx] + offsets[methods.index(method)]
            ax_seed.scatter(
                best_x,
                star_y,
                marker="*",
                s=145,
                color=METHOD_COLORS[method],
                zorder=4,
            )

    ax_seed.set_title(f"{_regime_label(regime)} sequences by seed")
    ax_seed.set_xticks(x)
    ax_seed.set_xticklabels([f"seed {seed}" for seed in seeds])
    ax_seed.set_xlabel("Sequence seed")
    ax_seed.set_ylabel("Cumulative total cost")
    ax_seed.set_ylim(0.0, y_max)
    polish_axes(ax_seed)
    legend_y = 1.00 if regime == "continuous_sudden" else 1.04
    ax_seed.legend(
        ncol=2,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.0, legend_y),
        borderaxespad=0.0,
        frameon=False,
    )

    means = np.array([float(np.mean(values[method])) for method in methods], dtype=float)
    best_counts = np.array(
        [
            sum(
                1
                for idx in range(len(seeds))
                if abs(values[method][idx] - min(values[candidate][idx] for candidate in methods)) <= 1e-9
            )
            for method in methods
        ],
        dtype=int,
    )
    y_positions = np.arange(len(methods), dtype=float)
    bars = ax_mean.barh(y_positions, means, color=[METHOD_COLORS[method] for method in methods], alpha=0.9)
    for bar, method, mean_value, bests in zip(bars, methods, means.tolist(), best_counts.tolist(), strict=True):
        ax_mean.text(
            float(bar.get_width()) + 0.02 * y_max,
            bar.get_y() + bar.get_height() / 2.0,
            f"{mean_value:.3f} | best/tied-best {bests}/{len(seeds)}",
            va="center",
            fontsize=9,
        )
    seed_title_y = 0.98 if regime == "continuous_sudden" else 1.0
    mean_title_y = 0.95 if regime == "continuous_sudden" else 1.0
    ax_seed.set_title(f"{_regime_label(regime)} sequences by seed", y=seed_title_y)
    ax_mean.set_title(f"Average over {len(seeds)} {_regime_label(regime).lower()} seeds", y=mean_title_y)
    ax_mean.set_yticks(y_positions)
    ax_mean.set_yticklabels(["Greedy", "Greedy +\nLocal Search", "Cold", "Parameter\nTransfer", "State", "Combined"])
    ax_mean.set_xlabel("Mean cumulative cost")
    ax_mean.set_xlim(0.0, y_max)
    ax_mean.invert_yaxis()
    polish_axes(ax_mean)
    caption_y = 0.01 if regime == "continuous_sudden" else -0.01
    fig.text(
        0.5,
        caption_y,
        "Star marks the best or tied-best method for each seed. Lower cost is better.",
        ha="center",
        fontsize=10,
    )
    fig.suptitle(f"{_regime_label(regime)} Regime", fontsize=15, y=0.98, weight="bold")
    save_figure(fig, output_dir / output_name)


def make_f10_gradual_quantum_vs_classical(
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    make_regime_quantum_vs_classical(
        summary_rows,
        metadata,
        output_dir,
        regime="gradual",
        output_name="F10_gradual_quantum_vs_classical.png",
    )


def make_f11_stationary_quantum_vs_classical(
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    make_regime_quantum_vs_classical(
        summary_rows,
        metadata,
        output_dir,
        regime="stationary",
        output_name="F11_stationary_quantum_vs_classical.png",
    )


def make_f12_continuous_sudden_quantum_vs_classical(
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    make_regime_quantum_vs_classical(
        summary_rows,
        metadata,
        output_dir,
        regime="continuous_sudden",
        output_name="F12_continuous_sudden_quantum_vs_classical.png",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_question_dataset(args.input_prefix)
    metadata = dataset["metadata"]

    make_f1_main_result(dataset["summary_rows"], metadata, output_dir)
    make_f2_factorial_heatmaps(dataset["summary_rows"], metadata, output_dir)
    make_f3_transfer_gain(dataset["transfer_rows"], metadata, output_dir)
    make_f4_adaptation_traces(dataset["trace_rows"], metadata, output_dir)
    make_f5_allocation_timelines(dataset["allocation_rows"], dataset["step_rows"], metadata, output_dir)
    make_f6_tradeoff(dataset["tradeoff_rows"], metadata, output_dir)
    make_f7_network_animation(
        dataset["position_rows"],
        dataset["edge_rows"],
        dataset["allocation_rows"],
        dataset["step_rows"],
        metadata,
        output_dir,
        args.fps,
    )
    make_f8_optimization_race(dataset["trace_rows"], metadata, output_dir, args.fps)
    make_f9_combined_vs_state_seed_gaps(dataset["summary_rows"], metadata, output_dir)
    make_f10_gradual_quantum_vs_classical(dataset["summary_rows"], metadata, output_dir)
    make_f11_stationary_quantum_vs_classical(dataset["summary_rows"], metadata, output_dir)
    make_f12_continuous_sudden_quantum_vs_classical(dataset["summary_rows"], metadata, output_dir)

    for path in sorted(output_dir.iterdir()):
        if path.suffix.lower() in {".png", ".gif"}:
            print(path.name)


if __name__ == "__main__":
    main()
