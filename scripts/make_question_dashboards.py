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

from respec.question_study import QUESTION_METHODS
from respec.visualization import CHANNEL_COLORS, METHOD_COLORS, REGIME_COLORS, apply_style, polish_axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render linked regime dashboard GIFs for ReSpec-QAOA.")
    parser.add_argument("--input-prefix", default="question_study")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "question_dashboards"))
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
    }


def _regime_label(regime: str) -> str:
    return regime.replace("_", " ").title()


def _study_regimes(metadata: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(regime) for regime in metadata.get("regimes", ("stationary", "gradual", "sudden")))


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
    candidates = [
        row
        for row in step_rows
        if row["regime"] == regime and row["seed"] == seed and row["method"] == "Cold" and int(row["snapshot_t"]) > 0
    ]
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


def _regime_step_rows(step_rows: list[dict[str, object]], regime: str, seed: int) -> dict[str, list[dict[str, object]]]:
    data: dict[str, list[dict[str, object]]] = {method: [] for method in QUESTION_METHODS}
    for row in step_rows:
        if row["regime"] == regime and row["seed"] == seed and row["method"] in data:
            data[str(row["method"])].append(row)
    for method in data:
        data[method].sort(key=lambda row: int(row["snapshot_t"]))
    return data


def _plot_network_panel(
    ax,
    *,
    regime: str,
    seed: int,
    snapshot_t: int,
    n_users: int,
    time_steps: int,
    allocation_rows: list[dict[str, object]],
    position_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    peak_snapshot: int | None,
) -> None:
    ax.clear()
    positions = _positions(position_rows, regime, seed, snapshot_t, n_users)
    weights = _edge_matrix(edge_rows, regime, seed, snapshot_t, n_users)
    combined_allocations = _allocation_matrix(
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
                alpha=0.14 + 0.75 * weights[u, v],
                linewidth=1.0 + 3.1 * weights[u, v],
                zorder=1,
            )

    moved_user = None
    if peak_snapshot is not None and snapshot_t == peak_snapshot and snapshot_t > 0:
        previous_positions = _positions(position_rows, regime, seed, snapshot_t - 1, n_users)
        displacement = np.linalg.norm(positions - previous_positions, axis=1)
        moved_user = int(np.argmax(displacement))

    for user, (x, y) in enumerate(positions):
        halo = REGIME_COLORS[regime] if moved_user == user else "#FCFBF8"
        ax.scatter(x, y, s=365, color=halo, edgecolor="none", zorder=2)
        ax.scatter(
            x,
            y,
            s=220,
            color=CHANNEL_COLORS[int(combined_allocations[user])],
            edgecolor="#1F1F1F",
            linewidth=1.0,
            zorder=3,
        )
        ax.text(x, y, f"U{user}", ha="center", va="center", color="white", weight="bold", fontsize=9)

    if moved_user is not None:
        marker = "jump" if regime == "sudden" else "peak drift"
        ax.text(
            positions[moved_user, 0] + 0.03,
            positions[moved_user, 1] + 0.05,
            marker,
            color=REGIME_COLORS[regime],
            fontsize=9,
            weight="bold",
        )

    ax.set_title(f"User positions and interference graph\n{_regime_label(regime)} | Combined allocation | t={snapshot_t}")
    ax.set_xlim(0.03, 0.97)
    ax.set_ylim(0.03, 0.97)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_cumulative_panel(ax, *, frame: int, regime_rows: dict[str, list[dict[str, object]]], y_max: float) -> None:
    ax.clear()
    for method in QUESTION_METHODS:
        rows = regime_rows[method][: frame + 1]
        xs = [int(row["snapshot_t"]) for row in rows]
        ys = [float(row["cumulative_cost"]) for row in rows]
        ax.plot(
            xs,
            ys,
            color=METHOD_COLORS[method],
            linewidth=2.3 if method == "Combined" else 1.9,
            marker="o",
            markersize=4.5,
            label=method,
        )
        if rows:
            ax.scatter(xs[-1], ys[-1], s=54, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.7, zorder=4)
    ax.set_title("Cumulative total cost")
    ax.set_xlabel("Snapshot t")
    ax.set_ylabel("Cumulative cost")
    ax.set_xlim(-0.2, max(len(regime_rows["Cold"]) - 0.8, 0.8))
    ax.set_ylim(0.0, y_max)
    polish_axes(ax)
    ax.legend(ncol=2, fontsize=8, loc="upper right")


def _plot_current_panel(
    ax,
    *,
    frame: int,
    regime: str,
    regime_rows: dict[str, list[dict[str, object]]],
    peak_snapshot: int | None,
    step_cost_ymax: float,
) -> None:
    ax.clear()
    methods = list(QUESTION_METHODS)
    current_rows = [regime_rows[method][frame] for method in methods]
    values = np.array([float(row["step_cost"]) for row in current_rows], dtype=float)
    colors = [METHOD_COLORS[method] for method in methods]
    positions = np.arange(len(methods))
    bars = ax.bar(positions, values, color=colors, width=0.65, alpha=0.88)

    for bar, row in zip(bars, current_rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02 * step_cost_ymax,
            f"{float(row['step_cost']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

    cold_cost = float(regime_rows["Cold"][frame]["step_cost"])
    combined_cost = float(regime_rows["Combined"][frame]["step_cost"])
    gain = cold_cost - combined_cost
    delta = float(regime_rows["Cold"][frame]["delta"])
    winner = min(current_rows, key=lambda row: float(row["step_cost"]))["method"]
    if regime == "continuous_sudden":
        tag = "peak drift" if peak_snapshot is not None and frame == peak_snapshot else "high drift"
    elif peak_snapshot is not None and frame == peak_snapshot:
        tag = "jump"
    else:
        tag = "steady"

    info_lines = [
        f"Delta_t = {delta:.3f}",
        f"Gain(Combined vs Cold) = {gain:+.3f}",
        f"Best strategy = {winner}",
        f"Event tag = {tag}",
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#FCFBF8", "edgecolor": "#D9D3C7"},
    )

    ax.set_title("Current snapshot cost comparison")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Cold", "Param", "State", "Combined"], rotation=14)
    ax.set_ylabel("Step cost J_t")
    ax.set_ylim(0.0, step_cost_ymax)
    if regime in {"sudden", "continuous_sudden"} and peak_snapshot is not None:
        ax.axhline(combined_cost, color=REGIME_COLORS[regime], linestyle=":", linewidth=1.1, alpha=0.75)
    polish_axes(ax)


def render_regime_dashboard(
    *,
    regime: str,
    metadata: dict[str, object],
    step_rows: list[dict[str, object]],
    allocation_rows: list[dict[str, object]],
    position_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    output_dir: Path,
    fps: int,
) -> None:
    apply_style()
    seed = _representative_seed(metadata, regime)
    n_users = int(metadata["n_users"])
    time_steps = int(metadata["time_steps"])
    peak_snapshot = _peak_change_snapshot(step_rows, regime, seed)
    regime_rows = _regime_step_rows(step_rows, regime, seed)

    cumulative_values = [
        float(row["cumulative_cost"])
        for method in QUESTION_METHODS
        for row in regime_rows[method]
    ]
    step_values = [
        float(row["step_cost"])
        for method in QUESTION_METHODS
        for row in regime_rows[method]
    ]
    cumulative_ymax = max(cumulative_values) * 1.08
    step_cost_ymax = max(step_values) * 1.25

    fig = plt.figure(figsize=(13.4, 6.9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.0, 0.78],
        wspace=0.22,
        hspace=0.26,
    )
    ax_network = fig.add_subplot(grid[:, 0])
    ax_cumulative = fig.add_subplot(grid[0, 1])
    ax_current = fig.add_subplot(grid[1, 1])

    def _update(frame: int) -> None:
        _plot_network_panel(
            ax_network,
            regime=regime,
            seed=seed,
            snapshot_t=frame,
            n_users=n_users,
            time_steps=time_steps,
            allocation_rows=allocation_rows,
            position_rows=position_rows,
            edge_rows=edge_rows,
            peak_snapshot=peak_snapshot,
        )
        _plot_cumulative_panel(
            ax_cumulative,
            frame=frame,
            regime_rows=regime_rows,
            y_max=cumulative_ymax,
        )
        _plot_current_panel(
            ax_current,
            frame=frame,
            regime=regime,
            regime_rows=regime_rows,
            peak_snapshot=peak_snapshot,
            step_cost_ymax=step_cost_ymax,
        )
        fig.suptitle(
            f"{_regime_label(regime)} dynamics dashboard | linked motion + data view",
            fontsize=16,
            y=0.98,
            weight="bold",
            color=REGIME_COLORS[regime],
        )

    animation = FuncAnimation(fig, _update, frames=range(time_steps), interval=max(220, 1000 // max(fps, 1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{regime}_dashboard.gif"
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_question_dataset(args.input_prefix)
    output_dir = Path(args.output_dir).expanduser()

    for regime in _study_regimes(dataset["metadata"]):
        render_regime_dashboard(
            regime=regime,
            metadata=dataset["metadata"],
            step_rows=dataset["step_rows"],
            allocation_rows=dataset["allocation_rows"],
            position_rows=dataset["position_rows"],
            edge_rows=dataset["edge_rows"],
            output_dir=output_dir,
            fps=args.fps,
        )

    for path in sorted(output_dir.glob("*.gif")):
        print(path.name)


if __name__ == "__main__":
    main()
