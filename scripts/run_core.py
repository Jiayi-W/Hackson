from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.runner import build_demo_rollouts


def main() -> None:
    _, methods = build_demo_rollouts()
    for name, rollout in methods.items():
        print(
            f"{name:12s} final_cost={rollout.cumulative_costs[-1]:.3f} "
            f"interference={rollout.cumulative_interference[-1]:.3f} "
            f"switches={rollout.cumulative_switches[-1]:.3f}"
        )


if __name__ == "__main__":
    main()

