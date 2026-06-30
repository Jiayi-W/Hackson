from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.metrics import build_tradeoff_table


def main() -> None:
    table = build_tradeoff_table([0.00, 0.15, 0.30, 0.60])
    for method, points in table.items():
        print(method)
        for lambda_value, (switches, interference) in points.items():
            print(f"  lambda={lambda_value} switches={switches:.3f} interference={interference:.3f}")


if __name__ == "__main__":
    main()

