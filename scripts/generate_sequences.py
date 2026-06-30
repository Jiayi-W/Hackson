from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from respec.environment import generate_sequence


def main() -> None:
    out_dir = ROOT / "artifacts" / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    for regime, seed in [("stationary", 3), ("gradual", 11), ("sudden", 19)]:
        sequence = generate_sequence(5, 10, regime=regime, seed=seed)
        positions = np.stack([snapshot.positions for snapshot in sequence.snapshots], axis=0)
        weights = np.stack([snapshot.weights for snapshot in sequence.snapshots], axis=0)
        np.savez(out_dir / f"{regime}_seed_{seed}.npz", positions=positions, weights=weights)
        print(f"saved {regime} seed={seed}")


if __name__ == "__main__":
    main()

