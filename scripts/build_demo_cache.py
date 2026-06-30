from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respec.cache import make_cache_key, write_json


def main() -> None:
    payload = {
        "sequence_hash": "demo-seq-v1",
        "snapshot_id": 5,
        "user_position": [0.52, 0.48],
        "method": "Combined",
        "lambda_switch": 0.30,
        "p": 2,
        "budget": 24,
        "shots": 256,
        "noise": "ideal",
        "optimizer_seed": 11,
        "transpiler_seed": 11,
        "git_commit": "local-uncommitted",
    }
    key = make_cache_key(payload)
    write_json(ROOT / "artifacts" / "cache" / f"{key}.json", payload)
    print(f"wrote cache stub {key}")


if __name__ == "__main__":
    main()

