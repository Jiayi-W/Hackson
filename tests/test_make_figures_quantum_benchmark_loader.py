from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from make_figures import load_quantum_benchmark_dataset


def test_quantum_benchmark_loader_reads_smoke_exports() -> None:
    metadata, tradeoff_rows, transfer_rows = load_quantum_benchmark_dataset("quantum_benchmarks_smoke")

    assert metadata["time_steps"] == 5
    assert metadata["methods"] == ["Cold", "Combined"]
    assert len(tradeoff_rows) == 8
    assert len(transfer_rows) == 12
    assert {row["method"] for row in tradeoff_rows} == {"Cold", "Combined"}
    assert {row["regime"] for row in transfer_rows} == {"stationary", "gradual", "sudden"}
