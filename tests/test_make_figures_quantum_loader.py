from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from make_figures import load_quantum_rollout_dataset


def test_quantum_rollout_loader_reads_smoke_exports() -> None:
    metadata, methods, traces = load_quantum_rollout_dataset("quantum_suite_smoke")

    assert metadata["regime"] == "gradual"
    assert metadata["time_steps"] == 5
    assert set(methods.keys()) == {"Cold", "Param", "State", "Combined"}
    assert len(methods["Combined"].allocations) == 5
    assert len(methods["Combined"].cumulative_costs) == 5
    assert 1 in traces["Cold"]
    assert len(traces["Cold"][1]["evaluation"]) >= 1
