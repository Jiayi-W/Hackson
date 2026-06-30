# Quantum Smoke Figure Set

This directory stores a quantum-backed figure variant generated with:

```bash
python scripts/make_figures.py \
  --rollout-source quantum \
  --quantum-prefix quantum_suite_smoke \
  --quantum-benchmark-prefix quantum_benchmarks_smoke \
  --output-dir artifacts/figures_quantum_smoke
```

Figure provenance:

- `F1` uses the actual dynamic sequence and `Combined` allocations from
  `quantum_suite_smoke`.
- `F2`, `F3`, and `F6` use the real Qiskit rollout exports from
  `artifacts/raw_results/quantum_suite_smoke_*`.
- `F4` and `F5` use the real Qiskit benchmark exports from
  `artifacts/raw_results/quantum_benchmarks_smoke_*`.
- `F7` and `F8` remain illustrative metric curves from the current
  plan/demo scaffold and are not yet backed by full hardware/noise sweeps.
