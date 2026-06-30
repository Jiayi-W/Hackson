# Quantum Continuous-Sudden Figure Guide

This folder contains the four-regime shot-based Qiskit figure suite with the
extra `continuous_sudden` dynamic case.

## Run Setting

- Network size: `5 users`, `3 channels`, `12 snapshots`
- Dynamic regimes: `stationary`, `gradual`, `sudden`, `continuous_sudden`
- Dynamic strategies: `Cold`, `Parameter Transfer`, `State Warm Start`,
  `Combined`
- QAOA budget: `6 COBYLA evaluations` per snapshot
- Sampling budget: `64 optimization shots`, `256 final shots`
- Seeds: `101`, `201`, `304`, `401`

## Commands

Generate raw quantum tables:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/run_quantum_question_study.py \
  --include-continuous-sudden
```

Render the eight figures:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/make_question_figures.py \
  --input-prefix quantum_question_study_continuous_smoke \
  --output-dir artifacts/question_figures_quantum_continuous_smoke
```

## Data Source

- Raw CSV/JSON: `artifacts/raw_results/quantum_question_study_continuous_smoke_*`
- Renderer: `scripts/make_question_figures.py`
