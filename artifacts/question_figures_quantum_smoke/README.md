# Quantum Question Figure Guide

This folder contains the eight question-study figures generated from the
shot-based Qiskit pipeline rather than the surrogate scaffold.

## Smoke Run Setting

- Network size: `5 users`, `3 channels`, `12 snapshots`
- Dynamic regimes: `stationary`, `gradual`, `sudden`
- Dynamic strategies: `Cold`, `Parameter Transfer`, `State Warm Start`,
  `Combined`
- QAOA budget: `6 COBYLA evaluations` per snapshot
- Sampling budget: `64 optimization shots`, `256 final shots`
- Seeds: `stationary=101`, `gradual=201`, `sudden=304`

## Commands

Generate raw quantum tables:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/run_quantum_question_study.py
```

Render the eight figures:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/make_question_figures.py \
  --input-prefix quantum_question_study_smoke \
  --output-dir artifacts/question_figures_quantum_smoke
```

## Data Source

- Raw CSV/JSON: `artifacts/raw_results/quantum_question_study_smoke_*`
- Renderer: `scripts/make_question_figures.py`
- Study driver: `scripts/run_quantum_question_study.py`

## Full-Budget Remote Command

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/run_quantum_question_study.py \
  --full-budget
```
