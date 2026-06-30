# Quantum Continuous-Sudden Dashboards

This folder contains the linked dashboard GIFs for the four-regime
shot-based Qiskit smoke study:

- `stationary_dashboard.gif`
- `gradual_dashboard.gif`
- `sudden_dashboard.gif`
- `continuous_sudden_dashboard.gif`

## Commands

Generate the source tables:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/run_quantum_question_study.py \
  --include-continuous-sudden
```

Render the dashboards:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/make_question_dashboards.py \
  --input-prefix quantum_question_study_continuous_smoke \
  --output-dir artifacts/question_dashboards_quantum_continuous_smoke
```
