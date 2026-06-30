# Quantum Linked Dashboards

This folder contains the three linked dashboard GIFs rendered from the same
shot-based Qiskit smoke study:

- `stationary_dashboard.gif`
- `gradual_dashboard.gif`
- `sudden_dashboard.gif`

Each dashboard synchronizes:

- Left: user motion and weighted interference graph
- Right top: cumulative total cost
- Right bottom: current-snapshot step cost and `Delta_t`

## Commands

Generate the source tables:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/run_quantum_question_study.py
```

Render the three dashboards:

```bash
/Users/jiayiwu/respec-qaoa/.venv/bin/python scripts/make_question_dashboards.py \
  --input-prefix quantum_question_study_smoke \
  --output-dir artifacts/question_dashboards_quantum_smoke
```

## Data Source

- Raw CSV/JSON: `artifacts/raw_results/quantum_question_study_smoke_*`
- Renderer: `scripts/make_question_dashboards.py`
