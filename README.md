# ReSpec-QAOA

Plan-aligned scaffold for the **ReSpec-QAOA** hackathon project:
warm-started feasible-subspace QAOA for dynamic spectrum allocation.

This repository currently focuses on two deliverables from the project plan:

1. A clean project scaffold that mirrors the proposed repository layout.
2. A reproducible `make_figures.py` pipeline that generates the eight
   presentation figures listed in Section 7.1 of the PDF plan.
3. Exportable CSV and JSON datasets for rollouts, allocations, transfer-gain
   points, tradeoff sweeps, and other figure-driving artifacts.

## Important Scope Note

The generated figures are **plan-driven illustrative assets** built from a
deterministic communication scenario plus lightweight surrogate method data.
They are useful for project framing, slides, and demo design, but they are
**not a claim that the full QAOA experiment suite has already been executed**.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_core.py
python scripts/run_ablations.py
python scripts/run_quantum_rollout.py
python scripts/run_quantum_benchmarks.py
python scripts/make_figures.py
python scripts/run_question_study.py
python scripts/make_question_figures.py
python scripts/make_question_dashboards.py
```

Generated figures will be written to:

`artifacts/figures/`

Structured experiment exports will be written to:

`artifacts/raw_results/`

Research-question figures and GIFs will be written to:

`artifacts/question_figures/`

Linked regime dashboard GIFs will be written to:

`artifacts/question_dashboards/`

## Figure Set

The script renders the exact eight figures requested by the plan:

- F1. Network evolution
- F2. Allocation timeline heatmap
- F3. Cumulative total cost
- F4. Interference-switching tradeoff
- F5. Transfer gain vs graph change
- F6. Optimization adaptation traces
- F7. Feasibility vs CX budget
- F8. Ideal vs noisy

## Repository Layout

The directory structure follows the PDF plan closely, with working utilities in
`environment.py`, `objective.py`, `exact.py`, `heuristics.py`,
`strategies.py`, `metrics.py`, and `visualization.py`. QAOA-specific modules
that are not implemented yet are scaffolded with explicit placeholders so the
team can fill them in incrementally during the hackathon.

## Current Working Outputs

- `scripts/run_core.py` exports the main sudden-sequence rollout datasets.
- `scripts/run_ablations.py` exports the figure-driving sweep datasets.
- `scripts/run_quantum_rollout.py` runs a real Qiskit-backed smoke rollout by
  default, with `--full-budget` available for a heavier plan-style run.
- `scripts/run_quantum_benchmarks.py` builds real-QAOA benchmark exports for
  F4/F5 under a smoke budget by default, with `--full-budget` available when
  you want a heavier run.
- `scripts/run_question_study.py` exports the regime-study tables used to
  answer the warm-start research questions across stationary, gradual, and
  sudden dynamics.
- `scripts/make_question_figures.py` renders a question-driven figure set with
  six static figures and two GIF animations under `artifacts/question_figures/`.
- `scripts/make_question_dashboards.py` renders three linked dashboard GIFs
  for `stationary`, `gradual`, and `sudden`, with user-position motion on the
  left and synchronized data panels on the right.
- `scripts/make_figures.py` renders the eight final figures and can switch
  F2/F3/F6 to real QAOA rollout exports with
  `--rollout-source quantum --quantum-prefix <prefix>`, and can switch F4/F5
  to real QAOA benchmark exports with
  `--quantum-benchmark-prefix <prefix>`. Use `--output-dir <path>` if you want
  to render a separate quantum-backed figure directory without overwriting the
  baseline illustrative set.
