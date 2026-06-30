# ReSpec-QAOA

Shot-based Qiskit study and figure pipeline for the **ReSpec-QAOA** hackathon
project: warm-started feasible-subspace QAOA for dynamic wireless spectrum
allocation.

The repository now contains both:

- plan-aligned scaffold code for the broader project, and
- a working research-question pipeline with exported datasets, static figures,
  and animated dashboards.

## Current Experiment Setup

The main study in this repo uses:

- `5` users
- `3` channels
- `12` time steps
- switching weight `lambda_switch = 0.30`
- four quantum strategies:
  - `Cold`
  - `Param` (`Parameter Transfer`)
  - `State` (`State Warm Start`)
  - `Combined`

The current dynamic regimes are:

- `stationary`
- `gradual`
- `sudden`
  - one large jump event at `t = 5`
- `continuous_sudden`
  - every step includes a large jump by one rotating user, plus global drift

## Main Outputs

The most up-to-date shot-based question-study artifacts are:

- raw tables:
  - `artifacts/raw_results/quantum_question_study_continuous_full_*.csv`
  - `artifacts/raw_results/quantum_question_study_continuous_full_metadata.json`
- full static figure set:
  - `artifacts/question_figures_quantum_continuous_full/`
- linked animated dashboards:
  - `artifacts/question_dashboards_quantum_continuous_full/`

The regime-extension runs used to expand the seed count are also included under
`artifacts/raw_results/`, for example:

- `stationary_extension_*`
- `gradual_extension_*`
- `continuous_sudden_extension_*`

## Reproducing the Current Full Study

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/run_quantum_question_study.py \
  --include-continuous-sudden \
  --full-budget

python scripts/make_question_figures.py \
  --input-prefix quantum_question_study_continuous_full \
  --output-dir artifacts/question_figures_quantum_continuous_full

python scripts/make_question_dashboards.py \
  --input-prefix quantum_question_study_continuous_full \
  --output-dir artifacts/question_dashboards_quantum_continuous_full
```

## Figure Inventory

The question-study figure pipeline currently renders:

- `F1_main_result_by_regime.png`
- `F2_factorial_reuse_heatmaps.png`
- `F3_transfer_gain_vs_change.png`
- `F4_trace_adaptation.png`
- `F5_allocation_trajectories.png`
- `F6_tradeoff_scan.png`
- `F7_user_motion_and_heatmap.gif`
- `F8_optimization_race.gif`
- `F9_combined_vs_state_seed_gaps.png`
- `F10_gradual_quantum_vs_classical.png`
- `F11_stationary_quantum_vs_classical.png`
- `F12_continuous_sudden_quantum_vs_classical.png`

The dashboard pipeline renders synchronized regime GIFs where user-position
motion is shown on the left and quantitative panels evolve on the right.

## Script Guide

- `scripts/run_quantum_question_study.py`
  - exports the shot-based Qiskit regime-study tables
- `scripts/make_question_figures.py`
  - renders the static figures and two GIFs from exported study tables
- `scripts/make_question_dashboards.py`
  - renders side-by-side animated dashboards for selected regimes
- `scripts/watch_quantum_question_study_continuous_full.sh`
  - helper for monitoring a longer remote full-budget run

Legacy scaffold and plan-figure scripts are still present:

- `scripts/run_core.py`
- `scripts/run_ablations.py`
- `scripts/run_quantum_rollout.py`
- `scripts/run_quantum_benchmarks.py`
- `scripts/make_figures.py`
- `scripts/run_question_study.py`

## Data Interpretation Notes

- `question_figures_quantum_continuous_full/` and
  `question_dashboards_quantum_continuous_full/` are based on real shot-based
  Qiskit exports from the current study pipeline.
- Some older directories such as `artifacts/figures/`,
  `artifacts/question_figures/`, and smoke-check outputs remain useful for
  prototyping and presentation iteration, but they should not be confused with
  the latest full-budget `continuous_full` study.

## Repository Layout

- `src/respec/`
  - environment generation, objectives, exact solvers, heuristics, warm-start
    strategies, Qiskit-backed question-study logic, and plotting utilities
- `scripts/`
  - experiment runners and figure/dashboard rendering entrypoints
- `artifacts/raw_results/`
  - exported CSV/JSON tables
- `artifacts/question_figures_quantum_continuous_full/`
  - latest static figures and GIFs for the main study
- `artifacts/question_dashboards_quantum_continuous_full/`
  - latest synchronized dashboard GIFs
- `tests/`
  - reproducibility and question-study checks
