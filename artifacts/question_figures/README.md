# Research-Question Figure Guide

This folder contains the question-driven figure suite for the two core project
questions:

1. When the wireless network changes over time, how much can warm-start
   strategies improve feasible-subspace QAOA?
2. Under what type of network dynamics do warm-start strategies actually help?

## Experiment Setting

- Network size: `5 users`, `3 channels`, `10 snapshots`
- Dynamic regimes: `stationary`, `gradual`, `sudden`
- Dynamic strategies: `Cold`, `Parameter Transfer`, `State Warm Start`,
  `Combined`
- Objective: normalized interference plus switching cost
- Main comparison budget: fixed dynamic-QAOA-style budget shared across methods

Strategy meanings:

- `Cold`: fresh feasible state + fresh parameters
- `Parameter Transfer`: fresh state + previous parameters
- `State Warm Start`: previous allocation state + fresh parameters
- `Combined`: previous allocation state + previous parameters

## Important Note

These figures are generated from the current `question_study` pipeline:

- data export: `scripts/run_question_study.py`
- rendering: `scripts/make_question_figures.py`
- raw tables: `artifacts/raw_results/question_study_*`

The figure logic follows the final project plan and uses the same four dynamic
strategies and regime split, but the current `question_study` pipeline is still
based on the repository's surrogate warm-start study scaffold rather than a
full Qiskit optimization run for every sequence. The same figure schema can be
reused later with quantum-backed tables.

## Figure-by-Figure Explanation

### [F1_main_result_by_regime.png](F1_main_result_by_regime.png)

What it shows:

- Three panels for `stationary`, `gradual`, and `sudden`
- Y-axis is `cumulative gap above offline DP`
- Lower is better

How to read it:

- `Offline DP` is the clairvoyant lower bound, so a smaller gap means the
  strategy tracks the sequence optimum more closely.
- Each box summarizes multiple sequences within one regime.

Main message:

- `Combined` is the strongest strategy in all three regimes.
- In the current run, median gain of `Combined` over `Cold` is about
  `+1.620` in stationary, `+1.322` in gradual, and `+1.279` in sudden.
- Pure `State` and pure `Param` help, but they are clearly weaker than using
  both together.

### [F2_factorial_reuse_heatmaps.png](F2_factorial_reuse_heatmaps.png)

What it shows:

- A `2 x 2` reuse decomposition for each regime
- Horizontal axis: parameter reuse on/off
- Vertical axis: state reuse on/off
- Cell value: median improvement versus `Cold`

How to read it:

- The bottom-right cell is `Combined`.
- The top-right cell is `Parameter Transfer`.
- The bottom-left cell is `State Warm Start`.

Main message:

- Both reuse channels matter.
- In `gradual` and `sudden`, `State Warm Start` is slightly stronger than
  `Parameter Transfer` alone in the current run.
- The largest jump appears when both are used together, which supports the
  interaction effect in the original project design.

### [F3_transfer_gain_vs_graph_change.png](F3_transfer_gain_vs_graph_change.png)

What it shows:

- X-axis is graph-change magnitude `Delta_t`
- Y-axis is step gain versus `Cold`
- Positive values mean the warm-start strategy beats `Cold` on that snapshot

How to read it:

- Each panel focuses on one warm-start strategy:
  `Parameter Transfer`, `State Warm Start`, or `Combined`.
- Colors separate `stationary`, `gradual`, and `sudden` points.
- The line is a coarse binned trend, not a fitted model.

Main message:

- `Combined` keeps a much stronger positive gain band than the other two.
- Pure `Param` and pure `State` are more fragile: their gains shrink and can
  even become negative as `Delta_t` grows.
- This is the clearest evidence that warm-start benefit is regime-dependent,
  not universal.

### [F4_adaptation_traces.png](F4_adaptation_traces.png)

What it shows:

- Best-so-far cost versus evaluation count under the same optimization budget
- Left panel: representative `gradual` change
- Right panel: representative `sudden` change

How to read it:

- Faster downward movement means the method adapts more quickly under the same
  budget.
- All methods are compared under the same evaluation count.

Main message:

- In the `gradual` panel, `Combined` starts with the strongest inductive bias
  and reaches low cost fastest.
- In the `sudden` panel, warm-start still helps, but the separation between
  methods is smaller and pure transfer can be less stable.
- This figure explains why some methods win in final quality: they also adapt
  faster.

### [F5_allocation_timeline_heatmaps.png](F5_allocation_timeline_heatmaps.png)

What it shows:

- Two representative sequences: one `gradual`, one `sudden`
- For each sequence, compare `Cold` and `Combined`
- Rows are users, columns are time steps, colors are channels

How to read it:

- Stable horizontal bands mean the allocation trajectory is temporally smooth.
- Frequent color changes mean the strategy is reconfiguring more often.
- The red dashed line marks the strongest sudden-change point.

Main message:

- `Combined` produces much smoother allocation trajectories.
- `Cold` switches channels more aggressively, especially around hard changes.
- This gives an intuitive explanation for the reduced switching cost seen in
  the quantitative plots.

### [F6_interference_switching_tradeoff.png](F6_interference_switching_tradeoff.png)

What it shows:

- X-axis: cumulative switches
- Y-axis: cumulative interference
- Curves scan `lambda_sw in {0.00, 0.15, 0.30, 0.60}`
- Separate panels for `gradual` and `sudden`

How to read it:

- Moving left means fewer reconfigurations.
- Moving down means less interference.
- If two lambda values land on the same point, they produced the same median
  trajectory outcome in the current scan.

Main message:

- `Combined` tends to stay in the low-switch region while maintaining low or
  moderate interference.
- `Cold` explores a wider switch range and can pay much more switching cost
  before improving interference.
- This figure shows that warm-start is not only about final cost; it also
  changes the interference-switching tradeoff surface.

### [F7_network_evolution.gif](F7_network_evolution.gif)

What it shows:

- A GIF of a representative `sudden` sequence
- The graph changes over time, edge thickness shows interference weight
- Node color shows the `Combined` allocation
- The overlay reports `Delta_t` and cumulative cost

How to use it:

- Use this GIF when introducing the problem setup.
- It is especially useful for explaining what the sudden regime means in a
  visually concrete way.

Main message:

- Transfer does not happen in an abstract state space only; it reacts to actual
  topology drift.
- The GIF makes the jump event and the need for adaptation immediately visible.

### [F8_optimization_race.gif](F8_optimization_race.gif)

What it shows:

- A side-by-side optimization race for representative `gradual` and `sudden`
  changes
- Curves are revealed frame by frame as evaluation budget is spent

How to use it:

- Use this GIF in a demo or talk when explaining fixed-budget adaptation.
- It is a dynamic companion to `F4`.

Main message:

- Under the same budget, `Combined` builds advantage earlier.
- In abrupt changes, the gain is still present but the gap narrows, which is
  consistent with the regime-dependence story.

## Takeaways for the Two Research Questions

Question 1: how much can warm-start help?

- In the current study, warm-start helps substantially.
- The strongest and most consistent improvement comes from `Combined`.
- Using only one reuse channel helps, but leaves a large part of the benefit
  on the table.

Question 2: under what dynamics does it help?

- Warm-start helps most when graph changes are small to moderate.
- As `Delta_t` increases, pure parameter transfer and pure state reuse become
  more brittle.
- `Combined` remains the most robust option, but the sudden regime still
  compresses the benefit, which matches the original hypothesis that abrupt
  changes may require partial or full reset.
