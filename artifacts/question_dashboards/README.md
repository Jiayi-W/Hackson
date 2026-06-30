# Linked Regime Dashboards

This directory contains three synchronized dashboard GIFs:

- `stationary_dashboard.gif`
- `gradual_dashboard.gif`
- `sudden_dashboard.gif`

Each GIF uses the same layout:

- Left: moving user positions and weighted interference graph
- Right top: cumulative total cost for `Cold`, `Param`, `State`, `Combined`
- Right bottom: current-snapshot step-cost comparison plus `Delta_t`,
  `Gain(Combined vs Cold)`, and the current best strategy

Rendering command:

```bash
python scripts/make_question_dashboards.py
```

Data source:

- `artifacts/raw_results/question_study_*`

Visual convention:

- Node color encodes the `Combined` allocation channel
- Edge thickness/opacity encodes interference weight
- Red jump highlighting appears only in the `sudden` dashboard
