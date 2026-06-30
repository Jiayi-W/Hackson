#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="cf_jiayi"
REMOTE_ROOT="~/GitHub/respec-qaoa-full"
LOCAL_ROOT="/Users/jiayiwu/respec-qaoa"
PREFIX="quantum_question_study_continuous_full"
REMOTE_PATTERN="python scripts/run_quantum_question_study.py --include-continuous-sudden --full-budget --time-steps 12 --n-jobs 14"

LOCAL_FIG_DIR="$LOCAL_ROOT/artifacts/question_figures_quantum_continuous_full"
LOCAL_DASH_DIR="$LOCAL_ROOT/artifacts/question_dashboards_quantum_continuous_full"
LOCAL_RAW_DIR="$LOCAL_ROOT/artifacts/raw_results"

render_local() {
  "$LOCAL_ROOT/.venv/bin/python" "$LOCAL_ROOT/scripts/make_question_figures.py" \
    --input-prefix "$PREFIX" \
    --output-dir "$LOCAL_FIG_DIR"
  "$LOCAL_ROOT/.venv/bin/python" "$LOCAL_ROOT/scripts/make_question_dashboards.py" \
    --input-prefix "$PREFIX" \
    --output-dir "$LOCAL_DASH_DIR"
}

while true; do
  if ssh "$REMOTE_HOST" "test -f $REMOTE_ROOT/artifacts/raw_results/${PREFIX}_metadata.json"; then
    mkdir -p "$LOCAL_RAW_DIR"
    rsync -az "$REMOTE_HOST:$REMOTE_ROOT/artifacts/raw_results/${PREFIX}_"* "$LOCAL_RAW_DIR/"
    if [ ! -f "$LOCAL_FIG_DIR/F1_main_result_by_regime.png" ]; then
      render_local
    fi
  fi

  if ! ssh "$REMOTE_HOST" "pgrep -f '$REMOTE_PATTERN' >/dev/null"; then
    if ssh "$REMOTE_HOST" "test -d $REMOTE_ROOT/artifacts/question_figures_quantum_continuous_full"; then
      rsync -az "$REMOTE_HOST:$REMOTE_ROOT/artifacts/question_figures_quantum_continuous_full" "$LOCAL_ROOT/artifacts/"
    fi
    if ssh "$REMOTE_HOST" "test -d $REMOTE_ROOT/artifacts/question_dashboards_quantum_continuous_full"; then
      rsync -az "$REMOTE_HOST:$REMOTE_ROOT/artifacts/question_dashboards_quantum_continuous_full" "$LOCAL_ROOT/artifacts/"
    fi
    if [ -f "$LOCAL_RAW_DIR/${PREFIX}_metadata.json" ] && [ ! -f "$LOCAL_FIG_DIR/F1_main_result_by_regime.png" ]; then
      render_local
    fi
    echo "done $(date '+%Y-%m-%d %H:%M:%S')"
    break
  fi

  echo "waiting $(date '+%Y-%m-%d %H:%M:%S')"
  sleep 60
done
