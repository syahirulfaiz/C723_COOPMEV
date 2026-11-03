#!/usr/bin/env bash
# COSC723 CoopMEV-Lite: force a full from-scratch plan regeneration (92 slots).
#
#   bash regenerate_plans.sh scratch   # RECOMMENDED: build a separate campaign under
#                                      # results_scratch/ (the frozen results/ is never touched)
#   bash regenerate_plans.sh inplace   # archive the previous campaign under
#                                      # results/archive/<stamp>/ and rebuild results/ from scratch
#
# Requires: export OPENROUTER_API_KEY='...' (optionally OPENROUTER_MODEL=...).
# Knobs: PLAN_MAX_REQUESTS (300), PLAN_ATTEMPTS (3), PLAN_BACKOFF_SECONDS (2), PLAN_STALLED_PASSES (3).
set -euo pipefail
MODE="${1:-scratch}"
ROOT="${2:-.}"
cd "$ROOT"
PY="${PY:-python}"
# driver.py needs Python >= 3.10 (from __future__ import annotations + typing syntax).
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "note: '${PY}' is older than Python 3.10; using ${CONDA_PREFIX}/bin/python" >&2
    PY="${CONDA_PREFIX}/bin/python"
  else
    echo "error: driver.py needs Python >= 3.10; activate the conda environment or export PY=/path/to/python" >&2
    exit 1
  fi
fi
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-openai/gpt-oss-120b}"
export OPENROUTER_MAX_COMPLETION_TOKENS="${OPENROUTER_MAX_COMPLETION_TOKENS:-512}"
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-low}"

PLAN_FLAGS=(
  --plan-fill-complete
  --plan-max-requests "${PLAN_MAX_REQUESTS:-300}"
  --plan-attempts "${PLAN_ATTEMPTS:-3}"
  --plan-backoff-seconds "${PLAN_BACKOFF_SECONDS:-2}"
  --plan-stalled-passes "${PLAN_STALLED_PASSES:-3}"
)

case "$MODE" in
  scratch)
    OUT="results_scratch"
    echo "[regenerate] mode=scratch: fresh campaign in ${OUT}/ (frozen results/ is untouched)"
    "$PY" driver.py --plans-only --dataset-dir DATASET --model best_model.zip --output-dir "$OUT" \
      --plan-budget 92 "${PLAN_FLAGS[@]}"
    echo "[regenerate] plans ready; running the full evaluation into ${OUT}/"
    "$PY" driver.py --dataset-dir DATASET --model best_model.zip --output-dir "$OUT"
    "$PY" chart.py --input "$OUT/raw_results.csv" --output-dir "$OUT/figures"
    echo "[regenerate] done: ${OUT}/raw_results.csv and ${OUT}/figures/COSC723_*.png"
    ;;
  inplace)
    STAMP="$(date +%Y%m%dT%H%M%S)"
    ARCHIVE="results/archive/${STAMP}"
    mkdir -p "$ARCHIVE"
    for entry in plans prompts figures raw_results.csv transactions.csv config.json \
                 freeze_manifest.json scenarios.csv summary_by_cell.csv overall_by_condition.csv \
                 checker_selftest.csv week4_rival_first_case.json plan_campaign_summary.json \
                 plan_usage.csv; do
      [[ -e "results/${entry}" ]] && mv "results/${entry}" "${ARCHIVE}/${entry}"
    done
    echo "[regenerate] mode=inplace: previous campaign archived under ${ARCHIVE}/"
    "$PY" driver.py --plans-only --dataset-dir DATASET --model best_model.zip --output-dir results \
      --plan-budget 92 "${PLAN_FLAGS[@]}"
    echo "[regenerate] plans rebuilt; re-running the full evaluation into results/"
    "$PY" driver.py --dataset-dir DATASET --model best_model.zip --output-dir results
    "$PY" chart.py --input results/raw_results.csv --output-dir results/figures
    echo "[regenerate] done: results/ rebuilt from scratch (archive kept at ${ARCHIVE}/)"
    ;;
  *)
    echo "usage: bash regenerate_plans.sh [scratch|inplace] [project-root]" >&2
    exit 2
    ;;
esac
