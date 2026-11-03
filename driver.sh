#!/usr/bin/env bash
# COSC723 CoopMEV-Lite: one-command pipeline (checker self-test -> offline plans -> evaluation -> figures).
#
#   export OPENROUTER_API_KEY='YOUR_OWN_OPENROUTER_KEY'
#   export OPENROUTER_MODEL='openai/gpt-oss-120b'
#   export OPENROUTER_MAX_COMPLETION_TOKENS='512'
#   export OPENROUTER_REASONING_EFFORT='low'
#   bash driver.sh                 # run in the current directory
#   bash driver.sh /path/to/proj   # or point at the project root
#
# Development smoke without an API key: python driver.py --conditions PPO_no_plan \
#   --seeds 40 --only-dimensions gas --max-scenarios 1 --allow-dev-plans
#
# Planning knobs (env-var overridable, forwarded by sbatch --export=ALL):
#   PLAN_MAX_REQUESTS=300   hard request ceiling in fill-complete mode (92 plan slots + retry slack)
#   PLAN_ATTEMPTS=3         attempts per agent/setting slot in one pass
#   PLAN_BACKOFF_SECONDS=2  retry backoff base in seconds
#   PLAN_STALLED_PASSES=3   stop only after this many consecutive passes that fill no slot
#   PLAN_USAGE_RESET=1      archive plan_usage.csv and count this campaign's requests from zero
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
PY="${PY:-python}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "warning: OPENROUTER_API_KEY is unset; cached plans are reused and missing plans stop the run." >&2
fi
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-openai/gpt-oss-120b}"
export OPENROUTER_MAX_COMPLETION_TOKENS="${OPENROUTER_MAX_COMPLETION_TOKENS:-512}"
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-low}"

"$PY" driver.py --self-test
PLAN_FLAGS=(
  --plan-fill-complete
  --plan-max-requests "${PLAN_MAX_REQUESTS:-300}"
  --plan-attempts "${PLAN_ATTEMPTS:-3}"
  --plan-backoff-seconds "${PLAN_BACKOFF_SECONDS:-2}"
  --plan-stalled-passes "${PLAN_STALLED_PASSES:-3}"
)
if [[ "${PLAN_USAGE_RESET:-0}" == "1" ]]; then
  PLAN_FLAGS+=(--plan-usage-reset)
fi
"$PY" driver.py --plans-only --dataset-dir DATASET --model best_model.zip --output-dir results \
  --plan-budget 92 "${PLAN_FLAGS[@]}"
"$PY" driver.py --dataset-dir DATASET --model best_model.zip --output-dir results --require-frozen \
  --plan-budget 92 "${PLAN_FLAGS[@]}"
"$PY" chart.py --input results/raw_results.csv --output-dir results/figures
echo "COSC723 pipeline complete: results/raw_results.csv and results/figures/COSC723_*.png"
