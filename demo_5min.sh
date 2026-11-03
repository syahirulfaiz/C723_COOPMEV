#!/usr/bin/env bash
# COSC723 CoopMEV-Lite: five-minute live demo (contract deliverable 5).
#
# Safety: the demo writes only under results_demo/ (plus results/dev/ for the Week-4 case).
# The frozen campaign under results/ (1150 cases, manifest, plans, figures) is never written,
# and the planner stage cannot spend a model request (--plan-max-requests 0).
#
# Usage:  bash demo_5min.sh          # run from the project root
#         bash demo_5min.sh /path/to/project
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
PY="${PY:-python}"
# driver.py needs Python >= 3.10 (from __future__ import annotations + typing syntax).
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "note: '${PY}' is older than Python 3.10; using ${CONDA_PREFIX}/bin/python" >&2
    PY="${CONDA_PREFIX}/bin/python"
  else
    echo "error: driver.py needs Python >= 3.10; activate a conda environment or export PY=/path/to/python" >&2
    exit 1
  fi
fi
DEMO_OUT="results_demo"
mkdir -p "$DEMO_OUT" logs

echo "=== 1/7 frozen project layout ==="
ls -la | grep -v '^total'
echo
echo "=== 2/7 deterministic execution checker: 23 known-answer cases ==="
"$PY" driver.py --self-test --output-dir "$DEMO_OUT"
echo
echo "=== 3/7 offline planner artefacts: checked plans (no new requests here) ==="
"$PY" driver.py --plans-status
"$PY" - <<'PYEOF'
import json, pathlib
summary = json.loads(pathlib.Path('results/plan_campaign_summary.json').read_text())
plan = json.loads(pathlib.Path('results/plans/gas_1x/C1.json').read_text())
print(f"plan campaign: {summary['plans_present']}/{summary['planned_slots']} checked plans, "
      f"{summary['requests_actual_total']} recorded requests, model {summary['model']}")
print("example C1 plan (gas:1x):", [p['path_index'] for p in plan['route_preferences']])
PYEOF
echo
echo "=== 4/7 live episode: B2 (communication OFF) vs Proposed (ON), gas:1x, seed 40 ==="
export OPENROUTER_MODEL="$("$PY" -c "import json;print(json.load(open('results/plan_campaign_summary.json'))['model'])")"
ln -sfn "$(cd results && pwd)/plans" "$DEMO_OUT/plans"
ln -sfn "$(cd results && pwd)/prompts" "$DEMO_OUT/prompts"
"$PY" driver.py --conditions B2 Proposed --seeds 40 --only-dimensions gas --max-scenarios 1 \
  --skip-plans --plan-max-requests 0 --output-dir "$DEMO_OUT"
"$PY" - <<'PYEOF'
import csv
for row in csv.DictReader(open('results_demo/raw_results.csv')):
    print(f"{row['condition']:9s} team=${float(row['team_after_gas_profit_usd']):8.2f}  "
          f"rival=${float(row['rival_profit_usd']):8.2f}  team trades={row['successful_trades_team']}  "
          f"duplicate-path proposals={row['duplicate_path_proposals']}  rounds={row['rounds']}")
tx = list(csv.DictReader(open('results_demo/transactions.csv')))
print("round 1 (shared reserves, atomic commits):",
      [(t['agent'], f"route {t['path_index']}", f"+{float(t['after_gas_profit_usd']):.2f}")
       for t in tx if t['round'] == '1'])
PYEOF
echo
echo "=== 5/7 figures rendered live from the demo run ==="
"$PY" chart.py --input "$DEMO_OUT/raw_results.csv" --output-dir "$DEMO_OUT/figures" | tail -3
ls "$DEMO_OUT/figures" | grep -c '\.png$' || true
echo
echo "=== 6/7 Week-4 gate: one repeatable rival-first case ==="
"$PY" demo_rival_first.py --setting gas:1x --condition B2 --seed 40
echo
echo "=== 7/7 frozen campaign evidence (untouched by this demo) ==="
"$PY" - <<'PYEOF'
import csv, json
from collections import Counter
manifest = json.load(open('results/freeze_manifest.json'))
print("protocol:", manifest['protocol_id'], "| seeds:", manifest['seeds'][0], "-", manifest['seeds'][-1])
print("plan policy:", manifest['plan_policy'], "| actual requests:", manifest['plan_usage']['requests_actual_total'])
rows = list(csv.DictReader(open('results/raw_results.csv')))
print("campaign rows:", len(rows), "| per condition:", dict(Counter(r['condition'] for r in rows)))
print("invalid committed trades:", sum(int(r['invalid_committed_trades']) for r in rows))
PYEOF
ls results/figures/COSC723_*.png | sed -n '1,18p'
echo
echo "=== demo complete: results/ (frozen) was written only by the campaign jobs ==="
