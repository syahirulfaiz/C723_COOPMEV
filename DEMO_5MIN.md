# COSC723 CoopMEV-Lite — five-minute demo (live or recorded)

Deliverable of the approved contract ("five-minute live or recorded demo"). The demo never
touches the frozen campaign: it writes only under `results_demo/` (plus `results/dev/` for the
Week-4 case) and it cannot spend a planner request (`--plan-max-requests 0`).

## Preconditions

```bash
conda activate /path/to/your/env      # Python >= 3.10 with numpy, pandas, torch,
cd /path/to/C723_COOPMEV              # stable-baselines3, gymnasium, matplotlib, openpyxl
# nothing to export: the demo reuses the 92 checked plans and makes no OpenRouter call
```

`driver.py` needs Python >= 3.10. If `python` on your `PATH` is older (for example the system
`/usr/bin/python` is 3.6), the demo script automatically falls back to the active conda
interpreter (`CONDA_PREFIX`); override explicitly with `PY=/path/to/python bash demo_5min.sh`.

## Run

```bash
bash demo_5min.sh                       # ~2-3 minutes of compute, the rest is narration
# optional: bash demo_5min.sh /path/to/project
```

## Steps and Remarks

| Steps | Remarks |
|---|---|
| 1/7 layout | "One frozen checkpoint, one simulator, four agents. `results/` holds the frozen campaign: 1150 cases, 92 checked plans, manifest with code/model/data hashes." |
| 2/7 `--self-test` | "The execution checker is the ground truth; 23 independently calculated known-answer cases cover fees, gas, sequential pool changes and atomic rejection." |
| 3/7 `--plans-status` | "Plans are produced offline before any timed episode, schema-checked, hashed and reused across seeds; 92/92 present, 101 recorded requests (9 retries), model `openai/gpt-oss-20b`." |
| 4/7 live B2 vs Proposed | "Same frozen PPO, same plans, same rivals; only the C1→C2 route message differs. Team profit and rival profit are printed per case; round 1 shows four atomic commits against shared reserves." |
| 5/7 live charts | "The same `chart.py` renders the ten approved metric atlases plus the profit/time cluster and the paired bootstrap figure." |
| 6/7 Week-4 gate | "One repeatable case where a rival takes an opportunity first: B2, gas:1x, seed 40, round 2 — A2 commits route 16 (`DAI>OHM>ETH>DAI`), which was C1's own current best candidate, for +$155.55 after gas. The frozen campaign reports the strict definition (zero) separately; we never conflate the two." |
| 7/7 frozen evidence | "The campaign itself is untouched by this demo: manifest hashes, 1150 rows, zero invalid committed trades, 17 `COSC723_*` figures." |


## Headline numbers to quote

- Communication (primary, paired `Proposed − B2`, complete seed groups): **+31.3 USD, 95% CI [3.5, 61.0]** → positive interval supports the communication hypothesis.
- Plan value (paired `Proposed − PPO_no_plan`): **−7.1 USD, 95% CI [−33.2, +20.7]** → null finding (the LLM plans add no measurable value in this snapshot); the contract states this is a valid result.
- Execution quality: **0 invalid proposals, 0 rejected trades, 0 invalid committed trades** across 1150 cases; message overhead **0.11 ms**; mean online decision **≈ 9–11 ms**.
- Campaign scale: 23 settings × seeds 40–49 × 5 conditions = **1150 cases**, **16,025** proposal records (≈ 5.2 rounds, 7.4 team commits and 6.5 rival commits per case).

## Troubleshooting

- Slow login node: run the demo on a compute node (`srun --ntasks=1 --cpus-per-task=1 bash demo_5min.sh`) or show the recorded log `logs/cosc723.10692632.out.txt`.
- Demo output directory missing: the script recreates `results_demo/` and re-links `results/plans`.
- Do **not** delete `results/plans`, `results/prompts`, `results/plan_usage.csv` or `results/freeze_manifest.json`; they are the frozen planning evidence.

## Regenerating the 92 plans from scratch (optional)

By default the driver **reuses** `results/plans` when the recorded planner model matches the
current `OPENROUTER_MODEL`. To force a complete regeneration (for example when a reviewer wants a
fresh end-to-end run):

```bash
export OPENROUTER_API_KEY='...' OPENROUTER_MODEL='openai/gpt-oss-20b'
bash regenerate_plans.sh scratch     # RECOMMENDED: new campaign under results_scratch/ (results/ untouched)
bash regenerate_plans.sh inplace     # archives results/ under results/archive/<stamp>/ and rebuilds it
```

Manual equivalent of `scratch` (no helper script):

```bash
python driver.py --plans-only --dataset-dir DATASET --model best_model.zip --output-dir results_scratch \
  --plan-fill-complete --plan-max-requests 300 --plan-attempts 3 --plan-backoff-seconds 2 \
  --plan-stalled-passes 3 --plan-budget 92
python driver.py --dataset-dir DATASET --model best_model.zip --output-dir results_scratch
python chart.py --input results_scratch/raw_results.csv --output-dir results_scratch/figures
```

Notes:

- a different `OPENROUTER_MODEL` invalidates the cache automatically (only slots whose recorded
  model does not match are re-requested);
- 92 slots need roughly 92-120 requests depending on retries; every attempt is logged in
  `plan_usage.csv`;
- `inplace` keeps the previous campaign under `results/archive/<stamp>/`, so nothing is lost.

## Safety notes

- The frozen campaign (`results/raw_results.csv`, `transactions.csv`, `freeze_manifest.json`, plans, prompts, figures) is byte-identical before and after the demo (verified with sha256).
- `results_demo/` is disposable; delete it after the demo if the repository should stay minimal.
