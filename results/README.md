# `results/` — index of every file

Everything in this folder comes from the single frozen campaign
(**1150 cases = 23 settings × seeds 40–49 × 5 conditions**) produced by `driver.sh` / `slurm_driver.sh`.
Nothing here is a hand-edited copy, and there is no earlier (superseded) campaign on disk: each re-run
overwrites these files in place.

| Path | What it is |
|---|---|
| `raw_results.csv` | one row per condition × setting × seed (1150 rows) — the source of every number in `../README.md` |
| `transactions.csv` | one row per proposal (~16k rows) with the checker verdict, fees, gas and wallet/reserve updates |
| `summary_by_cell.csv` | per-metric mean for each condition × setting (the data behind the 1×4 atlases) |
| `overall_by_condition.csv` | per-metric mean for each condition (the headline comparison) |
| `checker_selftest.csv` | known-answer self-test of the deterministic CPMM checker (step 1 of the pipeline) |
| `scenarios.csv` | the 23 settings: dimension, value, wallet, gas, depth and the C6 block-time budget |
| `config.json` | frozen configuration of the run (paths, model ids, request ceiling, seeds, hashes) |
| `plan_campaign_summary.json` | planner accounting: slots, retries, requests used, model id |
| `plan_usage.csv` | per-request planner record: model, prompt hash, tokens, seconds, failures |
| `plans/` (92 files) | the checked route-allocation plans that were executed — **provenance**, not extra results |
| `prompts/` (92 files) | the verbatim prompt sent to the planner for each plan — **provenance**, not extra results |
| `figures/` (17 png + 17 pdf + 2 csv) | metric atlases, the paired bootstrap figure and the atlas/paired CSVs |
| `week4_rival_first_case.json` | the strict-definition `rival_consumed_first` metric (no case observed) |
| `dev/` | Week-4 gate demo output: one repeatable *rival-first* case |

Local only (git-ignored, not part of the public repository): `freeze_manifest.json`, the provenance
record of the machine, interpreter and hashes that produced this campaign.
