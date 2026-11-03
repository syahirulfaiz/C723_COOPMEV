# CoopMEV-Lite — Cooperative vs Adversarial Integrated Agents in a Simulated DeFi Market

Reproducible code for the **COSC 723 Cybersecurity in IoT and Applications** project
The experiment follows the revised
project contract, **Revision 1.1** (2026-11-01), in which the offline planner is served by
OpenRouter chat-completions (model `openai/gpt-oss-20b` at submission).

## 1. What this project does

- Two cooperating agents (**C1**, **C2**) compete against two independent competing searchers
  (**A1**, **A2**) in a frozen Uniswap V2 / SushiSwap V2-style CPMM simulator
  (**174 pools**, **114 ordered triangular paths**, snapshot **2026-06-27**).
- Every integrated agent uses the **same frozen PPO checkpoint** (`best_model.zip`) plus a
  **checked, offline LLM route-allocation plan**; the simulator retains wallet, reserve, gas and
  block-time (C6) constraints, and each episode is a **sequence of atomic triangular transactions**.
- Research question: *does route sharing between C1 and C2 improve their combined after-gas profit
  compared with no sharing, against the same two rivals, with the same pre-generated plans,
  starting resources and execution rules?*
- Five conditions are evaluated: **B1** (context only), **B2** (communication OFF, primary
  control), **B3** (message sent and ignored), **Proposed** (communication ON) and
  **PPO_no_plan** (no LLM plans for C1/C2).
- Scale: **23 settings × seeds 40–49 × 5 conditions = 1150 cases**. A **deterministic CPMM
  execution checker** is the ground truth for every proposal (fees, gas, wallet/reserve updates,
  atomic rejection).

## 2. Headline results (frozen campaign, 1150 cases)

| Metric | Value |
|---|---|
| Communication value (paired `Proposed − B2`, complete seed groups) | **+31.3 USD, 95% CI [3.5, 61.0]** |
| Plan value (paired `Proposed − PPO_no_plan`) | **−7.1 USD, 95% CI [−33.2, +20.7]** (null finding) |
| Execution quality | **0** invalid proposals · **0** rejected trades · **0** invalid committed trades |
| Message overhead (B3) | ≈ **0.11 ms** per message; mean online decision ≈ **9–11 ms** |
| Week-4 gate | one repeatable *rival-first* case: `python demo_rival_first.py --setting gas:1x --condition B2 --seed 40` |

the primary paired interval supports the communication hypothesis and
that the LLM-plan contribution is evaluated **separately**, where null or negative findings remain
valid results. Execution-quality metrics can legitimately be zero because the checker prevents
invalid transactions.

## 3. Repository layout

```
C723_COOPMEV/
├── driver.py                    # multi-agent evaluation driver (5 conditions, checker, planner)
├── environment.py               # frozen CPMM simulator + PPO policy symbols (needed by best_model.zip)
├── chart.py                     # figures: metric atlases, paired bootstrap, seed-group unit
├── driver.sh                    # one-command pipeline: self-test -> plans -> evaluation -> figures
├── slurm_driver.sh              # Slurm template (adjust partition/account, or pass them on sbatch)
├── demo_rival_first.py          # Week-4 gate demo: one repeatable rival-first case
├── demo_5min.sh, DEMO_5MIN.md   # five-minute live demo script + narration/timing
├── regenerate_plans.sh          # force full plan regeneration (scratch | inplace)
├── requirements.txt             # pinned Python dependencies
├── best_model.zip               # frozen PPO checkpoint (stable-baselines3 2.9 save format)
├── DATASET/                     # pool/prices/wallet/gas snapshot + regeneration scripts
└── results/                     # frozen campaign evidence and figures
    ├── raw_results.csv          # one row per condition × setting × seed (1150 rows)
    ├── transactions.csv         # one row per proposal (16,025 rows, checker verdicts included)
    ├── plans/, prompts/         # 92 checked LLM plans + the exact prompts (hashed)
    ├── plan_usage.csv           # every planner request: tokens, seconds, failures
    ├── config.json, summary_by_cell.csv, overall_by_condition.csv, scenarios.csv
    ├── checker_selftest.csv     # 23 independent known-answer checker cases
    ├── week4_rival_first_case.json
    ├── dev/week4_rival_first_demo.{json,txt}   # Week-4 preemption demo case
    └── figures/                 # 17 COSC723_* figures (png + pdf) + atlas/paired CSVs
```


## 4. Requirements

Python **≥ 3.10** and the pinned packages in `requirements.txt`:

```bash
conda create -n coopmev python=3.10 -y && conda activate coopmev
pip install -r requirements.txt        # numpy, pandas, torch, stable-baselines3, gymnasium, matplotlib, openpyxl
```

The `environment.py` script also loads the paired gas data from `DATASET/Book1.xlsx` using `pandas` and `openpyxl`, while `driver.py` retrieves the `OPENROUTER_API_KEY` from the environment variables.

**No API key is required to reproduce the campaign**, as the 92 previously evaluated plans are already included in `results/plans/` and can be reused, provided that the recorded planner model matches the value specified in `OPENROUTER_MODEL`.

To regenerate the plans or run the entire process from scratch, please refer to **Section 5.3** for further instructions.

## 5. Reproduce

### 5.1 Smoke test (no API key, ~1 minute)

```bash
python driver.py --conditions PPO_no_plan --seeds 40 --only-dimensions gas --max-scenarios 1 --allow-dev-plans
python chart.py --input results/raw_results.csv --output-dir results/figures
```

`--allow-dev-plans` writes clearly labelled development fallback plans (`source: dev_fallback`);
they are never reused once an API key is present.

### 5.2 Full campaign from the checked plans (the shipped evidence)

```bash
bash driver.sh                       # self-test -> plan reuse -> 1150 cases -> figures
# or step by step
python driver.py --self-test
python driver.py --dataset-dir DATASET --model best_model.zip --output-dir results --require-frozen
python chart.py --input results/raw_results.csv --output-dir results/figures
```

### 5.3 Full Regenerate the 92 plans, and Campaign from scratch (requires an API key)

```bash
export OPENROUTER_API_KEY='...' OPENROUTER_MODEL='openai/gpt-oss-20b'
bash regenerate_plans.sh scratch     # new campaign under results_scratch/ (results/ untouched)
bash regenerate_plans.sh inplace     # archive results/ under results/archive/<stamp>/ and rebuild it
```

Manual equivalent of `scratch`:

```bash
python driver.py --plans-only --dataset-dir DATASET --model best_model.zip --output-dir results_scratch \
  --plan-fill-complete --plan-max-requests 300 --plan-attempts 3 --plan-backoff-seconds 2 \
  --plan-stalled-passes 3 --plan-budget 92
python driver.py --dataset-dir DATASET --model best_model.zip --output-dir results_scratch
python chart.py --input results_scratch/raw_results.csv --output-dir results_scratch/figures
```

Plan generation is **offline** (before any timed episode) and bounded: 92 slots (4 agents × 23
settings), retried until every slot holds a checked plan, with a documented request ceiling and
full usage reporting in `results/plan_usage.csv`. The shipped campaign used 101 requests
(92 plans + 9 retries) with the model recorded in `results/plan_campaign_summary.json`.

### 5.4 Slurm template

`slurm_driver.sh` is a template: create `logs/` first, point it at your environment and cluster,
then submit.

```bash
mkdir -p logs
export COSC723_ENV=/path/to/conda-env      # Python >= 3.10 with requirements.txt installed
sbatch --export=ALL --partition=<PARTITION> --account=<ACCOUNT> slurm_driver.sh
# allocation smoke test without an API key:
SMOKE=1 sbatch --export=ALL --partition=<PARTITION> --account=<ACCOUNT> slurm_driver.sh
```

The job runs the same four stages as `driver.sh` and writes `logs/cosc723.<jobid>.{out,err}.txt`.


## 6. What the code does

- **Agents.** C1 and C2 are teammates; A1 and A2 are independent competing searchers. Each agent
  has its own wallet and all four start with the scenario's per-agent balance. Team profit is
  C1 + C2 after gas; rival profit is A1 + A2.
- **Frozen PPO checkpoint.** `best_model.zip` is a two-head residual policy: one categorical head
  selects an ordered triangular path, one continuous head outputs the first-swap input fraction for
  the current state. The same checkpoint is used by all four agents.
- **Offline LLM planner.** Before any timed episode the driver asks OpenRouter chat-completions for
  a route-allocation plan per agent and setting (JSON schema), checks it (schema, agent/setting,
  route existence, distinct routes), hashes it and reuses it across seeds. Saved plans only change
  **candidate priority**; the amount for the selected route still comes from the policy for the
  current state and is checked before execution.
- **Deterministic CPMM execution checker.** An independent re-derivation of every proposal (three
  swaps, fees, gas, wallet/reserve updates, atomicity) is the ground truth; it is validated by
  **23 independently calculated known-answer cases** (`results/checker_selftest.csv`) covering
  fees, gas, sequential pool changes and atomic rejection. Each commit is cross-checked against the
  simulator; `invalid_committed_trades` must stay zero.
- **Episode flow.** A fixed proposal queue (defined in `results/config.json`) orders proposals; the
  checker always uses the latest shared market state before a commit; accepted trades update wallets
  and shared reserves atomically, while a rejection leaves them unchanged. An episode ends when no
  profitable feasible transaction remains or the block-time budget is exhausted.
- **Timing.** C6 is the per-agent wall-clock decision time (observation, message handling, policy
  inference, candidate screening, CPMM simulation, checker validation, bookkeeping); offline plan
  generation is reported separately from online decision and execution time.
- **Communication conditions.** B3 sends a route message that the receiver reads and ignores (pure
  overhead measurement); Proposed lets the receiver use that message plus the saved plans to avoid
  the duplicate route whenever a feasible alternative exists. `PPO_no_plan` keeps the identical
  duplicate-route rule but replaces the C1/C2 plans with the first feasible candidate in PPO order.

## 7. Outputs

| Artifact | Content |
|---|---|
| `results/raw_results.csv` | one row per condition × setting × seed (1150 rows, 59 metric columns) |
| `results/transactions.csv` | one row per proposal: route, amount, checker verdict, constraint, profits |
| `results/plans/`, `results/prompts/` | 92 checked plans and the exact prompts (hashes recorded) |
| `results/plan_usage.csv`, `plan_campaign_summary.json` | planner requests, tokens, seconds, failures |
| `results/config.json`, `scenarios.csv` | frozen settings, condition mapping, C6 semantics |
| `results/checker_selftest.csv` | the 23 known-answer checker cases |
| `results/summary_by_cell.csv`, `overall_by_condition.csv` | aggregated metrics |
| `results/week4_rival_first_case.json`, `results/dev/` | Week-4 gate evidence |
| `results/figures/COSC723_*.png or pdf` | 17 figures (metric atlases, profit–time, paired bootstrap) |

Figures: `COSC723_Fig04_mean_profit_performance_atlas`, `COSC723_Fig09_inference_time_performance_atlas`,
`COSC723_Fig10_profit_time_expanded_cluster`, `COSC723_arbitrage_type_heat_1x4`, `COSC723_team_win_1x4`,
`COSC723_team_profit_share_1x4`, `COSC723_rival_profit_1x4`, `COSC723_successful_trades_1x4`,
`COSC723_duplicate_path_proposals_1x4`, `COSC723_shared_pool_overlap_1x4`,
`COSC723_invalid_proposals_1x4`, `COSC723_rejected_trades_1x4`, `COSC723_invalid_committed_trades_1x4`,
`COSC723_plans_changed_route_1x4`, `COSC723_rival_consumed_first_1x4`,
`COSC723_paired_proposed_minus_B2_bootstrap95`, `COSC723_complete_seed_group_unit`
(plus `cosc723_atlas_cells.csv` and `cosc723_paired_proposed_minus_B2.csv`).

## 8. Five-minute demo

```bash
bash demo_5min.sh          # 7 steps; writes only to results_demo/ and cannot spend a planner request
```

`DEMO_5MIN.md` contains the timing table, narration and troubleshooting notes. The Week-4 gate case
(a rival takes an opportunity first) is reproduced separately:

```bash
python demo_rival_first.py --setting gas:1x --condition B2 --seed 40
# writes results/dev/week4_rival_first_demo.{json,txt} and prints the case
```

The frozen campaign reports `rival_consumed_first` under the strict definition (a team proposal
fails to commit and a rival commits that same route in the same round); the demo records the same
phenomenon under a preemption definition (a rival commits the team agent's current best candidate).
Both definitions are documented in the demo artifact and are never conflated.


## 9. Contract mapping (Revision 1.1)

| Contract element | Artifact / implementation |
|---|---|
| 23 settings × seeds 40–49 | `results/scenarios.csv`, `results/config.json`, `results/raw_results.csv` |
| Conditions B1 / B2 / B3 / Proposed / PPO_no_plan | `results/config.json` (`conditions`, `condition_agents`, `condition_plans`) + `raw_results.condition` |
| Checked plans produced offline and reused | `results/plans/`, `results/prompts/`, `results/plan_usage.csv` |
| Planning-call accounting | 92 plan slots, retry-until-complete, documented ceiling, actual usage reported |
| Deterministic execution checker + known answers | `driver.py` (`CPMMChecker`), `results/checker_selftest.csv`, `results/transactions.csv` |
| Metrics (profit, rival profit, win/share, trades, duplicate paths, pool overlap, invalid/rejected/invalid-committed, plan-changed route, rival-first) | `results/raw_results.csv` + the 1×4 atlases in `results/figures/` |
| Paired ON-minus-OFF statistics with complete seed groups | `COSC723_paired_proposed_minus_B2_bootstrap95.*`, `COSC723_complete_seed_group_unit.*`, `cosc723_paired_proposed_minus_B2.csv` |
| Week-4 gate (B2 baseline, rival-first case, checker) | B2 in `raw_results.csv`; `demo_rival_first.py` → `results/dev/`; `checker_selftest.csv` |
| One-command experiment | `driver.sh` (local) and `slurm_driver.sh` (cluster) |
| Five-minute demo | `demo_5min.sh` + `DEMO_5MIN.md` |
| Freeze and provenance | `freeze_manifest.json` (regenerated by every run; excluded from the repository), `config.json` |

## 10. Provenance, freeze and limitations

- **Provenance.** Every run writes `results/freeze_manifest.json` with the SHA-256 hashes of the
  code files, the checkpoint, the dataset inputs, the checked plans and the prompts, plus the
  protocol id, seeds, conditions and planner policy. `--require-frozen` refuses to start an
  evaluation whose manifest no longer matches. The manifest is intentionally **not committed**
  because it records absolute paths of the machine that produced the campaign.
- **Planner record.** Model id and generation date per plan, prompt hashes, per-request tokens and
  seconds (`plan_usage.csv`), and the campaign summary (`plan_campaign_summary.json`).
- **Limitations.** (i) One fixed dataset snapshot (2026-06-27) and one frozen checkpoint; the
  results are a controlled simulation, not a live-market or on-chain claim. (ii) The block-time
  budget C6 (4–12 s) rarely binds because decisions take ≈10 ms. (iii) The rival can only
  "take an opportunity first" by preempting the team's next-best route (demonstrated in
  `results/dev/`). (iv) Planning used 101 requests for 92 plans (9 recorded retries); a fresh
  campaign may use a different number. (v) The LLM-plan contribution is null in this snapshot and
  is reported separately from the communication result.

## 11. Security notes

- No credentials are stored in this repository. The driver reads `OPENROUTER_API_KEY` from the
  environment only; `.openrouter.env` (local, optional) is git-ignored.
- `results/plans/`, `results/prompts/`, `results/plan_usage.csv` contain prompts, route ids, token
  counts and hashes only — no keys, no wallet secrets.
- The experiment is a local simulation with no real funds, keys, signing or broadcast.

## 12. Version history

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-09-16 | Submitted project contract (approved by the course instructor). |
| **v1.1** | 2026-11-01 | Contract Revision 1.1: (a) planner provider — the discontinued campus Codex interface is replaced by OpenRouter chat-completions (model `openai/gpt-oss-20b` at submission); (b) planning-call accounting — 92 plan slots, retry-until-complete, actual request usage reported; (c) metric note — `rival_consumed_first` reported under the strict definition while the Week-4 rival-first case is demonstrated separately under a preemption definition. Scope, claim boundary and success criteria are unchanged. |

Git tag: `v1.1`.

## 13. Citation and references

```bibtex
@misc{faiz2026coopmev,
  title  = {CoopMEV-Lite: Cooperative vs Adversarial Integrated Agents in a Simulated DeFi Market},
  author = {Faiz, Syahirul and Materwala, Huned and Svetinovic, Davor},
  year   = {2026},
}
```

Related peer-reviewed literature used for the agentic-AI framing and the MEV context:

- Wooldridge, M., & Jennings, N. R. (1995). *Intelligent Agents: Theory and Practice.* Knowledge Engineering Review, 10(2), 115–152.
- Busoniu, L., Babuska, R., & De Schutter, B. (2008). *A Comprehensive Survey of Multiagent Reinforcement Learning.* IEEE Transactions on Systems, Man, and Cybernetics—Part C, 38(2), 156–172.
- Wang, L., et al. (2024). *A Survey on Large Language Model based Autonomous Agents.* Frontiers of Computer Science, 18(6), 186345.
- Xi, Z., et al. (2025). *The Rise and Potential of Large Language Model Based Agents: A Survey.* Science China Information Sciences (preprint arXiv:2309.07864).
- Yao, S., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023.
- Daian, P., et al. (2020). *Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability.* IEEE S&P 2020.
- Qin, K., Zhou, L., & Gervais, A. (2022). *Quantifying Blockchain Extractable Value: How Dark is the Forest?* IEEE S&P 2022.
- Faiz, S., Materwala, H., & Svetinovic, D. (2026). *A Constrained Deep Reinforcement Learning Framework for Atomic DeFi Triangular Arbitrage.* IEEE Transactions on Network Science and Engineering (under review).

