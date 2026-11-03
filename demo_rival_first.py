#!/usr/bin/env python3
"""Week-4 gate demo: one repeatable case where a rival takes an opportunity first.

Development-only tool. It reuses the frozen driver helpers (identical simulator, checker,
queue order, saved plans) but writes only under results/dev/, so the frozen campaign
artifacts (raw_results.csv, transactions.csv, freeze_manifest.json, figures, plans, prompts)
stay untouched and their hashes are unchanged.

Case definition used here (preemption):
  after a team agent's turn in round r, its own PPO candidate order is recomputed on the
  shared state it leaves behind; a case is recorded when a competing searcher (A1/A2)
  commits exactly that route later in the same round while the team agent did not commit
  that route itself. The frozen campaign column ``rival_consumed_first`` keeps its own
  stricter definition (team proposal fails, then a rival commits that route).

Run:
  python demo_rival_first.py --setting gas:1x --condition B2 --seeds 40 41 42 43 44
"""

from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from stable_baselines3 import PPO

import environment as sim  # registers the legacy 'train' alias required by PPO.load
import driver as D

CASE_DEFINITION = (
    "Preemption: after the team agent's turn in round r its own PPO candidate order is "
    "recomputed on the shared state; a case is recorded when a rival commits exactly that "
    "route later in the same round and the team agent did not commit that route itself. "
    "This closes the Week-4 gate ('one repeatable case where a rival takes an opportunity "
    "first'); it is a development artifact, not part of the frozen campaign."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("DATASET"))
    parser.add_argument("--model", type=Path, default=Path("best_model.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--setting", default="gas:1x", help="scenario setting_id to demo")
    parser.add_argument("--condition", default="B2", choices=list(D.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plan-prefs", type=int, default=8)
    return parser.parse_args()


def best_candidate(env: Any, model: Any, plan_routes: Optional[List[int]]) -> Optional[Tuple[int, float]]:
    """First feasible candidate in the agent's own order on the current env state."""
    obs = env._timed_observation()
    logits, fractions = D.policy_route_scores(env, obs, model)
    return D.first_feasible(env, D.candidate_order(logits, plan_routes), fractions)


def route_tokens(env: Any, path_index: int) -> str:
    token_a, token_b, token_c, _pools = env.triangular_paths[int(path_index)]
    return f"{token_a}>{token_b}>{token_c}>{token_a}"



def run_demo_episode(
    model: Any,
    checker: Any,
    args: argparse.Namespace,
    scenario: Any,
    seed: int,
    condition: str,
    plans: Mapping[str, Mapping[str, Mapping[str, Any]]],
    envs: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """One episode under the frozen rules with the preemption probe; returns the first case."""
    condition_agents = D.CONDITION_AGENTS[condition]
    plan_agents = set(D.CONDITION_PLANS[condition])
    scenario_plans = plans.get(scenario.setting_id, {})
    team_agents = [agent for agent in condition_agents if agent in D.TEAM]

    for agent in condition_agents:
        sim.reset_env_compat(envs[agent], seed=seed)
    shared_pools = copy.deepcopy(envs["C1"].current_pools)
    shared_gas = copy.copy(envs["C1"].current_gas)

    def plan_routes_for(agent: str) -> Optional[List[int]]:
        plan = scenario_plans.get(agent) if agent in plan_agents else None
        if not plan:
            return None
        return [int(item["path_index"]) for item in plan["route_preferences"]]

    active = list(condition_agents)
    for round_index in range(1, int(args.max_rounds) + 1):
        if not active:
            break
        committed_this_round: Dict[str, int] = {}
        team_best: Dict[str, int] = {}
        for agent in [name for name in condition_agents if name in active]:
            env = envs[agent]
            D.check_shared_state(env, shared_pools)
            env.current_gas = copy.copy(shared_gas)
            if env.cumulative_decision_time_seconds > float(scenario.c6_seconds):
                active.remove(agent)
                continue
            env.resume_decision_clock()
            pick = best_candidate(env, model, plan_routes_for(agent))
            if pick is None:
                env._finish_selection_timing()
                active.remove(agent)
                continue
            path_index, fraction = pick
            action = env._action_from_path_amount(int(path_index), float(fraction))
            path_index, fraction, _audit = env._decode_action(action)
            path = env.triangular_paths[path_index]
            proposal = D.Proposal(
                agent=agent,
                round_index=round_index,
                path_index=int(path_index),
                tokens=(path[0], path[1], path[2], path[0]),
                pool_ids=tuple(str(pool) for pool in path[3]),
                amount_fraction=float(fraction),
                amount_in=float(fraction) * float(env.current_balance.get(path[0], 0.0)),
                source="demo",
            )
            verdict = checker.check(
                proposal, shared_pools, env.current_balance, env.prices, env.current_gas.gas_cost_usd
            )
            stop_reason = ""
            committed_net = 0.0
            if verdict.status == "committed":
                _obs, _reward, done, info = sim.step_env_compat(env, action)
                if bool(info.get("success", False)):
                    committed_net = float(info.get("net_profit_usd", 0.0))
                    shared_pools = copy.deepcopy(env.current_pools)
                    shared_gas = copy.copy(env.current_gas)
                    committed_this_round[agent] = int(path_index)
                stop_reason = str(info.get("stop_reason", "")) or ("terminated" if done else "")
            if agent in D.RIVALS and committed_this_round.get(agent) == int(path_index):
                preempted = [team for team, route in team_best.items() if route == int(path_index)]
                if preempted and committed_this_round.get(preempted[0]) != int(path_index):
                    team_agent = preempted[0]
                    probe_env = envs[team_agent]
                    D.check_shared_state(probe_env, shared_pools)
                    probe_env.current_gas = copy.copy(shared_gas)
                    next_pick = best_candidate(probe_env, model, plan_routes_for(team_agent))
                    return {
                        "condition": condition,
                        "setting_id": scenario.setting_id,
                        "seed": int(seed),
                        "round": int(round_index),
                        "team_agent": team_agent,
                        "team_top_route_before_rival": int(path_index),
                        "team_top_route_tokens": route_tokens(env, int(path_index)),
                        "team_committed_this_round": committed_this_round.get(team_agent),
                        "rival": agent,
                        "rival_route": int(path_index),
                        "rival_amount_fraction": float(fraction),
                        "rival_after_gas_profit_usd": committed_net,
                        "team_next_best_after_rival": None if next_pick is None else int(next_pick[0]),
                        "note": (
                            "the rival committed the route that was the team agent's own best "
                            "candidate on the shared state right after its turn"
                        ),
                    }
            if agent in team_agents:
                probe_env = envs[agent]
                D.check_shared_state(probe_env, shared_pools)
                probe_env.current_gas = copy.copy(shared_gas)
                probe_pick = best_candidate(probe_env, model, plan_routes_for(agent))
                if probe_pick is not None:
                    team_best[agent] = int(probe_pick[0])
            if stop_reason:
                active.remove(agent)
    return None


def render_text(payload: Mapping[str, Any]) -> str:
    case = payload["case"]
    lines = [
        "COSC723 Week-4 gate demo: a rival takes an opportunity first",
        "=" * 64,
        f"command        : {payload['command']}",
        f"condition      : {case['condition']}",
        f"setting        : {case['setting_id']}",
        f"seed           : {case['seed']}",
        f"round          : {case['round']}",
        f"team agent     : {case['team_agent']} (its own best route was "
        f"{case['team_top_route_before_rival']} {case['team_top_route_tokens']})",
        f"rival          : {case['rival']} committed route {case['rival_route']} for "
        f"+{case['rival_after_gas_profit_usd']:.2f} USD after gas",
        f"team committed : {case['team_committed_this_round']}",
        f"team next best : {case['team_next_best_after_rival']}",
        f"planner model  : "
        f"{payload['planner']['plan_campaign'].get('model') or payload['planner']['environment_model']}",
        "",
        "definition     : " + payload["case_definition"].splitlines()[0],
        "note           : " + payload["frozen_campaign_note"],
    ]
    return "\n".join(str(line) for line in lines) + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    all_scenarios = D.build_scenarios()
    scenarios = [scenario for scenario in all_scenarios if scenario.setting_id == args.setting]
    if not scenarios:
        raise SystemExit(
            f"unknown --setting {args.setting!r}; known: {[s.setting_id for s in all_scenarios]}"
        )
    scenario = scenarios[0]
    catalogue = D.build_catalogue_env(args, scenarios)
    agents_needed = sorted({agent for agent in D.CONDITION_PLANS[args.condition]})
    plans = D.load_saved_plans(
        args.output_dir, scenarios, agents_needed, catalogue, int(args.plan_prefs)
    )
    missing = D.missing_plan_slots(scenarios, agents_needed, plans, D.openrouter_settings())
    if missing:
        raise SystemExit(
            f"missing checked plans for the demo: {missing}; run 'python driver.py --plans-only' first"
        )
    checker = D.CPMMChecker()
    print(f"[demo] loading frozen checkpoint {args.model} on {args.device} ...")
    model = PPO.load(str(args.model), device=str(args.device))
    started = time.time()
    for seed in args.seeds:
        envs = D.build_agent_envs(model, args.dataset_dir, scenario, int(seed))
        case = run_demo_episode(
            model, checker, args, scenario, int(seed), args.condition, plans, envs
        )
        if case is None:
            print(f"[demo] seed {seed}: no preemption case found")
            continue
        payload = {
            "case_definition": CASE_DEFINITION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": (
                f"python demo_rival_first.py --setting {scenario.setting_id} "
                f"--condition {args.condition} --seed {int(seed)}"
            ),
            "frozen_campaign_note": (
                "this tool writes only under results/dev/; the frozen campaign artifacts are untouched"
            ),
            "planner": {
                "environment_model": D.openrouter_settings()["model"],
                "plan_campaign": D.load_plan_campaign_summary(args.output_dir),
            },
            "code_sha256": {
                name: D.sha256_file(Path(name))
                for name in ("driver.py", "environment.py", "chart.py")
                if Path(name).is_file()
            },
            "model_sha256": D.sha256_file(args.model),
            "case": case,
            "search_seconds": round(time.time() - started, 2),
        }
        output_dir = args.output_dir / "dev"
        D.dump_json(output_dir / "week4_rival_first_demo.json", payload)
        (output_dir / "week4_rival_first_demo.txt").write_text(render_text(payload), encoding="utf-8")
        print(render_text(payload))
        print(f"[demo] saved {output_dir / 'week4_rival_first_demo.json'}")
        return 0
    print("[demo] no case found in the searched seeds; try more seeds or another setting")
    return 3


if __name__ == "__main__":
    sys.exit(main())

