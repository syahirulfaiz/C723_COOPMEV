#!/usr/bin/env python3
"""COSC723 CoopMEV-Lite driver: communication-value evaluation for a cooperative pair.

Approved contract experiment: 23 settings x seeds 40-49 per condition.
  B1           C1 (frozen PPO + offline LLM plan) versus A1; context only.
  B2           C1+C2 versus A1+A2 with saved plans, communication OFF (primary control).
  B3           as B2, but C1 sends a route message that C2 reads and ignores (overhead).
  Proposed     communication ON: C2 uses C1's route message, plans and PPO candidates to
               avoid the duplicate route whenever a feasible alternative exists.
  PPO_no_plan  Proposed with the identical duplicate-route rule but no LLM plans for C1/C2
               (first feasible candidate in PPO order); A1/A2 plans remain unchanged.

Stages:
  --self-test    independently calculated known-answer cases for the execution checker;
  --plans-only   <= --plan-budget OpenRouter requests generate, check, hash and cache the
                 offline route-allocation plans before any timed episode;
  evaluation     fixed-queue episodes against the shared market state; the deterministic
                 CPMM checker is the ground truth; raw CSV results feed chart.py.

Offline planning env vars: OPENROUTER_API_KEY, OPENROUTER_MODEL,
OPENROUTER_MAX_COMPLETION_TOKENS, OPENROUTER_REASONING_EFFORT.
"""

from __future__ import annotations
import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO

import environment as sim  # frozen simulator + PPO symbols (registers itself as 'train')

AGENTS = ("C1", "C2", "A1", "A2")
TEAM = ("C1", "C2")
RIVALS = ("A1", "A2")
CONDITIONS = ("B1", "B2", "B3", "Proposed", "PPO_no_plan")
CONDITION_AGENTS = {
    "B1": ("C1", "A1"),
    "B2": AGENTS,
    "B3": AGENTS,
    "Proposed": AGENTS,
    "PPO_no_plan": AGENTS,
}
CONDITION_PLANS = {  # agents whose saved LLM plans are in effect for that condition
    "B1": ("C1", "A1"),
    "B2": AGENTS,
    "B3": AGENTS,
    "Proposed": AGENTS,
    "PPO_no_plan": ("A1", "A2"),
}
CONDITION_NOTES = {
    "B1": "one integrated agent versus one integrated rival; context only",
    "B2": "four agents with saved plans; communication OFF (primary control)",
    "B3": "C1 route message read and ignored; measures message overhead",
    "Proposed": "communication ON; C2 avoids C1's route when a feasible alternative exists",
    "PPO_no_plan": "Proposed without C1/C2 plans; identical duplicate-route rule",
}
PLAN_SCHEMA = "cosc723-route-plan-v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
@dataclass(frozen=True)
class Scenario:
    dimension: str
    value: float
    label: str
    balance_usd: int
    gas_multiplier: float
    liquidity_depth: float
    c6_seconds: float

    @property
    def setting_id(self) -> str:
        return f"{self.dimension}:{self.label}"


def build_scenarios() -> List[Scenario]:
    """The exact 23-setting table of the approved contract."""
    out: List[Scenario] = []
    for usd in range(1000, 10001, 1000):
        out.append(Scenario("balance", float(usd), f"{usd // 1000}K", usd, 1.0, 1.0, 12.0))
    for c6 in (4.0, 6.0, 8.0, 10.0, 12.0):
        out.append(Scenario("latency", c6, f"{c6:g}s", 2000, 1.0, 1.0, c6))
    for gas in (1.0, 2.0, 4.0, 8.0):
        out.append(Scenario("gas", gas, f"{gas:g}x", 2000, gas, 1.0, 12.0))
    for depth in (0.25, 0.5, 0.75, 1.0):
        out.append(Scenario("liquidity", depth, f"{depth:g}", 2000, 1.0, depth, 12.0))
    return out


def select_scenarios(scenarios: Sequence[Scenario], args: argparse.Namespace) -> List[Scenario]:
    out = list(scenarios)
    if args.only_dimensions:
        allowed = {str(x).strip().lower() for x in args.only_dimensions}
        out = [s for s in out if s.dimension.lower() in allowed]
    if args.max_scenarios and args.max_scenarios > 0:
        out = out[: int(args.max_scenarios)]
    return out


def make_env_args(dataset_dir: Any, scenario: Scenario) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=str(dataset_dir),
        amount_min=0.0,
        amount_max=1.0,
        gas_multiplier=float(scenario.gas_multiplier),
        max_paths=5000,
        block_time_seconds=float(scenario.c6_seconds),
        reward_reference_usd=10000.0,
        action_mode=ACTION_MODE,
        amount_grid=sim.DEFAULT_AMOUNT_GRID_TEXT,
        protocol_id=sim.PROTOCOL_ID,
        c1_rejection_reward=0.0,
        ppo_path_feature_mode=sim.FIXED_FEATURE_MODE,
        c1_counterfactual_reward=False,
    )


def force_wallet_balance(env: Any, balance_usd: int) -> None:
    exact = [
        (name, wallet)
        for name, wallet in env.wallets
        if str(name).endswith(f".{int(balance_usd)}.json")
    ]
    if len(exact) != 1:
        raise ValueError(f"Expected one wallet file for USD {int(balance_usd):,}; found {len(exact)}.")
    env.wallets = exact
    env.reset_counter = 0
    env.wallet_start_tokens = sorted(
        {token for _, wallet in env.wallets for token, amount in wallet.items() if float(amount) > 0.0}
    )


def scale_liquidity_depth(env: Any, depth: float) -> None:
    """C3 liquidity-depth sweep: scale the shared reserve template before episode resets."""
    depth = float(depth)
    if depth <= 0.0:
        raise ValueError("C3 liquidity depth must be positive.")
    env.liquidity_depth_multiplier = depth
    env.pool_list = [
        dataclasses.replace(
            pool, reserve0=float(pool.reserve0) * depth, reserve1=float(pool.reserve1) * depth
        )
        for pool in env.pool_list
    ]
    env.pool_template = {pool.pool_id: pool for pool in env.pool_list}
    env.current_pools = env._pool_state_from_template()

ACTION_MODE = "hybrid_ppo"
EPS = 1e-9
QUEUE_RULE = (
    "readiness schedule fixed by role rank C1->C2->A1->A2; the checker uses the latest "
    "shared state before every commit"
)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames) if fieldnames is not None else (list(rows[0].keys()) if rows else [])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class CsvStream:
    """Streaming CSV writer so long evaluations never hold every row in memory."""

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.handle = None
        self.writer = None

    def __enter__(self) -> "CsvStream":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        return self

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.writerow(row)

    def __exit__(self, *_exc: Any) -> None:
        self.handle.close()


def safe_div(numerator: Any, denominator: Any) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else float("nan")


@dataclass
class Verdict:
    """Outcome of one checker evaluation (invalid_proposal | rejected | committed)."""

    status: str
    constraint: str
    reason: str
    gross_profit_usd: float = 0.0
    gas_cost_usd: float = 0.0
    after_gas_profit_usd: float = 0.0
    final_amount: float = 0.0
    wallet_after: Optional[Dict[str, float]] = None
    post_pools: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class Proposal:
    """One agent proposal: route, route-specific amount, and provenance flags."""

    agent: str
    round_index: int
    path_index: int
    tokens: Tuple[str, ...]
    pool_ids: Tuple[str, ...]
    amount_fraction: float
    amount_in: float
    source: str
    plan_used: bool = False
    message_used: bool = False
    message_ignored: bool = False
    plan_changed_route: bool = False
    inference_ms: float = 0.0
    decision_ms: float = 0.0
    message_overhead_ms: float = 0.0

    @property
    def start_token(self) -> str:
        return str(self.tokens[0])


def pool_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    return bool(set(left) & set(right))


class CPMMChecker:
    """Deterministic execution checker: the independently derived ground truth.

    Re-derives the sequential swaps, fees, gas and the wallet/reserve updates from the
    verified initial state; it never calls the simulator's execution functions. Invalid
    or rejected proposals leave wallets and reserves unchanged (atomicity). Driver
    proposals always hold the simulator's three legs; the checker itself accepts any
    leg count so known-answer cases can exercise arbitrary directed sequences.
    """

    @staticmethod
    def swap_out(amount_in: float, reserve_in: float, reserve_out: float, fee: float) -> float:
        if amount_in <= 0.0 or reserve_in <= 0.0 or reserve_out <= 0.0:
            return 0.0
        paid = amount_in * (1.0 - float(fee))
        return paid * reserve_out / (reserve_in + paid)

    def check(
        self,
        proposal: Proposal,
        pools: Mapping[str, Mapping[str, Any]],
        wallet: Mapping[str, float],
        prices: Mapping[str, float],
        gas_cost_usd: float,
        c6_exceeded: bool = False,
    ) -> Verdict:
        tokens = tuple(proposal.tokens)
        pool_ids = tuple(proposal.pool_ids)
        if len(pool_ids) < 1 or len(tokens) != len(pool_ids) + 1:
            return Verdict("invalid_proposal", "C4", "malformed_path_schema")
        if tokens[-1] != tokens[0]:
            return Verdict("invalid_proposal", "C5", "path_not_closed")
        if not (0.0 <= float(proposal.amount_fraction) <= 1.0):
            return Verdict("invalid_proposal", "C2", "amount_fraction_out_of_bounds")
        start_balance = float(wallet.get(tokens[0], 0.0))
        first_input = float(proposal.amount_in)
        if first_input <= EPS:
            return Verdict("invalid_proposal", "C2", "nonpositive_first_swap_input")
        if first_input > start_balance + 1e-06:
            return Verdict("invalid_proposal", "C2", "first_swap_input_exceeds_wallet")
        state: Dict[str, Dict[str, float]] = {}
        for pool_id in pool_ids:
            pool = pools.get(pool_id)
            if pool is None:
                return Verdict("invalid_proposal", "C3", f"unknown_pool:{pool_id}")
            if pool_id not in state:
                state[pool_id] = dict(pool)
        amount = first_input
        for leg, pool_id in enumerate(pool_ids):
            pool = state[pool_id]
            token_in, token_out = tokens[leg], tokens[leg + 1]
            token0, token1 = str(pool["token0"]), str(pool["token1"])
            if token_in == token0:
                reserve_in, reserve_out = float(pool["reserve0"]), float(pool["reserve1"])
            elif token_in == token1:
                reserve_in, reserve_out = float(pool["reserve1"]), float(pool["reserve0"])
            else:
                return Verdict("invalid_proposal", "C4", f"leg{leg}_token_not_in_pool")
            if token_out not in (token0, token1) or token_out == token_in:
                return Verdict("invalid_proposal", "C4", f"leg{leg}_token_output_mismatch")
            if reserve_in <= EPS or reserve_out <= EPS:
                return Verdict("invalid_proposal", "C3", f"leg{leg}_reserve_not_positive")
            out = self.swap_out(amount, reserve_in, reserve_out, float(pool.get("fee", 0.003)))
            if out <= 0.0:
                return Verdict("rejected", "C3", f"leg{leg}_swap_not_executable")
            if token_in == token0:
                pool["reserve0"], pool["reserve1"] = reserve_in + amount, reserve_out - out
            else:
                pool["reserve1"], pool["reserve0"] = reserve_in + amount, reserve_out - out
            if pool["reserve0"] <= EPS or pool["reserve1"] <= EPS:
                return Verdict("rejected", "C3", f"leg{leg}_pool_drained")
            amount = out
        gross = (amount - first_input) * float(prices.get(tokens[0], 0.0))
        net = gross - float(gas_cost_usd)
        if net <= 0.0:
            return Verdict("rejected", "C1", "not_profitable_after_gas", gross, gas_cost_usd, net, amount)
        if c6_exceeded:
            return Verdict(
                "rejected", "C6", "wall_clock_budget_exhausted_before_commit", gross, gas_cost_usd, net, amount
            )
        wallet_after = dict(wallet)
        wallet_after[tokens[0]] = start_balance - first_input + amount
        return Verdict("committed", "", "ok", gross, gas_cost_usd, net, amount, wallet_after, state)


def finite_mean(values: Iterable[Any]) -> float:
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.fmean(data)) if data else float("nan")

def _qa_pool(token0: str, token1: str, reserve0: float, reserve1: float, fee: float = 0.003) -> Dict[str, Any]:
    return {"token0": token0, "token1": token1, "reserve0": reserve0, "reserve1": reserve1, "fee": fee}


def checker_self_test() -> List[Dict[str, Any]]:
    """Known-answer cases with expected values derived independently of this code."""
    checker = CPMMChecker()
    rows: List[Dict[str, Any]] = []

    def record(case: str, expected: Any, actual: Any, tolerance: float = 1e-09) -> None:
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                passed = math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance)
            except (TypeError, ValueError):
                passed = False
        else:
            passed = expected == actual
        rows.append({"case": case, "expected": expected, "actual": actual, "passed": bool(passed)})

    prices = {"ETH": 1.0}
    wallet = {"ETH": 5000.0}

    # 1. CPMM fee maths: 1000 * 0.997 * 1e6 / (1e6 + 997) = 996.0069810399032.
    record("single_swap_fee", 996.0069810399032, checker.swap_out(1000.0, 1_000_000.0, 1_000_000.0, 0.003))

    # 2. Gas conversion: 20 gwei * 150000 gas * USD 1590.44 per ETH = 4.77132 USD.
    record("gas_cost_conversion", 4.77132, 20.0 * 150_000.0 * 1590.44 * 1e-09)

    # 3. Three sequential legs over three pools; committed wallet and reserve updates.
    pools = {
        "p1": _qa_pool("ETH", "DAI", 1_000_000.0, 2_000_000.0),
        "p2": _qa_pool("DAI", "USDC", 500_000.0, 400_000.0),
        "p3": _qa_pool("USDC", "ETH", 800_000.0, 600_000.0),
    }
    committed = checker.check(
        Proposal("QA", 0, 0, ("ETH", "DAI", "USDC", "ETH"), ("p1", "p2", "p3"), 0.2, 1000.0, "qa"),
        pools,
        wallet,
        prices,
        0.5,
    )
    record("three_leg_status", "committed", committed.status)
    record("three_leg_final_amount", 1181.0182792400155, committed.final_amount)
    record("three_leg_after_gas_profit", 180.51827924001554, committed.after_gas_profit_usd)
    record("three_leg_wallet_eth", 5181.0182792400155, float(committed.wallet_after["ETH"]))
    record("three_leg_p1_reserves", (1_001_000.0, 1_998_007.9860379202),
           (float(committed.post_pools["p1"]["reserve0"]), float(committed.post_pools["p1"]["reserve1"])), 1e-06)
    record("three_leg_p2_reserves", (501_992.0139620798, 398_417.455650024),
           (float(committed.post_pools["p2"]["reserve0"]), float(committed.post_pools["p2"]["reserve1"])), 1e-06)
    record("three_leg_p3_reserves", (801_582.544349976, 598_818.98172076),
           (float(committed.post_pools["p3"]["reserve0"]), float(committed.post_pools["p3"]["reserve1"])), 1e-06)

    # 4. Sequential pool change on a reused pool: the second leg consumes updated reserves.
    repeated = checker.check(
        Proposal("QA", 0, 0, ("ETH", "DAI", "ETH"), ("q1", "q1"), 0.2, 1000.0, "qa"),
        {"q1": _qa_pool("ETH", "DAI", 1_000_000.0, 2_000_000.0)},
        wallet,
        prices,
        0.5,
    )
    record("reused_pool_status", "rejected", repeated.status)
    record("reused_pool_constraint", "C1", repeated.constraint)
    record("reused_pool_final_amount", 994.0149491943662, repeated.final_amount)
    record("reused_pool_after_gas_profit", -6.4850508056338185, repeated.after_gas_profit_usd)

    # 5. Atomic rejection: a profitable route rejected by an unaffordable gas bill.
    expensive_gas = checker.check(
        Proposal("QA", 0, 0, ("ETH", "DAI", "USDC", "ETH"), ("p1", "p2", "p3"), 0.2, 1000.0, "qa"),
        pools,
        wallet,
        prices,
        1000.0,
    )
    record("c1_rejection_status", "rejected", expensive_gas.status)
    record("c1_rejection_constraint", "C1", expensive_gas.constraint)
    record("c1_rejection_after_gas_profit", -818.9817207599845, expensive_gas.after_gas_profit_usd)
    record("c1_rejection_state_unchanged", True, expensive_gas.post_pools is None)

    # 6. C6 expiry: the profitable trade is rejected before commit and leaves state intact.
    expired = checker.check(
        Proposal("QA", 0, 0, ("ETH", "DAI", "USDC", "ETH"), ("p1", "p2", "p3"), 0.2, 1000.0, "qa"),
        pools,
        wallet,
        prices,
        0.5,
        c6_exceeded=True,
    )
    record("c6_rejection_status", "rejected", expired.status)
    record("c6_rejection_constraint", "C6", expired.constraint)
    record("c6_rejection_state_unchanged", True, expired.post_pools is None)

    # 7. Continuity violation: the first leg's token is not part of the pool.
    continuity = checker.check(
        Proposal("QA", 0, 0, ("ETH", "DAI", "ETH"), ("s2", "s2"), 0.2, 1000.0, "qa"),
        {"s2": _qa_pool("DAI", "USDC", 500_000.0, 400_000.0)},
        wallet,
        prices,
        0.5,
    )
    record("continuity_status", "invalid_proposal", continuity.status)
    record("continuity_constraint", "C4", continuity.constraint)
    record("continuity_state_unchanged", True, continuity.post_pools is None)
    return rows

def policy_route_scores(env: Any, obs: np.ndarray, model: Any) -> Tuple[np.ndarray, np.ndarray]:
    """PPO candidate source: categorical path logits and route-specific amount fractions."""
    policy = model.policy
    with th.no_grad():
        logits, mu, _log_std, _values = policy._policy_outputs(
            th.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        )
    logits = logits[0].detach().cpu().numpy().astype(float)
    mu = mu[0].detach().cpu().numpy().astype(float)
    distribution = policy.action_dist
    fractions = np.clip(
        float(distribution.sampling_min)
        + float(distribution.amount_span) / (1.0 + np.exp(-np.clip(mu, -60.0, 60.0))),
        float(env.amount_min),
        float(env.amount_max),
    )
    return logits, fractions


def candidate_order(logits: np.ndarray, plan_routes: Optional[Sequence[int]]) -> List[int]:
    """PPO order, with saved-plan routes promoted ahead of the remaining candidates."""
    order = [int(i) for i in np.argsort(-np.asarray(logits, dtype=float), kind="stable")]
    if not plan_routes:
        return order
    preferred = [int(i) for i in plan_routes if 0 <= int(i) < len(order)]
    rest = [index for index in order if index not in set(preferred)]
    return preferred + rest


def first_feasible(
    env: Any, order: Sequence[int], fractions: np.ndarray, exclude_route: Optional[int] = None
) -> Optional[Tuple[int, float]]:
    """First candidate in order whose route-specific amount trades profitably right now."""
    excluded = set() if exclude_route is None else {int(exclude_route)}
    for index in order:
        index = int(index)
        if index in excluded:
            continue
        fraction = float(fractions[index])
        path = env.triangular_paths[index]
        amount = fraction * float(env.current_balance.get(path[0], 0.0))
        if amount <= 0.0:
            continue
        if env._simulate_path(path, amount, mutate=False).get("success"):
            return (index, fraction)
    return None


def encode_route_message(sender: str, route_index: int, round_index: int) -> Tuple[int, str, float]:
    """C1 -> C2 route message; returns (bytes, payload, encode milliseconds)."""
    started = time.perf_counter()
    text = json.dumps(
        {"sender": sender, "route_index": int(route_index), "round": int(round_index)},
        separators=(",", ":"),
    )
    return len(text.encode("utf-8")), text, (time.perf_counter() - started) * 1000.0


def decode_route_message(text: str) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    payload = json.loads(text)
    return dict(payload), (time.perf_counter() - started) * 1000.0


def build_agent_envs(model: Any, dataset_dir: Any, scenario: Scenario, seed: int) -> Dict[str, Any]:
    """Four agent environments: own wallets, one matched initial market state."""
    envs: Dict[str, Any] = {}
    for agent in AGENTS:
        env = sim.make_env(make_env_args(dataset_dir, scenario), seed=seed, wallet_mode="cycle", algo="PPO")
        force_wallet_balance(env, scenario.balance_usd)
        scale_liquidity_depth(env, scenario.liquidity_depth)
        sim.configure_model_for_env(model, env, "PPO")
        sim.reset_env_compat(env, seed=seed)
        envs[agent] = env
    shared_gas = copy.copy(envs["C1"].current_gas)
    for env in envs.values():
        env.current_gas = copy.copy(shared_gas)
    return envs


def check_shared_state(env: Any, shared_pools: Mapping[str, Mapping[str, float]]) -> None:
    env.current_pools = copy.deepcopy(shared_pools)


def pools_match(
    expected: Mapping[str, Mapping[str, float]],
    actual: Mapping[str, Mapping[str, float]],
    pool_ids: Sequence[str],
    tolerance: float = 1e-06,
) -> bool:
    for pool_id in set(pool_ids):
        if pool_id not in actual or pool_id not in expected:
            return False
        for key in ("reserve0", "reserve1"):
            reference = float(expected[pool_id][key])
            observed = float(actual[pool_id][key])
            if abs(observed - reference) > tolerance * max(1.0, abs(reference)):
                return False
    return True


def arbitrage_kind(env: Any, pool_ids: Sequence[str]) -> str:
    dexes = {str(getattr(env.pool_template[pool_id], "dex_type", "unknown")) for pool_id in pool_ids}
    return "single_dex" if len(dexes) <= 1 else "cross_dex"


def common_stop_reason(env: Any) -> str:
    if env._timed_common_stop_state_key() != env._cached_common_stop_state_key:
        reason, best = env._timed_common_stop_check()
        env._cached_common_stop_reason = str(reason)
        env._cached_common_stop_best = copy.deepcopy(best)
        env._cached_common_stop_state_key = env._timed_common_stop_state_key()
    return str(env._cached_common_stop_reason) or "no_feasible_candidate"

def run_episode(
    model: Any,
    checker: CPMMChecker,
    args: argparse.Namespace,
    scenario: Scenario,
    seed: int,
    condition: str,
    plans: Mapping[str, Mapping[str, Any]],
    envs: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """One condition x setting x seed episode over the shared market state."""
    condition_agents = CONDITION_AGENTS[condition]
    plan_agents = set(CONDITION_PLANS[condition])
    team = [a for a in condition_agents if a in TEAM]
    rivals = [a for a in condition_agents if a in RIVALS]
    scenario_plans = plans.get(scenario.setting_id, {})

    for agent in condition_agents:
        sim.reset_env_compat(envs[agent], seed=seed)
    shared_pools = copy.deepcopy(envs["C1"].current_pools)
    shared_gas = copy.copy(envs["C1"].current_gas)

    active = list(condition_agents)
    stop_reasons = {a: "" for a in AGENTS}
    profit = {a: 0.0 for a in AGENTS}
    gross = {a: 0.0 for a in AGENTS}
    gas_paid = {a: 0.0 for a in AGENTS}
    commits = {a: 0 for a in AGENTS}
    proposals = {a: 0 for a in AGENTS}
    kinds = {"single_dex": 0, "cross_dex": 0}
    invalid_proposals = rejected_trades = invalid_committed = checker_mismatches = 0
    duplicate_paths = pool_overlaps = plans_changed = message_changed = 0
    messages_sent = rival_first = 0
    message_overhead_ms = 0.0
    rival_first_case: Optional[Dict[str, Any]] = None
    inference_ms: List[float] = []
    decision_ms: List[float] = []
    transactions: List[Dict[str, Any]] = []
    round_index = 0

    while active and round_index < int(args.max_rounds):
        round_index += 1
        round_route: Dict[str, int] = {}
        round_pools: Dict[str, Tuple[str, ...]] = {}
        failed_team_routes: set = set()
        c1_message: Optional[Tuple[int, str, float]] = None
        round_overhead_ms = 0.0
        first_transaction = len(transactions)
        for agent in [a for a in condition_agents if a in active]:
            env = envs[agent]
            check_shared_state(env, shared_pools)
            env.current_gas = copy.copy(shared_gas)
            if env.cumulative_decision_time_seconds > float(scenario.c6_seconds):
                stop_reasons[agent] = "C6_wall_clock_budget_exhausted"
                active.remove(agent)
                continue
            plan = scenario_plans.get(agent) if agent in plan_agents else None
            plan_routes = [int(p["path_index"]) for p in plan["route_preferences"]] if plan else None
            exclude_route: Optional[int] = None
            message_used = message_ignored = False
            overhead_ms = 0.0
            obs = env._timed_observation()  # observation construction counts toward C6
            selection_started = time.perf_counter()
            env.resume_decision_clock()
            if agent == "C2" and condition in ("B3", "Proposed") and c1_message is not None:
                payload, decode_ms = decode_route_message(c1_message[1])
                overhead_ms = c1_message[2] + decode_ms
                if condition == "Proposed":
                    message_used = True
                    exclude_route = int(payload.get("route_index", -1))
                else:
                    message_ignored = True
            inference_started = time.perf_counter()
            logits, fractions = policy_route_scores(env, obs, model)
            inference_elapsed = (time.perf_counter() - inference_started) * 1000.0
            order = candidate_order(logits, plan_routes)
            pick = first_feasible(env, order, fractions, exclude_route=exclude_route)
            if pick is None and exclude_route is not None:
                pick = first_feasible(env, order, fractions)  # duplicate avoidance best effort
            if pick is None:
                env._finish_selection_timing()
                stop_reasons[agent] = common_stop_reason(env)
                active.remove(agent)
                continue
            path_index, fraction = pick
            action = env._action_from_path_amount(int(path_index), float(fraction))
            path_index, fraction, _action_audit = env._decode_action(action)  # exact submitted values
            plan_changed = False
            if plan_routes:
                base_pick = first_feasible(env, candidate_order(logits, None), fractions)
                plan_changed = bool(base_pick is not None and base_pick[0] != path_index)
                plans_changed += int(plan_changed)
            if message_used:
                plain_pick = first_feasible(env, order, fractions)
                message_changed += int(plain_pick is not None and plain_pick[0] != path_index)

            path = env.triangular_paths[path_index]
            proposal = Proposal(
                agent=agent,
                round_index=round_index,
                path_index=int(path_index),
                tokens=(path[0], path[1], path[2], path[0]),
                pool_ids=tuple(str(p) for p in path[3]),
                amount_fraction=float(fraction),
                amount_in=float(fraction) * float(env.current_balance.get(path[0], 0.0)),
                source=("plan" if plan_routes and int(path_index) in set(plan_routes) else "ppo"),
                plan_used=bool(plan_routes),
                message_used=bool(message_used),
                message_ignored=bool(message_ignored),
                plan_changed_route=bool(plan_changed),
                inference_ms=float(inference_elapsed),
                message_overhead_ms=float(overhead_ms),
            )
            proposals[agent] += 1
            round_route[agent] = int(path_index)
            round_pools[agent] = tuple(proposal.pool_ids)
            if agent == "C1" and condition in ("B3", "Proposed"):
                _bytes, text, encode_ms = encode_route_message("C1", int(path_index), round_index)
                c1_message = (_bytes, text, encode_ms)
                messages_sent += 1
            if agent == "C2":
                round_overhead_ms = overhead_ms
            c6_expired = env.cumulative_decision_time_seconds > float(scenario.c6_seconds)
            verdict = checker.check(
                proposal,
                shared_pools,
                env.current_balance,
                env.prices,
                env.current_gas.gas_cost_usd,
                c6_exceeded=c6_expired,
            )
            if verdict.status == "invalid_proposal":
                invalid_proposals += 1
            elif verdict.status == "rejected":
                rejected_trades += 1
            env_status, env_constraint, env_reason = "", "", ""
            rival_first_flag = False
            if verdict.status == "committed":
                _obs, _reward, done, info = sim.step_env_compat(env, action)
                env_status = str(info.get("execution_status", ""))
                env_constraint = str(info.get("constraint", ""))
                env_reason = str(info.get("reason", ""))
                if bool(info.get("success", False)):
                    net = float(info.get("net_profit_usd", 0.0))
                    if not pools_match(
                        verdict.post_pools, env.current_pools, proposal.pool_ids
                    ) or not math.isclose(net, verdict.after_gas_profit_usd, rel_tol=0.0, abs_tol=1e-06):
                        invalid_committed += 1
                    shared_pools = copy.deepcopy(env.current_pools)
                    shared_gas = copy.copy(env.current_gas)
                    profit[agent] += net
                    gross[agent] += float(info.get("gross_profit_usd", 0.0))
                    gas_paid[agent] += float(info.get("gas_charged_usd", env.current_gas.gas_cost_usd))
                    commits[agent] += 1
                    kinds[arbitrage_kind(env, proposal.pool_ids)] += 1
                    if agent in RIVALS and int(path_index) in failed_team_routes:
                        rival_first += 1
                        rival_first_flag = True
                        if rival_first_case is None:
                            rival_first_case = {
                                "condition": condition,
                                "setting_id": scenario.setting_id,
                                "seed": int(seed),
                                "round": int(round_index),
                                "rival": agent,
                                "route_index": int(path_index),
                                "route_tokens": ">".join(proposal.tokens),
                                "first_swap_input": float(proposal.amount_in),
                                "after_gas_profit_usd": net,
                                "note": "a competing searcher consumed an opportunity the team agent could not commit first",
                            }
                elif env_constraint == "C6":
                    rejected_trades += 1
                else:
                    checker_mismatches += 1  # the simulator refused a trade the checker accepted
                if bool(done) or info.get("stop_reason"):
                    stop_reasons[agent] = str(info.get("stop_reason") or "terminated")
                    active.remove(agent)
            elif agent in TEAM:
                failed_team_routes.add(int(path_index))
            decision_ms.append((time.perf_counter() - selection_started) * 1000.0)
            inference_ms.append(float(inference_elapsed))

            transactions.append(
                {
                    "condition": condition,
                    "dimension": scenario.dimension,
                    "ablation_value": scenario.value,
                    "ablation_label": scenario.label,
                    "seed": int(seed),
                    "round": int(round_index),
                    "agent": agent,
                    "role": ("team" if agent in TEAM else "rival"),
                    "proposal_source": proposal.source,
                    "plan_used": proposal.plan_used,
                    "message_used": proposal.message_used,
                    "message_ignored": proposal.message_ignored,
                    "path_index": int(path_index),
                    "path_tokens": ">".join(proposal.tokens),
                    "pool_ids": "|".join(proposal.pool_ids),
                    "arbitrage_type": arbitrage_kind(env, proposal.pool_ids),
                    "amount_fraction": float(fraction),
                    "amount_in": float(proposal.amount_in),
                    "checker_status": verdict.status,
                    "checker_constraint": verdict.constraint,
                    "checker_reason": verdict.reason,
                    "simulator_execution_status": env_status,
                    "simulator_constraint": env_constraint,
                    "simulator_reason": env_reason,
                    "gross_profit_usd": verdict.gross_profit_usd,
                    "gas_cost_usd": verdict.gas_cost_usd,
                    "after_gas_profit_usd": verdict.after_gas_profit_usd,
                    "plan_changed_route": bool(plan_changed),
                    "duplicate_of_team_route": False,
                    "pool_overlap_with_team": False,
                    "rival_consumed_first": bool(rival_first_flag),
                    "inference_ms": float(inference_elapsed),
                    "message_overhead_ms": float(overhead_ms),
                    "decision_ms": float(decision_ms[-1]),
                }
            )
        if "C1" in round_route and "C2" in round_route:
            same_route = round_route["C1"] == round_route["C2"]
            overlap = pool_overlap(round_pools["C1"], round_pools["C2"])
            if same_route:
                duplicate_paths += 1
            elif overlap:
                pool_overlaps += 1
            for row in transactions[first_transaction:]:
                if row["agent"] in TEAM:
                    row["duplicate_of_team_route"] = bool(same_route)
                    row["pool_overlap_with_team"] = bool(overlap)
        message_overhead_ms += round_overhead_ms

    max_rounds_hit = bool(active) and round_index >= int(args.max_rounds)
    for agent in active:
        stop_reasons[agent] = "loop_guard_max_rounds" if max_rounds_hit else stop_reasons[agent]
    team_profit = sum(profit[a] for a in team)
    rival_profit = sum(profit[a] for a in rivals)
    c6_values = [float(envs[a].cumulative_decision_time_seconds) for a in condition_agents]
    plan_sources = sorted(
        {str(scenario_plans[a].get("source", "unknown")) for a in plan_agents if a in scenario_plans}
    )
    row = {
        "condition": condition,
        "condition_label": CONDITION_NOTES[condition],
        "dimension": scenario.dimension,
        "ablation_value": scenario.value,
        "ablation_label": scenario.label,
        "balance_usd": scenario.balance_usd,
        "gas_multiplier": scenario.gas_multiplier,
        "liquidity_depth": scenario.liquidity_depth,
        "c6_seconds": scenario.c6_seconds,
        "seed": int(seed),
        "plan_source": "|".join(plan_sources) if plan_sources else "none",
        "rounds": round_index,
        "max_rounds_hit": bool(max_rounds_hit),
        "episode_valid": not (invalid_committed or checker_mismatches),
        "team_after_gas_profit_usd": team_profit,
        "c1_profit_usd": profit["C1"],
        "c2_profit_usd": profit["C2"],
        "a1_profit_usd": profit["A1"],
        "a2_profit_usd": profit["A2"],
        "rival_profit_usd": rival_profit,
        "team_gross_usd": sum(gross[a] for a in team),
        "team_gas_usd": sum(gas_paid[a] for a in team),
        "rival_gross_usd": sum(gross[a] for a in rivals),
        "rival_gas_usd": sum(gas_paid[a] for a in rivals),
        "team_win": bool(team_profit > rival_profit),
        "team_profit_share_pct": 100.0 * safe_div(team_profit, team_profit + rival_profit),
        "successful_trades_team": sum(commits[a] for a in team),
        "successful_trades_c1": commits["C1"],
        "successful_trades_c2": commits["C2"],
        "successful_trades_rival": sum(commits[a] for a in rivals),
        "successful_trades_a1": commits["A1"],
        "successful_trades_a2": commits["A2"],
        "single_dex_commits_team": kinds["single_dex"],
        "cross_dex_commits_team": kinds["cross_dex"],
        "cross_dex_share_pct_team": 100.0
        * safe_div(kinds["cross_dex"], kinds["single_dex"] + kinds["cross_dex"]),
        "proposals_team": sum(proposals[a] for a in team),
        "proposals_rival": sum(proposals[a] for a in rivals),
        "invalid_proposals": invalid_proposals,
        "rejected_trades": rejected_trades,
        "invalid_committed_trades": invalid_committed,
        "duplicate_path_proposals": float(duplicate_paths) if len(team) == 2 else float("nan"),
        "shared_pool_overlap_events": float(pool_overlaps) if len(team) == 2 else float("nan"),
        "plans_changed_route": plans_changed,
        "message_changed_route": message_changed,
        "messages_sent": messages_sent,
        "message_overhead_ms": message_overhead_ms,
        "rival_consumed_first": rival_first,
        "inference_ms_mean": finite_mean(inference_ms),
        "decision_ms_mean": finite_mean(decision_ms),
        "decision_ms_max": max(decision_ms) if decision_ms else float("nan"),
        "c6_used_s_mean": finite_mean(c6_values),
        "c6_used_s_max": max(c6_values) if c6_values else float("nan"),
        "checker_env_mismatches": checker_mismatches,
        "stop_reason_c1": stop_reasons["C1"],
        "stop_reason_c2": stop_reasons["C2"],
        "stop_reason_a1": stop_reasons["A1"],
        "stop_reason_a2": stop_reasons["A2"],
        "error_type": "",
        "error_message": "",
    }
    return row, transactions, rival_first_case


# ------------------------------------------------------------------ offline LLM planning
class PlanRequestError(RuntimeError):
    """Raised when an OpenRouter planning request or its checked reply cannot be used."""


def openrouter_settings() -> Dict[str, Any]:
    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)) or default)
        except ValueError:
            return default

    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY", "").strip(),
        "model": os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b",
        "max_tokens": _int_env("OPENROUTER_MAX_COMPLETION_TOKENS", 512),
        "reasoning_effort": os.environ.get("OPENROUTER_REASONING_EFFORT", "low").strip() or "low",
    }


def openrouter_chat(settings: Mapping[str, Any], system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """One OpenRouter chat-completions call (counted as one model request)."""
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": int(settings["max_tokens"]),
        "temperature": 0.0,
        "reasoning": {"effort": settings["reasoning_effort"]},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cosc723.coopmev-lite.local",
            "X-Title": "COSC723 CoopMEV-Lite planner",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001 - diagnostics only
            detail = ""
        raise PlanRequestError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PlanRequestError(f"transport error: {exc}") from exc
    message = ((body.get("choices") or [{}])[0]).get("message", {})
    return {
        "text": str(message.get("content") or ""),
        "usage": dict(body.get("usage") or {}),
        "id": str(body.get("id", "")),
    }


def route_catalogue(env: Any) -> List[Dict[str, Any]]:
    """Compact public route-pool catalogue handed to the offline planner."""
    catalogue: List[Dict[str, Any]] = []
    for index, (token_a, token_b, token_c, pool_ids) in enumerate(env.triangular_paths):
        dexes, fees = [], []
        for pool_id in pool_ids:
            pool = env.pool_template[pool_id]
            dexes.append("U" if str(pool.dex_type).startswith("UNI") else "S")
            fees.append(round(float(pool.fee), 4))
        catalogue.append(
            {
                "path_index": int(index),
                "tokens": f"{token_a}>{token_b}>{token_c}>{token_a}",
                "dexes": "".join(dexes),
                "fees": fees,
                "pools": [str(pool_id) for pool_id in pool_ids],
            }
        )
    return catalogue


def build_plan_prompt(
    agent: str, scenario: Scenario, env: Any, catalogue: Sequence[Mapping[str, Any]], limit: int
) -> Tuple[str, str]:
    if agent in TEAM:
        role = "team member; you may coordinate with your teammate through bounded route messages"
    else:
        role = "independent competing searcher; you cannot see private team messages"
    system = (
        f"You are {agent}, an integrated route-planning agent ({role}) in a local Uniswap "
        "V2 / SushiSwap V2 triangular-arbitrage simulation. You receive only public scenario and "
        "route-pool information, never outcomes, balances or evaluation seeds. Reply with exactly one "
        'JSON object and nothing else. The object must use the key "route_preferences": a list of at '
        f'most {limit} objects, each with an integer "path_index" copied from the route catalogue and a '
        'short "reason" (<=12 words). All path_index values must be distinct. Example reply: '
        '{"route_preferences":[{"path_index":12,"reason":"cheapest fees on all three legs"},'
        '{"path_index":30,"reason":"deepest funded starting reserve"}],'
        '"notes":"prefer liquid cross-DEX cycles"} '
        "Never invent route ids and never repeat a route id."
    )
    user_payload = {
        "schema_version": PLAN_SCHEMA,
        "agent": agent,
        "role": role,
        "setting": {
            "setting_id": scenario.setting_id,
            "dimension": scenario.dimension,
            "value": scenario.value,
            "balance_usd_per_agent": scenario.balance_usd,
            "gas_multiplier": scenario.gas_multiplier,
            "liquidity_depth_multiplier": scenario.liquidity_depth,
            "c6_block_time_budget_seconds": scenario.c6_seconds,
        },
        "market": {
            "pool_count": len(env.pool_list),
            "route_count": len(env.triangular_paths),
            "funded_tokens": list(env.wallet_start_tokens),
            "route_catalogue": list(catalogue),
        },
        "task": f"Rank at most {limit} distinct route ids you would prioritise for this setting.",
    }
    return system, json.dumps(user_payload, separators=(",", ":"))


PLAN_LIST_KEYS = (
    "route_preferences",
    "route_preference",
    "routes",
    "route_plan",
    "preferences",
    "plan",
    "paths",
)
PLAN_INDEX_KEYS = ("path_index", "path_id", "route_index", "route_id", "id", "index", "route")
PLAN_TOKEN_KEYS = ("tokens", "path_tokens", "route_tokens")
PLAN_REASON_KEYS = ("reason", "rationale", "why", "note", "explanation")


def _coerce_route_item(item: Any) -> Optional[Dict[str, Any]]:
    """Accept an int, a numeric string, or an object that carries a route id under a known key."""
    if isinstance(item, (int, np.integer)):
        return {"path_index": int(item)}
    if isinstance(item, str):
        try:
            return {"path_index": int(float(item))}
        except ValueError:
            return None
    if isinstance(item, Mapping):
        for key in PLAN_INDEX_KEYS:
            if key not in item:
                continue
            try:
                index = int(float(item[key]))
            except (TypeError, ValueError):
                continue
            entry: Dict[str, Any] = {"path_index": index}
            for token_key in PLAN_TOKEN_KEYS:
                if isinstance(item.get(token_key), str):
                    entry["tokens"] = str(item[token_key])
                    break
            for reason_key in PLAN_REASON_KEYS:
                if isinstance(item.get(reason_key), str):
                    entry["reason"] = str(item[reason_key])
                    break
            return entry
    return None


def _find_route_payload(payload: Any, depth: int = 3) -> Optional[Any]:
    """Depth-first search for the planner's route list under any known key."""
    if depth < 0:
        return None
    if isinstance(payload, Mapping):
        for key in PLAN_LIST_KEYS:
            if key in payload:
                return payload[key]
        for value in payload.values():
            if isinstance(value, Mapping):
                found = _find_route_payload(value, depth - 1)
                if found is not None:
                    return found
        return None
    if isinstance(payload, list):
        objects = [item for item in payload if isinstance(item, Mapping)]
        if objects and any(any(key in item for key in PLAN_INDEX_KEYS) for item in objects):
            return payload
        return None
    return None


def parse_plan_text(text: str) -> Dict[str, Any]:
    """Tolerantly normalise a planner reply into {"route_preferences": [...], "notes": str}."""
    raw_text = str(text or "")
    start, end = raw_text.find("{"), raw_text.rfind("}")
    list_start, list_end = raw_text.find("["), raw_text.rfind("]")
    candidates = []
    if 0 <= start < end:
        candidates.append(raw_text[start : end + 1])
    if 0 <= list_start < list_end:
        candidates.append(raw_text[list_start : list_end + 1])
    payload: Any = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        raise PlanRequestError("response contains no parseable JSON")
    notes = ""
    if isinstance(payload, Mapping):
        notes = str(payload.get("notes", payload.get("note", "")) or "")
    raw_list = payload if isinstance(payload, list) else _find_route_payload(payload)
    if raw_list is None:
        raise PlanRequestError("response JSON has no route list")
    entries: List[Dict[str, Any]] = []
    if isinstance(raw_list, list):
        for item in raw_list:
            coerced = _coerce_route_item(item)
            if coerced is not None:
                entries.append(coerced)
    if not entries:
        raise PlanRequestError("response JSON has no usable route ids")
    return {"route_preferences": entries, "notes": notes[:200]}

def normalize_plan(
    plan: Mapping[str, Any], agent: str, scenario: Scenario, env: Any, limit: int
) -> Dict[str, Any]:
    """Canonical plan: coerced route ids, duplicates removed, truncated, unmatched tokens dropped."""
    entries: List[Dict[str, Any]] = []
    for item in list(plan.get("route_preferences", []) or []):
        coerced = _coerce_route_item(item)
        if coerced is not None:
            entries.append(coerced)
    canonical: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in entries:
        index = int(entry["path_index"])
        if index in seen:
            continue
        seen.add(index)
        if 0 <= index < len(env.triangular_paths):
            token_a, token_b, token_c, _pools = env.triangular_paths[index]
            expected = f"{token_a}>{token_b}>{token_c}>{token_a}"
            if "tokens" in entry and str(entry["tokens"]) != expected:
                entry = {key: value for key, value in entry.items() if key != "tokens"}
        canonical.append(entry)
        if len(canonical) >= max(1, int(limit)):
            break
    normalized = dict(plan)
    normalized["route_preferences"] = canonical
    return normalized


def check_plan(
    plan: Mapping[str, Any], agent: str, scenario: Scenario, env: Any, limit: int
) -> List[str]:
    """Hard checks on a normalised plan: schema, agent, setting, route existence, coverage."""
    errors: List[str] = []
    prefs = plan.get("route_preferences")
    if not isinstance(prefs, list):
        return ["missing_route_preferences"]
    if str(plan.get("schema_version", "")) != PLAN_SCHEMA:
        errors.append("schema_version_mismatch")
    if str(plan.get("agent", "")) != agent:
        errors.append("agent_mismatch")
    if str(plan.get("setting_id", scenario.setting_id)) != scenario.setting_id:
        errors.append("setting_id_mismatch")
    if not prefs:
        errors.append("empty_route_preferences")
        return errors
    if len(prefs) > max(1, int(limit)):
        errors.append("too_many_preferences")
    seen: set = set()
    for position, item in enumerate(prefs):
        if not isinstance(item, Mapping):
            errors.append(f"preference_{position}_not_object")
            continue
        try:
            route = int(item.get("path_index"))
        except (TypeError, ValueError):
            errors.append(f"preference_{position}_bad_path_index")
            continue
        if not 0 <= route < len(env.triangular_paths):
            errors.append(f"preference_{position}_out_of_range")
            continue
        if route in seen:
            errors.append(f"preference_{position}_duplicate")
            continue
        seen.add(route)
    return errors


def plan_file(output_dir: Path, scenario: Scenario, agent: str) -> Path:
    return Path(output_dir) / "plans" / scenario.setting_id.replace(":", "_") / f"{agent}.json"


def prompt_file(output_dir: Path, scenario: Scenario, agent: str) -> Path:
    return Path(output_dir) / "prompts" / f"{scenario.setting_id.replace(':', '_')}_{agent}.json"


def load_saved_plans(
    output_dir: Path, scenarios: Sequence[Scenario], agents_needed: Sequence[str], env: Any, limit: int
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    saved: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for scenario in scenarios:
        entry: Dict[str, Mapping[str, Any]] = {}
        for agent in agents_needed:
            path = plan_file(output_dir, scenario, agent)
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                normalized = normalize_plan(payload, agent, scenario, env, limit)
                if not check_plan(normalized, agent, scenario, env, limit):
                    entry[agent] = normalized
        saved[scenario.setting_id] = entry
    return saved


def dev_fallback_plan(agent: str, scenario: Scenario, env: Any, limit: int) -> Dict[str, Any]:
    """Development smoke plan from the public pool snapshot; explicitly not an LLM decision."""
    ranked = []
    for index, (token_a, token_b, token_c, pool_ids) in enumerate(env.triangular_paths):
        product = 1.0
        for token_in, pool_id in zip((token_a, token_b, token_c), pool_ids):
            pool = env.pool_template[pool_id]
            if token_in == pool.token0:
                reserve_in, reserve_out = float(pool.reserve0), float(pool.reserve1)
            else:
                reserve_in, reserve_out = float(pool.reserve1), float(pool.reserve0)
            if reserve_in <= 0.0 or reserve_out <= 0.0:
                product = 0.0
                break
            product *= (1.0 - float(pool.fee)) * reserve_out / reserve_in
        ranked.append((product, index))
    top = [i for _score, i in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]
    return {
        "schema_version": PLAN_SCHEMA,
        "agent": agent,
        "setting_id": scenario.setting_id,
        "model": "dev_fallback(no_llm)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "dev_fallback",
        "route_preferences": [
            {
                "path_index": index,
                "tokens": ">".join(
                    (
                        env.triangular_paths[index][0],
                        env.triangular_paths[index][1],
                        env.triangular_paths[index][2],
                        env.triangular_paths[index][0],
                    )
                ),
                "reason": "development smoke fallback ranking",
            }
            for index in top
        ],
        "notes": "deterministic development plan; not an LLM decision",
    }


def read_usage_rows(usage_path: Path) -> List[Dict[str, Any]]:
    if not Path(usage_path).is_file():
        return []
    try:
        with Path(usage_path).open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return []


def archived_usage_rows(output_dir: Path) -> int:
    """Attempts recorded in archived usage files (previous campaigns), for honest reporting."""
    log_path = Path(output_dir) / "plan_usage_archive_log.json"
    if not log_path.is_file():
        return 0
    try:
        entries = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(sum(int(entry.get("rows", 0)) for entry in entries if isinstance(entry, Mapping)))


def rotate_usage_file(output_dir: Path) -> Optional[Path]:
    """Archive plan_usage.csv so a fresh campaign counts requests from zero."""
    usage_path = Path(output_dir) / "plan_usage.csv"
    if not usage_path.is_file():
        return None
    rows = read_usage_rows(usage_path)
    archived = Path(output_dir) / f"plan_usage.archived.{time.strftime('%Y%m%dT%H%M%S')}.csv"
    usage_path.rename(archived)
    log_path = Path(output_dir) / "plan_usage_archive_log.json"
    try:
        entries = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    except (OSError, json.JSONDecodeError):
        entries = []
    entries.append(
        {
            "archived_file": archived.name,
            "rows": len(rows),
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    dump_json(log_path, entries)
    print(f"[plans] archived {len(rows)} previous attempts to {archived.name}")
    return archived


def missing_plan_slots(
    scenarios: Sequence[Scenario],
    agents_needed: Sequence[str],
    saved: Mapping[str, Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> List[Tuple[Scenario, str]]:
    return [
        (scenario, agent)
        for scenario in scenarios
        for agent in agents_needed
        if saved.get(scenario.setting_id, {}).get(agent) is None
        or not plan_is_reusable(saved[scenario.setting_id][agent], settings)
    ]


def write_plan_campaign_summary(
    output_dir: Path,
    *,
    mode: str,
    planned_slots: int,
    request_ceiling: int,
    requests_this_campaign: int,
    failures_this_campaign: int,
    archived_requests: int,
    plans_present: int,
    missing: Sequence[Sequence[str]],
    seconds: float,
    model: str,
    usage_rows: Sequence[Mapping[str, Any]],
    plan_models: Sequence[str] = (),
) -> Dict[str, Any]:
    """Report actual planning usage: slots are the fairness unit, requests are counted exactly."""
    requests_by_agent: Dict[str, int] = {}
    ok_by_agent: Dict[str, int] = {}
    requests_by_setting: Dict[str, int] = {}
    for row in usage_rows:
        agent = str(row.get("agent", ""))
        setting = str(row.get("setting_id", ""))
        requests_by_agent[agent] = requests_by_agent.get(agent, 0) + 1
        requests_by_setting[setting] = requests_by_setting.get(setting, 0) + 1
        if str(row.get("ok", "")).strip().lower() in {"true", "1"}:
            ok_by_agent[agent] = ok_by_agent.get(agent, 0) + 1
    summary = {
        "mode": mode,
        "planned_slots": int(planned_slots),
        "request_ceiling": int(request_ceiling),
        "requests_this_campaign": int(requests_this_campaign),
        "requests_failed_this_campaign": int(failures_this_campaign),
        "requests_in_usage_file": int(len(list(usage_rows))),
        "requests_archived_previous_campaigns": int(archived_requests),
        "requests_actual_total": int(len(list(usage_rows))) + int(archived_requests),
        "plans_present": int(plans_present),
        "slots_missing": [list(item) for item in missing],
        "checked_plans_by_agent": ok_by_agent,
        "requests_by_agent": requests_by_agent,
        "requests_by_setting": requests_by_setting,
        "campaign_seconds": float(seconds),
        "model": ", ".join(sorted(set(plan_models))) if plan_models else model,
        "planner_environment_model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    dump_json(Path(output_dir) / "plan_campaign_summary.json", summary)
    return summary


def load_plan_campaign_summary(output_dir: Path) -> Dict[str, Any]:
    path = Path(output_dir) / "plan_campaign_summary.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def plan_policy_from_summary(args: argparse.Namespace, scenarios: Sequence[Scenario]) -> Dict[str, Any]:
    """Stable planning-policy record: the policy actually used, read from the persisted summary."""
    summary = load_plan_campaign_summary(args.output_dir)
    fallback_mode = "fill_complete" if bool(args.plan_fill_complete) else "strict"
    fallback_ceiling = (
        int(args.plan_max_requests) if bool(args.plan_fill_complete) else int(args.plan_budget)
    )
    return {
        "mode": str(summary.get("mode", fallback_mode)),
        "planned_slots": int(summary.get("planned_slots", len(scenarios) * len(AGENTS))),
        "request_ceiling": int(summary.get("request_ceiling", fallback_ceiling)),
    }


def plan_usage_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    """Non-frozen snapshot of actual planning usage, disclosed for the report."""
    summary = load_plan_campaign_summary(args.output_dir)
    return {
        key: summary.get(key, 0)
        for key in (
            "requests_this_campaign",
            "requests_failed_this_campaign",
            "requests_in_usage_file",
            "requests_archived_previous_campaigns",
            "requests_actual_total",
            "campaign_seconds",
            "model",
            "generated_at",
        )
    }


def plan_is_reusable(plan: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    """Cached plans are reused only from the same source; dev fallbacks never leak into real runs."""
    source = str(plan.get("source", ""))
    if source == "openrouter":
        return not settings["api_key"] or str(plan.get("model", "")) == str(settings["model"])
    return source == "dev_fallback" and not settings["api_key"]


def generate_or_load_plans(
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
    agents_needed: Sequence[str],
    env: Any,
    usage_path: Path,
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    """Reuse checked plans when present, otherwise request them until every slot is filled.

    Slots (one checked plan per agent x setting, 92 for the full campaign) are the fairness
    unit. By default the driver keeps retrying with backoff until all slots hold a checked
    plan, bounded by --plan-max-requests and by --plan-stalled-passes consecutive useless
    passes; --no-plan-fill-complete restores the legacy strict request ceiling. Request
    attempts, failures and tokens are reported separately in plan_usage.csv and
    plan_campaign_summary.json.
    """
    settings = openrouter_settings()
    limit = int(args.plan_prefs)
    catalogue = route_catalogue(env)
    if args.plan_usage_reset:
        rotate_usage_file(args.output_dir)
    saved = load_saved_plans(args.output_dir, scenarios, agents_needed, env, limit)
    usage_rows = read_usage_rows(usage_path)
    archived_requests = archived_usage_rows(args.output_dir)
    requests_this_campaign = 0
    failures_this_campaign = 0
    fill_mode = bool(args.plan_fill_complete)
    ceiling = int(args.plan_max_requests) if fill_mode else int(args.plan_budget)
    planned_slots = len(scenarios) * len(agents_needed)
    started_campaign = time.time()
    print(
        f"[plans] mode={'fill_complete' if fill_mode else 'strict'} slots={planned_slots} "
        f"cached={sum(len(entry) for entry in saved.values())} request_ceiling={ceiling} "
        f"archived_attempts={archived_requests}"
    )
    if fill_mode:
        print(
            "[plans] fill-complete: retrying (with backoff) until all "
            f"{planned_slots} plans are checked; every request is recorded in plan_usage.csv"
        )
    stalled_passes = 0
    while True:
        missing = missing_plan_slots(scenarios, agents_needed, saved, settings)
        if not missing:
            break
        progress = False
        for scenario, agent in missing:
            if requests_this_campaign >= ceiling:
                break
            if not settings["api_key"]:
                if args.allow_dev_plans:
                    plan = normalize_plan(
                        dev_fallback_plan(agent, scenario, env, limit), agent, scenario, env, limit
                    )
                    dump_json(plan_file(args.output_dir, scenario, agent), plan)
                    saved.setdefault(scenario.setting_id, {})[agent] = plan
                    progress = True
                    print(f"[plans] {scenario.setting_id} {agent}: development fallback plan (no LLM call)")
                    continue
                raise SystemExit(
                    "OPENROUTER_API_KEY is required to generate plans; export it or use "
                    "--allow-dev-plans for a clearly marked development smoke run."
                )
            system, user = build_plan_prompt(agent, scenario, env, catalogue, limit)
            prompt_sha = sha256_text(system + "\n" + user)
            dump_json(
                prompt_file(args.output_dir, scenario, agent),
                {"agent": agent, "setting_id": scenario.setting_id, "sha256": prompt_sha,
                 "system": system, "user": user, "model": settings["model"]},
            )
            per_slot_attempts = max(1, int(args.plan_attempts))
            for attempt in range(1, per_slot_attempts + 1):
                if requests_this_campaign >= ceiling:
                    break
                requests_this_campaign += 1
                started = time.perf_counter()
                ok, error, usage, plan = False, "", {}, None
                try:
                    reply = openrouter_chat(settings, system, user)
                    usage = dict(reply["usage"])
                    candidate = parse_plan_text(reply["text"])
                    candidate.update(
                        {
                            "schema_version": PLAN_SCHEMA,
                            "agent": agent,
                            "setting_id": scenario.setting_id,
                            "model": settings["model"],
                            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "source": "openrouter",
                            "response_id": reply.get("id", ""),
                            "prompt_sha256": prompt_sha,
                        }
                    )
                    plan = normalize_plan(candidate, agent, scenario, env, limit)
                    errors = check_plan(plan, agent, scenario, env, limit)
                    if errors:
                        raise PlanRequestError("plan check failed: " + "; ".join(errors))
                    ok = True
                except (PlanRequestError, ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                seconds = round(time.perf_counter() - started, 4)
                usage_rows.append(
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "setting_id": scenario.setting_id,
                        "agent": agent,
                        "model": settings["model"],
                        "attempt": attempt,
                        "ok": bool(ok),
                        "prompt_tokens": usage.get("prompt_tokens", ""),
                        "completion_tokens": usage.get("completion_tokens", ""),
                        "total_tokens": usage.get("total_tokens", ""),
                        "seconds": seconds,
                        "error": error[:300],
                        "prompt_sha256": prompt_sha,
                    }
                )
                write_csv(usage_path, usage_rows)  # persist every attempt for an auditable budget
                if ok:
                    plan["usage"] = {
                        key: usage.get(key, "") for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                    plan["generation_seconds"] = seconds
                    dump_json(plan_file(args.output_dir, scenario, agent), plan)
                    saved.setdefault(scenario.setting_id, {})[agent] = plan
                    progress = True
                    filled = sum(len(entry) for entry in saved.values())
                    print(
                        f"[plans] {scenario.setting_id} {agent}: checked in {seconds}s "
                        f"(plans {filled}/{planned_slots}, request {requests_this_campaign}/{ceiling})"
                    )
                    break
                failures_this_campaign += 1
                print(
                    f"[plans] {scenario.setting_id} {agent}: attempt {attempt} failed: {error} "
                    f"(request {requests_this_campaign}/{ceiling})"
                )
                time.sleep(min(10.0, max(0.0, float(args.plan_backoff_seconds)) * attempt))
        if requests_this_campaign >= ceiling:
            break
        if not progress:
            stalled_passes += 1
            still_missing = len(missing_plan_slots(scenarios, agents_needed, saved, settings))
            print(
                f"[plans] retry pass {stalled_passes} filled no slot ({still_missing} still missing); "
                f"request {requests_this_campaign}/{ceiling}"
            )
            if stalled_passes >= max(1, int(args.plan_stalled_passes)):
                break
            time.sleep(min(15.0, max(1.0, float(args.plan_backoff_seconds))))
        else:
            stalled_passes = 0
    remaining = missing_plan_slots(scenarios, agents_needed, saved, settings)
    plans_present = sum(len(entry) for entry in saved.values())
    plan_models = sorted(
        {
            str(plan.get("model", ""))
            for entry in saved.values()
            for plan in entry.values()
            if isinstance(plan, Mapping) and plan.get("model")
        }
    )
    write_plan_campaign_summary(
        args.output_dir,
        mode="fill_complete" if fill_mode else "strict",
        planned_slots=planned_slots,
        request_ceiling=ceiling,
        requests_this_campaign=requests_this_campaign,
        failures_this_campaign=failures_this_campaign,
        archived_requests=archived_requests,
        plans_present=plans_present,
        missing=[(scenario.setting_id, agent) for scenario, agent in remaining],
        seconds=round(time.time() - started_campaign, 2),
        model=settings["model"],
        usage_rows=usage_rows,
        plan_models=plan_models,
    )
    if remaining:
        listing = [f"{scenario.setting_id}:{agent}" for scenario, agent in remaining]
        raise SystemExit(
            f"PLANNING INCOMPLETE (a planner/API condition, not a code crash): {plans_present}/{planned_slots} "
            f"plans are checked after {requests_this_campaign} requests (ceiling {ceiling}) with model "
            f"{settings['model']!r}; no valid route JSON was obtained for {len(listing)} slot(s): "
            f"{listing[:10]}{' ...' if len(listing) > 10 else ''}. Checked plans are cached, so re-running "
            "'python driver.py --plans-only' only re-requests the missing slots; if this repeats, raise "
            "--plan-max-requests and/or change OPENROUTER_MODEL. The timed evaluation is not started."
        )
    return saved

# ------------------------------------------------------------------ schemas and reporting
RAW_FIELDS = (
    "condition", "condition_label", "dimension", "ablation_value", "ablation_label",
    "balance_usd", "gas_multiplier", "liquidity_depth", "c6_seconds", "seed", "plan_source",
    "rounds", "max_rounds_hit", "episode_valid", "team_after_gas_profit_usd",
    "c1_profit_usd", "c2_profit_usd", "a1_profit_usd", "a2_profit_usd", "rival_profit_usd",
    "team_gross_usd", "team_gas_usd", "rival_gross_usd", "rival_gas_usd", "team_win",
    "team_profit_share_pct", "successful_trades_team", "successful_trades_c1", "successful_trades_c2",
    "successful_trades_rival", "successful_trades_a1", "successful_trades_a2",
    "single_dex_commits_team", "cross_dex_commits_team", "cross_dex_share_pct_team",
    "proposals_team", "proposals_rival", "invalid_proposals", "rejected_trades",
    "invalid_committed_trades", "duplicate_path_proposals", "shared_pool_overlap_events",
    "plans_changed_route", "message_changed_route", "messages_sent", "message_overhead_ms",
    "rival_consumed_first", "inference_ms_mean", "decision_ms_mean", "decision_ms_max",
    "c6_used_s_mean", "c6_used_s_max", "checker_env_mismatches",
    "stop_reason_c1", "stop_reason_c2", "stop_reason_a1", "stop_reason_a2",
    "error_type", "error_message",
)
TRANSACTION_FIELDS = (
    "condition", "dimension", "ablation_value", "ablation_label", "seed", "round", "agent", "role",
    "proposal_source", "plan_used", "message_used", "message_ignored", "path_index", "path_tokens",
    "pool_ids", "arbitrage_type", "amount_fraction", "amount_in", "checker_status",
    "checker_constraint", "checker_reason", "simulator_execution_status", "simulator_constraint",
    "simulator_reason", "gross_profit_usd", "gas_cost_usd", "after_gas_profit_usd",
    "plan_changed_route", "duplicate_of_team_route", "pool_overlap_with_team",
    "rival_consumed_first", "inference_ms", "message_overhead_ms", "decision_ms",
)
ATLAS_METRICS = (
    "team_after_gas_profit_usd", "rival_profit_usd", "team_win", "team_profit_share_pct",
    "successful_trades_team", "cross_dex_share_pct_team", "duplicate_path_proposals",
    "shared_pool_overlap_events", "invalid_proposals", "rejected_trades",
    "invalid_committed_trades", "plans_changed_route", "rival_consumed_first",
    "decision_ms_mean", "message_overhead_ms",
)


def blank_episode_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {field: "" for field in RAW_FIELDS}
    integers = (
        "balance_usd", "seed", "rounds", "plans_changed_route", "message_changed_route",
        "messages_sent", "rival_consumed_first", "invalid_proposals", "rejected_trades",
        "invalid_committed_trades", "proposals_team", "proposals_rival", "successful_trades_team",
        "successful_trades_c1", "successful_trades_c2", "successful_trades_rival",
        "successful_trades_a1", "successful_trades_a2", "single_dex_commits_team",
        "cross_dex_commits_team", "checker_env_mismatches",
    )
    floats = (
        "team_after_gas_profit_usd", "c1_profit_usd", "c2_profit_usd", "a1_profit_usd",
        "a2_profit_usd", "rival_profit_usd", "team_gross_usd", "team_gas_usd", "rival_gross_usd",
        "rival_gas_usd", "team_profit_share_pct", "cross_dex_share_pct_team", "message_overhead_ms",
        "inference_ms_mean", "decision_ms_mean", "decision_ms_max", "c6_used_s_mean", "c6_used_s_max",
    )
    for field in integers:
        row[field] = 0
    for field in floats:
        row[field] = float("nan")
    row["gas_multiplier"] = 1.0
    row["liquidity_depth"] = 1.0
    row["c6_seconds"] = 12.0
    row["team_win"] = False
    row["episode_valid"] = False
    row["max_rounds_hit"] = False
    return row


def error_episode_row(scenario: Scenario, seed: int, condition: str, exc: BaseException) -> Dict[str, Any]:
    row = blank_episode_row()
    row.update(
        {
            "condition": condition,
            "condition_label": CONDITION_NOTES.get(condition, condition),
            "dimension": scenario.dimension,
            "ablation_value": scenario.value,
            "ablation_label": scenario.label,
            "balance_usd": scenario.balance_usd,
            "gas_multiplier": scenario.gas_multiplier,
            "liquidity_depth": scenario.liquidity_depth,
            "c6_seconds": scenario.c6_seconds,
            "seed": int(seed),
            "plan_source": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        }
    )
    return row

def aggregate_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, items in groups.items():
        record: Dict[str, Any] = dict(zip(keys, key))
        record["episodes"] = len(items)
        record["episodes_valid"] = sum(1 for item in items if item.get("episode_valid") is True)
        for metric in ATLAS_METRICS:
            record[metric] = finite_mean(
                item[metric] for item in items if isinstance(item.get(metric), (int, float))
            )
        out.append(record)
    return sorted(out, key=lambda item: tuple(str(item.get(key, "")) for key in keys))


def build_catalogue_env(args: argparse.Namespace, scenarios: Sequence[Scenario]) -> Any:
    base = scenarios[0] if scenarios else Scenario("balance", 2000.0, "2K", 2000, 1.0, 1.0, 12.0)
    env = sim.make_env(
        make_env_args(args.dataset_dir, base), seed=int(args.seeds[0]), wallet_mode="cycle", algo="PPO"
    )
    force_wallet_balance(env, base.balance_usd)
    scale_liquidity_depth(env, base.liquidity_depth)
    return env


def build_freeze_manifest(
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
    model_path: Path,
) -> Dict[str, Any]:
    code_files = [Path(__file__).resolve(), Path(sim.__file__).resolve()]
    for extra in ("chart.py", "driver.sh"):
        candidate = Path(__file__).with_name(extra)
        if candidate.is_file():
            code_files.append(candidate.resolve())
    dataset_dir = Path(args.dataset_dir)
    dataset_files = sorted(
        path.resolve()
        for path in dataset_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".xlsx", ".py"}
    )
    plan_hashes, prompt_hashes = {}, {}
    for scenario in scenarios:
        for agent in AGENTS:
            saved = plan_file(args.output_dir, scenario, agent)
            prompt = prompt_file(args.output_dir, scenario, agent)
            if saved.is_file():
                plan_hashes[str(saved)] = sha256_file(saved)
            if prompt.is_file():
                prompt_hashes[str(prompt)] = sha256_file(prompt)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_id": sim.PROTOCOL_ID,
        "seeds": [int(seed) for seed in args.seeds],
        "conditions": list(args.conditions),
        "queue_rule": QUEUE_RULE,
        "c6_semantics": (
            "C6 counts observation construction, message handling, PPO inference, candidate screening, "
            "CPMM simulation, checker validation, tentative state transition and execution bookkeeping "
            "per agent; offline plan generation is excluded"
        ),
        "plan_schema": PLAN_SCHEMA,
        "plan_budget": int(args.plan_budget),
        "plan_policy": plan_policy_from_summary(args, scenarios),
        "plan_usage": plan_usage_snapshot(args),
        "model_file": {"path": str(Path(model_path)), "sha256": sha256_file(model_path)},
        "code": {str(path): sha256_file(path) for path in code_files},
        "dataset": {str(path): sha256_file(path) for path in dataset_files},
        "plans": plan_hashes,
        "prompts": prompt_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": th.__version__,
        },
        "scenarios": [
            dataclasses.asdict(scenario) | {"setting_id": scenario.setting_id} for scenario in scenarios
        ],
    }


def verify_frozen(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if not manifest_path.is_file():
        return
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--require-frozen: unreadable {manifest_path}: {exc}") from exc
    for key in (
        "protocol_id",
        "seeds",
        "conditions",
        "model_file",
        "code",
        "dataset",
        "plans",
        "prompts",
        "plan_policy",
    ):
        if previous.get(key) != manifest.get(key):
            raise SystemExit(f"--require-frozen: recorded {key!r} differs from the frozen manifest")

def evaluate(
    args: argparse.Namespace,
    model: Any,
    checker: CPMMChecker,
    scenarios: Sequence[Scenario],
    conditions: Sequence[str],
    seeds: Sequence[int],
    plans: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Every condition x setting x seed case, streamed to raw_results.csv and transactions.csv."""
    rows: List[Dict[str, Any]] = []
    rival_first_case: Optional[Dict[str, Any]] = None
    total = len(scenarios) * len(seeds) * len(conditions)
    done = 0
    with CsvStream(args.output_dir / "raw_results.csv", RAW_FIELDS) as raw_stream, CsvStream(
        args.output_dir / "transactions.csv", TRANSACTION_FIELDS
    ) as tx_stream:
        for scenario in scenarios:
            envs = build_agent_envs(model, args.dataset_dir, scenario, int(seeds[0]))
            for seed in seeds:
                for condition in conditions:
                    try:
                        row, transactions, case = run_episode(
                            model, checker, args, scenario, int(seed), condition, plans, envs
                        )
                    except Exception as exc:  # noqa: BLE001 - one failed case must not stop the campaign
                        row = error_episode_row(scenario, int(seed), condition, exc)
                        transactions, case = [], None
                        print(f"[episode] {condition} {scenario.setting_id} seed {seed} ERROR: {exc!r}")
                    raw_stream.write(row)
                    for transaction in transactions:
                        tx_stream.write(transaction)
                    if case is not None and rival_first_case is None:
                        rival_first_case = case
                    rows.append(row)
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"[episode] {done}/{total} cases recorded")
    return rows, rival_first_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COSC723 CoopMEV-Lite driver (see the module docstring for the contract mapping).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("DATASET"))
    parser.add_argument("--model", type=Path, default=Path("best_model.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(40, 50)))
    parser.add_argument("--only-dimensions", nargs="+", default=None, help="development subset, e.g. gas balance")
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 keeps all 23 settings")
    parser.add_argument("--max-rounds", type=int, default=40, help="loop guard per episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-test", action="store_true", help="run the checker known-answer cases")
    parser.add_argument("--plans-only", action="store_true", help="generate/check plans and stop")
    parser.add_argument("--skip-plans", action="store_true", help="reuse cached plans; never call OpenRouter")
    parser.add_argument(
        "--allow-dev-plans",
        action="store_true",
        help="development smoke only: deterministic fallback plans when no API key is set",
    )
    parser.add_argument(
        "--plan-budget",
        type=int,
        default=92,
        help="planned request allowance for the campaign; also the strict-mode ceiling",
    )
    parser.add_argument("--plan-prefs", type=int, default=8, help="route preferences per saved plan")
    parser.add_argument("--plan-attempts", type=int, default=3, help="attempts per agent/setting slot in one pass")
    parser.add_argument(
        "--plan-fill-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retry until every agent x setting slot holds a checked plan (default; --no-plan-fill-complete restores the legacy strict ceiling)",
    )
    parser.add_argument(
        "--plan-max-requests",
        type=int,
        default=300,
        help="hard request ceiling in fill-complete mode (92 slots plus generous retry slack)",
    )
    parser.add_argument(
        "--plan-stalled-passes",
        type=int,
        default=3,
        help="stop only after this many consecutive retry passes that fill no slot (model or API appears broken)",
    )
    parser.add_argument("--plan-backoff-seconds", type=float, default=2.0, help="retry backoff base in seconds")
    parser.add_argument(
        "--plan-usage-reset",
        action="store_true",
        help="archive plan_usage.csv and count this campaign's requests from zero",
    )
    parser.add_argument("--plans-status", action="store_true", help="print checked-plan slot status and exit")
    parser.add_argument("--require-frozen", action="store_true", help="abort when the frozen manifest differs")
    return parser

def main() -> int:
    args = build_parser().parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.model).is_file():
        raise SystemExit(f"model file not found: {args.model} (pass --model best_model.zip)")
    scenarios = select_scenarios(build_scenarios(), args)
    seeds = [int(seed) for seed in args.seeds]
    conditions = [str(condition) for condition in args.conditions]
    write_csv(
        args.output_dir / "scenarios.csv",
        [dataclasses.asdict(scenario) | {"setting_id": scenario.setting_id} for scenario in scenarios],
    )
    checker = CPMMChecker()
    if args.self_test:
        rows = checker_self_test()
        write_csv(args.output_dir / "checker_selftest.csv", rows, ("case", "expected", "actual", "passed"))
        failures = [row for row in rows if not row["passed"]]
        print(f"[checker] {len(rows) - len(failures)}/{len(rows)} known-answer cases passed")
        for row in failures:
            print(f"[checker] FAIL {row['case']}: expected={row['expected']!r} actual={row['actual']!r}")
        if failures:
            return 1
        if not args.plans_only:
            return 0
    catalogue_env = build_catalogue_env(args, scenarios)
    agents_needed = sorted({agent for condition in conditions for agent in CONDITION_PLANS[condition]})
    if args.plans_status:
        status_settings = openrouter_settings()
        saved_plans = load_saved_plans(
            args.output_dir, scenarios, agents_needed, catalogue_env, int(args.plan_prefs)
        )
        pending = missing_plan_slots(scenarios, agents_needed, saved_plans, status_settings)
        total_slots = len(scenarios) * len(agents_needed)
        present = sum(len(entry) for entry in saved_plans.values())
        print(f"[plans] checked plans present: {present}/{total_slots}")
        if pending:
            print(
                "[plans] missing slots: "
                f"{[f'{scenario.setting_id}:{agent}' for scenario, agent in pending]}"
            )
        return 0
    plans: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    if agents_needed:
        usage_path = args.output_dir / "plan_usage.csv"
        if args.skip_plans:
            plans = load_saved_plans(
                args.output_dir, scenarios, agents_needed, catalogue_env, int(args.plan_prefs)
            )
            missing = [
                (scenario.setting_id, agent)
                for scenario in scenarios
                for agent in agents_needed
                if agent not in plans.get(scenario.setting_id, {})
            ]
            if missing:
                print(f"[plans] {len(missing)} cached plans missing; plan-dependent conditions drop out", file=sys.stderr)
                for example in missing[:5]:
                    print(f"[plans]   missing {example}", file=sys.stderr)
                conditions = [condition for condition in conditions if not CONDITION_PLANS[condition]]
                agents_needed = []
                if not conditions:
                    print("[plans] no runnable condition left", file=sys.stderr)
                    return 2
        else:
            plans = generate_or_load_plans(args, scenarios, agents_needed, catalogue_env, usage_path)
    if args.plans_only:
        summary = load_plan_campaign_summary(args.output_dir)
        print(
            f"[plans] saved under {args.output_dir / 'plans'} | plans {summary.get('plans_present', '?')}"
            f"/{summary.get('planned_slots', '?')} | requests this campaign "
            f"{summary.get('requests_this_campaign', '?')} (failed "
            f"{summary.get('requests_failed_this_campaign', '?')}, ceiling "
            f"{summary.get('request_ceiling', '?')})"
        )
        print(f"[plans] campaign summary: {args.output_dir / 'plan_campaign_summary.json'}")
        return 0
    args.conditions = conditions
    manifest = build_freeze_manifest(args, scenarios, args.model)
    if args.require_frozen:
        verify_frozen(manifest, args.output_dir / "freeze_manifest.json")
    dump_json(args.output_dir / "freeze_manifest.json", manifest)
    settings = openrouter_settings()
    dump_json(
        args.output_dir / "config.json",
        {
            "protocol_id": sim.PROTOCOL_ID,
            "observation_dim": int(sim.OBSERVATION_DIM),
            "conditions": {condition: CONDITION_NOTES[condition] for condition in conditions},
            "condition_agents": {condition: list(CONDITION_AGENTS[condition]) for condition in conditions},
            "condition_plans": {condition: list(CONDITION_PLANS[condition]) for condition in conditions},
            "queue_rule": QUEUE_RULE,
            "c6_semantics": manifest["c6_semantics"],
            "seeds": seeds,
            "settings": manifest["scenarios"],
            "planner": {
                "model": settings["model"],
                "max_completion_tokens": settings["max_tokens"],
                "reasoning_effort": settings["reasoning_effort"],
                "plan_budget": int(args.plan_budget),
                "plan_preferences": int(args.plan_prefs),
                "offline": True,
                "plan_policy": manifest["plan_policy"],
                "plan_campaign": load_plan_campaign_summary(args.output_dir),
            },
            "checker_selftest": "results/checker_selftest.csv",
            "max_rounds": int(args.max_rounds),
        },
    )
    print(f"[driver] loading frozen checkpoint {args.model} on {args.device}")
    model = PPO.load(str(args.model), device=str(args.device))
    rows, rival_first_case = evaluate(args, model, checker, scenarios, conditions, seeds, plans)
    dump_json(
        args.output_dir / "week4_rival_first_case.json",
        rival_first_case
        if rival_first_case is not None
        else {"note": "no rival-first case observed in this run"},
    )
    write_csv(
        args.output_dir / "summary_by_cell.csv",
        aggregate_rows(rows, ("condition", "dimension", "ablation_value", "ablation_label")),
    )
    write_csv(args.output_dir / "overall_by_condition.csv", aggregate_rows(rows, ("condition",)))
    error_cases = sum(1 for row in rows if str(row.get("error_type", "")))
    invalid_committed = sum(int(row.get("invalid_committed_trades") or 0) for row in rows)
    invalid_proposals = sum(int(row.get("invalid_proposals") or 0) for row in rows)
    rejected = sum(int(row.get("rejected_trades") or 0) for row in rows)
    print(
        f"[done] {len(rows)} cases | error cases: {error_cases} | invalid proposals: {invalid_proposals} "
        f"| rejected trades: {rejected} | invalid committed trades: {invalid_committed}"
    )
    print(f"[done] raw results written to {args.output_dir / 'raw_results.csv'}")
    return 0 if rows else 3


if __name__ == "__main__":
    sys.exit(main())

