#!/usr/bin/env python3
"""
The total is
split across a selected set of funding tokens, and the generated JSON validates
that sum(value_usd) == budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

DEFAULT_PRICE_FILE = "token_prices_2026-06-27.json"
DEFAULT_BUDGETS = "1000:10000:1000"
DEFAULT_TOKENS = ("ethereum", "usd-coin", "tether", "dai", "wrapped-bitcoin")


def parse_budgets(text: str) -> List[int]:
    text = str(text).strip()
    if not text:
        raise ValueError("--budgets must not be empty")
    if ":" in text:
        parts = [int(float(x)) for x in text.split(":")]
        if len(parts) != 3:
            raise ValueError("Budget range must be start:stop:step, for example 1000:10000:1000")
        start, stop, step = parts
        if step <= 0 or stop < start:
            raise ValueError("Invalid budget range")
        return list(range(start, stop + 1, step))
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def parse_tokens(text: str) -> List[str]:
    if not text:
        return list(DEFAULT_TOKENS)
    tokens = [x.strip() for x in text.split(",") if x.strip()]
    if not tokens:
        raise ValueError("--tokens resolved to an empty list")
    return tokens


def parse_weights(text: str, n: int) -> List[float]:
    if not text:
        return [1.0] * n
    weights = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(weights) != n:
        raise ValueError(f"--weights has {len(weights)} entries but {n} tokens were selected")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("--weights must be non-negative and sum to a positive value")
    return weights


def load_prices(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    prices: Dict[str, float] = {}
    for token, price_data in data.get("prices", {}).items():
        try:
            price = float(price_data.get("usd", 0.0))
        except Exception:
            price = 0.0
        if price > 0:
            prices[token] = price
    if not prices:
        raise ValueError(f"No positive USD prices found in {path}")
    return prices


def select_available_tokens(requested_tokens: Sequence[str], prices: Mapping[str, float]) -> List[str]:
    selected = [token for token in requested_tokens if token in prices and prices[token] > 0]
    if selected:
        return selected
    # Conservative fallback: choose the most standard stable/blue-chip tokens available.
    priority = list(DEFAULT_TOKENS) + sorted(prices)
    seen = set()
    fallback: List[str] = []
    for token in priority:
        if token in seen:
            continue
        seen.add(token)
        if token in prices and prices[token] > 0:
            fallback.append(token)
        if len(fallback) >= min(5, len(prices)):
            break
    if not fallback:
        raise ValueError("Could not select any wallet funding tokens from the price file")
    return fallback


def make_wallet(
    budget_usd: float,
    selected_tokens: Sequence[str],
    weights: Sequence[float],
    prices: Mapping[str, float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    total_weight = float(sum(weights))
    wallet: Dict[str, Dict[str, float]] = {}
    normalized_weights: Dict[str, float] = {}
    running_total = 0.0
    for idx, token in enumerate(selected_tokens):
        weight = float(weights[idx]) / total_weight
        value_usd = float(budget_usd) * weight
        price_usd = float(prices[token])
        amount = value_usd / price_usd
        wallet[token] = {
            "amount": amount,
            "value_usd": value_usd,
            "price_usd": price_usd,
            "allocation_weight": weight,
        }
        normalized_weights[token] = weight
        running_total += value_usd
    tolerance = max(1e-6, float(budget_usd) * 1e-9)
    if abs(running_total - float(budget_usd)) > tolerance:
        raise AssertionError(f"Wallet total {running_total} does not match budget {budget_usd}")
    return wallet, normalized_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wallet_balance_YYYY-MM-DD.BUDGET.json files with total-wallet budget semantics.")
    parser.add_argument("--token-prices", default=DEFAULT_PRICE_FILE, help="Token price JSON file")
    parser.add_argument("--output-dir", default=".", help="Directory to write wallet JSON files")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS, help="Budget list '1000,2000' or inclusive range '1000:10000:1000'")
    parser.add_argument("--tokens", default=",".join(DEFAULT_TOKENS), help="Comma-separated CoinGecko token IDs to fund")
    parser.add_argument("--weights", default="", help="Optional comma-separated allocation weights matching --tokens. Default equal weight.")
    args = parser.parse_args()

    price_path = Path(args.token_prices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with price_path.open("r", encoding="utf-8") as handle:
        input_data = json.load(handle)
    prices = load_prices(price_path)
    requested_tokens = parse_tokens(args.tokens)
    selected_tokens = select_available_tokens(requested_tokens, prices)
    weights = parse_weights(args.weights, len(requested_tokens))
    if len(selected_tokens) != len(requested_tokens):
        # Drop weights for unavailable requested tokens and revert to equal weights.
        weights = [1.0] * len(selected_tokens)

    budgets = parse_budgets(args.budgets)
    date = input_data.get("date", "unknown-date")
    for budget in budgets:
        wallet, normalized_weights = make_wallet(float(budget), selected_tokens, weights, prices)
        validation_total = float(sum(v["value_usd"] for v in wallet.values()))
        output_data = {
            "date": date,
            "wallet_metadata": {
                "generator": "wallet_balance_generator_1K_10K.py",
                "capital_semantics": "total_wallet_budget_usd_split_across_selected_tokens",
                "total_budget_usd": float(budget),
                "allocation_tokens": list(selected_tokens),
                "allocation_weights": normalized_weights,
                "validation_total_value_usd": validation_total,
                "note": "The total budget is split across selected tokens; it is not repeated for every token in the price universe.",
            },
            "wallet": wallet,
        }
        output_filename = output_dir / f"wallet_balance_{date}.{budget}.json"
        with output_filename.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=4, sort_keys=True)
        print(f"Created: {output_filename} | total=${validation_total:.2f} | tokens={','.join(selected_tokens)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
