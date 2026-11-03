#!/usr/bin/env python3
"""COSC723 CoopMEV-Lite figures.

Reads results/raw_results.csv (and, when present, results/transactions.csv) and writes the
approved COSC723_* atlas heatmaps, the paired Proposed-minus-B2 bootstrap intervals using
complete seed groups, and the resampling-unit figure. No evaluation is executed here.

Run: python chart.py --input results/raw_results.csv --output-dir results/figures
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CONDITIONS = ("B1", "B2", "B3", "Proposed", "PPO_no_plan")
CONDITION_LABELS = {
    "B1": "B1: integrated agent vs rival (context)",
    "B2": "B2: communication OFF (control)",
    "B3": "B3: message sent and ignored",
    "Proposed": "Proposed: communication ON",
    "PPO_no_plan": "Proposed without LLM plans",
}
DIMENSIONS = ("balance", "latency", "gas", "liquidity")
DIMENSION_TITLES = {
    "balance": "Wallet balance (USD per agent)",
    "latency": "Block-time budget C6 (s)",
    "gas": "Gas price multiplier",
    "liquidity": "Liquidity depth multiplier",
}
VALUE_ORDER = {
    "balance": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0],
    "latency": [4.0, 6.0, 8.0, 10.0, 12.0],
    "gas": [1.0, 2.0, 4.0, 8.0],
    "liquidity": [0.25, 0.5, 0.75, 1.0],
}
ALPHA = 0.05
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 723
REQUIRED_COLUMNS = (
    "condition", "dimension", "ablation_value", "ablation_label", "seed",
    "team_after_gas_profit_usd", "rival_profit_usd", "team_win", "team_profit_share_pct",
    "successful_trades_team", "cross_dex_share_pct_team", "duplicate_path_proposals",
    "shared_pool_overlap_events", "invalid_proposals", "rejected_trades",
    "invalid_committed_trades", "plans_changed_route", "rival_consumed_first",
    "decision_ms_mean", "message_overhead_ms",
)


def load_runs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    frame["condition"] = frame["condition"].astype(str)
    frame["ablation_value"] = pd.to_numeric(frame["ablation_value"], errors="coerce")
    return frame


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per condition x setting with the mean of every approved metric."""
    keys = ["condition", "dimension", "ablation_value", "ablation_label"]
    grouped = raw.groupby(keys, as_index=False)
    cells = grouped.agg(
        episodes=("seed", "size"),
        mean_profit=("team_after_gas_profit_usd", "mean"),
        mean_rival_profit=("rival_profit_usd", "mean"),
        team_win_pct=("team_win", lambda values: 100.0 * float(pd.to_numeric(values, errors="coerce").mean())),
        mean_profit_share_pct=("team_profit_share_pct", "mean"),
        mean_successful_trades=("successful_trades_team", "mean"),
        mean_cross_dex_pct=("cross_dex_share_pct_team", "mean"),
        mean_duplicate_paths=("duplicate_path_proposals", "mean"),
        mean_pool_overlaps=("shared_pool_overlap_events", "mean"),
        mean_invalid_proposals=("invalid_proposals", "mean"),
        mean_rejected_trades=("rejected_trades", "mean"),
        mean_invalid_committed=("invalid_committed_trades", "mean"),
        mean_plans_changed=("plans_changed_route", "mean"),
        mean_rival_first=("rival_consumed_first", "mean"),
        mean_decision_ms=("decision_ms_mean", "mean"),
        mean_message_overhead_ms=("message_overhead_ms", "mean"),
    )
    return cells


def configure_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure: Any, output_dir: Path, filename: str, formats: Sequence[str], dpi: int) -> None:
    import matplotlib.pyplot as plt

    for suffix in formats:
        figure.savefig(output_dir / f"{filename}.{suffix}", dpi=dpi)
    plt.close(figure)
    print(f"[chart] wrote {filename} ({','.join(formats)})")


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)

def atlas(
    cells: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    cmap: str = "YlOrRd",
    fmt: str = "{:,.2f}",
    conditions: Sequence[str] = CONDITIONS,
) -> None:
    """One 1x4 heatmap row per metric: rows are conditions, columns are sweep values."""
    import matplotlib.pyplot as plt

    values = pd.to_numeric(cells[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad("#e6e6e6")
    widths = [len(VALUE_ORDER[dimension]) for dimension in DIMENSIONS]
    figure, axes = plt.subplots(
        1,
        len(DIMENSIONS),
        figsize=(3.05 * len(DIMENSIONS) + 0.9, 0.60 * len(conditions) + 2.1),
        gridspec_kw={"width_ratios": widths, "wspace": 0.14},
        dpi=dpi,
    )
    images = []
    for axis, dimension in zip(axes, DIMENSIONS):
        levels = VALUE_ORDER[dimension]
        matrix = np.full((len(conditions), len(levels)), np.nan)
        for row_index, condition in enumerate(conditions):
            for column_index, value in enumerate(levels):
                match = cells[
                    (cells.condition == condition)
                    & (cells.dimension == dimension)
                    & np.isclose(cells.ablation_value, value)
                ]
                if len(match):
                    matrix[row_index, column_index] = float(
                        pd.to_numeric(match[column], errors="coerce").iloc[0]
                    )
        images.append(axis.imshow(matrix, cmap=colour_map, vmin=vmin, vmax=vmax, aspect="auto"))
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    colour = "white" if value > vmin + 0.68 * (vmax - vmin) else "black"
                    axis.text(column_index, row_index, fmt.format(value), ha="center", va="center",
                              fontsize=7.4, color=colour)
                else:
                    axis.text(column_index, row_index, "n/a", ha="center", va="center",
                              fontsize=7.0, color="#666666")
        labels = (
            [condition_label(condition) for condition in conditions]
            if dimension == DIMENSIONS[0]
            else ["" for _ in conditions]
        )
        axis.set_yticks(range(len(conditions)))
        axis.set_yticklabels(labels, fontsize=7.6)
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels([f"{level:g}" for level in levels], fontsize=7.6)
        axis.set_title(DIMENSION_TITLES[dimension], fontsize=9.0)
        axis.set_xlabel("setting value", fontsize=7.8)
    figure.colorbar(images[-1], ax=list(axes), fraction=0.012, pad=0.01)
    figure.suptitle(title, fontsize=10.0, y=1.0)
    save_figure(figure, output_dir, filename, formats, dpi)


def profit_time_figure(
    raw: pd.DataFrame, output_dir: Path, formats: Sequence[str], dpi: int, conditions: Sequence[str]
) -> None:
    import matplotlib.pyplot as plt

    overall = (
        raw.groupby("condition", as_index=False)
        .agg(
            profit=("team_after_gas_profit_usd", "mean"),
            decision_ms=("decision_ms_mean", "mean"),
            episodes=("seed", "size"),
        )
        .set_index("condition")
        .reindex(list(conditions))
        .reset_index()
    )
    markers = {"B1": "s", "B2": "o", "B3": "^", "Proposed": "D", "PPO_no_plan": "v"}
    colours = {
        "B1": "#0072B2",
        "B2": "#00A087",
        "B3": "#E69F00",
        "Proposed": "#D62728",
        "PPO_no_plan": "#7A5195",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=dpi)
    for axis in axes:
        for _, point in overall.iterrows():
            condition = str(point["condition"])
            axis.scatter(
                point["decision_ms"],
                point["profit"],
                s=95,
                marker=markers.get(condition, "o"),
                color=colours.get(condition, "#333333"),
                edgecolor="white",
                linewidth=0.6,
                label=condition_label(condition),
                zorder=3,
            )
        axis.grid(alpha=0.25, linestyle=":")
        axis.set_xlabel("Mean online decision time (ms)")
        axis.set_ylabel("Mean combined team after-gas profit (USD)")
    axes[0].set_title("All conditions")
    cluster = overall[overall.condition != "B1"]
    if cluster.empty:
        cluster = overall

    def _limits(series: pd.Series) -> Optional[Tuple[float, float]]:
        data = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return None
        low, high = float(data.min()), float(data.max())
        span = high - low
        if span <= 0.0:
            span = max(abs(high), 1.0) * 0.05
        return low - 0.35 * span, high + 0.35 * span

    xlim, ylim = _limits(cluster["decision_ms"]), _limits(cluster["profit"])
    if xlim:
        axes[1].set_xlim(*xlim)
    if ylim:
        axes[1].set_ylim(*ylim)
    axes[1].set_title("Expanded cluster (communication comparisons)")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("Profit against online decision time by condition", fontsize=10.0)
    save_figure(figure, output_dir, "COSC723_Fig10_profit_time_expanded_cluster", formats, dpi)

def paired_frame(raw: pd.DataFrame, condition_a: str, condition_b: str) -> pd.DataFrame:
    """Matched ON-minus-OFF differences per setting and seed (the final evaluation contract)."""
    columns = ["dimension", "ablation_value", "ablation_label", "seed", "team_after_gas_profit_usd"]
    left = raw.loc[raw.condition == condition_a, columns]
    right = raw.loc[raw.condition == condition_b, columns]
    merged = left.merge(
        right, on=["dimension", "ablation_value", "ablation_label", "seed"], suffixes=("_a", "_b")
    )
    if merged.empty:
        raise SystemExit(f"no matched rows for {condition_a} minus {condition_b}")
    merged["difference"] = merged["team_after_gas_profit_usd_a"] - merged["team_after_gas_profit_usd_b"]
    return merged


def seed_group_means(paired: pd.DataFrame) -> pd.Series:
    """The bootstrap resampling unit: the complete seed group of all paired settings."""
    return paired.groupby("seed")["difference"].mean()


def bootstrap_interval(
    values: Iterable[float], reps: int = BOOTSTRAP_REPS, alpha: float = ALPHA
) -> Tuple[float, float, float, np.ndarray]:
    data = np.asarray([float(value) for value in values], dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan"), float("nan"), np.asarray([])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = data[generator.integers(0, data.size, size=(reps, data.size))].mean(axis=1)
    return (
        float(data.mean()),
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
        draws,
    )


def paired_bootstrap_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows: List[Dict[str, Any]] = []
    for dimension in DIMENSIONS:
        subset = paired[paired.dimension == dimension]
        if subset.empty:
            continue
        mean, low, high, _draws = bootstrap_interval(seed_group_means(subset))
        rows.append(
            {
                "sweep": dimension,
                "label": DIMENSION_TITLES[dimension],
                "paired_cases": int(len(subset)),
                "seed_groups": int(subset.seed.nunique()),
                "mean_difference_usd": mean,
                "ci95_low_usd": low,
                "ci95_high_usd": high,
            }
        )
    mean, low, high, draws = bootstrap_interval(seed_group_means(paired))
    rows.append(
        {
            "sweep": "overall",
            "label": "All paired settings (complete seed groups)",
            "paired_cases": int(len(paired)),
            "seed_groups": int(paired.seed.nunique()),
            "mean_difference_usd": mean,
            "ci95_low_usd": low,
            "ci95_high_usd": high,
        }
    )
    table = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), dpi=dpi, gridspec_kw={"width_ratios": [1.3, 1.0]})
    positions = np.arange(len(table))
    axes[0].errorbar(
        table.mean_difference_usd,
        positions,
        xerr=[
            table.mean_difference_usd - table.ci95_low_usd,
            table.ci95_high_usd - table.mean_difference_usd,
        ],
        fmt="o",
        color="#D62728",
        ecolor="#444444",
        capsize=4,
    )
    axes[0].axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(table.label, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"Paired {condition_a} minus {condition_b} (USD per case)")
    axes[0].set_title("Paired differences with 95% bootstrap intervals\n(complete seed-group resampling)")
    axes[0].grid(axis="x", alpha=0.25, linestyle=":")
    if draws.size:
        axes[1].hist(draws, bins=60, color="#4C72B0", alpha=0.85)
        axes[1].axvline(mean, color="#D62728", linewidth=1.6, label=f"observed mean {mean:,.1f}")
        axes[1].axvline(low, color="#333333", linestyle="--", linewidth=1.1)
        axes[1].axvline(high, color="#333333", linestyle="--", linewidth=1.1,
                        label=f"95% CI [{low:,.1f}, {high:,.1f}]")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Bootstrap mean of complete seed groups (USD)")
    axes[1].set_ylabel("bootstrap draws")
    axes[1].set_title("Bootstrap distribution")
    figure.suptitle(f"Communication value: {condition_a} minus {condition_b} (paired ON minus OFF)", fontsize=10.0)
    save_figure(figure, output_dir, "COSC723_paired_proposed_minus_B2_bootstrap95", formats, dpi)
    return table

def seed_group_unit_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    condition_a: str,
    condition_b: str,
) -> None:
    import matplotlib.pyplot as plt

    groups = seed_group_means(paired)
    mean, low, high, draws = bootstrap_interval(groups)
    pivot = paired.pivot_table(index=["dimension", "ablation_label"], columns="seed", values="difference")
    matrix = pivot.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    figure = plt.figure(figsize=(13.0, 8.4), dpi=dpi)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.36, wspace=0.22)
    hot = figure.add_subplot(grid[:, 0])
    image = hot.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    hot.set_xticks(range(len(pivot.columns)))
    hot.set_xticklabels([str(seed) for seed in pivot.columns], fontsize=8)
    hot.set_yticks(range(len(pivot.index)))
    hot.set_yticklabels([f"{dimension}:{label}" for dimension, label in pivot.index], fontsize=6.6)
    hot.set_xlabel("seed: each column is one complete seed group (23 paired settings)")
    hot.set_title(f"Paired {condition_a} minus {condition_b} per setting and seed (USD)")
    figure.colorbar(image, ax=hot, fraction=0.028, pad=0.012)
    points = figure.add_subplot(grid[0, 1])
    positions = np.arange(len(groups))
    points.plot(positions, groups.to_numpy(dtype=float), "o-", color="#D62728", markersize=5, linewidth=1.1)
    points.axhline(mean, color="#333333", linestyle="--", linewidth=1.0)
    if np.isfinite(low):
        points.axhspan(low, high, color="#4C72B0", alpha=0.12, label=f"95% CI [{low:,.1f}, {high:,.1f}]")
        points.legend(fontsize=7.5)
    points.set_xticks(positions)
    points.set_xticklabels([str(int(seed)) for seed in groups.index], rotation=45, fontsize=8)
    points.set_xlabel("complete seed group (seed)")
    points.set_ylabel("group mean difference (USD)")
    points.set_title("Resampling unit: the complete seed group")
    points.grid(alpha=0.25, linestyle=":")
    hist = figure.add_subplot(grid[1, 1])
    if draws.size:
        hist.hist(draws, bins=55, color="#4C72B0", alpha=0.85)
        hist.axvline(mean, color="#D62728", linewidth=1.6)
        hist.axvline(low, color="#333333", linestyle="--", linewidth=1.1)
        hist.axvline(high, color="#333333", linestyle="--", linewidth=1.1)
    hist.set_xlabel("mean of resampled complete seed groups (USD)")
    hist.set_ylabel("bootstrap draws")
    hist.set_title(f"Bootstrap over seed groups (95% CI [{low:,.1f}, {high:,.1f}])")
    figure.suptitle(
        f"Bootstrap unit check: {condition_a} minus {condition_b} across complete seed groups",
        fontsize=10.5,
    )
    save_figure(figure, output_dir, "COSC723_complete_seed_group_unit", formats, dpi)

ATLAS_SPECS = (
    ("mean_profit", "COSC723_Fig04_mean_profit_performance_atlas",
     "Mean combined team after-gas profit (USD)", "YlOrRd", "{:,.0f}"),
    ("mean_decision_ms", "COSC723_Fig09_inference_time_performance_atlas",
     "Successful atomic complete-decision time (ms)", "PuBu", "{:,.1f}"),
    ("mean_cross_dex_pct", "COSC723_arbitrage_type_heat_1x4",
     "Cross-DEX share of committed team trades (%)", "cividis", "{:,.1f}"),
    ("mean_profit_share_pct", "COSC723_team_profit_share_1x4",
     "Mean team profit share (%)", "YlGn", "{:,.1f}"),
    ("team_win_pct", "COSC723_team_win_1x4", "Team win rate (%)", "YlGn", "{:,.1f}"),
    ("mean_rival_profit", "COSC723_rival_profit_1x4",
     "Mean rival after-gas profit (USD)", "YlOrBr", "{:,.0f}"),
    ("mean_successful_trades", "COSC723_successful_trades_1x4",
     "Mean committed team trades per case", "BuGn", "{:,.2f}"),
    ("mean_duplicate_paths", "COSC723_duplicate_path_proposals_1x4",
     "Mean duplicate-path proposals per case", "Reds", "{:,.2f}"),
    ("mean_pool_overlaps", "COSC723_shared_pool_overlap_1x4",
     "Mean shared-pool overlap events per case", "OrRd", "{:,.2f}"),
    ("mean_invalid_proposals", "COSC723_invalid_proposals_1x4",
     "Mean invalid proposals per case", "Reds", "{:,.2f}"),
    ("mean_rejected_trades", "COSC723_rejected_trades_1x4",
     "Mean rejected trades per case", "OrRd", "{:,.2f}"),
    ("mean_invalid_committed", "COSC723_invalid_committed_trades_1x4",
     "Mean invalid committed trades per case", "Reds", "{:,.2f}"),
    ("mean_plans_changed", "COSC723_plans_changed_route_1x4",
     "Mean plan-changed route decisions per case", "Purples", "{:,.2f}"),
    ("mean_rival_first", "COSC723_rival_consumed_first_1x4",
     "Mean rival-consumed-first events per case", "PuRd", "{:,.2f}"),
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Render the COSC723 figures from raw_results.csv.")
    parser.add_argument("--input", type=Path, default=Path("results/raw_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--condition-a", default="Proposed")
    parser.add_argument("--condition-b", default="B2")
    args = parser.parse_args()
    formats = [suffix.strip() for suffix in str(args.formats).split(",") if suffix.strip()]
    if set(formats) - {"png", "pdf"} or args.dpi < 1:
        parser.error("use png,pdf formats and a positive DPI")
    configure_style()
    raw = load_runs(Path(args.input))
    cells = summarize(raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / "cosc723_atlas_cells.csv", index=False)
    conditions = [condition for condition in CONDITIONS if condition in set(raw["condition"])]
    if not conditions:
        conditions = list(CONDITIONS)
    for column, filename, title, cmap, fmt in ATLAS_SPECS:
        atlas(cells, column, title, filename, output_dir, formats, args.dpi, cmap=cmap, fmt=fmt,
              conditions=conditions)
    profit_time_figure(raw, output_dir, formats, args.dpi, conditions)
    available = set(raw["condition"])
    if {args.condition_a, args.condition_b} <= available:
        paired = paired_frame(raw, args.condition_a, args.condition_b)
        table = paired_bootstrap_figure(
            paired, output_dir, formats, args.dpi, args.condition_a, args.condition_b
        )
        table.to_csv(output_dir / "cosc723_paired_proposed_minus_B2.csv", index=False)
        seed_group_unit_figure(paired, output_dir, formats, args.dpi, args.condition_a, args.condition_b)
    else:
        print(
            f"[chart] skipping paired figures: {args.condition_a} and {args.condition_b} "
            f"are both required (available: {sorted(available)})"
        )
    print(f"[chart] figures saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

