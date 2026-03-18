from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RANKING_SUMMARY_JSON_FILENAME = "ranking_summary.json"
STRATEGY_RANKING_CSV_FILENAME = "strategy_ranking.csv"

_REQUIRED_RANKING_COLUMNS = {
    "run_name",
    "run_id",
    "strategy_type",
    "variant_name",
    "sharpe_ratio",
    "cumulative_return",
    "max_drawdown",
}
_NUMERIC_RANKING_COLUMNS = {"sharpe_ratio", "cumulative_return", "max_drawdown"}
_RANKING_OUTPUT_COLUMNS = [
    "rank",
    "run_name",
    "run_id",
    "strategy_type",
    "variant_name",
    "sharpe_ratio",
    "cumulative_return",
    "max_drawdown",
    "max_drawdown_magnitude",
    "return_over_max_drawdown",
    "volatility",
    "average_daily_return",
    "trade_count",
    "win_rate",
    "activity_metrics_status",
]
_SELECTION_RULE = {
    "description": (
        "Rank runs by higher sharpe_ratio, then higher cumulative_return, "
        "then lower absolute max_drawdown, then lexical run_name ordering."
    ),
    "primary_metric": {"name": "sharpe_ratio", "direction": "higher_is_better"},
    "tie_breakers": [
        {"name": "cumulative_return", "direction": "higher_is_better"},
        {"name": "absolute_max_drawdown", "direction": "lower_is_better"},
        {"name": "run_name", "direction": "lexical_ascending"},
    ],
}


def _load_comparison_metrics(comparison_metrics_csv_path: Path) -> pd.DataFrame:
    metrics_path = Path(comparison_metrics_csv_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing comparison metrics input for ranking: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        raise ValueError("Comparison metrics input is empty; cannot build ranking summary.")

    missing_columns = sorted(_REQUIRED_RANKING_COLUMNS.difference(metrics_df.columns))
    if missing_columns:
        raise ValueError(f"Ranking input is missing required columns: {missing_columns}")

    for column in sorted(_NUMERIC_RANKING_COLUMNS):
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")
        if metrics_df[column].isna().any():
            raise ValueError(f"Ranking input has missing or non-numeric values in required metric '{column}'.")

    return metrics_df


def build_comparison_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        raise ValueError("Comparison metrics input is empty; cannot build ranking summary.")

    ranked = metrics_df.copy()
    ranked["run_name"] = ranked["run_name"].astype(str)
    ranked["variant_name"] = ranked["variant_name"].fillna("").astype(str)
    ranked["max_drawdown_magnitude"] = ranked["max_drawdown"].abs()
    ranked = ranked.sort_values(
        by=["sharpe_ratio", "cumulative_return", "max_drawdown_magnitude", "run_name"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    for column in _RANKING_OUTPUT_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = pd.NA

    return ranked.loc[:, _RANKING_OUTPUT_COLUMNS]


def write_comparison_ranking(
    *,
    comparison_dir: Path,
    comparison_run_id: str,
    generated_at: str,
    comparison_metrics_csv_path: Path,
) -> dict[str, Any]:
    metrics_df = _load_comparison_metrics(comparison_metrics_csv_path)
    ranked = build_comparison_ranking(metrics_df)

    ranking_summary_path = Path(comparison_dir) / RANKING_SUMMARY_JSON_FILENAME
    strategy_ranking_csv_path = Path(comparison_dir) / STRATEGY_RANKING_CSV_FILENAME
    ranking_summary_path.parent.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(strategy_ranking_csv_path, index=False)

    ranked_runs = ranked.where(pd.notna(ranked), None).to_dict(orient="records")
    preferred_run = dict(ranked_runs[0])
    payload = {
        "comparison_run_id": comparison_run_id,
        "generated_at": generated_at,
        "selection_rule": _SELECTION_RULE,
        "source_artifacts": {
            "comparison_metrics_csv": Path(comparison_metrics_csv_path).name,
            "strategy_ranking_csv": STRATEGY_RANKING_CSV_FILENAME,
        },
        "preferred_run": preferred_run,
        "ranked_runs": ranked_runs,
    }

    with ranking_summary_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    return {
        "ranking_summary_json_path": ranking_summary_path,
        "strategy_ranking_csv_path": strategy_ranking_csv_path,
        "preferred_run": preferred_run,
    }
