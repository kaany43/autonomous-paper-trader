from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine.metrics import compute_equity_metrics
from src.engine.simulator import (
    BENCHMARK_EQUITY_FILENAME,
    EQUAL_WEIGHT_EQUITY_FILENAME,
    PORTFOLIO_SNAPSHOT_FILENAME,
    TRADE_LOG_FILENAME,
)

COMPARISON_METRICS_JSON_FILENAME = "comparison_metrics.json"
COMPARISON_METRICS_CSV_FILENAME = "comparison_metrics.csv"
NOT_APPLICABLE = "not_applicable"


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _equity_input_for_run(run: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    output_dir = Path(run["output_dir"])
    run_name = str(run.get("name", ""))

    if run_name == "buy_and_hold":
        return _load_csv_if_exists(output_dir / BENCHMARK_EQUITY_FILENAME), "benchmark_equity"

    if run_name == "equal_weight":
        return _load_csv_if_exists(output_dir / EQUAL_WEIGHT_EQUITY_FILENAME), "equal_weight_equity"

    return _load_csv_if_exists(output_dir / PORTFOLIO_SNAPSHOT_FILENAME), "total_equity"


def _compute_activity_metrics(run: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(run["output_dir"])
    run_name = str(run.get("name", ""))

    if run_name in {"buy_and_hold", "equal_weight"}:
        return {"trade_count": None, "win_rate": None, "activity_metrics_status": NOT_APPLICABLE}

    trade_log = _load_csv_if_exists(output_dir / TRADE_LOG_FILENAME)
    if trade_log.empty:
        return {"trade_count": 0, "win_rate": None, "activity_metrics_status": "missing_trade_log"}

    trades = trade_log.copy()
    if "execution_status" in trades.columns:
        trades = trades.loc[trades["execution_status"].astype(str).str.lower() == "filled"]

    if "quantity" in trades.columns:
        qty = pd.to_numeric(trades["quantity"], errors="coerce").abs().fillna(0.0)
        trades = trades.loc[qty > 0.0]

    trade_count = int(len(trades))

    if "realized_pnl" not in trades.columns:
        return {"trade_count": trade_count, "win_rate": None, "activity_metrics_status": "win_rate_unavailable"}

    realized = pd.to_numeric(trades["realized_pnl"], errors="coerce").dropna()
    if realized.empty:
        win_rate = None
    else:
        win_rate = float((realized > 0.0).sum() / len(realized))
    return {"trade_count": trade_count, "win_rate": win_rate, "activity_metrics_status": "computed"}


def build_comparison_metrics_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run in sorted(runs, key=lambda item: str(item.get("name", ""))):
        run_name = str(run.get("name", ""))
        curve, equity_column = _equity_input_for_run(run)
        equity_metrics = compute_equity_metrics(curve, equity_column=equity_column)
        activity_metrics = _compute_activity_metrics(run)

        cumulative_return = float(equity_metrics["cumulative_return"])
        row = {
            "run_name": run_name,
            "strategy_type": str(run.get("strategy_type", "")),
            "variant_name": str(run.get("variant_name", "")),
            "equity_column": equity_column,
            "cumulative_return": cumulative_return,
            # Period return is intentionally equivalent to cumulative return for a single comparison period.
            "period_return": cumulative_return,
            "max_drawdown": float(equity_metrics["max_drawdown"]),
            "volatility": float(equity_metrics["volatility"]),
            "average_daily_return": float(equity_metrics["avg_daily_return"]),
            "sharpe_ratio": float(equity_metrics["sharpe_ratio"]),
            "return_over_max_drawdown": float(equity_metrics["return_over_max_drawdown"]),
            "trade_count": activity_metrics["trade_count"],
            "win_rate": activity_metrics["win_rate"],
            "activity_metrics_status": activity_metrics["activity_metrics_status"],
        }
        rows.append(row)

    return rows


def write_comparison_metrics(
    *,
    comparison_dir: Path,
    comparison_run_id: str,
    created_at: str,
    runs: list[dict[str, Any]],
) -> dict[str, Path]:
    rows = build_comparison_metrics_rows(runs)
    payload = {
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "metric_methodology": {
            "period_return_definition": "same_as_cumulative_return",
            "sharpe_ratio_risk_free_rate": 0.0,
            "volatility_frequency": "daily",
            "trade_metric_non_applicable": None,
        },
        "rows": rows,
    }

    json_path = comparison_dir / COMPARISON_METRICS_JSON_FILENAME
    csv_path = comparison_dir / COMPARISON_METRICS_CSV_FILENAME
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {"json_path": json_path, "csv_path": csv_path}
