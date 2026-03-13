from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

METRICS_FILENAME = "backtest_metrics.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _to_daily_returns(equity_curve: pd.DataFrame, equity_column: str) -> pd.Series:
    if equity_curve.empty or equity_column not in equity_curve.columns:
        return pd.Series(dtype="float64")

    curve = equity_curve.copy()
    if "date" in curve.columns:
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce").dt.normalize()
        curve = curve.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    curve[equity_column] = pd.to_numeric(curve[equity_column], errors="coerce")
    curve = curve.dropna(subset=[equity_column]).reset_index(drop=True)

    if curve.empty:
        return pd.Series(dtype="float64")

    returns = curve[equity_column].pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return pd.to_numeric(returns, errors="coerce").fillna(0.0)


def compute_equity_metrics(equity_curve: pd.DataFrame, equity_column: str) -> dict[str, float]:
    if equity_curve.empty or equity_column not in equity_curve.columns:
        return {
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "avg_daily_return": 0.0,
            "sharpe_ratio": 0.0,
            "return_over_max_drawdown": 0.0,
        }

    curve = equity_curve.copy()
    if "date" in curve.columns:
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce").dt.normalize()
        curve = curve.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    curve[equity_column] = pd.to_numeric(curve[equity_column], errors="coerce")
    curve = curve.dropna(subset=[equity_column]).reset_index(drop=True)

    if curve.empty:
        return {
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "avg_daily_return": 0.0,
            "sharpe_ratio": 0.0,
            "return_over_max_drawdown": 0.0,
        }

    first_equity = _safe_float(curve.iloc[0][equity_column], default=0.0)
    last_equity = _safe_float(curve.iloc[-1][equity_column], default=0.0)

    if first_equity == 0.0:
        cumulative_return = 0.0
    else:
        cumulative_return = float(last_equity / first_equity - 1.0)

    running_peak = curve[equity_column].cummax()
    drawdowns = (curve[equity_column] / running_peak) - 1.0
    max_drawdown = _safe_float(drawdowns.min(), default=0.0)

    daily_returns = _to_daily_returns(curve, equity_column)
    avg_daily_return = _safe_float(daily_returns.mean(), default=0.0)
    volatility = _safe_float(daily_returns.std(ddof=0), default=0.0)

    if volatility == 0.0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(avg_daily_return / volatility)

    if max_drawdown == 0.0:
        return_over_max_drawdown = 0.0
    else:
        return_over_max_drawdown = float(cumulative_return / abs(max_drawdown))

    return {
        "cumulative_return": float(cumulative_return),
        "max_drawdown": float(max_drawdown),
        "volatility": float(volatility),
        "avg_daily_return": float(avg_daily_return),
        "sharpe_ratio": float(sharpe_ratio),
        "return_over_max_drawdown": float(return_over_max_drawdown),
    }


def compute_trade_metrics(trade_history: pd.DataFrame) -> dict[str, float | int]:
    if trade_history.empty:
        return {"trade_count": 0, "win_rate": 0.0}

    trades = trade_history.copy()
    if "success" in trades.columns:
        trades = trades.loc[trades["success"].astype(bool)]
    if "executed_quantity" in trades.columns:
        trades = trades.loc[pd.to_numeric(trades["executed_quantity"], errors="coerce").fillna(0.0) > 0.0]

    if trades.empty:
        return {"trade_count": 0, "win_rate": 0.0}

    trade_count = int(len(trades))

    realized = pd.to_numeric(trades.get("realized_pnl", pd.Series(dtype="float64")), errors="coerce")
    realized = realized.dropna()

    if realized.empty:
        win_rate = 0.0
    else:
        wins = (realized > 0.0).sum()
        win_rate = float(wins / len(realized))

    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
    }


def compute_backtest_metrics(
    strategy_equity_curve: pd.DataFrame,
    benchmark_equity_curve: pd.DataFrame,
    trade_history: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    strategy_metrics = compute_equity_metrics(strategy_equity_curve, equity_column="total_equity")
    strategy_metrics.update(compute_trade_metrics(trade_history))

    benchmark_metrics = compute_equity_metrics(benchmark_equity_curve, equity_column="benchmark_equity")

    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
    }


def write_metrics_json(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return output_path
