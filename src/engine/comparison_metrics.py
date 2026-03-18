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
ALIGNED_EQUITY_CURVES_CSV_FILENAME = "aligned_equity_curves.csv"
ALIGNED_DRAWDOWNS_CSV_FILENAME = "aligned_drawdowns.csv"
COMPARISON_SUMMARY_JSON_FILENAME = "comparison_summary.json"
NOT_APPLICABLE = "not_applicable"
_EXECUTED_STATUSES = {"filled", "executed"}


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _equity_input_for_run(run: dict[str, Any], *, require_existing: bool = False) -> tuple[pd.DataFrame, str]:
    output_dir = Path(run["output_dir"])
    run_name = str(run.get("name", ""))
    file_name = PORTFOLIO_SNAPSHOT_FILENAME

    if run_name == "buy_and_hold":
        file_name = BENCHMARK_EQUITY_FILENAME
        column_name = "benchmark_equity"
    elif run_name == "equal_weight":
        file_name = EQUAL_WEIGHT_EQUITY_FILENAME
        column_name = "equal_weight_equity"
    else:
        column_name = "total_equity"

    curve_path = output_dir / file_name
    curve = _load_csv_if_exists(curve_path)
    if require_existing and curve.empty:
        raise FileNotFoundError(f"Missing required run artifact for {run_name}: {curve_path}")
    return curve, column_name


def _build_aligned_equity_curves(runs: list[dict[str, Any]]) -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    for run in sorted(runs, key=lambda item: str(item.get("name", ""))):
        run_name = str(run.get("name", ""))
        curve, equity_column = _equity_input_for_run(run, require_existing=True)
        if "date" not in curve.columns:
            raise ValueError(f"Run {run_name} artifact is missing required 'date' column.")
        if equity_column not in curve.columns:
            raise ValueError(f"Run {run_name} artifact is missing required '{equity_column}' column.")

        run_curve = curve[["date", equity_column]].copy()
        run_curve["date"] = pd.to_datetime(run_curve["date"], errors="coerce").dt.normalize()
        run_curve[equity_column] = pd.to_numeric(run_curve[equity_column], errors="coerce")
        run_curve = run_curve.dropna(subset=["date", equity_column]).sort_values("date")
        run_curve = run_curve.drop_duplicates(subset=["date"], keep="last")
        run_curve = run_curve.rename(columns={equity_column: run_name})

        if aligned is None:
            aligned = run_curve
            continue
        aligned = aligned.merge(run_curve, on="date", how="inner")

    if aligned is None:
        return pd.DataFrame(columns=["date"])
    return aligned.sort_values("date").reset_index(drop=True)


def _build_aligned_drawdowns(equity_curves: pd.DataFrame) -> pd.DataFrame:
    if equity_curves.empty:
        return equity_curves.copy()
    drawdowns = equity_curves.copy()
    run_columns = [column for column in drawdowns.columns if column != "date"]
    for column in run_columns:
        series = pd.to_numeric(drawdowns[column], errors="coerce")
        running_peak = series.cummax()
        drawdowns[column] = (series / running_peak) - 1.0
    return drawdowns


def _equity_artifact_path_for_run(run: dict[str, Any]) -> tuple[Path, str]:
    output_dir = Path(run["output_dir"])
    run_name = str(run.get("name", ""))

    if run_name == "buy_and_hold":
        return output_dir / BENCHMARK_EQUITY_FILENAME, "benchmark_equity"

    if run_name == "equal_weight":
        return output_dir / EQUAL_WEIGHT_EQUITY_FILENAME, "equal_weight_equity"

    return output_dir / PORTFOLIO_SNAPSHOT_FILENAME, "total_equity"


def _load_required_equity_curve(run: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    run_name = str(run.get("name", ""))
    artifact_path, equity_column = _equity_artifact_path_for_run(run)
    if not artifact_path.exists():
        raise ValueError(
            f"Missing required equity artifact for run '{run_name}' at '{artifact_path}'. "
            f"Expected equity column '{equity_column}'."
        )

    curve = pd.read_csv(artifact_path)
    if "date" not in curve.columns:
        raise ValueError(f"Run '{run_name}' equity artifact is missing required 'date' column: '{artifact_path}'.")
    if equity_column not in curve.columns:
        raise ValueError(
            f"Run '{run_name}' equity artifact is missing required '{equity_column}' column: '{artifact_path}'."
        )
    return curve, equity_column


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
        normalized_status = trades["execution_status"].astype(str).str.lower()
        trades = trades.loc[normalized_status.isin(_EXECUTED_STATUSES)]

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
            "run_id": str(run.get("run_id", "")),
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


def build_aligned_equity_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    comparable_runs = [run for run in runs if str(run.get("status", "completed")).lower() == "completed"]
    if not comparable_runs:
        raise ValueError("No comparable runs exist to build aligned equity and drawdown exports.")

    aligned: pd.DataFrame | None = None
    for run in comparable_runs:
        run_name = str(run.get("name", ""))
        curve, equity_column = _load_required_equity_curve(run)

        series = curve.loc[:, ["date", equity_column]].copy()
        series["date"] = pd.to_datetime(series["date"], errors="coerce").dt.normalize()
        series[equity_column] = pd.to_numeric(series[equity_column], errors="coerce")
        run_series = series.rename(columns={equity_column: run_name}).set_index("date")

        if aligned is None:
            aligned = run_series
        else:
            aligned = aligned.join(run_series, how="outer")

    assert aligned is not None
    aligned = aligned.sort_index()
    aligned.index = aligned.index.strftime("%Y-%m-%d")
    aligned.index.name = "date"
    return aligned.reset_index()


def build_aligned_drawdowns_table(aligned_equity: pd.DataFrame) -> pd.DataFrame:
    drawdowns = aligned_equity.copy()
    run_columns = [col for col in drawdowns.columns if col != "date"]
    for column in run_columns:
        # Drawdown formula for notebook consumers:
        # running_peak = cumulative maximum of aligned equity values for the run;
        # drawdown = (equity / running_peak) - 1.0.
        running_peak = drawdowns[column].cummax()
        drawdowns[column] = (drawdowns[column] / running_peak) - 1.0
    return drawdowns


def write_comparison_metrics(
    *,
    comparison_dir: Path,
    comparison_run_id: str,
    created_at: str,
    runs: list[dict[str, Any]],
) -> dict[str, Path]:
    rows = build_comparison_metrics_rows(runs)
    aligned_equity_curves = _build_aligned_equity_curves(runs)
    aligned_drawdowns_export = _build_aligned_drawdowns(aligned_equity_curves)
    build_aligned_drawdowns_table(build_aligned_equity_table(runs))
    payload = {
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "metric_methodology": {
            "period_return_definition": "same_as_cumulative_return",
            "sharpe_ratio_risk_free_rate": 0.0,
            "volatility_frequency": "daily",
            "drawdown_formula": "(equity / running_peak) - 1.0",
            "trade_metric_non_applicable": None,
        },
        "rows": rows,
    }

    json_path = comparison_dir / COMPARISON_METRICS_JSON_FILENAME
    csv_path = comparison_dir / COMPARISON_METRICS_CSV_FILENAME
    aligned_equity_curves_path = comparison_dir / ALIGNED_EQUITY_CURVES_CSV_FILENAME
    aligned_drawdowns_path = comparison_dir / ALIGNED_DRAWDOWNS_CSV_FILENAME
    comparison_summary_path = comparison_dir / COMPARISON_SUMMARY_JSON_FILENAME
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    aligned_equity_curves.to_csv(aligned_equity_curves_path, index=False)
    aligned_drawdowns_export.to_csv(aligned_drawdowns_path, index=False)

    summary_payload = {
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "run_count": len(runs),
        "runs": [{"run_name": str(run.get("name", "")), "run_id": str(run.get("run_id", ""))} for run in sorted(runs, key=lambda item: str(item.get("name", "")))],
        "exports": {
            "comparison_metrics_json": COMPARISON_METRICS_JSON_FILENAME,
            "comparison_metrics_json_path": COMPARISON_METRICS_JSON_FILENAME,
            "comparison_metrics_csv": COMPARISON_METRICS_CSV_FILENAME,
            "comparison_metrics_csv_path": COMPARISON_METRICS_CSV_FILENAME,
            "aligned_equity_curves_csv": ALIGNED_EQUITY_CURVES_CSV_FILENAME,
            "aligned_equity_curves_csv_path": ALIGNED_EQUITY_CURVES_CSV_FILENAME,
            "aligned_drawdowns_csv": ALIGNED_DRAWDOWNS_CSV_FILENAME,
            "aligned_drawdowns_csv_path": ALIGNED_DRAWDOWNS_CSV_FILENAME,
        },
    }
    with comparison_summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, sort_keys=True)

    return {
        "json_path": json_path,
        "comparison_metrics_json_path": json_path,
        "csv_path": csv_path,
        "comparison_metrics_csv_path": csv_path,
        "aligned_equity_csv_path": aligned_equity_curves_path,
        "aligned_equity_curves_csv_path": aligned_equity_curves_path,
        "aligned_drawdowns_csv_path": aligned_drawdowns_path,
        "comparison_summary_path": comparison_summary_path,
        "comparison_summary_json_path": comparison_summary_path,
    }
