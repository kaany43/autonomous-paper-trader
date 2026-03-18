from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.engine.simulator import (
    BENCHMARK_EQUITY_FILENAME,
    EQUAL_WEIGHT_EQUITY_FILENAME,
    PORTFOLIO_SNAPSHOT_FILENAME,
)

ALIGNED_EQUITY_CURVES_FILENAME = "aligned_equity_curves.csv"

_RUN_TYPE_CONFIG: dict[str, tuple[str, str]] = {
    "momentum": (PORTFOLIO_SNAPSHOT_FILENAME, "total_equity"),
    "buy_and_hold": (BENCHMARK_EQUITY_FILENAME, "benchmark_equity"),
    "equal_weight": (EQUAL_WEIGHT_EQUITY_FILENAME, "equal_weight_equity"),
}


def _resolve_run_type(run_record: dict[str, Any]) -> str:
    run_name = str(run_record.get("name", "")).strip().lower()
    if run_name == "buy_and_hold":
        return "buy_and_hold"
    if run_name == "equal_weight":
        return "equal_weight"

    strategy_type = str(run_record.get("strategy_type", "")).strip().lower()
    if strategy_type == "momentum":
        return "momentum"

    raise ValueError(f"Unsupported run type for aligned equity export: run_name={run_name!r}, strategy_type={strategy_type!r}.")


def _load_equity_curve_for_run(run_record: dict[str, Any], run_name: str) -> pd.DataFrame:
    run_type = _resolve_run_type(run_record)
    filename, value_column = _RUN_TYPE_CONFIG[run_type]

    output_dir = Path(str(run_record.get("output_dir", ""))).expanduser()
    curve_path = output_dir / filename
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing equity input for run {run_name!r}: {curve_path}")

    curve_df = pd.read_csv(curve_path)
    required_columns = {"date", value_column}
    missing_columns = required_columns.difference(curve_df.columns)
    if missing_columns:
        raise ValueError(
            f"Equity input for run {run_name!r} is missing columns {sorted(missing_columns)} in {curve_path}."
        )

    normalized = curve_df[["date", value_column]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized[value_column] = pd.to_numeric(normalized[value_column], errors="coerce")
    normalized = normalized.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    normalized = normalized.rename(columns={value_column: run_name})
    return normalized


def write_aligned_equity_curves(
    *,
    comparison_dir: Path,
    comparison_run_id: str,
    created_at: str,
    run_records: list[dict[str, Any]],
) -> dict[str, Path]:
    del comparison_run_id, created_at

    ordered_records = sorted(run_records, key=lambda item: str(item.get("name", "")))

    if not ordered_records:
        aligned = pd.DataFrame(columns=["date"])
    else:
        aligned = pd.DataFrame(columns=["date"])
        for run_record in ordered_records:
            run_name = str(run_record.get("name", "")).strip()
            if not run_name:
                raise ValueError("All run records must include a non-empty 'name'.")
            run_curve = _load_equity_curve_for_run(run_record, run_name)
            aligned = aligned.merge(run_curve, on="date", how="outer")

        aligned = aligned.sort_values("date").reset_index(drop=True)
        run_columns = [str(item["name"]).strip() for item in ordered_records]
        aligned[run_columns] = aligned[run_columns].ffill()

    output_path = Path(comparison_dir) / ALIGNED_EQUITY_CURVES_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)

    return {"aligned_equity_curves_csv_path": output_path}
