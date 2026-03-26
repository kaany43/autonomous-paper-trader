from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loader import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
M4_TARGET_CONFIG_PATH = REPO_ROOT / "config" / "modeling" / "m4_target.yaml"
TARGET_COLUMN_PREFIX = "target_"
TARGET_DATE_COLUMN = "target_date"
TARGET_VALID_COLUMN = "target_is_valid"


@dataclass(frozen=True)
class OfficialTargetDefinition:
    milestone: str
    contract_name: str
    version: int
    task_type: str
    official_target_column: str
    helper_return_column: str
    forecast_horizon_sessions: int
    price_column: str
    positive_return_threshold: float
    invalid_target_policy: str
    feature_timestamp: str
    target_timestamp: str


def load_m4_target_definition(
    config_path: Path = M4_TARGET_CONFIG_PATH,
) -> OfficialTargetDefinition:
    """Load the single official M4 target contract from config."""
    data = load_yaml(config_path)
    target_cfg = data.get("target")

    if not isinstance(target_cfg, dict):
        raise ValueError(f"Missing or invalid target config in: {config_path}")

    definition = OfficialTargetDefinition(
        milestone=str(target_cfg.get("milestone", "")).strip(),
        contract_name=str(target_cfg.get("contract_name", "")).strip(),
        version=int(target_cfg.get("version", 0) or 0),
        task_type=str(target_cfg.get("task_type", "")).strip().lower(),
        official_target_column=str(target_cfg.get("official_target_column", "")).strip(),
        helper_return_column=str(target_cfg.get("helper_return_column", "")).strip(),
        forecast_horizon_sessions=int(target_cfg.get("forecast_horizon_sessions", 0) or 0),
        price_column=str(target_cfg.get("price_column", "")).strip(),
        positive_return_threshold=float(target_cfg.get("positive_return_threshold", 0.0) or 0.0),
        invalid_target_policy=str(target_cfg.get("invalid_target_policy", "")).strip(),
        feature_timestamp=str(target_cfg.get("feature_timestamp", "")).strip(),
        target_timestamp=str(target_cfg.get("target_timestamp", "")).strip(),
    )

    if definition.milestone != "M4":
        raise ValueError("Official target config milestone must be 'M4'.")
    if definition.task_type != "classification":
        raise ValueError("Official M4 target must define exactly one classification task.")
    if definition.forecast_horizon_sessions != 1:
        raise ValueError("Official M4 target horizon must be exactly one next tradable session.")
    if not definition.official_target_column.startswith("target_"):
        raise ValueError("official_target_column must use the target_ prefix.")
    if not definition.helper_return_column.startswith("target_"):
        raise ValueError("helper_return_column must use the target_ prefix.")
    if not definition.price_column:
        raise ValueError("price_column is required for the M4 target contract.")
    if definition.invalid_target_policy != "null_and_exclude_from_training":
        raise ValueError(
            "Official M4 invalid target handling must be 'null_and_exclude_from_training'."
        )
    if not definition.feature_timestamp:
        raise ValueError("feature_timestamp is required for the M4 target contract.")
    if not definition.target_timestamp:
        raise ValueError("target_timestamp is required for the M4 target contract.")

    return definition


def add_m4_target_columns(
    df: pd.DataFrame,
    definition: OfficialTargetDefinition | None = None,
) -> pd.DataFrame:
    """
    Add the single official M4 supervised target to a processed daily dataframe.

    The row at session t is labeled using the next tradable session for the same
    symbol only after the dataframe is sorted and normalized by symbol/date.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    target_definition = definition or load_m4_target_definition()

    required_columns = {"date", "symbol", target_definition.price_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Target generation is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="coerce",
        format="mixed",
        utc=True,
    ).dt.tz_convert(None)
    if normalized["date"].isna().any():
        raise ValueError("Target generation requires valid non-null date values.")

    normalized["symbol"] = normalized["symbol"].astype(str).str.upper().str.strip()
    normalized = (
        # Preserve keep-last ingestion semantics for duplicate symbol/date corrections.
        normalized.drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    grouped = normalized.groupby("symbol", group_keys=False)
    current_price = pd.to_numeric(normalized[target_definition.price_column], errors="coerce")
    target_date = pd.to_datetime(
        grouped["date"].shift(-target_definition.forecast_horizon_sessions),
        errors="coerce",
    )
    future_price = pd.to_numeric(
        grouped[target_definition.price_column].shift(-target_definition.forecast_horizon_sessions),
        errors="coerce",
    )

    valid_target = (
        current_price.notna()
        & future_price.notna()
        & (current_price > 0.0)
        & (future_price > 0.0)
    )

    helper_return = pd.Series(pd.NA, index=normalized.index, dtype="Float64")
    if bool(valid_target.any()):
        helper_return.loc[valid_target] = (
            future_price.loc[valid_target] / current_price.loc[valid_target]
        ) - 1.0

    official_target = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
    if bool(valid_target.any()):
        official_target.loc[valid_target] = (
            helper_return.loc[valid_target].astype("float64")
            > target_definition.positive_return_threshold
        ).astype("int64")

    normalized[TARGET_DATE_COLUMN] = target_date
    normalized[TARGET_VALID_COLUMN] = valid_target.astype(bool)
    normalized[target_definition.helper_return_column] = helper_return
    normalized[target_definition.official_target_column] = official_target

    return normalized
