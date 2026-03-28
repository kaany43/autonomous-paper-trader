from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_yaml
from src.data.targets import (
    TARGET_DATE_COLUMN,
    TARGET_VALID_COLUMN,
    OfficialTargetDefinition,
    load_m4_target_definition,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
M4_SPLIT_CONFIG_PATH = REPO_ROOT / "config" / "modeling" / "m4_split.yaml"
MODELING_INPUT_FILE = PROCESSED_DATA_DIR / "m4_modeling_dataset.parquet"
TRAIN_OUTPUT_FILE = PROCESSED_DATA_DIR / "m4_train_dataset.parquet"
VALIDATION_OUTPUT_FILE = PROCESSED_DATA_DIR / "m4_validation_dataset.parquet"
SPLIT_METADATA_FILE = PROCESSED_DATA_DIR / "m4_train_validation_split.metadata.json"
OFFICIAL_M4_SPLIT_METHOD = "chronological_holdout_by_target_date_window"


@dataclass(frozen=True)
class OfficialM4SplitDefinition:
    milestone: str
    contract_name: str
    version: int
    method: str
    symbol_column: str
    feature_timestamp_column: str
    target_timestamp_column: str
    official_target_column: str
    validation_start_date: str
    validation_end_date: str

    @property
    def validation_start_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.validation_start_date).normalize()

    @property
    def validation_end_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.validation_end_date).normalize()


def load_m4_split_definition(
    config_path: Path = M4_SPLIT_CONFIG_PATH,
) -> OfficialM4SplitDefinition:
    """Load the single official M4 train/validation split contract from config."""
    data = load_yaml(config_path)
    split_cfg = data.get("split")

    if not isinstance(split_cfg, dict):
        raise ValueError(f"Missing or invalid split config in: {config_path}")

    definition = OfficialM4SplitDefinition(
        milestone=str(split_cfg.get("milestone", "")).strip(),
        contract_name=str(split_cfg.get("contract_name", "")).strip(),
        version=int(split_cfg.get("version", 0) or 0),
        method=str(split_cfg.get("method", "")).strip(),
        symbol_column=str(split_cfg.get("symbol_column", "")).strip(),
        feature_timestamp_column=str(split_cfg.get("feature_timestamp_column", "")).strip(),
        target_timestamp_column=str(split_cfg.get("target_timestamp_column", "")).strip(),
        official_target_column=str(split_cfg.get("official_target_column", "")).strip(),
        validation_start_date=str(split_cfg.get("validation_start_date", "")).strip(),
        validation_end_date=str(split_cfg.get("validation_end_date", "")).strip(),
    )

    target_definition = load_m4_target_definition()

    if definition.milestone != "M4":
        raise ValueError("Official M4 split config milestone must be 'M4'.")
    if definition.method != OFFICIAL_M4_SPLIT_METHOD:
        raise ValueError(f"Official M4 split method must be '{OFFICIAL_M4_SPLIT_METHOD}'.")
    if definition.symbol_column != "symbol":
        raise ValueError("Official M4 split symbol_column must be 'symbol'.")
    if definition.feature_timestamp_column != "date":
        raise ValueError("Official M4 split feature_timestamp_column must be 'date'.")
    if definition.target_timestamp_column != TARGET_DATE_COLUMN:
        raise ValueError(
            f"Official M4 split target_timestamp_column must be '{TARGET_DATE_COLUMN}'."
        )
    if definition.official_target_column != target_definition.official_target_column:
        raise ValueError(
            "Official M4 split official_target_column must match the official M4 target definition."
        )
    if not definition.contract_name:
        raise ValueError("Official M4 split contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official M4 split version must be >= 1.")
    if not definition.validation_start_date:
        raise ValueError("Official M4 split validation_start_date is required.")
    if not definition.validation_end_date:
        raise ValueError("Official M4 split validation_end_date is required.")

    validation_start = definition.validation_start_timestamp
    validation_end = definition.validation_end_timestamp
    if pd.isna(validation_start) or pd.isna(validation_end):
        raise ValueError("Official M4 split dates must be valid calendar dates.")
    if validation_start > validation_end:
        raise ValueError("Official M4 split validation_start_date must be on or before validation_end_date.")

    return definition


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed").dt.tz_localize(None)


def _normalize_timestamp_series_to_date(series: pd.Series) -> pd.Series:
    return _normalize_timestamp_series(series).dt.normalize()


def _validate_binary_target_values(
    values: pd.Series,
    *,
    column_name: str,
) -> None:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.isna().any():
        raise ValueError(
            f"M4 split requires '{column_name}' to contain only numeric 0/1 values."
        )

    if not bool((numeric_values == numeric_values.round()).all()):
        raise ValueError(
            f"M4 split requires '{column_name}' to contain exact integer 0/1 values."
        )

    observed_target_values = set(numeric_values.astype("int64").tolist())
    if not observed_target_values.issubset({0, 1}):
        raise ValueError(
            f"M4 split requires '{column_name}' to contain only 0/1 values."
        )


def _is_sorted_by_symbol_date(
    df: pd.DataFrame,
    *,
    symbol_column: str,
    feature_timestamp_column: str,
) -> bool:
    observed = df[[symbol_column, feature_timestamp_column]].reset_index(drop=True)
    expected = (
        df.sort_values([symbol_column, feature_timestamp_column])
        .reset_index(drop=True)[[symbol_column, feature_timestamp_column]]
    )
    return observed.equals(expected)


def _prepare_split_input(
    modeling_df: pd.DataFrame,
    split_definition: OfficialM4SplitDefinition,
    target_definition: OfficialTargetDefinition,
) -> tuple[pd.DataFrame, bool]:
    if modeling_df.empty:
        raise ValueError("M4 modeling dataset is empty.")

    required_columns = {
        split_definition.symbol_column,
        split_definition.feature_timestamp_column,
        split_definition.target_timestamp_column,
        split_definition.official_target_column,
    }
    missing_columns = required_columns - set(modeling_df.columns)
    if missing_columns:
        raise ValueError(
            "M4 modeling dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    normalized = modeling_df.copy()
    normalized[split_definition.feature_timestamp_column] = _normalize_timestamp_series(
        normalized[split_definition.feature_timestamp_column]
    )
    normalized[split_definition.target_timestamp_column] = _normalize_timestamp_series(
        normalized[split_definition.target_timestamp_column]
    )

    if normalized[split_definition.feature_timestamp_column].isna().any():
        raise ValueError("M4 split requires valid non-null feature timestamps.")
    if normalized[split_definition.target_timestamp_column].isna().any():
        raise ValueError("M4 split requires valid non-null target timestamps.")

    normalized[split_definition.symbol_column] = (
        normalized[split_definition.symbol_column].astype(str).str.upper().str.strip()
    )

    if normalized.duplicated(
        subset=[split_definition.symbol_column, split_definition.feature_timestamp_column]
    ).any():
        raise ValueError("M4 modeling dataset contains duplicate symbol/date rows.")

    if not bool(
        (
            normalized[split_definition.target_timestamp_column]
            > normalized[split_definition.feature_timestamp_column]
        ).all()
    ):
        raise ValueError("M4 split requires target timestamps to stay strictly after feature timestamps.")

    if normalized[split_definition.official_target_column].isna().any():
        raise ValueError("M4 split requires non-null official target values.")

    if TARGET_VALID_COLUMN in normalized.columns and not bool(normalized[TARGET_VALID_COLUMN].all()):
        raise ValueError("M4 split requires target_is_valid to be true for every modeling row.")

    _validate_binary_target_values(
        normalized[target_definition.official_target_column],
        column_name=target_definition.official_target_column,
    )

    input_was_sorted = _is_sorted_by_symbol_date(
        normalized,
        symbol_column=split_definition.symbol_column,
        feature_timestamp_column=split_definition.feature_timestamp_column,
    )

    normalized = (
        normalized.sort_values(
            [split_definition.symbol_column, split_definition.feature_timestamp_column]
        )
        .reset_index(drop=True)
    )

    return normalized, input_was_sorted


def split_m4_modeling_dataset(
    modeling_df: pd.DataFrame,
    split_definition: OfficialM4SplitDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Create the official leakage-safe M4 train/validation holdout split.

    The split boundary is applied to `target_date`, not just `date`, so rows
    whose labels belong to the validation window cannot remain inside training.
    """
    resolved_target_definition = target_definition or load_m4_target_definition()
    resolved_split_definition = split_definition or load_m4_split_definition()

    prepared_df, input_was_sorted = _prepare_split_input(
        modeling_df,
        resolved_split_definition,
        resolved_target_definition,
    )

    target_timestamp_column = resolved_split_definition.target_timestamp_column
    feature_timestamp_column = resolved_split_definition.feature_timestamp_column
    validation_start = resolved_split_definition.validation_start_timestamp
    validation_end = resolved_split_definition.validation_end_timestamp
    target_dates = _normalize_timestamp_series_to_date(prepared_df[target_timestamp_column])

    train_mask = target_dates < validation_start
    validation_mask = target_dates.between(
        validation_start,
        validation_end,
        inclusive="both",
    )
    excluded_mask = target_dates > validation_end

    train_df = prepared_df.loc[train_mask].reset_index(drop=True)
    validation_df = prepared_df.loc[validation_mask].reset_index(drop=True)
    excluded_df = prepared_df.loc[excluded_mask].reset_index(drop=True)

    if train_df.empty:
        raise ValueError(
            "Configured M4 split leaves the training partition empty before "
            f"{validation_start.strftime('%Y-%m-%d')}."
        )
    if validation_df.empty:
        raise ValueError(
            "Configured M4 split leaves the validation partition empty inside "
            f"{validation_start.strftime('%Y-%m-%d')} -> {validation_end.strftime('%Y-%m-%d')}."
        )
    if not bool(
        _normalize_timestamp_series_to_date(train_df[target_timestamp_column]).max()
        < _normalize_timestamp_series_to_date(validation_df[target_timestamp_column]).min()
    ):
        raise ValueError("M4 split failed to keep training targets strictly before validation targets.")

    summary = {
        "input_row_count": int(len(prepared_df)),
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(validation_df)),
        "excluded_future_row_count": int(len(excluded_df)),
        "input_was_sorted_by_symbol_date": bool(input_was_sorted),
        "train_feature_date_start": _format_date(train_df[feature_timestamp_column].min()),
        "train_feature_date_end": _format_date(train_df[feature_timestamp_column].max()),
        "train_target_date_start": _format_date(train_df[target_timestamp_column].min()),
        "train_target_date_end": _format_date(train_df[target_timestamp_column].max()),
        "validation_feature_date_start": _format_date(validation_df[feature_timestamp_column].min()),
        "validation_feature_date_end": _format_date(validation_df[feature_timestamp_column].max()),
        "validation_target_date_start": _format_date(validation_df[target_timestamp_column].min()),
        "validation_target_date_end": _format_date(validation_df[target_timestamp_column].max()),
        "excluded_feature_date_start": _format_date(excluded_df[feature_timestamp_column].min() if not excluded_df.empty else None),
        "excluded_feature_date_end": _format_date(excluded_df[feature_timestamp_column].max() if not excluded_df.empty else None),
        "excluded_target_date_start": _format_date(excluded_df[target_timestamp_column].min() if not excluded_df.empty else None),
        "excluded_target_date_end": _format_date(excluded_df[target_timestamp_column].max() if not excluded_df.empty else None),
    }

    return train_df, validation_df, summary
