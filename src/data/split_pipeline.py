from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.splits import (
    M4_SPLIT_CONFIG_PATH,
    MODELING_INPUT_FILE,
    SPLIT_METADATA_FILE,
    TRAIN_OUTPUT_FILE,
    VALIDATION_OUTPUT_FILE,
    OfficialM4SplitDefinition,
    load_m4_split_definition,
    split_m4_modeling_dataset,
)
from src.data.targets import OfficialTargetDefinition, load_m4_target_definition


PIPELINE_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    return value


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_m4_split_metadata(
    *,
    input_path: Path,
    train_output_path: Path,
    validation_output_path: Path,
    metadata_path: Path,
    config_path: Path,
    modeling_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    summary: dict[str, Any],
    split_definition: OfficialM4SplitDefinition,
    target_definition: OfficialTargetDefinition,
) -> dict[str, Any]:
    target_timestamp_column = split_definition.target_timestamp_column
    feature_timestamp_column = split_definition.feature_timestamp_column

    return {
        "pipeline_name": "m4_train_validation_split",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "entrypoint": "python -m src.data.split_pipeline",
        "split_config_path": str(config_path),
        "input_dataset": {
            "path": str(input_path),
            "row_count": int(len(modeling_df)),
            "columns": [str(column) for column in modeling_df.columns],
            "feature_date_start": _format_date(modeling_df[feature_timestamp_column].min()),
            "feature_date_end": _format_date(modeling_df[feature_timestamp_column].max()),
            "target_date_start": _format_date(modeling_df[target_timestamp_column].min()),
            "target_date_end": _format_date(modeling_df[target_timestamp_column].max()),
        },
        "train_dataset": {
            "path": str(train_output_path),
            "row_count": int(len(train_df)),
            "columns": [str(column) for column in train_df.columns],
            "feature_date_start": summary["train_feature_date_start"],
            "feature_date_end": summary["train_feature_date_end"],
            "target_date_start": summary["train_target_date_start"],
            "target_date_end": summary["train_target_date_end"],
        },
        "validation_dataset": {
            "path": str(validation_output_path),
            "row_count": int(len(validation_df)),
            "columns": [str(column) for column in validation_df.columns],
            "feature_date_start": summary["validation_feature_date_start"],
            "feature_date_end": summary["validation_feature_date_end"],
            "target_date_start": summary["validation_target_date_start"],
            "target_date_end": summary["validation_target_date_end"],
        },
        "metadata_output": {
            "path": str(metadata_path),
        },
        "split_definition": asdict(split_definition),
        "target_definition": asdict(target_definition),
        "split_rule": {
            "method": split_definition.method,
            "boundary_anchor_column": split_definition.target_timestamp_column,
            "train_inclusion_rule": (
                f"{split_definition.target_timestamp_column} < "
                f"{split_definition.validation_start_date}"
            ),
            "validation_inclusion_rule": (
                f"{split_definition.validation_start_date} <= "
                f"{split_definition.target_timestamp_column} <= "
                f"{split_definition.validation_end_date}"
            ),
            "excluded_row_rule": (
                f"{split_definition.target_timestamp_column} > "
                f"{split_definition.validation_end_date}"
            ),
        },
        "time_safety": {
            "input_was_sorted_by_symbol_date": summary["input_was_sorted_by_symbol_date"],
            "stable_sort_order": [
                split_definition.symbol_column,
                split_definition.feature_timestamp_column,
            ],
            "target_horizon_sessions": target_definition.forecast_horizon_sessions,
            "leakage_guard": (
                "Rows stay in training only when target_date falls strictly before the "
                "validation window. Rows whose labels land inside the validation window "
                "are kept out of training even if their feature date is near the boundary."
            ),
        },
        "excluded_rows": {
            "count": summary["excluded_future_row_count"],
            "feature_date_start": summary["excluded_feature_date_start"],
            "feature_date_end": summary["excluded_feature_date_end"],
            "target_date_start": summary["excluded_target_date_start"],
            "target_date_end": summary["excluded_target_date_end"],
            "reason": (
                "Rows with target_date after the official validation_end_date stay in the "
                "modeling dataset but are excluded from the official train/validation split."
            ),
        },
        "counts": {
            "input_row_count": summary["input_row_count"],
            "train_row_count": summary["train_row_count"],
            "validation_row_count": summary["validation_row_count"],
            "excluded_future_row_count": summary["excluded_future_row_count"],
        },
        "symbols": sorted(str(symbol) for symbol in modeling_df["symbol"].astype(str).unique()),
    }


def save_m4_train_validation_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    train_output_path: Path = TRAIN_OUTPUT_FILE,
    validation_output_path: Path = VALIDATION_OUTPUT_FILE,
    metadata: dict[str, Any] | None = None,
    metadata_path: Path = SPLIT_METADATA_FILE,
) -> tuple[Path, Path, Path]:
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_output_path, index=False)

    validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_parquet(validation_output_path, index=False)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(metadata or {}), fh, indent=2, sort_keys=True)

    return train_output_path, validation_output_path, metadata_path


def run_m4_train_validation_split(
    *,
    input_path: Path = MODELING_INPUT_FILE,
    train_output_path: Path = TRAIN_OUTPUT_FILE,
    validation_output_path: Path = VALIDATION_OUTPUT_FILE,
    metadata_path: Path = SPLIT_METADATA_FILE,
    config_path: Path = M4_SPLIT_CONFIG_PATH,
    split_definition: OfficialM4SplitDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"M4 modeling dataset not found: {input_path}. "
            "Run `python -m src.data.target_pipeline` first."
        )

    resolved_split_definition = split_definition or load_m4_split_definition(config_path)
    resolved_target_definition = target_definition or load_m4_target_definition()

    modeling_df = pd.read_parquet(input_path)
    train_df, validation_df, summary = split_m4_modeling_dataset(
        modeling_df,
        split_definition=resolved_split_definition,
        target_definition=resolved_target_definition,
    )
    metadata = build_m4_split_metadata(
        input_path=input_path,
        train_output_path=train_output_path,
        validation_output_path=validation_output_path,
        metadata_path=metadata_path,
        config_path=config_path,
        modeling_df=modeling_df,
        train_df=train_df,
        validation_df=validation_df,
        summary=summary,
        split_definition=resolved_split_definition,
        target_definition=resolved_target_definition,
    )
    saved_train_path, saved_validation_path, saved_metadata_path = save_m4_train_validation_split(
        train_df,
        validation_df,
        train_output_path=train_output_path,
        validation_output_path=validation_output_path,
        metadata=metadata,
        metadata_path=metadata_path,
    )

    return {
        "input_path": input_path,
        "train_output_path": saved_train_path,
        "validation_output_path": saved_validation_path,
        "metadata_path": saved_metadata_path,
        "train_dataframe": train_df,
        "validation_dataframe": validation_df,
        "metadata": metadata,
        **summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the official M4 time-aware train/validation split."
    )
    parser.add_argument("--input", default=str(MODELING_INPUT_FILE), help="Path to the M4 modeling parquet.")
    parser.add_argument("--train-output", default=str(TRAIN_OUTPUT_FILE), help="Path to the train split parquet.")
    parser.add_argument(
        "--validation-output",
        default=str(VALIDATION_OUTPUT_FILE),
        help="Path to the validation split parquet.",
    )
    parser.add_argument(
        "--metadata-output",
        default=str(SPLIT_METADATA_FILE),
        help="Path to the split metadata JSON sidecar.",
    )
    parser.add_argument(
        "--config",
        default=str(M4_SPLIT_CONFIG_PATH),
        help="Path to the official M4 split config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    train_output_path = Path(args.train_output)
    validation_output_path = Path(args.validation_output)
    metadata_path = Path(args.metadata_output)
    config_path = Path(args.config)

    try:
        result = run_m4_train_validation_split(
            input_path=input_path,
            train_output_path=train_output_path,
            validation_output_path=validation_output_path,
            metadata_path=metadata_path,
            config_path=config_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] M4 time-aware split failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 time-aware train/validation split completed")
    print(f"Input dataset: {result['input_path']}")
    print(f"Train dataset: {result['train_output_path']}")
    print(f"Validation dataset: {result['validation_output_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Train rows: {result['train_row_count']}")
    print(f"Validation rows: {result['validation_row_count']}")
    print(f"Excluded future rows: {result['excluded_future_row_count']}")
    print(
        "Validation target window: "
        f"{result['validation_target_date_start']} -> {result['validation_target_date_end']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
