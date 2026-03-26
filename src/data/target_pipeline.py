from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.targets import (
    M4_TARGET_CONFIG_PATH,
    TARGET_COLUMN_PREFIX,
    TARGET_DATE_COLUMN,
    TARGET_VALID_COLUMN,
    OfficialTargetDefinition,
    add_m4_target_columns,
    load_m4_target_definition,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
FEATURES_INPUT_FILE = PROCESSED_DATA_DIR / "market_features.parquet"
MODELING_OUTPUT_FILE = PROCESSED_DATA_DIR / "m4_modeling_dataset.parquet"
MODELING_METADATA_FILE = PROCESSED_DATA_DIR / "m4_modeling_dataset.metadata.json"
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


def _strip_existing_target_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    existing_target_columns = [
        str(column) for column in df.columns if str(column).startswith(TARGET_COLUMN_PREFIX)
    ]
    if not existing_target_columns:
        return df.copy(), []
    return df.drop(columns=existing_target_columns).copy(), existing_target_columns


def _is_sorted_by_symbol_date(df: pd.DataFrame) -> bool:
    observed = df[["symbol", "date"]].reset_index(drop=True)
    expected = df.sort_values(["symbol", "date"]).reset_index(drop=True)[["symbol", "date"]]
    return observed.equals(expected)


def _validate_modeling_dataset(
    df: pd.DataFrame,
    definition: OfficialTargetDefinition,
) -> None:
    if df.empty:
        raise ValueError("Target preparation produced no valid modeling rows.")
    if not _is_sorted_by_symbol_date(df):
        raise ValueError("Prepared modeling dataset must be sorted by symbol/date.")
    if df.duplicated(subset=["symbol", "date"]).any():
        raise ValueError("Prepared modeling dataset contains duplicate symbol/date rows.")
    if df[["date", TARGET_DATE_COLUMN]].isna().any().any():
        raise ValueError("Prepared modeling dataset contains null feature or target dates.")
    if not bool((df[TARGET_DATE_COLUMN] > df["date"]).all()):
        raise ValueError("Prepared modeling dataset contains non-forward target dates.")
    if not bool(df[TARGET_VALID_COLUMN].all()):
        raise ValueError("Prepared modeling dataset includes invalid target rows.")
    if df[definition.helper_return_column].isna().any():
        raise ValueError("Prepared modeling dataset contains null helper returns.")
    if df[definition.official_target_column].isna().any():
        raise ValueError("Prepared modeling dataset contains null official target values.")

    official_values = set(
        pd.to_numeric(df[definition.official_target_column], errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )
    if not official_values.issubset({0, 1}):
        raise ValueError("Official target column must contain only 0/1 values.")


def prepare_m4_modeling_dataset(
    features_df: pd.DataFrame,
    definition: OfficialTargetDefinition | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build the official M4 modeling dataset from processed daily market features.

    The returned dataframe is deterministic, sorted by symbol/date, stripped of
    stale input target columns, and contains only rows with valid future labels.
    """
    if features_df.empty:
        raise ValueError("Processed feature dataframe is empty.")

    target_definition = definition or load_m4_target_definition()
    stripped_features, dropped_input_target_columns = _strip_existing_target_columns(features_df)
    labeled_df = add_m4_target_columns(stripped_features, target_definition)
    modeling_df = labeled_df.loc[labeled_df[TARGET_VALID_COLUMN]].reset_index(drop=True)
    _validate_modeling_dataset(modeling_df, target_definition)

    summary = {
        "input_row_count": int(len(features_df)),
        "normalized_row_count": int(len(labeled_df)),
        "dropped_duplicate_row_count": int(len(stripped_features) - len(labeled_df)),
        "output_row_count": int(len(modeling_df)),
        "dropped_invalid_row_count": int(len(labeled_df) - len(modeling_df)),
        "dropped_input_target_columns": dropped_input_target_columns,
        "target_columns": [
            TARGET_DATE_COLUMN,
            TARGET_VALID_COLUMN,
            target_definition.helper_return_column,
            target_definition.official_target_column,
        ],
    }
    return modeling_df, summary


def build_target_preparation_metadata(
    *,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    features_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
    summary: dict[str, Any],
    definition: OfficialTargetDefinition,
) -> dict[str, Any]:
    return {
        "pipeline_name": "m4_target_preparation",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "entrypoint": "python -m src.data.target_pipeline",
        "target_config_path": str(M4_TARGET_CONFIG_PATH),
        "input_dataset": {
            "path": str(input_path),
            "row_count": int(len(features_df)),
            "columns": [str(column) for column in features_df.columns],
            "feature_date_start": _format_date(features_df["date"].min() if "date" in features_df.columns else None),
            "feature_date_end": _format_date(features_df["date"].max() if "date" in features_df.columns else None),
        },
        "output_dataset": {
            "path": str(output_path),
            "metadata_path": str(metadata_path),
            "row_count": int(len(modeling_df)),
            "columns": [str(column) for column in modeling_df.columns],
            "feature_date_start": _format_date(modeling_df["date"].min()),
            "feature_date_end": _format_date(modeling_df["date"].max()),
            "target_date_start": _format_date(modeling_df[TARGET_DATE_COLUMN].min()),
            "target_date_end": _format_date(modeling_df[TARGET_DATE_COLUMN].max()),
        },
        "target_definition": asdict(definition),
        "invalid_label_handling": {
            "definition_policy": definition.invalid_target_policy,
            "pipeline_rule": "drop rows where target_is_valid is false after label construction",
            "dropped_invalid_row_count": summary["dropped_invalid_row_count"],
        },
        "normalization": {
            "dropped_input_target_columns": list(summary["dropped_input_target_columns"]),
            "duplicate_rule": "keep last row by original input order for each symbol/date before sorting",
        },
        "counts": {
            "input_row_count": summary["input_row_count"],
            "normalized_row_count": summary["normalized_row_count"],
            "dropped_duplicate_row_count": summary["dropped_duplicate_row_count"],
            "output_row_count": summary["output_row_count"],
            "dropped_invalid_row_count": summary["dropped_invalid_row_count"],
        },
        "symbols": sorted(str(symbol) for symbol in modeling_df["symbol"].astype(str).unique()),
        "target_columns": list(summary["target_columns"]),
    }


def save_m4_modeling_dataset(
    modeling_df: pd.DataFrame,
    *,
    output_path: Path = MODELING_OUTPUT_FILE,
    metadata: dict[str, Any] | None = None,
    metadata_path: Path = MODELING_METADATA_FILE,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.to_parquet(output_path, index=False)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = metadata or {}
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)

    return output_path, metadata_path


def run_m4_target_preparation(
    *,
    input_path: Path = FEATURES_INPUT_FILE,
    output_path: Path = MODELING_OUTPUT_FILE,
    metadata_path: Path = MODELING_METADATA_FILE,
    definition: OfficialTargetDefinition | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed feature dataset not found: {input_path}. "
            "Run `python -m src.data.features` first."
        )

    target_definition = definition or load_m4_target_definition()
    features_df = pd.read_parquet(input_path)
    modeling_df, summary = prepare_m4_modeling_dataset(features_df, target_definition)
    metadata = build_target_preparation_metadata(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        features_df=features_df,
        modeling_df=modeling_df,
        summary=summary,
        definition=target_definition,
    )
    saved_output_path, saved_metadata_path = save_m4_modeling_dataset(
        modeling_df,
        output_path=output_path,
        metadata=metadata,
        metadata_path=metadata_path,
    )

    return {
        "input_path": input_path,
        "output_path": saved_output_path,
        "metadata_path": saved_metadata_path,
        "dataframe": modeling_df,
        "metadata": metadata,
        **summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the official M4 modeling dataset.")
    parser.add_argument("--input", default=str(FEATURES_INPUT_FILE), help="Path to processed feature parquet.")
    parser.add_argument("--output", default=str(MODELING_OUTPUT_FILE), help="Path to modeling dataset parquet.")
    parser.add_argument(
        "--metadata-output",
        default=str(MODELING_METADATA_FILE),
        help="Path to the metadata JSON sidecar.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)

    try:
        result = run_m4_target_preparation(
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Target preparation failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 target preparation completed")
    print(f"Input features: {result['input_path']}")
    print(f"Output dataset: {result['output_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Rows saved: {result['output_row_count']}")
    print(f"Dropped duplicate rows: {result['dropped_duplicate_row_count']}")
    print(f"Dropped invalid rows: {result['dropped_invalid_row_count']}")
    print("Target columns: " + ", ".join(result["target_columns"]))
    if result["dropped_input_target_columns"]:
        print("Dropped legacy input target columns: " + ", ".join(result["dropped_input_target_columns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
