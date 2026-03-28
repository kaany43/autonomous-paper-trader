from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_yaml
from src.data.splits import M4_SPLIT_CONFIG_PATH, OfficialM4SplitDefinition, load_m4_split_definition
from src.data.targets import (
    M4_TARGET_CONFIG_PATH,
    TARGET_DATE_COLUMN,
    TARGET_VALID_COLUMN,
    OfficialTargetDefinition,
    load_m4_target_definition,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
M4_MODELING_DATASET_CONFIG_PATH = REPO_ROOT / "config" / "modeling" / "m4_dataset.yaml"
FEATURES_INPUT_FILE = PROCESSED_DATA_DIR / "market_features.parquet"
MODELING_OUTPUT_FILE = PROCESSED_DATA_DIR / "m4_modeling_dataset.parquet"
MODELING_METADATA_FILE = PROCESSED_DATA_DIR / "m4_modeling_dataset.metadata.json"
MODELING_DATASET_EXPORT_VERSION = 1


@dataclass(frozen=True)
class OfficialM4ModelingDatasetDefinition:
    milestone: str
    contract_name: str
    version: int
    source_feature_dataset_path: str
    output_dataset_path: str
    metadata_output_path: str
    identifier_columns: tuple[str, ...]
    feature_timestamp_column: str
    target_timestamp_column: str
    target_valid_column: str
    passthrough_feature_columns: tuple[str, ...]
    engineered_feature_columns: tuple[str, ...]
    split_ready_sort_order: tuple[str, ...]
    inference_key_columns: tuple[str, ...]
    duplicate_row_policy: str
    invalid_target_row_policy: str


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


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed").dt.tz_localize(None)


def _ordered_unique_strings(values: list[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def load_m4_modeling_dataset_definition(
    config_path: Path = M4_MODELING_DATASET_CONFIG_PATH,
) -> OfficialM4ModelingDatasetDefinition:
    """Load the official M4 modeling dataset export contract from config."""
    data = load_yaml(config_path)
    dataset_cfg = data.get("dataset")

    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Missing or invalid dataset config in: {config_path}")

    definition = OfficialM4ModelingDatasetDefinition(
        milestone=str(dataset_cfg.get("milestone", "")).strip(),
        contract_name=str(dataset_cfg.get("contract_name", "")).strip(),
        version=int(dataset_cfg.get("version", 0) or 0),
        source_feature_dataset_path=str(dataset_cfg.get("source_feature_dataset_path", "")).strip(),
        output_dataset_path=str(dataset_cfg.get("output_dataset_path", "")).strip(),
        metadata_output_path=str(dataset_cfg.get("metadata_output_path", "")).strip(),
        identifier_columns=_ordered_unique_strings(list(dataset_cfg.get("identifier_columns", []))),
        feature_timestamp_column=str(dataset_cfg.get("feature_timestamp_column", "")).strip(),
        target_timestamp_column=str(dataset_cfg.get("target_timestamp_column", "")).strip(),
        target_valid_column=str(dataset_cfg.get("target_valid_column", "")).strip(),
        passthrough_feature_columns=_ordered_unique_strings(
            list(dataset_cfg.get("passthrough_feature_columns", []))
        ),
        engineered_feature_columns=_ordered_unique_strings(
            list(dataset_cfg.get("engineered_feature_columns", []))
        ),
        split_ready_sort_order=_ordered_unique_strings(
            list(dataset_cfg.get("split_ready_sort_order", []))
        ),
        inference_key_columns=_ordered_unique_strings(list(dataset_cfg.get("inference_key_columns", []))),
        duplicate_row_policy=str(dataset_cfg.get("duplicate_row_policy", "")).strip(),
        invalid_target_row_policy=str(dataset_cfg.get("invalid_target_row_policy", "")).strip(),
    )

    if definition.milestone != "M4":
        raise ValueError("Official modeling dataset config milestone must be 'M4'.")
    if not definition.contract_name:
        raise ValueError("Official modeling dataset contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official modeling dataset version must be >= 1.")
    if definition.source_feature_dataset_path != "data/processed/market_features.parquet":
        raise ValueError(
            "Official modeling dataset source_feature_dataset_path must be 'data/processed/market_features.parquet'."
        )
    if definition.output_dataset_path != "data/processed/m4_modeling_dataset.parquet":
        raise ValueError(
            "Official modeling dataset output_dataset_path must be 'data/processed/m4_modeling_dataset.parquet'."
        )
    if definition.metadata_output_path != "data/processed/m4_modeling_dataset.metadata.json":
        raise ValueError(
            "Official modeling dataset metadata_output_path must be 'data/processed/m4_modeling_dataset.metadata.json'."
        )
    if definition.identifier_columns != ("symbol",):
        raise ValueError("Official modeling dataset identifier_columns must be ['symbol'].")
    if definition.feature_timestamp_column != "date":
        raise ValueError("Official modeling dataset feature_timestamp_column must be 'date'.")
    if definition.target_timestamp_column != TARGET_DATE_COLUMN:
        raise ValueError(
            f"Official modeling dataset target_timestamp_column must be '{TARGET_DATE_COLUMN}'."
        )
    if definition.target_valid_column != TARGET_VALID_COLUMN:
        raise ValueError(
            f"Official modeling dataset target_valid_column must be '{TARGET_VALID_COLUMN}'."
        )
    if not definition.passthrough_feature_columns:
        raise ValueError("Official modeling dataset passthrough_feature_columns are required.")
    if not definition.engineered_feature_columns:
        raise ValueError("Official modeling dataset engineered_feature_columns are required.")
    if definition.split_ready_sort_order != ("symbol", "date"):
        raise ValueError("Official modeling dataset split_ready_sort_order must be ['symbol', 'date'].")
    if definition.inference_key_columns != ("symbol", "date", TARGET_DATE_COLUMN):
        raise ValueError(
            f"Official modeling dataset inference_key_columns must be ['symbol', 'date', '{TARGET_DATE_COLUMN}']."
        )
    if not definition.duplicate_row_policy:
        raise ValueError("Official modeling dataset duplicate_row_policy is required.")
    if not definition.invalid_target_row_policy:
        raise ValueError("Official modeling dataset invalid_target_row_policy is required.")

    return definition


def get_m4_modeling_feature_columns(
    definition: OfficialM4ModelingDatasetDefinition | None = None,
) -> list[str]:
    resolved_definition = definition or load_m4_modeling_dataset_definition()
    return list(resolved_definition.passthrough_feature_columns) + list(
        resolved_definition.engineered_feature_columns
    )


def get_m4_modeling_dataset_column_order(
    dataset_definition: OfficialM4ModelingDatasetDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
) -> list[str]:
    resolved_dataset_definition = dataset_definition or load_m4_modeling_dataset_definition()
    resolved_target_definition = target_definition or load_m4_target_definition()

    return (
        [resolved_dataset_definition.feature_timestamp_column]
        + list(resolved_dataset_definition.identifier_columns)
        + get_m4_modeling_feature_columns(resolved_dataset_definition)
        + [
            resolved_dataset_definition.target_timestamp_column,
            resolved_dataset_definition.target_valid_column,
            resolved_target_definition.helper_return_column,
            resolved_target_definition.official_target_column,
        ]
    )


def build_m4_modeling_dataset_schema(
    dataset_definition: OfficialM4ModelingDatasetDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
    split_definition: OfficialM4SplitDefinition | None = None,
) -> dict[str, Any]:
    resolved_dataset_definition = dataset_definition or load_m4_modeling_dataset_definition()
    resolved_target_definition = target_definition or load_m4_target_definition()
    resolved_split_definition = split_definition or load_m4_split_definition()

    return {
        "identifier_columns": list(resolved_dataset_definition.identifier_columns),
        "feature_timestamp_columns": [resolved_dataset_definition.feature_timestamp_column],
        "target_timestamp_columns": [resolved_dataset_definition.target_timestamp_column],
        "passthrough_feature_columns": list(resolved_dataset_definition.passthrough_feature_columns),
        "engineered_feature_columns": list(resolved_dataset_definition.engineered_feature_columns),
        "feature_columns": get_m4_modeling_feature_columns(resolved_dataset_definition),
        "target_metadata_columns": [resolved_dataset_definition.target_valid_column],
        "target_label_columns": [
            resolved_target_definition.helper_return_column,
            resolved_target_definition.official_target_column,
        ],
        "all_target_related_columns": [
            resolved_dataset_definition.target_timestamp_column,
            resolved_dataset_definition.target_valid_column,
            resolved_target_definition.helper_return_column,
            resolved_target_definition.official_target_column,
        ],
        "split_ready_columns": list(
            _ordered_unique_strings(
                [
                    *resolved_dataset_definition.identifier_columns,
                    resolved_dataset_definition.feature_timestamp_column,
                    resolved_dataset_definition.target_timestamp_column,
                ]
            )
        ),
        "inference_key_columns": list(resolved_dataset_definition.inference_key_columns),
        "official_sort_order": list(resolved_dataset_definition.split_ready_sort_order),
        "official_split_boundary_column": resolved_split_definition.target_timestamp_column,
        "column_order": get_m4_modeling_dataset_column_order(
            resolved_dataset_definition,
            resolved_target_definition,
        ),
    }


def normalize_m4_modeling_dataset(
    modeling_df: pd.DataFrame,
    dataset_definition: OfficialM4ModelingDatasetDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
) -> pd.DataFrame:
    resolved_dataset_definition = dataset_definition or load_m4_modeling_dataset_definition()
    resolved_target_definition = target_definition or load_m4_target_definition()
    expected_columns = get_m4_modeling_dataset_column_order(
        resolved_dataset_definition,
        resolved_target_definition,
    )

    missing_columns = [column for column in expected_columns if column not in modeling_df.columns]
    if missing_columns:
        raise ValueError(
            "Modeling dataset export is missing required columns: "
            + ", ".join(missing_columns)
        )
    unexpected_columns = [column for column in modeling_df.columns if column not in expected_columns]
    if unexpected_columns:
        raise ValueError(
            "Modeling dataset export contains unexpected columns outside the official schema: "
            + ", ".join(unexpected_columns)
        )

    normalized = modeling_df.copy()
    normalized[resolved_dataset_definition.feature_timestamp_column] = _normalize_timestamp_series(
        normalized[resolved_dataset_definition.feature_timestamp_column]
    )
    normalized[resolved_dataset_definition.target_timestamp_column] = _normalize_timestamp_series(
        normalized[resolved_dataset_definition.target_timestamp_column]
    )

    if normalized[resolved_dataset_definition.feature_timestamp_column].isna().any():
        raise ValueError("Modeling dataset export requires valid non-null feature timestamps.")
    if normalized[resolved_dataset_definition.target_timestamp_column].isna().any():
        raise ValueError("Modeling dataset export requires valid non-null target timestamps.")

    for column in resolved_dataset_definition.identifier_columns:
        if normalized[column].isna().any():
            raise ValueError(
                f"Modeling dataset export requires valid non-null identifier values in '{column}'."
            )
        normalized[column] = normalized[column].astype(str).str.upper().str.strip()
        if normalized[column].eq("").any():
            raise ValueError(
                f"Modeling dataset export requires non-empty identifier values in '{column}'."
            )

    normalized = normalized.loc[:, expected_columns].copy()
    normalized = normalized.sort_values(list(resolved_dataset_definition.split_ready_sort_order)).reset_index(
        drop=True
    )

    return normalized


def validate_m4_modeling_dataset_contract(
    modeling_df: pd.DataFrame,
    dataset_definition: OfficialM4ModelingDatasetDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
    split_definition: OfficialM4SplitDefinition | None = None,
) -> None:
    resolved_dataset_definition = dataset_definition or load_m4_modeling_dataset_definition()
    resolved_target_definition = target_definition or load_m4_target_definition()
    resolved_split_definition = split_definition or load_m4_split_definition()

    if modeling_df.empty:
        raise ValueError("Modeling dataset export produced no rows.")

    expected_columns = get_m4_modeling_dataset_column_order(
        resolved_dataset_definition,
        resolved_target_definition,
    )
    if list(modeling_df.columns) != expected_columns:
        raise ValueError("Modeling dataset export columns do not match the official schema order.")

    if modeling_df.duplicated(
        subset=[
            *resolved_dataset_definition.identifier_columns,
            resolved_dataset_definition.feature_timestamp_column,
        ]
    ).any():
        raise ValueError("Modeling dataset export contains duplicate identifier/timestamp rows.")

    feature_timestamp = modeling_df[resolved_dataset_definition.feature_timestamp_column]
    target_timestamp = modeling_df[resolved_dataset_definition.target_timestamp_column]
    if not bool((target_timestamp > feature_timestamp).all()):
        raise ValueError("Modeling dataset export contains non-forward target timestamps.")

    if not bool(modeling_df[resolved_dataset_definition.target_valid_column].all()):
        raise ValueError("Modeling dataset export includes invalid target rows.")

    if modeling_df[resolved_target_definition.helper_return_column].isna().any():
        raise ValueError("Modeling dataset export contains null helper target returns.")
    if modeling_df[resolved_target_definition.official_target_column].isna().any():
        raise ValueError("Modeling dataset export contains null official target values.")

    official_target = pd.to_numeric(
        modeling_df[resolved_target_definition.official_target_column],
        errors="coerce",
    )
    if official_target.isna().any():
        raise ValueError("Modeling dataset export official target values must be numeric 0/1.")
    if not bool((official_target == official_target.round()).all()):
        raise ValueError("Modeling dataset export official target values must be exact integer 0/1.")
    if not set(official_target.astype("int64").tolist()).issubset({0, 1}):
        raise ValueError("Modeling dataset export official target values must contain only 0/1.")

    observed = modeling_df[
        list(resolved_dataset_definition.split_ready_sort_order)
    ].reset_index(drop=True)
    expected = (
        modeling_df.sort_values(list(resolved_dataset_definition.split_ready_sort_order))
        .reset_index(drop=True)[list(resolved_dataset_definition.split_ready_sort_order)]
    )
    if not observed.equals(expected):
        raise ValueError("Modeling dataset export must be sorted by the official split-ready order.")

    if resolved_split_definition.target_timestamp_column != resolved_dataset_definition.target_timestamp_column:
        raise ValueError("Modeling dataset export target timestamp column must align with the split contract.")
    if resolved_split_definition.symbol_column not in resolved_dataset_definition.identifier_columns:
        raise ValueError("Modeling dataset export identifier columns must align with the split contract.")


def build_m4_modeling_dataset_metadata(
    *,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    features_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
    summary: dict[str, Any],
    dataset_definition: OfficialM4ModelingDatasetDefinition,
    target_definition: OfficialTargetDefinition,
    split_definition: OfficialM4SplitDefinition,
) -> dict[str, Any]:
    schema = build_m4_modeling_dataset_schema(
        dataset_definition=dataset_definition,
        target_definition=target_definition,
        split_definition=split_definition,
    )

    return {
        "pipeline_name": "m4_modeling_dataset_export",
        "pipeline_version": MODELING_DATASET_EXPORT_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "entrypoint": "python -m src.data.target_pipeline",
        "dataset_config_path": str(M4_MODELING_DATASET_CONFIG_PATH),
        "target_config_path": str(M4_TARGET_CONFIG_PATH),
        "split_config_path": str(M4_SPLIT_CONFIG_PATH),
        "input_dataset": {
            "path": str(input_path),
            "row_count": int(len(features_df)),
            "columns": [str(column) for column in features_df.columns],
            "feature_date_start": _format_date(
                features_df[dataset_definition.feature_timestamp_column].min()
                if dataset_definition.feature_timestamp_column in features_df.columns
                else None
            ),
            "feature_date_end": _format_date(
                features_df[dataset_definition.feature_timestamp_column].max()
                if dataset_definition.feature_timestamp_column in features_df.columns
                else None
            ),
        },
        "output_dataset": {
            "path": str(output_path),
            "metadata_path": str(metadata_path),
            "row_count": int(len(modeling_df)),
            "columns": [str(column) for column in modeling_df.columns],
            "feature_date_start": _format_date(modeling_df[dataset_definition.feature_timestamp_column].min()),
            "feature_date_end": _format_date(modeling_df[dataset_definition.feature_timestamp_column].max()),
            "target_date_start": _format_date(modeling_df[dataset_definition.target_timestamp_column].min()),
            "target_date_end": _format_date(modeling_df[dataset_definition.target_timestamp_column].max()),
        },
        "dataset_definition": asdict(dataset_definition),
        "target_definition": asdict(target_definition),
        "split_definition": asdict(split_definition),
        "schema": schema,
        "invalid_label_handling": {
            "definition_policy": target_definition.invalid_target_policy,
            "dataset_policy": dataset_definition.invalid_target_row_policy,
            "dropped_invalid_row_count": summary["dropped_invalid_row_count"],
        },
        "normalization": {
            "dropped_input_target_columns": list(summary["dropped_input_target_columns"]),
            "duplicate_rule": "keep last row by original input order for each symbol/date before sorting",
            "official_sort_order": list(dataset_definition.split_ready_sort_order),
        },
        "split_ready_metadata": {
            "boundary_column": split_definition.target_timestamp_column,
            "method": split_definition.method,
            "sort_order": list(dataset_definition.split_ready_sort_order),
            "validation_start_date": split_definition.validation_start_date,
            "validation_end_date": split_definition.validation_end_date,
        },
        "inference_ready_metadata": {
            "inference_key_columns": list(dataset_definition.inference_key_columns),
            "identifier_columns": list(dataset_definition.identifier_columns),
            "feature_timestamp_column": dataset_definition.feature_timestamp_column,
            "target_timestamp_column": dataset_definition.target_timestamp_column,
        },
        "counts": {
            "input_row_count": summary["input_row_count"],
            "normalized_row_count": summary["normalized_row_count"],
            "dropped_duplicate_row_count": summary["dropped_duplicate_row_count"],
            "output_row_count": summary["output_row_count"],
            "dropped_invalid_row_count": summary["dropped_invalid_row_count"],
            "feature_column_count": len(schema["feature_columns"]),
        },
        "symbols": sorted(str(symbol) for symbol in modeling_df["symbol"].astype(str).unique()),
        "feature_columns": list(schema["feature_columns"]),
        "target_columns": list(schema["all_target_related_columns"]),
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
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(metadata or {}), fh, indent=2, sort_keys=True)

    return output_path, metadata_path


def load_m4_modeling_dataset_bundle(
    *,
    dataset_path: Path = MODELING_OUTPUT_FILE,
    metadata_path: Path = MODELING_METADATA_FILE,
    validate: bool = True,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing modeling dataset artifact: {dataset_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing modeling dataset metadata artifact: {metadata_path}")

    dataset_definition = load_m4_modeling_dataset_definition()
    target_definition = load_m4_target_definition()
    split_definition = load_m4_split_definition()

    dataframe = pd.read_parquet(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Modeling dataset metadata file must contain a JSON object.")

    if validate:
        normalized = normalize_m4_modeling_dataset(
            dataframe,
            dataset_definition=dataset_definition,
            target_definition=target_definition,
        )
        validate_m4_modeling_dataset_contract(
            normalized,
            dataset_definition=dataset_definition,
            target_definition=target_definition,
            split_definition=split_definition,
        )
        dataframe = normalized

    return {
        "dataframe": dataframe,
        "metadata": metadata,
        "schema": metadata.get("schema", {}),
        "dataset_definition": dataset_definition,
        "target_definition": target_definition,
        "split_definition": split_definition,
    }
