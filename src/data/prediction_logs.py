from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_yaml
from src.data.targets import TARGET_DATE_COLUMN


REPO_ROOT = Path(__file__).resolve().parents[2]
M4_PREDICTION_LOG_CONFIG_PATH = REPO_ROOT / "config" / "modeling" / "m4_prediction_logs.yaml"
MODEL_OUTPUT_LOGGING_VERSION = 1


@dataclass(frozen=True)
class OfficialM4PredictionLogDefinition:
    milestone: str
    contract_name: str
    version: int
    output_filename: str
    metadata_filename: str
    identifier_columns: tuple[str, ...]
    feature_timestamp_column: str
    target_timestamp_column: str
    run_identity_columns: tuple[str, ...]
    model_identity_columns: tuple[str, ...]
    prediction_context_columns: tuple[str, ...]
    prediction_value_columns: tuple[str, ...]
    join_key_columns: tuple[str, ...]
    sort_order: tuple[str, ...]
    duplicate_row_policy: str


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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    return path


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_m4_prediction_log_definition(
    config_path: Path = M4_PREDICTION_LOG_CONFIG_PATH,
) -> OfficialM4PredictionLogDefinition:
    data = load_yaml(config_path)
    logging_cfg = data.get("logging")
    if not isinstance(logging_cfg, dict):
        raise ValueError(f"Missing or invalid prediction log config in: {config_path}")

    definition = OfficialM4PredictionLogDefinition(
        milestone=str(logging_cfg.get("milestone", "")).strip(),
        contract_name=str(logging_cfg.get("contract_name", "")).strip(),
        version=int(logging_cfg.get("version", 0) or 0),
        output_filename=str(logging_cfg.get("output_filename", "")).strip(),
        metadata_filename=str(logging_cfg.get("metadata_filename", "")).strip(),
        identifier_columns=_ordered_unique_strings(list(logging_cfg.get("identifier_columns", []))),
        feature_timestamp_column=str(logging_cfg.get("feature_timestamp_column", "")).strip(),
        target_timestamp_column=str(logging_cfg.get("target_timestamp_column", "")).strip(),
        run_identity_columns=_ordered_unique_strings(list(logging_cfg.get("run_identity_columns", []))),
        model_identity_columns=_ordered_unique_strings(list(logging_cfg.get("model_identity_columns", []))),
        prediction_context_columns=_ordered_unique_strings(
            list(logging_cfg.get("prediction_context_columns", []))
        ),
        prediction_value_columns=_ordered_unique_strings(list(logging_cfg.get("prediction_value_columns", []))),
        join_key_columns=_ordered_unique_strings(list(logging_cfg.get("join_key_columns", []))),
        sort_order=_ordered_unique_strings(list(logging_cfg.get("sort_order", []))),
        duplicate_row_policy=str(logging_cfg.get("duplicate_row_policy", "")).strip(),
    )

    if definition.milestone != "M4":
        raise ValueError("Official prediction log config milestone must be 'M4'.")
    if not definition.contract_name:
        raise ValueError("Official prediction log contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official prediction log version must be >= 1.")
    if definition.output_filename != "baseline_model_predictions.parquet":
        raise ValueError(
            "Official prediction log output_filename must be 'baseline_model_predictions.parquet'."
        )
    if definition.metadata_filename != "baseline_model_predictions.metadata.json":
        raise ValueError(
            "Official prediction log metadata_filename must be 'baseline_model_predictions.metadata.json'."
        )
    if definition.identifier_columns != ("symbol",):
        raise ValueError("Official prediction log identifier_columns must be ['symbol'].")
    if definition.feature_timestamp_column != "date":
        raise ValueError("Official prediction log feature_timestamp_column must be 'date'.")
    if definition.target_timestamp_column != TARGET_DATE_COLUMN:
        raise ValueError(
            f"Official prediction log target_timestamp_column must be '{TARGET_DATE_COLUMN}'."
        )
    if definition.run_identity_columns != ("prediction_run_id", "training_run_id", "inference_partition"):
        raise ValueError(
            "Official prediction log run_identity_columns must be "
            "['prediction_run_id', 'training_run_id', 'inference_partition']."
        )
    if definition.model_identity_columns != (
        "model_name",
        "estimator",
        "model_artifact_path",
        "model_metadata_path",
    ):
        raise ValueError(
            "Official prediction log model_identity_columns must be "
            "['model_name', 'estimator', 'model_artifact_path', 'model_metadata_path']."
        )
    if definition.prediction_context_columns != ("target_column", "task_type"):
        raise ValueError(
            "Official prediction log prediction_context_columns must be ['target_column', 'task_type']."
        )
    if definition.prediction_value_columns != ("predicted_class", "predicted_probability"):
        raise ValueError(
            "Official prediction log prediction_value_columns must be "
            "['predicted_class', 'predicted_probability']."
        )
    if definition.join_key_columns != ("model_name", "symbol", "date", TARGET_DATE_COLUMN):
        raise ValueError(
            f"Official prediction log join_key_columns must be ['model_name', 'symbol', 'date', '{TARGET_DATE_COLUMN}']."
        )
    if definition.sort_order != ("model_name", "symbol", "date", TARGET_DATE_COLUMN):
        raise ValueError(
            f"Official prediction log sort_order must be ['model_name', 'symbol', 'date', '{TARGET_DATE_COLUMN}']."
        )
    if not definition.duplicate_row_policy:
        raise ValueError("Official prediction log duplicate_row_policy is required.")

    return definition


def get_m4_prediction_log_column_order(
    definition: OfficialM4PredictionLogDefinition | None = None,
) -> list[str]:
    resolved_definition = definition or load_m4_prediction_log_definition()
    return (
        list(resolved_definition.run_identity_columns)
        + list(resolved_definition.model_identity_columns)
        + list(resolved_definition.identifier_columns)
        + [
            resolved_definition.feature_timestamp_column,
            resolved_definition.target_timestamp_column,
        ]
        + list(resolved_definition.prediction_context_columns)
        + list(resolved_definition.prediction_value_columns)
    )


def build_m4_prediction_log_schema(
    definition: OfficialM4PredictionLogDefinition | None = None,
) -> dict[str, Any]:
    resolved_definition = definition or load_m4_prediction_log_definition()
    return {
        "run_identity_columns": list(resolved_definition.run_identity_columns),
        "model_identity_columns": list(resolved_definition.model_identity_columns),
        "identifier_columns": list(resolved_definition.identifier_columns),
        "feature_timestamp_columns": [resolved_definition.feature_timestamp_column],
        "target_timestamp_columns": [resolved_definition.target_timestamp_column],
        "prediction_context_columns": list(resolved_definition.prediction_context_columns),
        "prediction_value_columns": list(resolved_definition.prediction_value_columns),
        "join_key_columns": list(resolved_definition.join_key_columns),
        "official_sort_order": list(resolved_definition.sort_order),
        "column_order": get_m4_prediction_log_column_order(resolved_definition),
    }


def normalize_m4_prediction_log(
    prediction_log_df: pd.DataFrame,
    definition: OfficialM4PredictionLogDefinition | None = None,
) -> pd.DataFrame:
    resolved_definition = definition or load_m4_prediction_log_definition()
    expected_columns = get_m4_prediction_log_column_order(resolved_definition)

    missing_columns = [column for column in expected_columns if column not in prediction_log_df.columns]
    if missing_columns:
        raise ValueError(
            "Prediction log is missing required columns: " + ", ".join(missing_columns)
        )
    unexpected_columns = [column for column in prediction_log_df.columns if column not in expected_columns]
    if unexpected_columns:
        raise ValueError(
            "Prediction log contains unexpected columns outside the official schema: "
            + ", ".join(unexpected_columns)
        )

    normalized = prediction_log_df.copy()
    normalized[resolved_definition.feature_timestamp_column] = _normalize_timestamp_series(
        normalized[resolved_definition.feature_timestamp_column]
    )
    normalized[resolved_definition.target_timestamp_column] = _normalize_timestamp_series(
        normalized[resolved_definition.target_timestamp_column]
    )
    if normalized[resolved_definition.feature_timestamp_column].isna().any():
        raise ValueError("Prediction log requires valid non-null feature timestamps.")
    if normalized[resolved_definition.target_timestamp_column].isna().any():
        raise ValueError("Prediction log requires valid non-null target timestamps.")

    for column in resolved_definition.identifier_columns:
        if normalized[column].isna().any():
            raise ValueError(f"Prediction log requires valid non-null identifier values in '{column}'.")
        normalized[column] = normalized[column].astype(str).str.upper().str.strip()
        if normalized[column].eq("").any():
            raise ValueError(f"Prediction log requires non-empty identifier values in '{column}'.")

    required_non_empty_columns = (
        list(resolved_definition.run_identity_columns)
        + list(resolved_definition.model_identity_columns)
        + list(resolved_definition.prediction_context_columns)
    )
    for column in required_non_empty_columns:
        if normalized[column].isna().any():
            raise ValueError(f"Prediction log requires non-null values in '{column}'.")
        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise ValueError(f"Prediction log requires non-empty values in '{column}'.")

    predicted_class = pd.to_numeric(normalized["predicted_class"], errors="coerce")
    if predicted_class.isna().any():
        raise ValueError("Prediction log requires predicted_class to contain only numeric 0/1 values.")
    if not bool((predicted_class == predicted_class.round()).all()):
        raise ValueError("Prediction log requires predicted_class to contain exact integer 0/1 values.")
    predicted_class = predicted_class.astype("int64")
    if not set(predicted_class.tolist()).issubset({0, 1}):
        raise ValueError("Prediction log requires predicted_class to contain only 0/1 values.")
    normalized["predicted_class"] = predicted_class

    predicted_probability = pd.to_numeric(normalized["predicted_probability"], errors="coerce")
    invalid_probability_mask = normalized["predicted_probability"].notna() & predicted_probability.isna()
    if invalid_probability_mask.any():
        raise ValueError(
            "Prediction log requires predicted_probability to be numeric or null."
        )
    out_of_bounds_probability = predicted_probability.dropna()
    if ((out_of_bounds_probability < 0.0) | (out_of_bounds_probability > 1.0)).any():
        raise ValueError("Prediction log requires predicted_probability values to stay within [0, 1].")
    normalized["predicted_probability"] = predicted_probability.astype("Float64")

    normalized = normalized.loc[:, expected_columns].copy()
    normalized = normalized.sort_values(list(resolved_definition.sort_order)).reset_index(drop=True)
    return normalized


def validate_m4_prediction_log_contract(
    prediction_log_df: pd.DataFrame,
    definition: OfficialM4PredictionLogDefinition | None = None,
) -> None:
    resolved_definition = definition or load_m4_prediction_log_definition()
    if prediction_log_df.empty:
        raise ValueError("Prediction log produced no rows.")

    expected_columns = get_m4_prediction_log_column_order(resolved_definition)
    if list(prediction_log_df.columns) != expected_columns:
        raise ValueError("Prediction log columns do not match the official schema order.")

    if prediction_log_df.duplicated(subset=list(resolved_definition.join_key_columns)).any():
        raise ValueError("Prediction log contains duplicate model/timestamp prediction rows.")

    observed = prediction_log_df[list(resolved_definition.sort_order)].reset_index(drop=True)
    expected = (
        prediction_log_df.sort_values(list(resolved_definition.sort_order))
        .reset_index(drop=True)[list(resolved_definition.sort_order)]
    )
    if not observed.equals(expected):
        raise ValueError("Prediction log must be sorted by the official sort order.")

    feature_timestamp = prediction_log_df[resolved_definition.feature_timestamp_column]
    target_timestamp = prediction_log_df[resolved_definition.target_timestamp_column]
    if not bool((target_timestamp > feature_timestamp).all()):
        raise ValueError("Prediction log contains non-forward target timestamps.")


def build_m4_prediction_log_metadata(
    *,
    output_path: Path,
    metadata_path: Path,
    prediction_log_df: pd.DataFrame,
    definition: OfficialM4PredictionLogDefinition,
    prediction_run_id: str,
    training_run_id: str,
    training_summary_path: Path,
    training_output_dir: Path,
    feature_schema_path: Path,
    split_summary_path: Path,
    prediction_config_path: Path,
    source_dataset_path: Path,
    source_dataset_metadata_path: Path,
    split_config_path: Path,
    split_metadata_path: Path,
    split_summary: dict[str, Any],
    logged_output_signature: str,
    source_models: list[dict[str, Any]],
    inference_partition: str,
    target_column: str,
    task_type: str,
    inference_key_columns: list[str],
) -> dict[str, Any]:
    schema = build_m4_prediction_log_schema(definition)
    return {
        "pipeline_name": "m4_model_output_logging",
        "pipeline_version": MODEL_OUTPUT_LOGGING_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "entrypoint": "python -m src.engine.generate_predictions",
        "prediction_log_config_path": str(M4_PREDICTION_LOG_CONFIG_PATH),
        "prediction_config_path": str(prediction_config_path),
        "logging_definition": asdict(definition),
        "output_log": {
            "path": str(output_path),
            "metadata_path": str(metadata_path),
            "row_count": int(len(prediction_log_df)),
            "columns": [str(column) for column in prediction_log_df.columns],
            "feature_date_start": _format_date(prediction_log_df[definition.feature_timestamp_column].min()),
            "feature_date_end": _format_date(prediction_log_df[definition.feature_timestamp_column].max()),
            "target_date_start": _format_date(prediction_log_df[definition.target_timestamp_column].min()),
            "target_date_end": _format_date(prediction_log_df[definition.target_timestamp_column].max()),
            "join_key_columns": list(definition.join_key_columns),
            "sort_order": list(definition.sort_order),
            "output_signature": logged_output_signature,
        },
        "prediction_context": {
            "prediction_run_id": str(prediction_run_id),
            "training_run_id": str(training_run_id),
            "inference_partition": str(inference_partition),
            "target_column": str(target_column),
            "task_type": str(task_type),
            "inference_key_columns": list(inference_key_columns),
        },
        "source_artifacts": {
            "training_summary_path": str(training_summary_path),
            "training_output_dir": str(training_output_dir),
            "feature_schema_path": str(feature_schema_path),
            "split_summary_path": str(split_summary_path),
            "source_dataset_path": str(source_dataset_path),
            "source_dataset_metadata_path": str(source_dataset_metadata_path),
            "split_config_path": str(split_config_path),
            "split_metadata_path": str(split_metadata_path),
            "split_metadata_exists": bool(split_metadata_path.exists()),
        },
        "official_split": {
            "summary": dict(split_summary),
        },
        "schema": schema,
        "counts": {
            "row_count": int(len(prediction_log_df)),
            "model_count": int(prediction_log_df["model_name"].nunique()),
            "symbol_count": int(prediction_log_df["symbol"].nunique()),
        },
        "models": list(source_models),
    }


def save_m4_prediction_log(
    prediction_log_df: pd.DataFrame,
    *,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
    metadata_path: Path,
    definition: OfficialM4PredictionLogDefinition | None = None,
) -> tuple[Path, Path]:
    resolved_definition = definition or load_m4_prediction_log_definition()
    normalized = normalize_m4_prediction_log(prediction_log_df, resolved_definition)
    validate_m4_prediction_log_contract(normalized, resolved_definition)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_path, metadata or {})
    return output_path, metadata_path


def load_m4_prediction_log_bundle(
    *,
    dataset_path: Path,
    metadata_path: Path,
    validate: bool = True,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing prediction log artifact: {dataset_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing prediction log metadata artifact: {metadata_path}")

    definition = load_m4_prediction_log_definition()
    dataframe = pd.read_parquet(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Prediction log metadata file must contain a JSON object.")

    if validate:
        normalized = normalize_m4_prediction_log(dataframe, definition)
        validate_m4_prediction_log_contract(normalized, definition)
        dataframe = normalized

    return {
        "dataframe": dataframe,
        "metadata": metadata,
        "schema": metadata.get("schema", {}),
        "definition": definition,
    }
