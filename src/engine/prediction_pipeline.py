from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.loader import load_yaml
from src.engine.model_evaluation import load_m4_baseline_training_run_bundle
from src.engine.run_artifacts import RunArtifactManager
from src.strategy.ml_baselines import (
    build_dataframe_signature,
    load_trained_baseline_model,
    prepare_m4_baseline_training_data,
)
from src.data.splits import OfficialM4SplitDefinition


REPO_ROOT = Path(__file__).resolve().parents[2]
M4_BATCH_PREDICTION_CONFIG_PATH = REPO_ROOT / "config" / "evaluation" / "m4_batch_prediction.yaml"
BATCH_PREDICTIONS_FILENAME = "baseline_model_predictions.parquet"
BATCH_PREDICTION_SUMMARY_FILENAME = "baseline_prediction_summary.json"
PIPELINE_VERSION = 1


@dataclass(frozen=True)
class OfficialM4BatchPredictionDefinition:
    milestone: str
    contract_name: str
    version: int
    training_config_path: str
    output_dir: str
    strategy_name: str
    run_label: str
    inference_partition: str
    prediction_task_type: str


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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    return path


def _resolve_repo_path(value: str) -> Path:
    path = Path(str(value).strip())
    if not str(path):
        raise ValueError("Configured path value cannot be empty.")
    return path if path.is_absolute() else REPO_ROOT / path


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _validate_binary_predictions(values: pd.Series, *, label: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.isna().any():
        raise ValueError(f"Baseline batch prediction requires '{label}' to contain only numeric 0/1 values.")
    if not bool((numeric_values == numeric_values.round()).all()):
        raise ValueError(
            f"Baseline batch prediction requires '{label}' to contain exact integer 0/1 values."
        )
    normalized = numeric_values.astype("int64")
    if not set(normalized.tolist()).issubset({0, 1}):
        raise ValueError(f"Baseline batch prediction requires '{label}' to contain only 0/1 values.")
    return normalized


def _extract_positive_class_probability(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    if not hasattr(model, "predict_proba"):
        return pd.Series(pd.array([pd.NA] * len(features), dtype="Float64"), index=features.index)

    probabilities = model.predict_proba(features)
    if len(probabilities) != len(features):
        raise ValueError("Batch prediction probability output length does not match the inference input rows.")

    estimator = model.named_steps.get("model", model)
    classes = list(getattr(estimator, "classes_", []))
    if 1 not in classes:
        raise ValueError("Batch prediction requires the fitted classifier to expose positive class '1'.")
    positive_index = classes.index(1)
    positive_probabilities = pd.Series(probabilities[:, positive_index], index=features.index, dtype="float64")
    if ((positive_probabilities < 0.0) | (positive_probabilities > 1.0)).any():
        raise ValueError("Batch prediction probabilities must stay within [0, 1].")
    return positive_probabilities


def _build_inference_context(
    inference_df: pd.DataFrame,
    split_definition: OfficialM4SplitDefinition,
    *,
    partition_name: str,
) -> dict[str, Any]:
    return {
        "partition_name": partition_name,
        "row_count": int(len(inference_df)),
        "feature_date_start": _format_date(inference_df[split_definition.feature_timestamp_column].min()),
        "feature_date_end": _format_date(inference_df[split_definition.feature_timestamp_column].max()),
        "target_date_start": _format_date(inference_df[split_definition.target_timestamp_column].min()),
        "target_date_end": _format_date(inference_df[split_definition.target_timestamp_column].max()),
    }


def load_m4_batch_prediction_definition(
    config_path: Path = M4_BATCH_PREDICTION_CONFIG_PATH,
) -> OfficialM4BatchPredictionDefinition:
    data = load_yaml(config_path)
    prediction_cfg = data.get("prediction")
    if not isinstance(prediction_cfg, dict):
        raise ValueError(f"Missing or invalid prediction config in: {config_path}")

    definition = OfficialM4BatchPredictionDefinition(
        milestone=str(prediction_cfg.get("milestone", "")).strip(),
        contract_name=str(prediction_cfg.get("contract_name", "")).strip(),
        version=int(prediction_cfg.get("version", 0) or 0),
        training_config_path=str(prediction_cfg.get("training_config_path", "")).strip(),
        output_dir=str(prediction_cfg.get("output_dir", "")).strip(),
        strategy_name=str(prediction_cfg.get("strategy_name", "")).strip(),
        run_label=str(prediction_cfg.get("run_label", "")).strip(),
        inference_partition=str(prediction_cfg.get("inference_partition", "")).strip().lower(),
        prediction_task_type=str(prediction_cfg.get("prediction_task_type", "")).strip().lower(),
    )

    if definition.milestone != "M4":
        raise ValueError("Official M4 batch prediction milestone must be 'M4'.")
    if not definition.contract_name:
        raise ValueError("Official M4 batch prediction contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official M4 batch prediction version must be >= 1.")
    if definition.training_config_path != "config/modeling/m4_baselines.yaml":
        raise ValueError(
            "Official M4 batch prediction training_config_path must be "
            "'config/modeling/m4_baselines.yaml'."
        )
    if definition.output_dir != "outputs/predictions/model_batches":
        raise ValueError(
            "Official M4 batch prediction output_dir must be 'outputs/predictions/model_batches'."
        )
    if not definition.strategy_name:
        raise ValueError("Official M4 batch prediction strategy_name is required.")
    if not definition.run_label:
        raise ValueError("Official M4 batch prediction run_label is required.")
    if definition.inference_partition != "validation":
        raise ValueError("Official M4 batch prediction inference_partition must be 'validation'.")
    if definition.prediction_task_type != "classification":
        raise ValueError("Official M4 batch prediction prediction_task_type must be 'classification'.")

    return definition


def run_m4_batch_prediction(
    *,
    training_summary_path: Path,
    config_path: Path = M4_BATCH_PREDICTION_CONFIG_PATH,
    prediction_definition: OfficialM4BatchPredictionDefinition | None = None,
) -> dict[str, Any]:
    resolved_prediction_definition = prediction_definition or load_m4_batch_prediction_definition(config_path)
    training_bundle = load_m4_baseline_training_run_bundle(training_summary_path)

    if resolved_prediction_definition.training_config_path != "config/modeling/m4_baselines.yaml":
        raise ValueError("Prediction contract training_config_path must stay aligned with official baseline training.")

    prepared = prepare_m4_baseline_training_data(
        training_definition=training_bundle["training_definition"],
        split_definition=training_bundle["split_definition"],
        target_definition=training_bundle["target_definition"],
    )
    if resolved_prediction_definition.inference_partition != "validation":
        raise ValueError(
            "Unsupported official inference partition. Expected 'validation' for M4 baseline batch prediction."
        )

    inference_df = prepared["validation_dataframe"].copy()
    x_inference = prepared["x_validation"]
    y_inference = prepared["y_validation"]
    split_summary = prepared["split_summary"]
    feature_columns = prepared["feature_columns"]
    feature_schema = training_bundle["feature_schema"]

    summary_split = training_bundle["training_summary"].get("official_split", {}).get("summary", {})
    if summary_split:
        expected_validation_rows = int(summary_split.get("validation_row_count", len(inference_df)))
        if int(len(inference_df)) != expected_validation_rows:
            raise ValueError(
                "Rebuilt validation partition does not match the stored training summary row count."
            )

    expected_validation_signature = str(
        training_bundle["split_summary"].get("validation_dataset_signature")
        or summary_split.get("validation_dataset_signature")
        or ""
    ).strip()
    rebuilt_validation_signature = build_dataframe_signature(inference_df)
    signature_check_status = "not_available_in_training_artifacts"
    if expected_validation_signature:
        signature_check_status = "matched"
    if expected_validation_signature and rebuilt_validation_signature != expected_validation_signature:
        raise ValueError(
            "Rebuilt validation partition signature does not match the stored training artifacts."
        )

    schema_feature_columns = [str(column) for column in feature_schema.get("feature_columns", [])]
    if schema_feature_columns != [str(column) for column in feature_columns]:
        raise ValueError("Training feature schema does not match the rebuilt inference feature order.")

    task_type = str(training_bundle["target_definition"].task_type).strip().lower()
    if task_type != resolved_prediction_definition.prediction_task_type:
        raise ValueError("Prediction task type does not match the trained M4 baseline target contract.")

    inference_key_columns = [str(column) for column in feature_schema.get("inference_key_columns", [])]
    if not inference_key_columns:
        raise ValueError("Training artifacts are missing inference_key_columns.")
    missing_inference_keys = [column for column in inference_key_columns if column not in inference_df.columns]
    if missing_inference_keys:
        raise ValueError(
            "Inference dataset is missing required key columns: " + ", ".join(missing_inference_keys)
        )

    output_root = _resolve_repo_path(resolved_prediction_definition.output_dir)
    manager = RunArtifactManager(
        base_output_dir=output_root,
        strategy_name=resolved_prediction_definition.strategy_name,
        start_date=str(split_summary.get("validation_feature_date_start") or ""),
        end_date=str(split_summary.get("validation_feature_date_end") or ""),
        run_label=resolved_prediction_definition.run_label,
        strategy_variant="batch_predictions",
    )

    config_snapshot = {
        "prediction_definition": asdict(resolved_prediction_definition),
        "training_summary_path": str(training_bundle["training_summary_path"]),
        "training_config_snapshot_path": str(training_bundle["training_config_snapshot_path"]),
        "training_run_id": training_bundle["training_summary"].get("run_id", ""),
        "training_manifest_path": str(training_bundle["training_manifest_path"]),
        "recreated_training_definition": asdict(training_bundle["training_definition"]),
        "recreated_split_definition": asdict(training_bundle["split_definition"]),
        "recreated_target_definition": asdict(training_bundle["target_definition"]),
    }
    manager.write_config_snapshot(config_snapshot)

    inference_context = _build_inference_context(
        inference_df,
        training_bundle["split_definition"],
        partition_name=resolved_prediction_definition.inference_partition,
    )
    model_summaries: list[dict[str, Any]] = []

    try:
        prediction_frames: list[pd.DataFrame] = []
        for model_record in sorted(
            training_bundle["training_summary"]["models"],
            key=lambda item: str(item.get("model_name", "")),
        ):
            model_name = str(model_record.get("model_name", "")).strip()
            if not model_name:
                raise ValueError("Training summary contains a model entry without model_name.")

            model_artifact_raw = str(model_record.get("artifact_path", "")).strip()
            model_metadata_raw = str(model_record.get("metadata_path", "")).strip()
            if not model_artifact_raw:
                raise ValueError(f"Training summary is missing artifact_path for model '{model_name}'.")
            if not model_metadata_raw:
                raise ValueError(f"Training summary is missing metadata_path for model '{model_name}'.")
            model_artifact_path = Path(model_artifact_raw)
            model_metadata_path = Path(model_metadata_raw)
            if not model_artifact_path.exists():
                raise FileNotFoundError(f"Missing trained model artifact: {model_artifact_path}")
            if not model_metadata_path.exists():
                raise FileNotFoundError(f"Missing trained model metadata artifact: {model_metadata_path}")

            with model_metadata_path.open("r", encoding="utf-8") as fh:
                model_metadata = json.load(fh)
            if not isinstance(model_metadata, dict):
                raise ValueError(f"Model metadata must be a JSON object: {model_metadata_path}")

            metadata_feature_columns = [str(column) for column in model_metadata.get("feature_columns", [])]
            if metadata_feature_columns != [str(column) for column in feature_columns]:
                raise ValueError(
                    f"Model artifact '{model_name}' feature schema does not match the official inference feature order."
                )
            if str(model_metadata.get("target_column", "")).strip() != training_bundle["training_definition"].target_column:
                raise ValueError(
                    f"Model artifact '{model_name}' target column does not match the training contract."
                )
            if str(model_metadata.get("task_type", "")).strip().lower() != task_type:
                raise ValueError(f"Model artifact '{model_name}' task_type does not match the training contract.")

            model = load_trained_baseline_model(model_artifact_path)
            predicted_class = _validate_binary_predictions(
                pd.Series(model.predict(x_inference), index=x_inference.index),
                label=f"{model_name}_predictions",
            )
            predicted_probability = _extract_positive_class_probability(model, x_inference)

            model_predictions = inference_df.loc[:, inference_key_columns].copy()
            model_predictions["model_name"] = model_name
            model_predictions["target_column"] = training_bundle["training_definition"].target_column
            model_predictions["task_type"] = task_type
            model_predictions["predicted_class"] = predicted_class.astype("int64")
            model_predictions["predicted_probability"] = predicted_probability.astype("Float64")
            prediction_frames.append(
                model_predictions.loc[
                    :,
                    [
                        "model_name",
                        *inference_key_columns,
                        "target_column",
                        "task_type",
                        "predicted_class",
                        "predicted_probability",
                    ],
                ]
            )

            model_summaries.append(
                {
                    "model_name": model_name,
                    "estimator": str(model_metadata.get("estimator", "")).strip(),
                    "row_count": int(len(model_predictions)),
                    "predicted_positive_rate": float(predicted_class.mean()),
                    "mean_predicted_probability": (
                        float(predicted_probability.dropna().mean())
                        if bool(predicted_probability.dropna().shape[0])
                        else None
                    ),
                    "model_artifact_path": str(model_artifact_path),
                    "model_metadata_path": str(model_metadata_path),
                }
            )

        predictions_df = pd.concat(prediction_frames, ignore_index=True)
        predictions_df = predictions_df.sort_values(
            ["model_name", *inference_key_columns]
        ).reset_index(drop=True)

        predictions_path = manager.artifact_path(BATCH_PREDICTIONS_FILENAME)
        predictions_df.to_parquet(predictions_path, index=False)
        manager.register_artifact("batch_predictions", predictions_path)

        output_signature = build_dataframe_signature(predictions_df)
        summary_payload = {
            "pipeline_name": "m4_baseline_batch_prediction",
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "entrypoint": "python -m src.engine.generate_predictions",
            "prediction_run_id": manager.run_id,
            "training_run_id": training_bundle["training_summary"].get("run_id", ""),
            "training_summary_path": str(training_bundle["training_summary_path"]),
            "training_output_dir": str(training_bundle["training_run_dir"]),
            "official_dataset": {
                "path": str(prepared["dataset_path"]),
                "metadata_path": str(prepared["dataset_metadata_path"]),
            },
            "official_split": {
                "config_path": str(prepared["split_config_path"]),
                "metadata_path": str(prepared["split_metadata_path"]),
                "metadata_exists": bool(prepared["split_metadata_path"].exists()),
                "method": training_bundle["split_definition"].method,
                "boundary_column": training_bundle["split_definition"].target_timestamp_column,
                "validation_start_date": training_bundle["split_definition"].validation_start_date,
                "validation_end_date": training_bundle["split_definition"].validation_end_date,
                "summary": split_summary,
                "stored_validation_dataset_signature": expected_validation_signature or None,
                "rebuilt_validation_dataset_signature": rebuilt_validation_signature,
                "validation_dataset_signature_check_status": signature_check_status,
            },
            "inference_dataset": {
                **inference_context,
                "actual_target_available": True,
                "actual_positive_rate": float(y_inference.mean()),
                "inference_key_columns": inference_key_columns,
                "feature_columns": feature_columns,
                "feature_column_count": len(feature_columns),
                "source_training_feature_schema_path": str(training_bundle["feature_schema_path"]),
            },
            "prediction_output": {
                "path": str(predictions_path),
                "format": "parquet",
                "row_count": int(len(predictions_df)),
                "columns": [str(column) for column in predictions_df.columns],
                "sort_order": ["model_name", *inference_key_columns],
                "prediction_output_signature": output_signature,
                "positive_probability_semantics": "Probability that target_next_session_direction == 1.",
            },
            "model_count": len(model_summaries),
            "models": model_summaries,
        }
        summary_json_path = _write_json(
            manager.artifact_path(BATCH_PREDICTION_SUMMARY_FILENAME),
            summary_payload,
        )
        manager.register_artifact("prediction_summary", summary_json_path)
        manifest_path = manager.write_manifest(status="completed", config_source=str(config_path))
    except Exception as exc:
        manifest_path = manager.write_manifest(
            status="failed",
            config_source=str(config_path),
            error_message=str(exc),
        )
        raise

    return {
        "run_id": manager.run_id,
        "output_dir": manager.output_dir,
        "manifest_path": manifest_path,
        "summary_json_path": summary_json_path,
        "predictions_path": predictions_path,
        "model_count": len(model_summaries),
        "prediction_row_count": int(len(predictions_df)),
    }
