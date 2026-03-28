from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.data.loader import load_yaml
from src.data.targets import OfficialTargetDefinition
from src.engine.run_artifacts import RunArtifactManager
from src.strategy.ml_baselines import (
    BaselineModelSpec,
    OfficialM4BaselineTrainingDefinition,
    SUPPORTED_METRICS,
    load_trained_baseline_model,
    prepare_m4_baseline_training_data,
)
from src.data.splits import OfficialM4SplitDefinition


REPO_ROOT = Path(__file__).resolve().parents[2]
M4_BASELINE_EVALUATION_CONFIG_PATH = REPO_ROOT / "config" / "evaluation" / "m4_baseline_evaluation.yaml"
EVALUATION_SUMMARY_JSON_FILENAME = "baseline_evaluation_summary.json"
EVALUATION_SUMMARY_CSV_FILENAME = "baseline_evaluation_summary.csv"
PIPELINE_VERSION = 1


@dataclass(frozen=True)
class OfficialM4BaselineEvaluationDefinition:
    milestone: str
    contract_name: str
    version: int
    training_config_path: str
    output_dir: str
    strategy_name: str
    run_label: str
    metrics: tuple[str, ...]


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


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


def load_m4_baseline_evaluation_definition(
    config_path: Path = M4_BASELINE_EVALUATION_CONFIG_PATH,
) -> OfficialM4BaselineEvaluationDefinition:
    data = load_yaml(config_path)
    evaluation_cfg = data.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        raise ValueError(f"Missing or invalid evaluation config in: {config_path}")

    metrics = tuple(
        str(metric).strip()
        for metric in evaluation_cfg.get("metrics", [])
        if str(metric).strip()
    )
    if not metrics:
        raise ValueError("Official M4 baseline evaluation metrics are required.")

    unsupported_metrics = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if unsupported_metrics:
        raise ValueError("Unsupported evaluation metric(s): " + ", ".join(unsupported_metrics))

    definition = OfficialM4BaselineEvaluationDefinition(
        milestone=str(evaluation_cfg.get("milestone", "")).strip(),
        contract_name=str(evaluation_cfg.get("contract_name", "")).strip(),
        version=int(evaluation_cfg.get("version", 0) or 0),
        training_config_path=str(evaluation_cfg.get("training_config_path", "")).strip(),
        output_dir=str(evaluation_cfg.get("output_dir", "")).strip(),
        strategy_name=str(evaluation_cfg.get("strategy_name", "")).strip(),
        run_label=str(evaluation_cfg.get("run_label", "")).strip(),
        metrics=metrics,
    )

    if definition.milestone != "M4":
        raise ValueError("Official M4 baseline evaluation milestone must be 'M4'.")
    if not definition.contract_name:
        raise ValueError("Official M4 baseline evaluation contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official M4 baseline evaluation version must be >= 1.")
    if definition.training_config_path != "config/modeling/m4_baselines.yaml":
        raise ValueError(
            "Official M4 baseline evaluation training_config_path must be "
            "'config/modeling/m4_baselines.yaml'."
        )
    if definition.output_dir != "outputs/reports/model_evaluations":
        raise ValueError(
            "Official M4 baseline evaluation output_dir must be 'outputs/reports/model_evaluations'."
        )
    if not definition.strategy_name:
        raise ValueError("Official M4 baseline evaluation strategy_name is required.")
    if not definition.run_label:
        raise ValueError("Official M4 baseline evaluation run_label is required.")

    return definition


def _build_baseline_model_spec(raw: dict[str, Any]) -> BaselineModelSpec:
    return BaselineModelSpec(
        name=str(raw.get("name", "")).strip(),
        estimator=str(raw.get("estimator", "")).strip(),
        params={str(key): value for key, value in dict(raw.get("params", {}) or {}).items()},
    )


def _build_training_definition(raw: dict[str, Any]) -> OfficialM4BaselineTrainingDefinition:
    models_raw = raw.get("models", [])
    return OfficialM4BaselineTrainingDefinition(
        milestone=str(raw.get("milestone", "")).strip(),
        contract_name=str(raw.get("contract_name", "")).strip(),
        version=int(raw.get("version", 0) or 0),
        modeling_dataset_path=str(raw.get("modeling_dataset_path", "")).strip(),
        modeling_dataset_metadata_path=str(raw.get("modeling_dataset_metadata_path", "")).strip(),
        split_config_path=str(raw.get("split_config_path", "")).strip(),
        split_metadata_path=str(raw.get("split_metadata_path", "")).strip(),
        output_dir=str(raw.get("output_dir", "")).strip(),
        strategy_name=str(raw.get("strategy_name", "")).strip(),
        run_label=str(raw.get("run_label", "")).strip(),
        target_column=str(raw.get("target_column", "")).strip(),
        metrics=tuple(str(metric).strip() for metric in raw.get("metrics", []) if str(metric).strip()),
        models=tuple(_build_baseline_model_spec(model) for model in models_raw),
    )


def _build_target_definition(raw: dict[str, Any]) -> OfficialTargetDefinition:
    return OfficialTargetDefinition(
        milestone=str(raw.get("milestone", "")).strip(),
        contract_name=str(raw.get("contract_name", "")).strip(),
        version=int(raw.get("version", 0) or 0),
        task_type=str(raw.get("task_type", "")).strip().lower(),
        official_target_column=str(raw.get("official_target_column", "")).strip(),
        helper_return_column=str(raw.get("helper_return_column", "")).strip(),
        forecast_horizon_sessions=int(raw.get("forecast_horizon_sessions", 0) or 0),
        price_column=str(raw.get("price_column", "")).strip(),
        positive_return_threshold=float(raw.get("positive_return_threshold", 0.0) or 0.0),
        invalid_target_policy=str(raw.get("invalid_target_policy", "")).strip(),
        feature_timestamp=str(raw.get("feature_timestamp", "")).strip(),
        target_timestamp=str(raw.get("target_timestamp", "")).strip(),
    )


def _build_split_definition(raw: dict[str, Any]) -> OfficialM4SplitDefinition:
    return OfficialM4SplitDefinition(
        milestone=str(raw.get("milestone", "")).strip(),
        contract_name=str(raw.get("contract_name", "")).strip(),
        version=int(raw.get("version", 0) or 0),
        method=str(raw.get("method", "")).strip(),
        symbol_column=str(raw.get("symbol_column", "")).strip(),
        feature_timestamp_column=str(raw.get("feature_timestamp_column", "")).strip(),
        target_timestamp_column=str(raw.get("target_timestamp_column", "")).strip(),
        official_target_column=str(raw.get("official_target_column", "")).strip(),
        validation_start_date=str(raw.get("validation_start_date", "")).strip(),
        validation_end_date=str(raw.get("validation_end_date", "")).strip(),
    )


def _validate_binary_series(values: pd.Series, *, label: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.isna().any():
        raise ValueError(f"Baseline evaluation requires '{label}' to contain only numeric 0/1 values.")
    if not bool((numeric_values == numeric_values.round()).all()):
        raise ValueError(
            f"Baseline evaluation requires '{label}' to contain exact integer 0/1 values."
        )
    normalized = numeric_values.astype("int64")
    if not set(normalized.tolist()).issubset({0, 1}):
        raise ValueError(f"Baseline evaluation requires '{label}' to contain only 0/1 values.")
    return normalized


def _compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    metric_values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return {metric_name: metric_values[metric_name] for metric_name in metric_names}


def _validate_training_summary(training_summary: dict[str, Any], training_summary_path: Path) -> None:
    if training_summary.get("pipeline_name") != "m4_baseline_model_training":
        raise ValueError(
            f"Training summary at '{training_summary_path}' is not an M4 baseline training artifact."
        )
    if not isinstance(training_summary.get("models"), list) or not training_summary["models"]:
        raise ValueError("Training summary is missing evaluated baseline model entries.")


def load_m4_baseline_training_run_bundle(training_summary_path: Path) -> dict[str, Any]:
    training_summary_path = Path(training_summary_path).resolve()
    if not training_summary_path.exists():
        raise FileNotFoundError(f"Missing baseline training summary artifact: {training_summary_path}")

    training_summary = _read_json(training_summary_path)
    _validate_training_summary(training_summary, training_summary_path)

    training_run_dir = training_summary_path.parent
    training_manifest_path = training_run_dir / "manifest.json"
    training_config_snapshot_path = training_run_dir / "config.json"
    feature_schema_path = training_run_dir / "feature_schema.json"
    split_summary_path = training_run_dir / "split_summary.json"

    for required_path, label in (
        (training_manifest_path, "training manifest"),
        (training_config_snapshot_path, "training config snapshot"),
        (feature_schema_path, "feature schema"),
        (split_summary_path, "split summary"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label} artifact: {required_path}")

    training_manifest = _read_json(training_manifest_path)
    training_config_snapshot = _read_json(training_config_snapshot_path)
    feature_schema = _read_json(feature_schema_path)
    split_summary = _read_json(split_summary_path)

    if "training_definition" not in training_config_snapshot:
        raise ValueError("Training config snapshot is missing 'training_definition'.")
    if "target_definition" not in training_config_snapshot:
        raise ValueError("Training config snapshot is missing 'target_definition'.")
    if "split_definition" not in training_config_snapshot:
        raise ValueError("Training config snapshot is missing 'split_definition'.")

    training_definition = _build_training_definition(training_config_snapshot["training_definition"])
    target_definition = _build_target_definition(training_config_snapshot["target_definition"])
    split_definition = _build_split_definition(training_config_snapshot["split_definition"])

    return {
        "training_summary_path": training_summary_path,
        "training_run_dir": training_run_dir,
        "training_summary": training_summary,
        "training_manifest_path": training_manifest_path,
        "training_manifest": training_manifest,
        "training_config_snapshot_path": training_config_snapshot_path,
        "training_config_snapshot": training_config_snapshot,
        "feature_schema_path": feature_schema_path,
        "feature_schema": feature_schema,
        "split_summary_path": split_summary_path,
        "split_summary": split_summary,
        "training_definition": training_definition,
        "target_definition": target_definition,
        "split_definition": split_definition,
    }


def _build_validation_context(
    validation_df: pd.DataFrame,
    split_definition: OfficialM4SplitDefinition,
) -> dict[str, Any]:
    return {
        "row_count": int(len(validation_df)),
        "feature_date_start": _format_date(validation_df[split_definition.feature_timestamp_column].min()),
        "feature_date_end": _format_date(validation_df[split_definition.feature_timestamp_column].max()),
        "target_date_start": _format_date(validation_df[split_definition.target_timestamp_column].min()),
        "target_date_end": _format_date(validation_df[split_definition.target_timestamp_column].max()),
    }


def run_m4_baseline_evaluation(
    *,
    training_summary_path: Path,
    config_path: Path = M4_BASELINE_EVALUATION_CONFIG_PATH,
    evaluation_definition: OfficialM4BaselineEvaluationDefinition | None = None,
) -> dict[str, Any]:
    resolved_evaluation_definition = evaluation_definition or load_m4_baseline_evaluation_definition(config_path)
    training_bundle = load_m4_baseline_training_run_bundle(training_summary_path)

    if resolved_evaluation_definition.training_config_path != "config/modeling/m4_baselines.yaml":
        raise ValueError("Evaluation contract training_config_path must stay aligned with official baseline training.")

    prepared = prepare_m4_baseline_training_data(
        training_definition=training_bundle["training_definition"],
        split_definition=training_bundle["split_definition"],
        target_definition=training_bundle["target_definition"],
    )

    x_validation = prepared["x_validation"]
    y_validation = prepared["y_validation"]
    validation_df = prepared["validation_dataframe"]
    split_summary = prepared["split_summary"]
    feature_columns = prepared["feature_columns"]
    feature_schema = training_bundle["feature_schema"]
    summary_split = training_bundle["training_summary"].get("official_split", {}).get("summary", {})
    if summary_split:
        expected_validation_rows = int(summary_split.get("validation_row_count", len(validation_df)))
        if int(len(validation_df)) != expected_validation_rows:
            raise ValueError(
                "Rebuilt validation partition does not match the stored training summary row count."
            )

    schema_feature_columns = [str(column) for column in feature_schema.get("feature_columns", [])]
    if schema_feature_columns != [str(column) for column in feature_columns]:
        raise ValueError("Training feature schema does not match the rebuilt validation feature order.")

    output_root = _resolve_repo_path(resolved_evaluation_definition.output_dir)
    manager = RunArtifactManager(
        base_output_dir=output_root,
        strategy_name=resolved_evaluation_definition.strategy_name,
        start_date=str(split_summary.get("validation_feature_date_start") or ""),
        end_date=str(split_summary.get("validation_feature_date_end") or ""),
        run_label=resolved_evaluation_definition.run_label,
        strategy_variant="baseline_reports",
    )

    config_snapshot = {
        "evaluation_definition": asdict(resolved_evaluation_definition),
        "training_summary_path": str(training_summary_path),
        "training_config_snapshot_path": str(training_bundle["training_config_snapshot_path"]),
        "training_run_id": training_bundle["training_summary"].get("run_id", ""),
        "training_manifest_path": str(training_bundle["training_manifest_path"]),
        "recreated_split_definition": asdict(training_bundle["split_definition"]),
        "recreated_target_definition": asdict(training_bundle["target_definition"]),
    }
    manager.write_config_snapshot(config_snapshot)

    validation_context = _build_validation_context(validation_df, training_bundle["split_definition"])
    model_reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    try:
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

            model_metadata = _read_json(model_metadata_path)
            metadata_feature_columns = [str(column) for column in model_metadata.get("feature_columns", [])]
            if metadata_feature_columns != [str(column) for column in feature_columns]:
                raise ValueError(
                    f"Model artifact '{model_name}' feature schema does not match the official validation feature order."
                )
            if str(model_metadata.get("target_column", "")).strip() != training_bundle["training_definition"].target_column:
                raise ValueError(
                    f"Model artifact '{model_name}' target column does not match the training contract."
                )

            model = load_trained_baseline_model(model_artifact_path)
            raw_predictions = pd.Series(model.predict(x_validation), index=y_validation.index)
            y_pred = _validate_binary_series(raw_predictions, label=f"{model_name}_predictions")
            metrics = _compute_classification_metrics(
                y_validation,
                y_pred,
                resolved_evaluation_definition.metrics,
            )

            tn, fp, fn, tp = confusion_matrix(y_validation, y_pred, labels=[0, 1]).ravel().tolist()
            report_payload = {
                "pipeline_name": "m4_baseline_model_evaluation_report",
                "pipeline_version": PIPELINE_VERSION,
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "evaluation_run_id": manager.run_id,
                "training_run_id": training_bundle["training_summary"].get("run_id", ""),
                "model_name": model_name,
                "estimator": str(model_metadata.get("estimator", "")),
                "task_type": training_bundle["target_definition"].task_type,
                "target_column": training_bundle["training_definition"].target_column,
                "validation_dataset": {
                    **validation_context,
                    "positive_rate": float(y_validation.mean()),
                },
                "prediction_summary": {
                    "predicted_positive_rate": float(y_pred.mean()),
                    "predicted_positive_count": int(y_pred.sum()),
                    "predicted_negative_count": int((1 - y_pred).sum()),
                },
                "confusion_matrix": {
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                },
                "metrics": metrics,
                "artifact_references": {
                    "model_artifact_path": str(model_artifact_path),
                    "model_metadata_path": str(model_metadata_path),
                    "training_summary_path": str(training_summary_path),
                    "training_config_snapshot_path": str(training_bundle["training_config_snapshot_path"]),
                    "feature_schema_path": str(training_bundle["feature_schema_path"]),
                    "split_summary_path": str(training_bundle["split_summary_path"]),
                    "split_config_path": training_bundle["training_summary"].get("official_split", {}).get("config_path", ""),
                    "split_metadata_path": training_bundle["training_summary"].get("official_split", {}).get("metadata_path", ""),
                },
            }
            report_path = _write_json(manager.artifact_path(f"{model_name}.evaluation.json"), report_payload)
            manager.register_artifact(f"{model_name}_evaluation", report_path)

            row = {
                "model_name": model_name,
                "estimator": str(model_metadata.get("estimator", "")),
                "task_type": training_bundle["target_definition"].task_type,
                "target_column": training_bundle["training_definition"].target_column,
                "validation_row_count": int(len(y_validation)),
                "validation_feature_date_start": validation_context["feature_date_start"],
                "validation_feature_date_end": validation_context["feature_date_end"],
                "validation_target_date_start": validation_context["target_date_start"],
                "validation_target_date_end": validation_context["target_date_end"],
                "actual_positive_rate": float(y_validation.mean()),
                "predicted_positive_rate": float(y_pred.mean()),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                **metrics,
                "model_artifact_path": str(model_artifact_path),
                "model_metadata_path": str(model_metadata_path),
                "evaluation_report_path": str(report_path),
            }

            model_reports.append(
                {
                    "model_name": model_name,
                    "report_path": str(report_path),
                    "metrics": metrics,
                }
            )
            summary_rows.append(row)

        summary_rows = sorted(summary_rows, key=lambda item: str(item["model_name"]))
        summary_payload = {
            "pipeline_name": "m4_baseline_model_evaluation",
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "entrypoint": "python -m src.engine.evaluate_baselines",
            "evaluation_run_id": manager.run_id,
            "training_run_id": training_bundle["training_summary"].get("run_id", ""),
            "training_summary_path": str(training_summary_path),
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
            },
            "validation_dataset": validation_context,
            "metrics": list(resolved_evaluation_definition.metrics),
            "model_count": len(summary_rows),
            "rows": summary_rows,
        }
        summary_json_path = _write_json(manager.artifact_path(EVALUATION_SUMMARY_JSON_FILENAME), summary_payload)
        manager.register_artifact("evaluation_summary_json", summary_json_path)

        summary_csv_path = manager.artifact_path(EVALUATION_SUMMARY_CSV_FILENAME)
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
        manager.register_artifact("evaluation_summary_csv", summary_csv_path)

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
        "summary_csv_path": summary_csv_path,
        "model_reports": model_reports,
        "validation_row_count": int(len(y_validation)),
    }
