from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_yaml
from src.data.modeling_dataset import (
    M4_MODELING_DATASET_CONFIG_PATH,
    build_m4_modeling_dataset_schema,
    load_m4_modeling_dataset_bundle,
)
from src.data.splits import (
    M4_SPLIT_CONFIG_PATH,
    OfficialM4SplitDefinition,
    load_m4_split_definition,
    split_m4_modeling_dataset,
)
from src.data.targets import M4_TARGET_CONFIG_PATH, OfficialTargetDefinition, load_m4_target_definition
from src.engine.run_artifacts import RunArtifactManager


REPO_ROOT = Path(__file__).resolve().parents[2]
M4_BASELINE_TRAINING_CONFIG_PATH = REPO_ROOT / "config" / "modeling" / "m4_baselines.yaml"
TRAINING_SUMMARY_FILENAME = "baseline_training_summary.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
SPLIT_SUMMARY_FILENAME = "split_summary.json"
PIPELINE_VERSION = 1
SUPPORTED_METRICS = ("accuracy", "precision", "recall", "f1")
SUPPORTED_ESTIMATORS = ("logistic_regression", "decision_tree_classifier")


@dataclass(frozen=True)
class BaselineModelSpec:
    name: str
    estimator: str
    params: dict[str, Any]


@dataclass(frozen=True)
class OfficialM4BaselineTrainingDefinition:
    milestone: str
    contract_name: str
    version: int
    modeling_dataset_path: str
    modeling_dataset_metadata_path: str
    split_config_path: str
    split_metadata_path: str
    output_dir: str
    strategy_name: str
    run_label: str
    target_column: str
    metrics: tuple[str, ...]
    models: tuple[BaselineModelSpec, ...]


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


def _normalize_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in normalized)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("_-")


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_m4_baseline_training_definition(
    config_path: Path = M4_BASELINE_TRAINING_CONFIG_PATH,
) -> OfficialM4BaselineTrainingDefinition:
    data = load_yaml(config_path)
    training_cfg = data.get("training")
    if not isinstance(training_cfg, dict):
        raise ValueError(f"Missing or invalid training config in: {config_path}")

    models_raw = training_cfg.get("models", [])
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("Official M4 baseline training config requires a non-empty models list.")

    model_specs: list[BaselineModelSpec] = []
    seen_model_names: set[str] = set()
    for index, raw_model in enumerate(models_raw):
        if not isinstance(raw_model, dict):
            raise ValueError(f"training.models[{index}] must be a mapping.")
        model_name = _normalize_name(raw_model.get("name"))
        if not model_name:
            raise ValueError(f"training.models[{index}].name is required.")
        if model_name in seen_model_names:
            raise ValueError(f"Duplicate baseline model name: {model_name}")
        seen_model_names.add(model_name)

        estimator = str(raw_model.get("estimator", "")).strip()
        if estimator not in SUPPORTED_ESTIMATORS:
            raise ValueError(
                "Unsupported baseline estimator: "
                f"{estimator!r}. Expected one of {', '.join(SUPPORTED_ESTIMATORS)}."
            )

        params = raw_model.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError(f"training.models[{index}].params must be a mapping.")

        model_specs.append(
            BaselineModelSpec(
                name=model_name,
                estimator=estimator,
                params={str(key): value for key, value in params.items()},
            )
        )

    metrics_raw = training_cfg.get("metrics", [])
    metrics = tuple(str(metric).strip() for metric in metrics_raw if str(metric).strip())
    if not metrics:
        raise ValueError("Official M4 baseline training metrics are required.")
    unsupported_metrics = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if unsupported_metrics:
        raise ValueError("Unsupported baseline metric(s): " + ", ".join(unsupported_metrics))

    definition = OfficialM4BaselineTrainingDefinition(
        milestone=str(training_cfg.get("milestone", "")).strip(),
        contract_name=str(training_cfg.get("contract_name", "")).strip(),
        version=int(training_cfg.get("version", 0) or 0),
        modeling_dataset_path=str(training_cfg.get("modeling_dataset_path", "")).strip(),
        modeling_dataset_metadata_path=str(
            training_cfg.get("modeling_dataset_metadata_path", "")
        ).strip(),
        split_config_path=str(training_cfg.get("split_config_path", "")).strip(),
        split_metadata_path=str(training_cfg.get("split_metadata_path", "")).strip(),
        output_dir=str(training_cfg.get("output_dir", "")).strip(),
        strategy_name=str(training_cfg.get("strategy_name", "")).strip(),
        run_label=str(training_cfg.get("run_label", "")).strip(),
        target_column=str(training_cfg.get("target_column", "")).strip(),
        metrics=metrics,
        models=tuple(model_specs),
    )

    target_definition = load_m4_target_definition()
    if definition.milestone != "M4":
        raise ValueError("Official M4 baseline training milestone must be 'M4'.")
    if not definition.contract_name:
        raise ValueError("Official M4 baseline training contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official M4 baseline training version must be >= 1.")
    if definition.modeling_dataset_path != "data/processed/m4_modeling_dataset.parquet":
        raise ValueError(
            "Official M4 baseline training modeling_dataset_path must be "
            "'data/processed/m4_modeling_dataset.parquet'."
        )
    if definition.modeling_dataset_metadata_path != "data/processed/m4_modeling_dataset.metadata.json":
        raise ValueError(
            "Official M4 baseline training modeling_dataset_metadata_path must be "
            "'data/processed/m4_modeling_dataset.metadata.json'."
        )
    if definition.split_config_path != "config/modeling/m4_split.yaml":
        raise ValueError(
            "Official M4 baseline training split_config_path must be "
            "'config/modeling/m4_split.yaml'."
        )
    if definition.split_metadata_path != "data/processed/m4_train_validation_split.metadata.json":
        raise ValueError(
            "Official M4 baseline training split_metadata_path must be "
            "'data/processed/m4_train_validation_split.metadata.json'."
        )
    if definition.output_dir != "outputs/models":
        raise ValueError("Official M4 baseline training output_dir must be 'outputs/models'.")
    if not definition.strategy_name:
        raise ValueError("Official M4 baseline training strategy_name is required.")
    if not definition.run_label:
        raise ValueError("Official M4 baseline training run_label is required.")
    if definition.target_column != target_definition.official_target_column:
        raise ValueError(
            "Official M4 baseline training target_column must match the official M4 target."
        )

    return definition


def _coerce_numeric_feature_frame(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    partition_name: str,
) -> pd.DataFrame:
    missing_columns = [column for column in feature_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Baseline training {partition_name} partition is missing feature columns: "
            + ", ".join(missing_columns)
        )

    numeric_frame = pd.DataFrame(index=dataframe.index)
    for column in feature_columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_mask = dataframe[column].notna() & series.isna()
        if invalid_mask.any():
            raise ValueError(
                f"Baseline training {partition_name} partition contains non-numeric values in '{column}'."
            )
        numeric_frame[column] = series.astype("float64")

    if partition_name == "training":
        empty_training_features = [
            column for column in feature_columns if not bool(numeric_frame[column].notna().any())
        ]
        if empty_training_features:
            raise ValueError(
                "Baseline training requires at least one observed training value for each feature column. "
                "Empty columns: "
                + ", ".join(empty_training_features)
            )

    return numeric_frame.loc[:, feature_columns]


def _coerce_binary_target(
    values: pd.Series,
    *,
    column_name: str,
    partition_name: str,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.isna().any():
        raise ValueError(
            f"Baseline training {partition_name} partition requires '{column_name}' "
            "to contain only numeric 0/1 values."
        )
    if not bool((numeric_values == numeric_values.round()).all()):
        raise ValueError(
            f"Baseline training {partition_name} partition requires '{column_name}' "
            "to contain exact integer 0/1 values."
        )

    normalized = numeric_values.astype("int64")
    if not set(normalized.tolist()).issubset({0, 1}):
        raise ValueError(
            f"Baseline training {partition_name} partition requires '{column_name}' "
            "to contain only 0/1 values."
        )

    return normalized


def _build_estimator_pipeline(model_spec: BaselineModelSpec) -> Pipeline:
    params = dict(model_spec.params)

    if model_spec.estimator == "logistic_regression":
        params.setdefault("solver", "liblinear")
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", 42)
        estimator = LogisticRegression(**params)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_spec.estimator == "decision_tree_classifier":
        params.setdefault("random_state", 42)
        estimator = DecisionTreeClassifier(**params)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    raise ValueError(f"Unsupported baseline estimator: {model_spec.estimator!r}")


def _compute_validation_metrics(
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


def _extract_interpretability_summary(
    fitted_pipeline: Pipeline,
    feature_columns: list[str],
) -> dict[str, Any]:
    model = fitted_pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0].tolist()
        return {
            "type": "linear_coefficients",
            "values": {
                column: float(value)
                for column, value in zip(feature_columns, coefficients, strict=False)
            },
        }

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        return {
            "type": "feature_importances",
            "values": {
                column: float(value)
                for column, value in zip(feature_columns, importances, strict=False)
            },
        }

    return {"type": "none", "values": {}}


def save_trained_baseline_model(model: Pipeline, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(model, fh)
    return output_path


def load_trained_baseline_model(model_path: Path) -> Pipeline:
    with model_path.open("rb") as fh:
        loaded = pickle.load(fh)
    if not isinstance(loaded, Pipeline):
        raise ValueError(f"Saved baseline artifact is not a sklearn Pipeline: {model_path}")
    return loaded


def prepare_m4_baseline_training_data(
    *,
    training_definition: OfficialM4BaselineTrainingDefinition,
    split_definition: OfficialM4SplitDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
) -> dict[str, Any]:
    resolved_target_definition = target_definition or load_m4_target_definition()
    resolved_split_definition = split_definition or load_m4_split_definition(
        _resolve_repo_path(training_definition.split_config_path)
    )

    dataset_path = _resolve_repo_path(training_definition.modeling_dataset_path)
    metadata_path = _resolve_repo_path(training_definition.modeling_dataset_metadata_path)
    bundle = load_m4_modeling_dataset_bundle(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        validate=True,
    )

    schema = build_m4_modeling_dataset_schema(
        dataset_definition=bundle["dataset_definition"],
        target_definition=bundle["target_definition"],
        split_definition=bundle["split_definition"],
    )
    metadata_schema = bundle.get("schema") or {}
    metadata_feature_columns = metadata_schema.get("feature_columns")
    if metadata_feature_columns is not None:
        metadata_feature_columns = [str(column) for column in metadata_feature_columns]
        official_feature_columns = [str(column) for column in schema.get("feature_columns", [])]
        if metadata_feature_columns != official_feature_columns:
            raise ValueError(
                "Modeling dataset metadata feature_columns do not match the official schema. "
                "Regenerate modeling dataset metadata before running baseline training."
            )

    feature_columns = list(schema.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Official modeling dataset schema is missing feature_columns.")
    if training_definition.target_column not in bundle["dataframe"].columns:
        raise ValueError(
            f"Baseline training target column missing from modeling dataset: {training_definition.target_column}"
        )

    train_df, validation_df, split_summary = split_m4_modeling_dataset(
        bundle["dataframe"],
        split_definition=resolved_split_definition,
        target_definition=resolved_target_definition,
    )
    if train_df.empty:
        raise ValueError("Official M4 split produced an empty training partition.")
    if validation_df.empty:
        raise ValueError("Official M4 split produced an empty validation partition.")

    x_train = _coerce_numeric_feature_frame(
        train_df,
        feature_columns=feature_columns,
        partition_name="training",
    )
    x_validation = _coerce_numeric_feature_frame(
        validation_df,
        feature_columns=feature_columns,
        partition_name="validation",
    )
    y_train = _coerce_binary_target(
        train_df[training_definition.target_column],
        column_name=training_definition.target_column,
        partition_name="training",
    )
    y_validation = _coerce_binary_target(
        validation_df[training_definition.target_column],
        column_name=training_definition.target_column,
        partition_name="validation",
    )

    if y_train.nunique(dropna=False) < 2:
        raise ValueError(
            "Baseline training requires at least two target classes in the training partition."
        )

    return {
        "bundle": bundle,
        "schema": schema,
        "feature_columns": feature_columns,
        "train_dataframe": train_df,
        "validation_dataframe": validation_df,
        "x_train": x_train,
        "x_validation": x_validation,
        "y_train": y_train,
        "y_validation": y_validation,
        "split_summary": split_summary,
        "dataset_path": dataset_path,
        "dataset_metadata_path": metadata_path,
        "split_config_path": _resolve_repo_path(training_definition.split_config_path),
        "split_metadata_path": _resolve_repo_path(training_definition.split_metadata_path),
        "output_root": _resolve_repo_path(training_definition.output_dir),
        "target_definition": resolved_target_definition,
        "split_definition": resolved_split_definition,
    }


def run_m4_baseline_training(
    *,
    config_path: Path = M4_BASELINE_TRAINING_CONFIG_PATH,
    training_definition: OfficialM4BaselineTrainingDefinition | None = None,
    split_definition: OfficialM4SplitDefinition | None = None,
    target_definition: OfficialTargetDefinition | None = None,
) -> dict[str, Any]:
    resolved_training_definition = training_definition or load_m4_baseline_training_definition(config_path)
    prepared = prepare_m4_baseline_training_data(
        training_definition=resolved_training_definition,
        split_definition=split_definition,
        target_definition=target_definition,
    )

    bundle = prepared["bundle"]
    schema = prepared["schema"]
    feature_columns = prepared["feature_columns"]
    x_train = prepared["x_train"]
    x_validation = prepared["x_validation"]
    y_train = prepared["y_train"]
    y_validation = prepared["y_validation"]
    split_summary = prepared["split_summary"]
    split_metadata_path = prepared["split_metadata_path"]
    resolved_split_definition = prepared["split_definition"]
    resolved_target_definition = prepared["target_definition"]
    output_root = prepared["output_root"]

    manager = RunArtifactManager(
        base_output_dir=output_root,
        strategy_name=resolved_training_definition.strategy_name,
        start_date=str(split_summary.get("train_feature_date_start") or ""),
        end_date=str(split_summary.get("validation_feature_date_end") or ""),
        run_label=resolved_training_definition.run_label,
        strategy_variant="tabular_baselines",
    )

    config_snapshot = {
        "training_definition": asdict(resolved_training_definition),
        "target_definition": asdict(resolved_target_definition),
        "dataset_definition": asdict(bundle["dataset_definition"]),
        "split_definition": asdict(resolved_split_definition),
        "dataset_config_path": str(M4_MODELING_DATASET_CONFIG_PATH),
        "target_config_path": str(M4_TARGET_CONFIG_PATH),
        "split_config_path": str(M4_SPLIT_CONFIG_PATH),
    }
    manager.write_config_snapshot(config_snapshot)

    feature_schema_payload = {
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "target_column": resolved_training_definition.target_column,
        "task_type": resolved_target_definition.task_type,
        "identifier_columns": list(schema.get("identifier_columns", [])),
        "feature_timestamp_columns": list(schema.get("feature_timestamp_columns", [])),
        "target_timestamp_columns": list(schema.get("target_timestamp_columns", [])),
        "inference_key_columns": list(schema.get("inference_key_columns", [])),
        "official_sort_order": list(schema.get("official_sort_order", [])),
    }
    feature_schema_path = _write_json(manager.artifact_path(FEATURE_SCHEMA_FILENAME), feature_schema_payload)
    manager.register_artifact("feature_schema", feature_schema_path)

    split_summary_payload = {
        "method": resolved_split_definition.method,
        "boundary_column": resolved_split_definition.target_timestamp_column,
        "validation_start_date": resolved_split_definition.validation_start_date,
        "validation_end_date": resolved_split_definition.validation_end_date,
        "split_config_path": str(prepared["split_config_path"]),
        "split_metadata_path": str(split_metadata_path),
        "split_metadata_exists": bool(split_metadata_path.exists()),
        "summary": split_summary,
        "time_safety": {
            "train_target_date_end": split_summary["train_target_date_end"],
            "validation_target_date_start": split_summary["validation_target_date_start"],
            "target_horizon_sessions": resolved_target_definition.forecast_horizon_sessions,
        },
    }
    split_summary_path = _write_json(manager.artifact_path(SPLIT_SUMMARY_FILENAME), split_summary_payload)
    manager.register_artifact("split_summary", split_summary_path)

    model_records: list[dict[str, Any]] = []

    try:
        for model_spec in resolved_training_definition.models:
            fitted_pipeline = _build_estimator_pipeline(model_spec)
            fitted_pipeline.fit(x_train, y_train)
            validation_predictions = pd.Series(
                fitted_pipeline.predict(x_validation),
                index=y_validation.index,
            ).astype("int64")
            metrics = _compute_validation_metrics(
                y_validation,
                validation_predictions,
                resolved_training_definition.metrics,
            )

            model_artifact_path = manager.artifact_path(f"{model_spec.name}.pkl")
            save_trained_baseline_model(fitted_pipeline, model_artifact_path)
            manager.register_artifact(f"model_{model_spec.name}", model_artifact_path)

            model_metadata = {
                "model_name": model_spec.name,
                "estimator": model_spec.estimator,
                "artifact_path": str(model_artifact_path),
                "feature_schema_path": str(feature_schema_path),
                "split_summary_path": str(split_summary_path),
                "target_column": resolved_training_definition.target_column,
                "task_type": resolved_target_definition.task_type,
                "feature_columns": feature_columns,
                "feature_column_count": len(feature_columns),
                "train_row_count": int(len(x_train)),
                "validation_row_count": int(len(x_validation)),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_validation.mean()),
                "metrics": metrics,
                "model_params": dict(model_spec.params),
                "preprocessing": {
                    "imputer": "median",
                    "scaler": "standard" if model_spec.estimator == "logistic_regression" else "none",
                },
                "interpretability": _extract_interpretability_summary(
                    fitted_pipeline,
                    feature_columns,
                ),
            }
            model_metadata_path = _write_json(
                manager.artifact_path(f"{model_spec.name}.metadata.json"),
                model_metadata,
            )
            manager.register_artifact(f"{model_spec.name}_metadata", model_metadata_path)

            model_records.append(
                {
                    "model_name": model_spec.name,
                    "estimator": model_spec.estimator,
                    "artifact_path": str(model_artifact_path),
                    "metadata_path": str(model_metadata_path),
                    "metrics": metrics,
                }
            )

        training_summary = {
            "pipeline_name": "m4_baseline_model_training",
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "entrypoint": "python -m src.strategy.train_baselines",
            "config_source": str(config_path),
            "run_id": manager.run_id,
            "output_dir": str(manager.output_dir),
            "official_dataset": {
                "path": str(prepared["dataset_path"]),
                "metadata_path": str(prepared["dataset_metadata_path"]),
                "row_count": int(len(bundle["dataframe"])),
                "feature_date_start": _format_date(
                    bundle["dataframe"][bundle["dataset_definition"].feature_timestamp_column].min()
                ),
                "feature_date_end": _format_date(
                    bundle["dataframe"][bundle["dataset_definition"].feature_timestamp_column].max()
                ),
            },
            "official_split": {
                "config_path": str(prepared["split_config_path"]),
                "metadata_path": str(split_metadata_path),
                "metadata_exists": bool(split_metadata_path.exists()),
                "method": resolved_split_definition.method,
                "boundary_column": resolved_split_definition.target_timestamp_column,
                "validation_start_date": resolved_split_definition.validation_start_date,
                "validation_end_date": resolved_split_definition.validation_end_date,
                "summary": split_summary,
            },
            "feature_schema_artifact": str(feature_schema_path),
            "split_summary_artifact": str(split_summary_path),
            "target_definition": asdict(resolved_target_definition),
            "model_count": len(model_records),
            "models": model_records,
        }
        training_summary_path = _write_json(
            manager.artifact_path(TRAINING_SUMMARY_FILENAME),
            training_summary,
        )
        manager.register_artifact("training_summary", training_summary_path)
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
        "feature_schema_path": feature_schema_path,
        "split_summary_path": split_summary_path,
        "training_summary_path": training_summary_path,
        "models": model_records,
        "train_row_count": int(len(x_train)),
        "validation_row_count": int(len(x_validation)),
    }
