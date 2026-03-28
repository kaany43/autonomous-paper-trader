# M4 Model Output Logging

The official M4 model output logging contract turns batch prediction rows into a stable simulation-ready artifact. The logging layer does not retrain models or redefine inference. It only normalizes, validates, and persists prediction rows plus their provenance metadata.

## Official contract

- logging config: `config/modeling/m4_prediction_logs.yaml`
- logged parquet artifact: `baseline_model_predictions.parquet`
- logged metadata artifact: `baseline_model_predictions.metadata.json`

The batch prediction pipeline writes both files inside each prediction run directory:

- `outputs/predictions/model_batches/<prediction_run_id>/baseline_model_predictions.parquet`
- `outputs/predictions/model_batches/<prediction_run_id>/baseline_model_predictions.metadata.json`

## Guaranteed row schema

Each logged prediction row includes:

- `prediction_run_id`
- `training_run_id`
- `inference_partition`
- `model_name`
- `estimator`
- `model_artifact_path`
- `model_metadata_path`
- `symbol`
- `date`
- `target_date`
- `target_column`
- `task_type`
- `predicted_class`
- `predicted_probability`

Rows are sorted by `model_name`, `symbol`, `date`, `target_date`.

## Why this is simulation-ready

The schema preserves:

- row-level join keys back to the modeling dataset via `symbol`, `date`, `target_date`
- model provenance via `model_name`, `estimator`, `model_artifact_path`, `model_metadata_path`
- run provenance via `prediction_run_id` and `training_run_id`
- prediction meaning via `target_column`, `task_type`, `predicted_class`, and `predicted_probability`

This lets later simulation code reload predictions without rerunning training or reverse-engineering notebook assumptions.

## Metadata sidecar

`baseline_model_predictions.metadata.json` records:

- schema version and contract name
- output path and row count
- official sort and join-key columns
- source training artifacts
- official modeling dataset and split references
- logged output signature
- source model list

## Reload path

Later simulation-oriented code should treat the parquet file as the canonical row-level input and use the metadata sidecar to confirm schema, provenance, and source artifact references.
