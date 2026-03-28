# M4 Batch Prediction Pipeline

The M4 batch prediction workflow loads a completed baseline training run, rebuilds the official validation partition, applies every saved baseline model without retraining, and writes structured per-row predictions for later simulation work.

## Official inputs

- training summary artifact: `outputs/models/<training_run_id>/baseline_training_summary.json`
- training config snapshot: `outputs/models/<training_run_id>/config.json`
- feature schema: `outputs/models/<training_run_id>/feature_schema.json`
- split summary: `outputs/models/<training_run_id>/split_summary.json`
- modeling dataset: `data/processed/m4_modeling_dataset.parquet`
- split contract: `config/modeling/m4_split.yaml`
- prediction config: `config/evaluation/m4_batch_prediction.yaml`

## Prediction dataset contract

The official M4 batch prediction flow uses the shared time-aware validation partition. This keeps inference rows aligned with the holdout period used by baseline evaluation while preserving the saved training feature schema and split assumptions.

Each prediction row includes:

- `model_name`
- `symbol`
- `date`
- `target_date`
- `target_column`
- `task_type`
- `predicted_class`
- `predicted_probability`

`predicted_probability` is the positive-class probability for `target_next_session_direction == 1`.

## Run command

```bash
python -m src.engine.generate_predictions --training-summary outputs/models/<training_run_id>/baseline_training_summary.json
```

## Output contract

Each batch prediction run writes a self-contained directory under `outputs/predictions/model_batches/<prediction_run_id>/` with:

- `config.json`
- `manifest.json`
- `baseline_prediction_summary.json`
- `baseline_model_predictions.parquet`

The parquet artifact keeps one row per model per validation row. Rows are sorted by `model_name`, `symbol`, and `date` so later simulation code can reload predictions deterministically and map them back to the original asset/time keys.
