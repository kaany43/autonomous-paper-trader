# M4 Baseline Model Evaluation Reports

The M4 baseline evaluation workflow loads a completed baseline training run, rebuilds the official validation partition, evaluates every saved baseline model on that same holdout window, and writes reproducible reports.

## Official inputs

- training summary artifact: `outputs/models/<training_run_id>/baseline_training_summary.json`
- training config snapshot: `outputs/models/<training_run_id>/config.json`
- feature schema: `outputs/models/<training_run_id>/feature_schema.json`
- split summary: `outputs/models/<training_run_id>/split_summary.json`
- modeling dataset: `data/processed/m4_modeling_dataset.parquet`
- split contract: `config/modeling/m4_split.yaml`
- evaluation config: `config/evaluation/m4_baseline_evaluation.yaml`

## Metrics

The report contract keeps the metric set aligned with the official M4 classification target:

- `accuracy`
- `precision`
- `recall`
- `f1`

Each per-model report also records confusion-matrix counts plus positive-rate context for the validation partition.

## Run command

```bash
python -m src.engine.evaluate_baselines --training-summary outputs/models/<training_run_id>/baseline_training_summary.json
```

## Output contract

Each evaluation run writes a self-contained directory under `outputs/reports/model_evaluations/<evaluation_run_id>/` with:

- `config.json`
- `manifest.json`
- `baseline_evaluation_summary.json`
- `baseline_evaluation_summary.csv`
- one `<model_name>.evaluation.json` report per trained baseline model

The combined summary keeps one row per model so later milestone summaries can compare baseline models without reloading notebooks. The per-model JSON reports preserve metric values, validation date range, artifact references, and split metadata needed for auditability.
