# M4 Baseline Tabular Model Training

The first official M4 training workflow fits a very small set of interpretable tabular classifiers on the reusable modeling dataset contract.

## Official inputs

- modeling dataset: `data/processed/m4_modeling_dataset.parquet`
- modeling metadata: `data/processed/m4_modeling_dataset.metadata.json`
- target contract: `config/modeling/m4_target.yaml`
- split contract: `config/modeling/m4_split.yaml`
- training config: `config/modeling/m4_baselines.yaml`

The workflow reuses the official `target_next_session_direction` classification target and applies the shared time-aware holdout split from `target_date`.

## Baseline model set

- `logistic_regression`
- `decision_tree`

Both models use deterministic settings from the official training config. Logistic regression uses median imputation plus standard scaling. The decision tree uses median imputation and a fixed `random_state`.

## Run command

```bash
python -m src.strategy.train_baselines
```

## Output contract

Each run writes a self-contained directory under `outputs/models/<run_id>/` with:

- `config.json`
- `manifest.json`
- `feature_schema.json`
- `split_summary.json`
- `baseline_training_summary.json`
- one `.pkl` artifact per model
- one `.metadata.json` sidecar per model

The saved metadata records the target column, feature order, split boundaries, row counts, model parameters, and validation metrics so later prediction code can reload the artifacts without reconstructing schema assumptions from notebooks.
