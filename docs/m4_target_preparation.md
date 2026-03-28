# M4 Target Preparation Pipeline

This document describes the dedicated, leakage-safe preparation step that turns processed market features into the official M4 modeling dataset.

## Source and outputs

- **Source feature dataset:** `data/processed/market_features.parquet`
- **Modeling-ready output:** `data/processed/m4_modeling_dataset.parquet`
- **Metadata sidecar:** `data/processed/m4_modeling_dataset.metadata.json`
- **Entrypoint:** `python -m src.data.target_pipeline`

## Pipeline contract

The target preparation step reuses the single official M4 target definition from `config/modeling/m4_target.yaml`.

It does not define a new task, new horizon, or alternate label rule.

The pipeline:

1. loads the processed feature dataset
2. removes any existing `target_` columns from the input so stale legacy labels cannot leak through
3. normalizes rows by `symbol, date` with the repo's keep-last duplicate rule
4. sorts each symbol series in timestamp order
5. computes `target_date`, `target_is_valid`, `target_next_session_return`, and `target_next_session_direction`
6. drops rows where `target_is_valid` is false
7. writes the modeling dataset and a metadata snapshot

## Time-safe alignment

- Feature rows stay anchored to `date = t`.
- `target_date` is the next tradable session for the same symbol after sorting.
- The helper return and official target are derived with a per-symbol forward shift only.
- The saved modeling dataset contains only rows where `target_date > date` and the official target is valid.

## Invalid label handling

The official contract still uses `null_and_exclude_from_training` as the invalid-target policy.

In practice, this pipeline first constructs labels with nulls for invalid rows and then excludes those invalid rows from the saved modeling dataset. This keeps the saved parquet ready for later train/validation splitting without a second target-cleaning pass.

Rows are excluded when:

- there is no next tradable session for the symbol
- the current `adj_close` is missing or non-positive
- the future `adj_close` is missing or non-positive

## Re-running the pipeline

Generate fresh feature inputs first if needed:

```bash
python -m src.data.features
```

Then prepare the official M4 modeling dataset:

```bash
python -m src.data.target_pipeline
```

Optional explicit paths:

```bash
python -m src.data.target_pipeline --input data/processed/market_features.parquet --output data/processed/m4_modeling_dataset.parquet --metadata-output data/processed/m4_modeling_dataset.metadata.json
```

## Relationship to later ML steps

- Later splitting and training steps should consume `data/processed/m4_modeling_dataset.parquet`.
- The official holdout split is documented in `docs/m4_train_validation_split.md`.
- The split entrypoint is `python -m src.data.split_pipeline`, which writes reusable train/validation parquet files plus split metadata.
- Later live or simulation workflows should continue using `data/processed/market_features.parquet`, which remains free of inline target columns.
