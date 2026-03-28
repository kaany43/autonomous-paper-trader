# M4 Train/Validation Split (Official Contract)

This document defines the single official holdout split for the first M4 baseline-model workflow.

The source parquet consumed here is the official modeling dataset described in `docs/m4_modeling_dataset.md`.

## Source and outputs

- **Source modeling dataset:** `data/processed/m4_modeling_dataset.parquet`
- **Train output:** `data/processed/m4_train_dataset.parquet`
- **Validation output:** `data/processed/m4_validation_dataset.parquet`
- **Metadata sidecar:** `data/processed/m4_train_validation_split.metadata.json`
- **Split config:** `config/modeling/m4_split.yaml`
- **Entrypoint:** `python -m src.data.split_pipeline`

## Official split window

The official validation window is the first full post-M3 calendar year:

- **Validation target window:** `2025-01-01` to `2025-12-31`

That window is fixed in `config/modeling/m4_split.yaml` so later baseline runs stay reproducible even if fresher market data is added to the modeling dataset.

## Split rule

The split is a single chronological holdout anchored on `target_date`, not only on `date`.

- **Training rows:** rows where `target_date < 2025-01-01`
- **Validation rows:** rows where `2025-01-01 <= target_date <= 2025-12-31`
- **Excluded future rows:** rows where `target_date > 2025-12-31`

This anchor matters because the official target predicts the next tradable session. A row close to the boundary can have `date` before the validation window while its label still belongs inside the validation window. Those rows must stay out of training.

## Time-safety contract

- The modeling dataset is re-sorted deterministically by `symbol, date` before splitting.
- No random shuffle is used.
- Duplicate `symbol, date` rows are rejected.
- Training targets must end strictly before validation targets begin.
- Rows with non-forward `target_date <= date` are rejected.
- Rows beyond the official validation window are not mixed into validation by default.

## Re-running the split

Prepare the modeling dataset first if needed:

```bash
python -m src.data.target_pipeline
```

Then create the official train/validation holdout:

```bash
python -m src.data.split_pipeline
```

Optional explicit paths:

```bash
python -m src.data.split_pipeline --input data/processed/m4_modeling_dataset.parquet --train-output data/processed/m4_train_dataset.parquet --validation-output data/processed/m4_validation_dataset.parquet --metadata-output data/processed/m4_train_validation_split.metadata.json --config config/modeling/m4_split.yaml
```

## Metadata contract

The saved metadata sidecar records:

- split method and config path
- train and validation date ranges
- the target-based boundary rule
- row counts, including excluded future rows
- target definition details
- stable sort order and leakage guard notes

## Guidance for later baseline models

Later baseline ML code should reuse the shared helper in `src/data/splits.py` instead of re-implementing split logic:

- load the modeling dataset
- call `split_m4_modeling_dataset(...)`
- or consume the saved train/validation parquet outputs produced by `python -m src.data.split_pipeline`

Do not introduce random train/validation splitting for the official M4 baseline workflow.
