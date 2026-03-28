# M4 Modeling Dataset (Official Contract)

This document defines the single official modeling dataset artifact for baseline ML experiments in M4.

## Source inputs and outputs

- **Processed feature input:** `data/processed/market_features.parquet`
- **Official modeling dataset:** `data/processed/m4_modeling_dataset.parquet`
- **Metadata sidecar:** `data/processed/m4_modeling_dataset.metadata.json`
- **Dataset config:** `config/modeling/m4_dataset.yaml`
- **Entrypoint:** `python -m src.data.target_pipeline`
- **Reusable loader:** `load_m4_modeling_dataset_bundle()` from `src/data/modeling_dataset.py`

## What the artifact contains

The official modeling dataset keeps one deterministic row per `symbol, date` after duplicate normalization and invalid-target removal.

Schema groups:

- **Identifier columns:** `symbol`
- **Feature timestamp columns:** `date`
- **Target timestamp columns:** `target_date`
- **Passthrough market feature columns:** `open`, `high`, `low`, `close`, `adj_close`, `volume`, `dividends`, `stock_splits`
- **Engineered feature columns:** `ret_1d`, `ret_5d`, `ret_10d`, `ma_10`, `ma_20`, `ma_50`, `vol_20`, `volume_change_1d`, `volume_ma_20`, `volume_ratio_20`, `price_vs_ma10`, `price_vs_ma20`, `price_vs_ma50`, `ma10_vs_ma20`, `ma20_vs_ma50`, `rolling_high_20`, `rolling_low_20`, `range_pos_20`
- **Target metadata columns:** `target_is_valid`
- **Official target label columns:** `target_next_session_return`, `target_next_session_direction`
- **Inference key columns:** `symbol`, `date`, `target_date`

The official column order is:

`date`, `symbol`, all feature columns in config order, `target_date`, `target_is_valid`, `target_next_session_return`, `target_next_session_direction`

## Why this export exists

This artifact is the stable handoff point between:

- the processed feature pipeline
- the official target contract
- the time-aware split workflow
- later baseline ML training code

Later baseline experiments should reuse this parquet and its metadata sidecar directly instead of rebuilding dataset assumptions in notebooks or model scripts.

## Deterministic export rules

- Input rows are normalized with the repo's keep-last duplicate rule before target generation.
- Exported rows are sorted by `symbol, date`.
- Only rows with valid official targets are saved.
- `target_date` must stay strictly after `date`.
- The export schema is validated against `config/modeling/m4_dataset.yaml`, `config/modeling/m4_target.yaml`, and `config/modeling/m4_split.yaml`.

## Split-ready metadata

The metadata sidecar records:

- explicit schema groups
- official target definition snapshot
- official split definition snapshot
- split boundary column: `target_date`
- official split method and validation window
- inference key columns for later prediction-to-row mapping

That keeps the modeling dataset reusable without embedding train/validation assignments into the parquet itself.

## Reuse guidance

To regenerate the artifact:

```bash
python -m src.data.target_pipeline
```

To reload it in later baseline code:

```python
from src.data.modeling_dataset import load_m4_modeling_dataset_bundle

bundle = load_m4_modeling_dataset_bundle()
df = bundle["dataframe"]
schema = bundle["schema"]
```

Use `schema["feature_columns"]`, `schema["target_label_columns"]`, and `schema["inference_key_columns"]` instead of hard-coding column assumptions in training scripts.
