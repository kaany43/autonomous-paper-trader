# M4 Target Definition (Official Contract)

This document defines the single official supervised prediction target for the first M4 modeling layer.

## Official task

- **Task type:** binary classification
- **Official target column:** `target_next_session_direction`
- **Helper derivation column:** `target_next_session_return`
- **Forecast horizon:** one next tradable session
- **Source price field:** `adj_close`

## Why this is the first M4 target

M4 needs one narrow, stable objective before model training starts. Binary next-session direction is the simplest contract that:
- stays fully time-safe on the existing daily feature pipeline
- maps cleanly to later buy/avoid or rank-by-probability decisions
- avoids introducing early dependence on noisy return-magnitude calibration
- can be reproduced directly from processed daily market data for every symbol

## Exact label construction rule

For each symbol and feature row at session `t`:

1. Compute the helper return:

```text
target_next_session_return[t] = adj_close[t+1] / adj_close[t] - 1.0
```

`t+1` means the next available tradable session for the same symbol after sorting by `symbol, date`.

2. Convert that helper return into the official binary target:

```text
target_next_session_direction[t] = 1 if target_next_session_return[t] > 0.0 else 0
```

Zero return is treated as non-positive and therefore maps to class `0`.

## Timestamp alignment and time safety

- A processed feature row with `date = t` represents the symbol state available by the close of trading session `t`.
- The official target for that row refers strictly to the next tradable session for the same symbol, `t+1`.
- Target generation uses a per-symbol forward shift after sorting, so the row at `t` never uses prices from `t+1` as input features.
- The simulator already strips all `target_` columns before strategy execution, which keeps model labels out of live decision inputs.
- The dedicated preparation step also records `target_date` and `target_is_valid` so alignment remains auditable in the saved modeling dataset.

## Missing and invalid target handling

Label construction first marks invalid rows with null targets. The dedicated preparation pipeline then excludes those invalid rows when it writes `data/processed/m4_modeling_dataset.parquet`.

Rows are invalid when any of the following is true:

- there is no next tradable session for that symbol
- the current `adj_close` is missing or non-positive
- the future `adj_close` is missing or non-positive

As a repo-consistent normalization step, duplicate `symbol, date` rows are collapsed with the existing keep-last convention before target alignment.

## Relationship to the current simulator

The current project operates on daily processed bars and current simulator code paths use `adj_close` as the effective pricing field. The M4 target is therefore anchored to `adj_close` so the first modeling layer inherits the same daily-bar convention already used elsewhere in the repository.

## Preparation pipeline

The reproducible target-preparation workflow lives in `src/data/target_pipeline.py` and reads from `data/processed/market_features.parquet`.

See `docs/m4_target_preparation.md` for the saved output paths, rerun command, and invalid-row handling in the modeling dataset.
