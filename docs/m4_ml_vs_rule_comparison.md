# M4 ML-vs-Rule Comparison

The M4 ML-vs-rule workflow compares the logged baseline model predictions against the current rule-based momentum strategy under one explicit, reproducible methodology.

## Official inputs

- prediction log artifact: `outputs/predictions/model_batches/<prediction_run_id>/baseline_model_predictions.parquet`
- prediction log metadata: `outputs/predictions/model_batches/<prediction_run_id>/baseline_model_predictions.metadata.json`
- training summary artifact referenced by the prediction metadata
- shared feature dataset for rule replay: `data/processed/market_features.parquet`
- current rule-strategy settings: `config/settings.yaml`
- comparison config: `config/evaluation/m4_ml_vs_rule_comparison.yaml`

## Methodology

The official methodology is `validation_signal_alignment`.

It works in five steps:

1. reload the official M4 prediction log and rebuild the same validation split used to generate those prediction rows
2. replay the live momentum strategy through the existing simulator on the shared feature history for the comparison symbols through the validation end date
3. align both sides on the shared row keys `symbol`, `date`, and `target_date`
4. keep the raw rule action as `BUY`, `SELL`, or `HOLD`, while mapping the common binary comparison label to `rule_entry_signal = 1` only for `BUY`
5. export row-level aligned artifacts plus small summary tables that show agreement, disagreement, and next-session outcome context

This keeps the comparison time-safe because:

- ML rows keep the official `date -> target_date` forward-looking contract
- rule decisions come from the existing simulator path that only exposes data up to the decision date
- the simulator still strips `target_` columns before strategy execution
- the replay still uses next-session execution behavior

## Why the mapping is explicit

The baseline M4 models predict next-session direction, not a full portfolio action. The current rule strategy emits `BUY`, `SELL`, or no action depending on portfolio state.

To keep the first comparison interpretable:

- raw rule actions are preserved in the aligned artifact
- the common binary label is `BUY = 1`, `SELL/HOLD = 0`

That makes the first comparison about entry alignment while still exposing rule exits for later simulation integration work.

## Run command

```bash
python -m src.engine.compare_ml_vs_rule --predictions outputs/predictions/model_batches/<prediction_run_id>/baseline_model_predictions.parquet
```

## Output contract

Each comparison run writes a self-contained directory under `outputs/reports/ml_vs_rule_comparisons/<comparison_run_id>/` with:

- `config.json`
- `manifest.json`
- `ml_vs_rule_aligned.parquet`
- `ml_vs_rule_aligned.metadata.json`
- `ml_vs_rule_summary.json`
- `ml_vs_rule_summary.csv`
- `ml_vs_rule_by_symbol.csv`
- `rule_strategy_runs/` containing the replayed rule-strategy run artifacts

## How to read the results

- `ml_vs_rule_aligned.parquet` is the canonical row-level artifact for later reuse
- `ml_vs_rule_summary.csv` shows one row per model with agreement counts, rule-action rates, and disagreement outcome context
- `ml_vs_rule_by_symbol.csv` highlights where the model and the rule differ most by asset
- `rule_strategy_runs/` keeps the replayed rule-strategy evidence bundle for later simulation-level follow-up
