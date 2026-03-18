# M3 Strategy Evaluation and Baseline Comparison

This page explains what the M3 comparison workflow is trying to answer, how the current implementation runs the comparison, and how contributors should read the exported results.

For the fixed protocol inputs, see `config/evaluation/m3_protocol.yaml`. For the high-level contract summary, see `docs/m3_evaluation_protocol.md`.

## Where M3 Sits in the Project

M3 builds directly on the milestone work that came before it:

- M1 provides the project structure, market data loading, feature generation, portfolio state, broker logic, and the initial momentum strategy.
- M2 provides the honest daily simulation loop, next-session execution behavior, benchmark comparison, backtest metrics, and run-level artifacts.
- M3 adds the comparison layer on top of those pieces so contributors can evaluate the momentum strategy against baselines and small parameter variations under one shared protocol.

That means this page is about how the existing trading engine is compared, not about a separate reporting-only workflow.

## Purpose of M3

M3 exists to answer a simple project question before later milestones build on top of it:

- does the current momentum strategy add value relative to simple baseline choices?
- do small, interpretable parameter changes materially change the conclusion?
- which configuration is the best candidate to carry forward?

The point of M3 is not to prove that one metric is perfect. It is to generate a fair side-by-side comparison under one shared protocol so later roadmap work builds on a clear baseline decision instead of isolated backtest results.

## What Gets Compared

The M3 comparison runner lives in `src/engine/comparison_runner.py` and evaluates the full comparison set under one shared config.

### Main momentum strategy

The main strategy is the momentum implementation in `src/strategy/momentum.py`. In comparison outputs, momentum runs are named like `momentum_baseline` or `momentum_faster_momentum`.

This is the strategy family the project is trying to improve. Its run artifacts include trades, portfolio snapshots, benchmark curves, and backtest metrics.

### Buy-and-hold benchmark

The buy-and-hold baseline uses the configured benchmark symbol from the protocol file and tracks its equity curve across the same evaluation window.

This answers the question: would a passive benchmark position have been simpler and just as good?

### Equal-weight baseline

The equal-weight baseline builds a simple basket from the configured universe and rebalances it using the implemented comparator logic.

This answers the question: does the active momentum selection beat a passive diversified exposure to the same universe?

### Momentum parameter variations

The protocol config can define small momentum variants under `strategy.variants`. These are intentionally lightweight variations of the same strategy family, not a broad hyperparameter search.

In the current protocol, the variants are:

- `baseline`
- `faster_momentum`
- `slower_momentum`

The goal is robustness checking. If a small parameter change completely reverses the comparison outcome, that is important M3 information.

## Fair Comparison Protocol

The comparison is meant to be fair because all compared runs share the same core setup:

- the same protocol file: `config/evaluation/m3_protocol.yaml`
- the same date window
- the same universe and benchmark symbol
- the same execution assumptions
- the same portfolio and cost settings
- the same market data loading path
- the same metrics pipeline and artifact structure

In practice, `run_m3_comparison()` loads the protocol once, builds shared market features, runs each momentum variant, then generates the buy-and-hold and equal-weight baselines inside the same comparison output directory.

This does not mean every run trades the same way. It means the evaluation environment is held constant so differences in the results come from strategy behavior rather than from moving assumptions.

## Parameter Variations

The M3 variants are deliberately small and interpretable:

- `top_k` changes how many symbols can be selected
- `min_score` changes how strict the entry threshold is
- `min_volume_ratio` changes how strict the liquidity filter is

These variants are not presented as a tuning framework. They are a compact sensitivity check around the core momentum idea.

## Reported Metrics

Comparison-level metrics are exported by `src/engine/comparison_metrics.py` into `comparison_metrics.json` and `comparison_metrics.csv`.

The key fields are:

- `cumulative_return`: total return over the evaluated window. Higher is better, but it should not be read alone.
- `period_return`: currently the same as cumulative return for a single M3 comparison window.
- `max_drawdown`: largest peak-to-trough decline. More negative values mean deeper capital drawdowns.
- `volatility`: day-to-day return variability. Higher volatility means a less stable equity path.
- `average_daily_return`: average daily change in equity across the run.
- `sharpe_ratio`: simple risk-adjusted return using the exported daily return series and a zero risk-free rate.
- `return_over_max_drawdown`: total return scaled by drawdown magnitude. This is a compact reward-versus-pain check.
- `trade_count`: number of qualifying executed trades. This is only meaningful for active strategies.
- `win_rate`: share of realized trades with positive `realized_pnl`. This is not applicable to passive baselines.

For `buy_and_hold` and `equal_weight`, `trade_count` and `win_rate` are intentionally recorded as non-applicable in comparison outputs.

## Exported Outputs and What They Are For

The comparison runner writes grouped outputs under:

`outputs/backtests/comparisons/<comparison_run_id>/`

The main comparison-level artifacts are:

- `comparison_config.json`: frozen snapshot of the comparison inputs.
- `comparison_metrics.json`: structured metrics payload for every compared run.
- `comparison_metrics.csv`: tabular version of the same comparison metrics.
- `aligned_equity_curves.csv`: run equity curves aligned by date for direct side-by-side inspection.
- `aligned_drawdowns.csv`: drawdown series derived from the aligned equity curves.
- `ranking_summary.json`: machine-readable ranking and preferred configuration summary.
- `strategy_ranking.csv`: ordered ranking table for quick inspection.
- `comparison_summary.json`: top-level summary that points to the key artifacts and preferred run.
- `manifest.json`: comparison manifest that records runs, artifact paths, and compared strategies.

The comparison directory also contains `runs/`, where each individual run keeps its own artifacts such as:

- `trade_log.csv`
- `daily_portfolio_snapshots.csv`
- `daily_position_snapshots.csv`
- `benchmark_equity_curve.csv`
- `equal_weight_equity_curve.csv`
- `backtest_metrics.json`
- `manifest.json`
- `config.json`

Use the run-level artifacts to inspect how a single strategy behaved. Use the comparison-level artifacts to decide which run performed best in the shared protocol.

## Ranking and Preferred Configuration

The current implementation includes a lightweight ranking layer in `src/engine/comparison_ranking.py`.

It ranks compared runs with a simple deterministic rule:

1. higher `sharpe_ratio`
2. higher `cumulative_return`
3. lower absolute `max_drawdown`
4. lexical `run_name` ordering as the final stable tie-break

The winner is written to:

- `ranking_summary.json` as `preferred_run`
- `comparison_summary.json` as `preferred_run`
- `manifest.json` as `preferred_run`

This ranking summary is a convenience decision layer, not a replacement for reading the curves and metrics together. A top-ranked run should still be checked against the aligned equity and drawdown exports before it is treated as the best candidate for later milestones.

## How to Interpret the Comparison

When reading M3 results:

- start with `comparison_metrics.csv` to see the full scorecard
- check `aligned_equity_curves.csv` to see whether a higher return came from a smoother or more erratic path
- check `aligned_drawdowns.csv` to understand how painful the worst decline was
- use `ranking_summary.json` to find the current preferred configuration quickly

Do not over-trust one metric in isolation. A run with the best return may still have an unacceptable drawdown. A run with the best Sharpe ratio may have a weaker absolute result. The ranking output is there to keep selection consistent, but it should be read as a summary of the comparison, not as proof that one metric tells the whole story.

For the roadmap, the preferred run is the configuration that should be carried into the next milestone unless a later review finds a practical reason to override it.

## Reproducibility and Re-running

The implemented M3 entrypoint is:

```bash
python -m src.engine.comparison_runner --config config/evaluation/m3_protocol.yaml
```

Optional `--start-date` and `--end-date` overrides exist for subset runs, but the default project reference point should stay the official protocol file whenever contributors want the standard M3 result.

If you are reviewing an existing comparison output instead of re-running it:

- open `comparison_config.json` to confirm the inputs
- open `comparison_summary.json` and `manifest.json` to find the key artifact paths
- inspect `comparison_metrics.csv`, `aligned_equity_curves.csv`, `aligned_drawdowns.csv`, and `ranking_summary.json` together

That is the shortest path for a new contributor to understand both what was run and what the results are saying.
