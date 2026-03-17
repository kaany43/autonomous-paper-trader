# M3 Evaluation Protocol (Official Contract)

The file `config/evaluation/m3_protocol.yaml` is the **single official evaluation setup for M3**.  
All momentum strategy variants and baseline strategies must run under this same protocol.

## Why this exists

M3 comparisons must be reproducible and fair. This protocol fixes:
- the date window
- the comparison universe
- the benchmark
- initial capital
- simulator assumptions
- required artifacts and metrics

## Official setup

- **Protocol file:** `config/evaluation/m3_protocol.yaml`
- **Backtest window:** `2020-01-01` to `2024-12-31`
- **Universe:** `AAPL, MSFT, NVDA, AMZN, META, TSLA, AMD, NFLX, QCOM, GOOGL`
- **Benchmark:** `QQQ`
- **Starting capital:** `10000.0`

## Fixed simulator assumptions for M3

M3 runs must use the same execution assumptions already implemented in the simulator:

1. Decision-date data isolation (no forward-looking leakage).
2. Next-session execution behavior for orders.
3. Configured portfolio constraints (position limits/weights).
4. Benchmark equity curve generation from the configured benchmark symbol.
5. Consistent metrics generation and run artifact export.

These assumptions are documented in the protocol file under `m3_requirements.simulator_assumptions` and enforced by existing simulator/metrics/artifact code paths.

## Required outputs for every M3 run

Each run must produce, at minimum:
- `trade_log.csv`
- `daily_portfolio_snapshots.csv`
- `daily_position_snapshots.csv`
- `benchmark_equity_curve.csv`
- `equal_weight_equity_curve.csv`
- `backtest_metrics.json`
- `manifest.json`
- `config.json`

## Required metrics for every M3 run

At minimum, reported metrics must include:

- **Strategy:** `cumulative_return`, `max_drawdown`, `volatility`, `trade_count`, `win_rate`, `sharpe_ratio`, `return_over_max_drawdown`
- **Benchmark:** `cumulative_return`, `max_drawdown`, `volatility`, `sharpe_ratio`

## How to run with the official protocol

Use the standard backtest entrypoint and point it to the protocol config:

```bash
python -m src.cli.backtest --config config/evaluation/m3_protocol.yaml
```

Optional date override (if you need a subset run) still uses the same protocol assumptions:

```bash
python -m src.cli.backtest --config config/evaluation/m3_protocol.yaml --start-date 2021-01-01 --end-date 2021-12-31
```

## Guidance for future baselines

Any future M3 baseline must reuse `config/evaluation/m3_protocol.yaml` directly (or a versioned successor file when the protocol itself is intentionally revised). Do not change Python source files for routine comparison runs.
