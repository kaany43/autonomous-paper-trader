# Autonomous Paper Trader

An autonomous, explainable, and time-safe paper trading system that simulates daily portfolio decisions using only historically available data.

## Why this project?

This project is built as a portfolio-focused learning project to explore:
- financial data pipelines
- backtesting systems
- portfolio simulation
- strategy development
- explainable AI workflows

The core idea is simple: the agent should manage a virtual wallet, act under real capital constraints, and make decisions without seeing the future.

## Current Status

Phase 1 in progress: core daily simulation engine is available with momentum strategy and execution components.

## Quick start

Run an end-to-end paper backtest:

```bash
python -m src.run_backtest
```

Outputs are written under `outputs/backtests/` based on `config/settings.yaml`.
