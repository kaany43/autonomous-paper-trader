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

## Current Milestone Coverage

The repository now includes the core pieces from the first three milestones:

### M1: Core Trading Foundation

- repository structure and tracked config/data/output folders
- parquet-based market data loading
- basic feature generation
- portfolio state tracking
- broker execution layer
- first momentum strategy implementation

### M2: Simulation and Backtest Engine

- daily simulator loop
- next-session execution logic
- benchmark comparison
- trade log export
- portfolio daily snapshots
- position daily snapshots
- backtest metrics
- self-contained run artifacts for later reporting

### M3: Strategy Evaluation and Baseline Comparison

- shared M3 protocol config
- multi-run comparison runner
- buy-and-hold benchmark
- equal-weight baseline
- momentum parameter variants
- comparison metrics and aligned equity exports
- ranking summary and preferred configuration selection

## M3 Comparison Workflow

The project's M3 milestone evaluates the momentum strategy against simple baselines under one shared protocol before later roadmap decisions are made.

- Methodology guide: `docs/m3_comparison_methodology.md`
- Results summary: `docs/m3_results_summary.md`
- Official protocol config: `config/evaluation/m3_protocol.yaml`
- Protocol summary: `docs/m3_evaluation_protocol.md`

Use the comparison runner to execute the full M3 workflow:

```bash
python -m src.engine.comparison_runner --config config/evaluation/m3_protocol.yaml
```

## M4 Data Preparation

The official M4 target contract, modeling dataset export, preparation pipeline, and time-aware holdout split are documented here:

- `docs/m4_target_definition.md`
- `docs/m4_modeling_dataset.md`
- `docs/m4_target_preparation.md`
- `docs/m4_train_validation_split.md`

Prepare the modeling dataset from the reusable processed feature parquet with:

```bash
python -m src.data.target_pipeline
```

Create the official M4 train/validation split with:

```bash
python -m src.data.split_pipeline
```
