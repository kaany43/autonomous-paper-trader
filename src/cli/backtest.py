from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.features import add_basic_features
from src.data.loader import get_benchmark_symbol, load_market_data, load_yaml
from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.engine.simulator import DailySimulator
from src.strategy.momentum import MomentumStrategy


def _parse_date(value: str, flag_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid {flag_name} date format: {value!r}. Use YYYY-MM-DD.") from exc


def _load_universe_symbols(settings_path: Path, settings: dict[str, Any]) -> list[str]:
    symbols = settings.get("universe", {}).get("symbols", [])
    source = str(settings_path)

    if not symbols:
        universe_path = settings_path.parent / "universe.yaml"
        data = load_yaml(universe_path)
        symbols = data.get("universe", {}).get("symbols", [])
        source = str(universe_path)

    if not isinstance(symbols, list) or not symbols:
        raise ValueError(f"No symbols found in universe config: {source}")

    cleaned = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not cleaned:
        raise ValueError(f"Universe symbols are empty after cleaning: {source}")

    return cleaned


def run_backtest(
    config_path: Path,
    start_date_override: str | None = None,
    end_date_override: str | None = None,
) -> dict[str, Any]:
    settings = load_yaml(config_path)

    portfolio_cfg = settings.get("portfolio", {})
    execution_cfg = settings.get("execution", {})
    strategy_cfg = settings.get("strategy", {})
    benchmark_symbol = get_benchmark_symbol(settings)

    start_date = start_date_override or settings.get("data", {}).get("start_date")
    end_date = end_date_override if end_date_override is not None else settings.get("data", {}).get("end_date")

    start_ts: pd.Timestamp | None = None
    end_ts: pd.Timestamp | None = None

    if start_date:
        start_ts = _parse_date(str(start_date), "--start-date")
    if end_date:
        end_ts = _parse_date(str(end_date), "--end-date")

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"Invalid date range: start date {start_ts.date()} is after end date {end_ts.date()}."
        )

    symbols = _load_universe_symbols(config_path, settings)
    market_df = load_market_data(symbols=symbols + ([benchmark_symbol] if benchmark_symbol and benchmark_symbol not in symbols else []))
    features_df = add_basic_features(market_df)

    portfolio = Portfolio(initial_cash=float(portfolio_cfg.get("initial_cash", 0.0)))
    broker = Broker(
        commission_rate=float(execution_cfg.get("commission_rate", 0.0)),
        slippage_rate=float(execution_cfg.get("slippage_rate", 0.0)),
        fractional_shares=bool(portfolio_cfg.get("fractional_shares", True)),
    )
    strategy = MomentumStrategy(
        max_open_positions=int(portfolio_cfg.get("max_open_positions", 1)),
        top_k=int(strategy_cfg.get("top_k", 1)),
        min_score=0.0,
    )

    simulator = DailySimulator(
        market_data=features_df,
        strategy=strategy,
        broker=broker,
        portfolio=portfolio,
        price_column="adj_close",
    )

    run_config = {
        "settings": settings,
        "cli_overrides": {
            "start_date": str(start_ts.date()) if start_ts is not None else None,
            "end_date": str(end_ts.date()) if end_ts is not None else None,
        },
    }

    results = simulator.run(
        start_date=start_ts,
        end_date=end_ts,
        benchmark_symbol=benchmark_symbol,
        run_config=run_config,
        config_source=str(config_path),
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full backtest pipeline.")
    parser.add_argument("--config", required=True, help="Path to settings YAML file.")
    parser.add_argument("--start-date", help="Optional override start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional override end date (YYYY-MM-DD).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        results = run_backtest(
            config_path=config_path,
            start_date_override=args.start_date,
            end_date_override=args.end_date,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Backtest failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    portfolio_snapshots = results.get("portfolio_snapshots", pd.DataFrame())
    sim_start = "N/A"
    sim_end = "N/A"
    final_equity = "N/A"

    if not portfolio_snapshots.empty:
        sim_start = str(pd.Timestamp(portfolio_snapshots.iloc[0]["date"]).date())
        sim_end = str(pd.Timestamp(portfolio_snapshots.iloc[-1]["date"]).date())
        final_equity = f"{float(portfolio_snapshots.iloc[-1]['total_equity']):.2f}"

    trade_count = int(len(results.get("trade_history", pd.DataFrame())))
    benchmark = ""
    benchmark_curve = results.get("benchmark_curve", pd.DataFrame())
    if isinstance(benchmark_curve, pd.DataFrame) and not benchmark_curve.empty:
        benchmark = str(benchmark_curve.iloc[0].get("benchmark_symbol", ""))

    print("-" * 60)
    print("Backtest completed")
    print(f"Run ID: {results.get('run_id')}")
    print(f"Output directory: {results.get('output_dir')}")
    print(f"Window: {sim_start} -> {sim_end}")
    print(f"Trade count: {trade_count}")
    print(f"Final equity: {final_equity}")
    print(f"Benchmark: {benchmark or 'N/A'}")
    print(f"Metrics file: {results.get('backtest_metrics_path')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
