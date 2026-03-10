from __future__ import annotations

from pathlib import Path

from src.data.features import add_basic_features
from src.data.loader import load_market_data, load_settings
from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.engine.simulator import DailySimulator, save_simulation_outputs
from src.strategy.momentum import MomentumStrategy


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "backtests"


def main() -> None:
    settings = load_settings()

    portfolio_cfg = settings.get("portfolio", {})
    execution_cfg = settings.get("execution", {})
    strategy_cfg = settings.get("strategy", {})
    output_cfg = settings.get("output", {})

    market_df = load_market_data()
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
        strategy=strategy,
        broker=broker,
        portfolio=portfolio,
        max_position_weight=float(portfolio_cfg.get("max_position_weight", 1.0)),
    )

    result = simulator.run(features_df)
    paths = save_simulation_outputs(
        result=result,
        output_dir=OUTPUT_DIR,
        save_trades_csv=bool(output_cfg.get("save_trades_csv", True)),
        save_portfolio_csv=bool(output_cfg.get("save_portfolio_csv", True)),
        save_positions_csv=bool(output_cfg.get("save_positions_csv", True)),
        save_metrics_json=bool(output_cfg.get("save_metrics_json", True)),
    )

    print("-" * 60)
    print("Backtest completed")
    print(f"Final equity: {result.metrics.get('final_equity'):.4f}")
    print(f"Total return: {result.metrics.get('total_return'):.2%}")
    print(f"Trade count: {result.metrics.get('trade_count')}")
    print(f"Max drawdown: {result.metrics.get('max_drawdown'):.2%}")

    if result.metrics.get("win_rate") is not None:
        print(f"Win rate: {result.metrics.get('win_rate'):.2%}")

    for key, path in paths.items():
        print(f"Saved {key}: {path}")


if __name__ == "__main__":
    main()
