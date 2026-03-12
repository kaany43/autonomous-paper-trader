from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pandas as pd

from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.engine.simulator import DailySimulator
from src.strategy.base import BaseStrategy, StrategySignal


class TwoStepStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        d = pd.to_datetime(decision_date)
        if d == pd.Timestamp("2024-01-01"):
            return [
                StrategySignal(date=d, symbol="AAA", action="BUY", score=1.0, reason_code="TEST_BUY"),
            ]
        if d == pd.Timestamp("2024-01-03") and portfolio.has_position("AAA"):
            return [
                StrategySignal(date=d, symbol="AAA", action="SELL", score=1.0, reason_code="TEST_SELL"),
            ]
        return []


def main() -> None:
    data = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
            {"date": "2024-01-03", "symbol": "AAA", "adj_close": 12.0},
            {"date": "2024-01-04", "symbol": "AAA", "adj_close": 11.0},
        ]
    )

    simulator = DailySimulator(
        market_data=data,
        strategy=TwoStepStrategy(),
        broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
        portfolio=Portfolio(initial_cash=100.0),
        price_column="adj_close",
    )

    result = simulator.run()
    trades = result["trade_history"]
    signals = result["signal_history"]

    assert len(trades) == 2, "Expected one buy and one sell trade"

    # Buy is decided on 2024-01-01 and executes on next trading day 2024-01-03.
    first_trade = trades.iloc[0]
    assert pd.Timestamp(first_trade["decision_date"]) == pd.Timestamp("2024-01-01")
    assert pd.Timestamp(first_trade["execution_date"]) == pd.Timestamp("2024-01-03")

    # Sell is decided on 2024-01-03 and executes on 2024-01-04.
    second_trade = trades.iloc[1]
    assert pd.Timestamp(second_trade["decision_date"]) == pd.Timestamp("2024-01-03")
    assert pd.Timestamp(second_trade["execution_date"]) == pd.Timestamp("2024-01-04")

    # Last-day signals are tracked but not executed when no future session exists.
    last_signal = signals.sort_values("decision_date").iloc[-1]
    assert last_signal["schedule_status"] in {"SCHEDULED", "NO_NEXT_TRADING_SESSION"}


    print("Simulator manual test passed.")


if __name__ == "__main__":
    main()
