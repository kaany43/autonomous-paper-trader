from __future__ import annotations

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
        if d == pd.Timestamp("2024-01-02") and portfolio.has_position("AAA"):
            return [
                StrategySignal(date=d, symbol="AAA", action="SELL", score=1.0, reason_code="TEST_SELL"),
            ]
        return []


def main() -> None:
    data = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 10.0, "close": 11.0},
            {"date": "2024-01-02", "symbol": "AAA", "open": 12.0, "close": 12.0},
        ]
    )

    simulator = DailySimulator(
        strategy=TwoStepStrategy(),
        broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
        portfolio=Portfolio(initial_cash=100.0),
        max_position_weight=0.5,
    )

    result = simulator.run(data)

    assert len(result.trades) == 2, "Expected one buy and one sell trade"
    assert result.metrics["trade_count"] == 2
    assert result.metrics["final_equity"] > 100.0
    print("Simulator manual test passed.")


if __name__ == "__main__":
    main()
