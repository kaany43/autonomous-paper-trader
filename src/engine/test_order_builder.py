from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.engine.broker import Broker
from src.engine.order_builder import OrderBuilder
from src.engine.portfolio import Portfolio
from src.engine.simulator import DailySimulator
from src.strategy.base import BaseStrategy, StrategySignal


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class OrderBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.price_map = {"AAA": 10.0, "BBB": 20.0, "CCC": 5.0}

    def test_buy_sell_hold_conversion_and_sell_requires_position(self) -> None:
        portfolio = Portfolio(initial_cash=100.0)
        portfolio.buy("AAA", quantity=2, price=10.0)

        signals = [
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="AAA", action="SELL", score=1.0),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="BBB", action="BUY", score=0.9),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="CCC", action="HOLD", score=0.8),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="ZZZ", action="SELL", score=0.7),
        ]

        builder = OrderBuilder(max_open_positions=3)
        orders = builder.build_orders(signals=signals, portfolio=portfolio, price_map=self.price_map)

        self.assertEqual([o["side"] for o in orders], ["SELL", "BUY"])
        self.assertEqual([o["symbol"] for o in orders], ["AAA", "BBB"])

    def test_respects_available_cash(self) -> None:
        portfolio = Portfolio(initial_cash=15.0)
        signals = [
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="AAA", action="BUY", score=1.0),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="BBB", action="BUY", score=0.9),
        ]

        builder = OrderBuilder(max_open_positions=2, max_position_size=1.0, fractional_shares=False)
        orders = builder.build_orders(signals=signals, portfolio=portfolio, price_map=self.price_map)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "AAA")
        self.assertEqual(orders[0]["quantity"], 1.0)

    def test_respects_max_open_positions(self) -> None:
        portfolio = Portfolio(initial_cash=100.0)
        portfolio.buy("AAA", quantity=1, price=10.0)

        signals = [
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="BBB", action="BUY", score=1.0),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="CCC", action="BUY", score=0.5),
        ]

        builder = OrderBuilder(max_open_positions=2)
        orders = builder.build_orders(signals=signals, portfolio=portfolio, price_map=self.price_map)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "BBB")

    def test_respects_max_position_size(self) -> None:
        portfolio = Portfolio(initial_cash=100.0)
        signals = [StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="AAA", action="BUY", score=1.0)]

        builder = OrderBuilder(max_open_positions=5, max_position_size=0.1, fractional_shares=False)
        orders = builder.build_orders(signals=signals, portfolio=portfolio, price_map=self.price_map)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["quantity"], 1.0)

    def test_filters_duplicates_and_invalid_orders(self) -> None:
        portfolio = Portfolio(initial_cash=100.0)
        portfolio.buy("AAA", quantity=1, price=10.0)

        signals = [
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="BBB", action="BUY", score=1.0),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="BBB", action="BUY", score=0.1),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="AAA", action="BUY", score=0.9),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="", action="BUY", score=0.9),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="CCC", action="UNKNOWN", score=0.9),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="ZZZ", action="BUY", score=0.9),
        ]

        builder = OrderBuilder(max_open_positions=3, fractional_shares=False)
        orders = builder.build_orders(signals=signals, portfolio=portfolio, price_map=self.price_map)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "BBB")

    def test_deterministic_order_output(self) -> None:
        portfolio = Portfolio(initial_cash=100.0)
        signals = [
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="CCC", action="BUY", score=1.0),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="AAA", action="BUY", score=1.0),
            StrategySignal(date=pd.Timestamp("2024-01-01"), symbol="BBB", action="BUY", score=1.0),
        ]

        builder = OrderBuilder(max_open_positions=3, fractional_shares=False)
        first = builder.build_orders(signals=signals, portfolio=portfolio, price_map=self.price_map)
        second = builder.build_orders(signals=list(reversed(signals)), portfolio=portfolio, price_map=self.price_map)

        self.assertEqual(first, second)
        self.assertEqual([o["symbol"] for o in first], ["AAA", "BBB", "CCC"])

    def test_broker_can_consume_generated_orders(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0},
            ]
        )
        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=20.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        orders = [
            {"symbol": "AAA", "side": "BUY", "quantity": 1.0, "market_price": 10.0},
            {"symbol": "AAA", "side": "SELL", "quantity": 1.0, "market_price": 11.0},
        ]

        simulator._execute_orders(  # pylint: disable=protected-access
            orders=orders,
            execution_date=pd.Timestamp("2024-01-02"),
            decision_date=pd.Timestamp("2024-01-01"),
        )

        self.assertEqual(len(simulator._trade_history), 2)  # pylint: disable=protected-access
        self.assertAlmostEqual(simulator.portfolio.cash, 21.0)


if __name__ == "__main__":
    unittest.main()
