from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

import src.engine.simulator as simulator_module
from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.engine.simulator import DailySimulator, PORTFOLIO_SNAPSHOT_COLUMNS
from src.strategy.base import BaseStrategy, StrategySignal


class BuyHoldThenSellStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        d = pd.to_datetime(decision_date)
        if d == pd.Timestamp("2024-01-01"):
            return [
                StrategySignal(date=d, symbol="AAA", action="BUY", score=1.0, reason_code="ENTRY"),
            ]
        if d == pd.Timestamp("2024-01-03") and portfolio.has_position("AAA"):
            return [
                StrategySignal(date=d, symbol="AAA", action="SELL", score=1.0, reason_code="EXIT"),
            ]
        return []


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class PortfolioSnapshotExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def test_one_snapshot_per_day_sorted_and_consistent(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 12.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 13.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=BuyHoldThenSellStrategy(),
            portfolio=Portfolio(initial_cash=100.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run()
        snapshots = results["portfolio_snapshots"]

        trading_days = pd.to_datetime(data["date"]).nunique()
        self.assertEqual(len(snapshots), trading_days)

        self.assertEqual(list(snapshots.columns), PORTFOLIO_SNAPSHOT_COLUMNS)
        self.assertTrue(snapshots["date"].is_monotonic_increasing)

        equity_check = snapshots["cash_balance"] + snapshots["invested_value"]
        self.assertTrue((snapshots["total_equity"] - equity_check).abs().lt(1e-9).all())

        portfolio_history = results["portfolio_history"].sort_values("date").reset_index(drop=True)
        self.assertEqual(
            snapshots["open_positions"].tolist(),
            portfolio_history["number_of_positions"].tolist(),
        )

        self.assertIn("realized_pnl", snapshots.columns)
        self.assertIn("unrealized_pnl", snapshots.columns)

        exported_path = results["portfolio_snapshots_path"]
        loaded = pd.read_csv(exported_path, parse_dates=["date"])
        self.assertEqual(list(loaded.columns), PORTFOLIO_SNAPSHOT_COLUMNS)
        self.assertEqual(len(loaded), len(snapshots))

    def test_days_with_no_trades_still_have_snapshot_rows(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 10.5},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 11.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=250.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run()
        snapshots = results["portfolio_snapshots"]

        self.assertEqual(len(snapshots), 3)
        self.assertTrue((snapshots["invested_value"] == 0.0).all())
        self.assertTrue((snapshots["open_positions"] == 0).all())
        self.assertTrue((snapshots["cash_balance"] == 250.0).all())


if __name__ == "__main__":
    unittest.main()
