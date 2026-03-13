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
from src.engine.simulator import DailySimulator, TRADE_LOG_COLUMNS
from src.strategy.base import BaseStrategy, StrategySignal


class MixedSignalStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        d = pd.to_datetime(decision_date)
        if d == pd.Timestamp("2024-01-01"):
            return [
                StrategySignal(date=d, symbol="AAA", action="BUY", score=1.0, reason_code="BUY_ENTRY"),
            ]
        if d == pd.Timestamp("2024-01-03"):
            return [
                StrategySignal(date=d, symbol="AAA", action="SELL", score=1.0, reason_code="SELL_EXIT"),
            ]
        return []


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class TradeLogExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def test_logs_executed_and_skipped_orders_and_writes_csv(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 12.0},
            ]
        )
        simulator = DailySimulator(
            market_data=data,
            strategy=MixedSignalStrategy(),
            portfolio=Portfolio(initial_cash=100.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run()

        trade_log = results["trade_log"]
        self.assertIn("EXECUTED", set(trade_log["execution_status"]))
        self.assertIn("SKIPPED_NO_NEXT_SESSION", set(trade_log["execution_status"]))

        executed = trade_log.loc[trade_log["execution_status"] == "EXECUTED"].iloc[0]
        self.assertEqual(executed["symbol"], "AAA")
        self.assertEqual(executed["side"], "BUY")
        self.assertEqual(executed["decision_date"], "2024-01-01T00:00:00")
        self.assertEqual(executed["execution_date"], "2024-01-02T00:00:00")

        exported_path = results["trade_log_path"]
        self.assertIn(str(Path("outputs") / "backtests"), str(exported_path))
        self.assertTrue(exported_path.exists())

        loaded = pd.read_csv(exported_path)
        self.assertGreaterEqual(len(loaded), 2)
        self.assertEqual(list(loaded.columns), TRADE_LOG_COLUMNS)

    def test_rejected_order_attempt_is_logged(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 10.0},
            ]
        )
        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=0.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        simulator._execute_orders(  # pylint: disable=protected-access
            orders=[{"symbol": "AAA", "side": "BUY", "quantity": 1.0, "market_price": 10.0}],
            execution_date=pd.Timestamp("2024-01-02"),
            decision_date=pd.Timestamp("2024-01-01"),
        )

        log = pd.DataFrame(simulator._trade_log_history)  # pylint: disable=protected-access
        self.assertEqual(log.iloc[0]["execution_status"], "REJECTED")
        self.assertIn("Insufficient cash", str(log.iloc[0]["reason"]))

    def test_empty_run_exports_machine_readable_headers(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 10.0},
            ]
        )
        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=100.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run()
        trade_log = results["trade_log"]

        self.assertTrue(trade_log.empty)
        exported_path = results["trade_log_path"]
        loaded = pd.read_csv(exported_path)
        self.assertTrue(loaded.empty)
        self.assertEqual(list(loaded.columns), TRADE_LOG_COLUMNS)


if __name__ == "__main__":
    unittest.main()
