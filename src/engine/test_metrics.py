from __future__ import annotations

import json
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
from src.engine.metrics import (
    compute_backtest_metrics,
    compute_equity_metrics,
    compute_trade_metrics,
)
from src.engine.portfolio import Portfolio
from src.engine.simulator import DailySimulator
from src.strategy.base import BaseStrategy


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class MetricsTests(unittest.TestCase):
    def test_compute_equity_metrics_core_values(self) -> None:
        curve = pd.DataFrame(
            [
                {"date": "2024-01-01", "total_equity": 100.0},
                {"date": "2024-01-02", "total_equity": 120.0},
                {"date": "2024-01-03", "total_equity": 90.0},
                {"date": "2024-01-04", "total_equity": 135.0},
            ]
        )

        metrics = compute_equity_metrics(curve, equity_column="total_equity")

        self.assertAlmostEqual(metrics["cumulative_return"], 0.35, places=8)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.25, places=8)
        self.assertAlmostEqual(metrics["volatility"], 0.2747157622, places=8)

    def test_compute_trade_metrics_from_realized_pnl(self) -> None:
        trade_history = pd.DataFrame(
            [
                {"success": True, "executed_quantity": 10.0, "realized_pnl": 100.0},
                {"success": True, "executed_quantity": 5.0, "realized_pnl": -50.0},
                {"success": True, "executed_quantity": 3.0, "realized_pnl": 0.0},
                {"success": False, "executed_quantity": 2.0, "realized_pnl": 10.0},
            ]
        )

        metrics = compute_trade_metrics(trade_history)

        self.assertEqual(metrics["trade_count"], 3)
        self.assertAlmostEqual(metrics["win_rate"], 1.0 / 3.0, places=8)

    def test_compute_backtest_metrics_for_strategy_and_benchmark(self) -> None:
        strategy_curve = pd.DataFrame(
            [
                {"date": "2024-01-01", "total_equity": 100.0},
                {"date": "2024-01-02", "total_equity": 110.0},
            ]
        )
        benchmark_curve = pd.DataFrame(
            [
                {"date": "2024-01-01", "benchmark_equity": 100.0},
                {"date": "2024-01-02", "benchmark_equity": 105.0},
            ]
        )
        trade_history = pd.DataFrame([
            {"success": True, "executed_quantity": 1.0, "realized_pnl": 5.0}
        ])

        metrics = compute_backtest_metrics(strategy_curve, benchmark_curve, trade_history)

        self.assertIn("strategy", metrics)
        self.assertIn("benchmark", metrics)
        self.assertIn("trade_count", metrics["strategy"])
        self.assertNotIn("trade_count", metrics["benchmark"])


class MetricsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def test_metrics_written_after_run_and_are_reproducible(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 100.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 101.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 102.0},
                {"date": "2024-01-02", "symbol": "SPY", "adj_close": 400.0},
                {"date": "2024-01-03", "symbol": "SPY", "adj_close": 404.0},
                {"date": "2024-01-04", "symbol": "SPY", "adj_close": 408.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=10000.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=True),
            price_column="adj_close",
        )

        first = simulator.run(start_date="2024-01-02", end_date="2024-01-04", benchmark_symbol="SPY")
        second = simulator.run(start_date="2024-01-02", end_date="2024-01-04", benchmark_symbol="SPY")

        self.assertEqual(first["backtest_metrics"], second["backtest_metrics"])

        metrics_path = first["backtest_metrics_path"]
        self.assertTrue(metrics_path.exists())

        with metrics_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.assertIn("strategy", payload)
        self.assertIn("benchmark", payload)
        self.assertIn("cumulative_return", payload["strategy"])
        self.assertIn("sharpe_ratio", payload["benchmark"])


if __name__ == "__main__":
    unittest.main()
