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
from src.engine.simulator import DailySimulator, EQUAL_WEIGHT_EQUITY_COLUMNS
from src.strategy.base import BaseStrategy


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class EqualWeightExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def test_builds_aligned_equal_weight_curve_and_writes_csv(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 100.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 110.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 121.0},
                {"date": "2024-01-02", "symbol": "BBB", "adj_close": 200.0},
                {"date": "2024-01-03", "symbol": "BBB", "adj_close": 220.0},
                {"date": "2024-01-04", "symbol": "BBB", "adj_close": 242.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=10000.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=True),
            price_column="adj_close",
        )

        results = simulator.run(
            start_date="2024-01-02",
            end_date="2024-01-04",
            equal_weight_universe=["AAA", "BBB"],
        )

        curve = results["equal_weight_curve"]
        self.assertEqual(list(curve.columns), EQUAL_WEIGHT_EQUITY_COLUMNS)
        self.assertEqual(len(curve), 3)
        self.assertEqual(
            list(pd.to_datetime(curve["date"])),
            list(pd.to_datetime(results["portfolio_snapshots"]["date"])),
        )

        self.assertEqual(float(curve.iloc[0]["equal_weight_return"]), 0.0)
        self.assertAlmostEqual(float(curve.iloc[1]["equal_weight_return"]), 0.1, places=8)
        self.assertAlmostEqual(float(curve.iloc[2]["equal_weight_return"]), 0.1, places=8)
        self.assertAlmostEqual(float(curve.iloc[2]["equal_weight_equity"]), 12100.0, places=8)

        output_path = results["equal_weight_curve_path"]
        self.assertTrue(output_path.exists())
        loaded = pd.read_csv(output_path)
        self.assertEqual(list(loaded.columns), EQUAL_WEIGHT_EQUITY_COLUMNS)
        self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()
