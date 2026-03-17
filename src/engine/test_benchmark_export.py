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
from src.data.loader import get_benchmark_symbol
from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.engine.simulator import BENCHMARK_EQUITY_COLUMNS, DailySimulator
from src.strategy.base import BaseStrategy


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class BenchmarkExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def test_reads_benchmark_symbol_from_new_config_field(self) -> None:
        settings = {"benchmark": {"benchmark_symbol": "spy"}}
        self.assertEqual(get_benchmark_symbol(settings), "SPY")

    def test_builds_aligned_normalized_benchmark_curve_and_writes_csv(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 50.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 51.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 52.0},
                {"date": "2024-01-05", "symbol": "AAA", "adj_close": 53.0},
                {"date": "2024-01-02", "symbol": "QQQ", "adj_close": 100.0},
                {"date": "2024-01-04", "symbol": "QQQ", "adj_close": 110.0},
                {"date": "2024-01-05", "symbol": "QQQ", "adj_close": 121.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=100000.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run(
            start_date="2024-01-02",
            end_date="2024-01-05",
            benchmark_symbol="QQQ",
        )



        benchmark_curve = results["benchmark_curve"]
        benchmark_curve = results["benchmark_curve"]
        self.assertEqual(list(benchmark_curve.columns), BENCHMARK_EQUITY_COLUMNS)
        self.assertEqual(list(benchmark_curve.columns), BENCHMARK_EQUITY_COLUMNS)
        self.assertEqual(len(benchmark_curve), 4)
        self.assertEqual(len(benchmark_curve), 4)


        dates = pd.to_datetime(benchmark_curve["date"])
        dates = pd.to_datetime(benchmark_curve["date"])
        self.assertEqual(dates.min(), pd.Timestamp("2024-01-02"))
        self.assertEqual(dates.min(), pd.Timestamp("2024-01-02"))
        self.assertEqual(dates.max(), pd.Timestamp("2024-01-05"))
        self.assertEqual(dates.max(), pd.Timestamp("2024-01-05"))


        self.assertEqual(float(benchmark_curve.iloc[0]["benchmark_equity"]), 100000.0)
        self.assertEqual(float(benchmark_curve.iloc[0]["benchmark_equity"]), 100000.0)
        self.assertEqual(float(benchmark_curve.iloc[1]["benchmark_return"]), 0.0)
        self.assertEqual(float(benchmark_curve.iloc[1]["benchmark_return"]), 0.0)
        self.assertAlmostEqual(float(benchmark_curve.iloc[2]["benchmark_return"]), 0.10, places=8)
        self.assertAlmostEqual(float(benchmark_curve.iloc[2]["benchmark_return"]), 0.10, places=8)
        self.assertAlmostEqual(float(benchmark_curve.iloc[3]["benchmark_equity"]), 121000.0, places=6)
        self.assertAlmostEqual(float(benchmark_curve.iloc[3]["benchmark_equity"]), 121000.0, places=6)


        self.assertEqual(
        self.assertEqual(
            list(pd.to_datetime(benchmark_curve["date"])),
            list(pd.to_datetime(benchmark_curve["date"])),
            list(pd.to_datetime(results["portfolio_snapshots"]["date"])),
            list(pd.to_datetime(results["portfolio_snapshots"]["date"])),
        )
        )


        output_path = results["benchmark_curve_path"]
        output_path = results["benchmark_curve_path"]
        self.assertTrue(output_path.exists())
        self.assertTrue(output_path.exists())
        loaded = pd.read_csv(output_path)
        loaded = pd.read_csv(output_path)
        self.assertEqual(list(loaded.columns), BENCHMARK_EQUITY_COLUMNS)
        self.assertEqual(list(loaded.columns), BENCHMARK_EQUITY_COLUMNS)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(len(loaded), 4)


    def test_benchmark_remains_flat_until_first_available_price(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 50.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 51.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 52.0},
                {"date": "2024-01-05", "symbol": "AAA", "adj_close": 53.0},
                {"date": "2024-01-03", "symbol": "QQQ", "adj_close": 100.0},
                {"date": "2024-01-04", "symbol": "QQQ", "adj_close": 105.0},
                {"date": "2024-01-05", "symbol": "QQQ", "adj_close": 110.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=100000.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run(
            start_date="2024-01-02",
            end_date="2024-01-05",
            benchmark_symbol="QQQ",
        )

        benchmark_curve = results["benchmark_curve"]

        self.assertTrue(pd.isna(benchmark_curve.iloc[0]["benchmark_price"]))
        self.assertEqual(float(benchmark_curve.iloc[0]["benchmark_equity"]), 100000.0)
        self.assertEqual(float(benchmark_curve.iloc[1]["benchmark_equity"]), 100000.0)
        self.assertAlmostEqual(float(benchmark_curve.iloc[2]["benchmark_equity"]), 105000.0, places=6)
        self.assertAlmostEqual(float(benchmark_curve.iloc[3]["benchmark_equity"]), 110000.0, places=6)



if __name__ == "__main__":
    unittest.main()
