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
from src.engine.portfolio import Portfolio
from src.engine.simulator import (
    BACKTEST_METRICS_FILENAME,
    BENCHMARK_EQUITY_FILENAME,
    DailySimulator,
    PORTFOLIO_SNAPSHOT_FILENAME,
    POSITION_SNAPSHOT_FILENAME,
    TRADE_LOG_FILENAME,
)
from src.strategy.base import BaseStrategy


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        return []


class RunArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    @staticmethod
    def _build_market_data() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 100.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 101.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 102.0},
                {"date": "2024-01-02", "symbol": "SPY", "adj_close": 400.0},
                {"date": "2024-01-03", "symbol": "SPY", "adj_close": 402.0},
                {"date": "2024-01-04", "symbol": "SPY", "adj_close": 404.0},
            ]
        )

    def _build_simulator(self) -> DailySimulator:
        return DailySimulator(
            market_data=self._build_market_data(),
            strategy=EmptyStrategy(),
            portfolio=Portfolio(initial_cash=10000.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=True),
            price_column="adj_close",
        )

    def test_backtest_run_writes_self_contained_artifacts_and_manifest(self) -> None:
        simulator = self._build_simulator()
        run_config = {
            "strategy": {"name": "EmptyStrategy"},
            "execution": {"commission_rate": 0.0, "slippage_rate": 0.0},
            "run": {"seed": 1},
        }

        results = simulator.run(
            start_date="2024-01-02",
            end_date="2024-01-04",
            benchmark_symbol="SPY",
            run_config=run_config,
            config_source="config/settings.yaml",
        )

        output_dir = results["output_dir"]
        self.assertTrue(output_dir.exists())
        self.assertEqual(output_dir.parent, simulator_module.BACKTEST_OUTPUTS_DIR)

        expected_files = {
            TRADE_LOG_FILENAME,
            PORTFOLIO_SNAPSHOT_FILENAME,
            POSITION_SNAPSHOT_FILENAME,
            BENCHMARK_EQUITY_FILENAME,
            BACKTEST_METRICS_FILENAME,
            "config.json",
            "manifest.json",
        }
        self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))

        config_path = results["config_path"]
        with config_path.open("r", encoding="utf-8") as fh:
            stored_config = json.load(fh)
        self.assertEqual(stored_config, run_config)

        manifest_path = results["manifest_path"]
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        self.assertEqual(manifest["run_id"], results["run_id"])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["strategy_name"], "EmptyStrategy")
        self.assertEqual(manifest["benchmark_symbol"], "SPY")
        self.assertEqual(manifest["config_source"], "config/settings.yaml")
        self.assertIn("trade_log", manifest["artifacts"])
        self.assertIn("backtest_metrics", manifest["artifacts"])
        self.assertIn("manifest.json", manifest["artifact_files"])
        self.assertIn("config.json", manifest["artifact_files"])

    def test_repeated_runs_create_unique_directories_without_overwrite(self) -> None:
        simulator = self._build_simulator()

        first = simulator.run(start_date="2024-01-02", end_date="2024-01-04", benchmark_symbol="SPY")
        second = simulator.run(start_date="2024-01-02", end_date="2024-01-04", benchmark_symbol="SPY")

        first_output_dir = first["output_dir"]
        second_output_dir = second["output_dir"]

        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(first_output_dir, second_output_dir)
        self.assertTrue(first_output_dir.exists())
        self.assertTrue(second_output_dir.exists())

        first_trade_log = pd.read_csv(first["trade_log_path"])
        second_trade_log = pd.read_csv(second["trade_log_path"])
        self.assertEqual(len(first_trade_log), len(second_trade_log))

        runs_on_disk = [p for p in simulator_module.BACKTEST_OUTPUTS_DIR.iterdir() if p.is_dir()]
        self.assertGreaterEqual(len(runs_on_disk), 2)


if __name__ == "__main__":
    unittest.main()
