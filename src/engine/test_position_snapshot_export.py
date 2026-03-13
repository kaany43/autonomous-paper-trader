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
from src.engine.simulator import DailySimulator, POSITION_SNAPSHOT_COLUMNS
from src.strategy.base import BaseStrategy, StrategySignal


class BuyThenSellStrategy(BaseStrategy):
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


class BuyOnlyStrategy(BaseStrategy):
    def generate_signals(self, decision_date, market_data, portfolio):
        d = pd.to_datetime(decision_date)
        if d == pd.Timestamp("2024-01-01"):
            return [
                StrategySignal(date=d, symbol="AAA", action="BUY", score=1.0, reason_code="ENTRY"),
            ]
        return []


class PositionSnapshotExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def test_position_snapshots_export_daily_history_with_consistent_valuation(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 12.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 13.0},
                {"date": "2024-01-05", "symbol": "AAA", "adj_close": 14.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=BuyThenSellStrategy(),
            portfolio=Portfolio(initial_cash=100.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run()
        position_snapshots = results["position_snapshots"]
        portfolio_snapshots = results["portfolio_snapshots"]

        self.assertEqual(list(position_snapshots.columns), POSITION_SNAPSHOT_COLUMNS)
        self.assertTrue(position_snapshots[["date", "symbol"]].equals(
            position_snapshots.sort_values(["date", "symbol"])[["date", "symbol"]].reset_index(drop=True)
        ))

        active_dates = set(pd.to_datetime(position_snapshots["date"]).dt.normalize())
        self.assertEqual(
            active_dates,
            {
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            },
        )

        self.assertTrue((position_snapshots["market_value"] == position_snapshots["quantity"] * position_snapshots["latest_price"]).all())
        self.assertTrue((position_snapshots["unrealized_pnl"] == position_snapshots["market_value"] - (position_snapshots["quantity"] * position_snapshots["average_cost"])).all())

        invested_by_day = position_snapshots.groupby("date")["market_value"].sum().sort_index()
        invested_reference = (
            portfolio_snapshots.set_index("date").loc[invested_by_day.index, "invested_value"].sort_index()
        )
        self.assertTrue((invested_by_day - invested_reference).abs().lt(1e-9).all())

        weight_sum_by_day = position_snapshots.groupby("date")["position_weight"].sum().sort_index()
        invested_weight_reference = (
            portfolio_snapshots.set_index("date").loc[weight_sum_by_day.index, "invested_value"]
            / portfolio_snapshots.set_index("date").loc[weight_sum_by_day.index, "total_equity"]
        ).sort_index()
        self.assertTrue((weight_sum_by_day - invested_weight_reference).abs().lt(1e-9).all())

        for day in [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]:
            self.assertFalse((pd.to_datetime(position_snapshots["date"]).dt.normalize() == day).any())

        exported_path = results["position_snapshots_path"]
        self.assertTrue(exported_path.exists())
        loaded = pd.read_csv(exported_path, parse_dates=["date"])
        self.assertEqual(list(loaded.columns), POSITION_SNAPSHOT_COLUMNS)
        self.assertEqual(len(loaded), len(position_snapshots))

    def test_missing_latest_price_for_open_position_is_handled_cleanly(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0},
                {"date": "2024-01-03", "symbol": "BBB", "adj_close": 20.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 13.0},
            ]
        )

        simulator = DailySimulator(
            market_data=data,
            strategy=BuyOnlyStrategy(),
            portfolio=Portfolio(initial_cash=100.0),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

        results = simulator.run()
        snapshots = results["position_snapshots"]

        jan03_row = snapshots.loc[pd.to_datetime(snapshots["date"]).dt.normalize() == pd.Timestamp("2024-01-03")]
        self.assertEqual(len(jan03_row), 1)
        self.assertTrue(pd.isna(jan03_row.iloc[0]["latest_price"]))
        self.assertTrue(pd.isna(jan03_row.iloc[0]["market_value"]))
        self.assertTrue(pd.isna(jan03_row.iloc[0]["unrealized_pnl"]))
        self.assertTrue(pd.isna(jan03_row.iloc[0]["position_weight"]))


if __name__ == "__main__":
    unittest.main()
