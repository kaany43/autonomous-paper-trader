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


class ScriptedStrategy(BaseStrategy):
    def __init__(self, signal_map: dict[str, list[dict[str, object]]]) -> None:
        self.signal_map = signal_map

    def generate_signals(self, decision_date, market_data, portfolio):
        key = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
        signals: list[StrategySignal] = []
        for raw in self.signal_map.get(key, []):
            signals.append(
                StrategySignal(
                    date=pd.Timestamp(raw.get("date", decision_date)),
                    symbol=str(raw["symbol"]),
                    action=str(raw["action"]),
                    score=float(raw.get("score", 0.0)),
                    reason_code=str(raw.get("reason_code", "")),
                    target_weight=raw.get("target_weight"),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        return signals


class SimulatorConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_output_dir = simulator_module.BACKTEST_OUTPUTS_DIR
        self._tmp_dir = tempfile.TemporaryDirectory()
        simulator_module.BACKTEST_OUTPUTS_DIR = Path(self._tmp_dir.name) / "outputs" / "backtests"

    def tearDown(self) -> None:
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_output_dir
        self._tmp_dir.cleanup()

    def _build_simulator(
        self,
        data: pd.DataFrame,
        signal_map: dict[str, list[dict[str, object]]],
        *,
        initial_cash: float = 100.0,
    ) -> DailySimulator:
        return DailySimulator(
            market_data=data,
            strategy=ScriptedStrategy(signal_map),
            portfolio=Portfolio(initial_cash=initial_cash),
            broker=Broker(commission_rate=0.0, slippage_rate=0.0, fractional_shares=False),
            price_column="adj_close",
        )

    def test_next_day_execution_uses_next_available_trading_session(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 11.0},
                {"date": "2024-01-05", "symbol": "AAA", "adj_close": 12.0},
            ]
        )
        decision_date = pd.Timestamp("2024-01-03")
        simulator = self._build_simulator(
            data,
            {
                "2024-01-03": [
                    {"date": decision_date, "symbol": "AAA", "action": "BUY", "score": 1.0},
                ]
            },
        )

        results = simulator.run()
        trades = results["trade_history"]
        signal_history = results["signal_history"]

        self.assertEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades.iloc[0]["decision_date"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(pd.Timestamp(trades.iloc[0]["execution_date"]), pd.Timestamp("2024-01-05"))
        self.assertNotEqual(
            pd.Timestamp(trades.iloc[0]["decision_date"]),
            pd.Timestamp(trades.iloc[0]["execution_date"]),
        )

        signal_row = signal_history.iloc[0]
        self.assertEqual(pd.Timestamp(signal_row["decision_date"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(pd.Timestamp(signal_row["scheduled_execution_date"]), pd.Timestamp("2024-01-05"))

        trade_log_row = results["trade_log"].iloc[0]
        self.assertEqual(trade_log_row["decision_date"], "2024-01-03T00:00:00")
        self.assertEqual(trade_log_row["execution_date"], "2024-01-05T00:00:00")

    def test_insufficient_cash_rejects_buy_and_cash_never_negative(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 10.0},
            ]
        )
        simulator = self._build_simulator(data, {}, initial_cash=5.0)

        simulator._execute_orders(  # pylint: disable=protected-access
            orders=[{"symbol": "AAA", "side": "BUY", "quantity": 1.0, "market_price": 10.0}],
            execution_date=pd.Timestamp("2024-01-02"),
            decision_date=pd.Timestamp("2024-01-01"),
        )

        self.assertEqual(simulator.portfolio.cash, 5.0)
        self.assertGreaterEqual(simulator.portfolio.cash, 0.0)

        trade_log = pd.DataFrame(simulator._trade_log_history)  # pylint: disable=protected-access
        self.assertEqual(len(trade_log), 1)
        self.assertEqual(trade_log.iloc[0]["execution_status"], "REJECTED")
        self.assertIn("Insufficient cash", str(trade_log.iloc[0]["reason"]))

    def test_sell_before_buy_does_not_create_invalid_execution(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 20.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 21.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 22.0},
            ]
        )
        simulator = self._build_simulator(
            data,
            {
                "2024-01-01": [
                    {
                        "date": pd.Timestamp("2024-01-01"),
                        "symbol": "AAA",
                        "action": "SELL",
                        "score": 1.0,
                        "reason_code": "SELL_WITHOUT_POSITION",
                    }
                ]
            },
            initial_cash=100.0,
        )

        results = simulator.run()

        self.assertTrue(results["trade_history"].empty)
        self.assertEqual(simulator.portfolio.cash, 100.0)
        self.assertEqual(len(simulator.portfolio.positions), 0)

        signal_history = results["signal_history"]
        self.assertEqual(len(signal_history), 1)
        self.assertEqual(signal_history.iloc[0]["action"], "SELL")
        self.assertEqual(signal_history.iloc[0]["schedule_status"], "SCHEDULED")

    def test_portfolio_snapshots_remain_internally_consistent(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 9.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 12.0},
            ]
        )
        simulator = self._build_simulator(
            data,
            {
                "2024-01-01": [{"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "action": "BUY", "score": 1.0}],
                "2024-01-03": [{"date": pd.Timestamp("2024-01-03"), "symbol": "AAA", "action": "SELL", "score": 1.0}],
            },
            initial_cash=100.0,
        )

        results = simulator.run()
        snapshots = results["portfolio_snapshots"]
        positions = results["position_snapshots"]

        self.assertEqual(len(snapshots), pd.to_datetime(data["date"]).nunique())
        self.assertTrue(snapshots["date"].is_monotonic_increasing)

        total = snapshots["cash_balance"] + snapshots["invested_value"]
        self.assertTrue((snapshots["total_equity"] - total).abs().lt(1e-9).all())

        open_positions_by_day = (
            positions.groupby("date")["symbol"].nunique().reindex(snapshots["date"], fill_value=0).tolist()
            if not positions.empty
            else [0] * len(snapshots)
        )
        self.assertEqual(snapshots["open_positions"].astype(int).tolist(), open_positions_by_day)

        self.assertIn("realized_pnl", snapshots.columns)
        self.assertIn("unrealized_pnl", snapshots.columns)

    def test_trade_log_includes_executed_and_skipped_with_required_fields(self) -> None:
        data = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 10.5},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 11.0},
            ]
        )
        simulator = self._build_simulator(
            data,
            {
                "2024-01-01": [{"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "action": "BUY", "score": 1.0}],
                "2024-01-03": [{"date": pd.Timestamp("2024-01-03"), "symbol": "AAA", "action": "SELL", "score": 1.0}],
            },
            initial_cash=100.0,
        )

        trade_log = simulator.run()["trade_log"]

        self.assertIn("EXECUTED", set(trade_log["execution_status"]))
        self.assertIn("SKIPPED_NO_NEXT_SESSION", set(trade_log["execution_status"]))
        self.assertEqual(list(trade_log.columns), TRADE_LOG_COLUMNS)

        required_fields = ["decision_date", "execution_date", "symbol", "side", "quantity", "execution_status"]
        for field in required_fields:
            self.assertIn(field, trade_log.columns)

        executed = trade_log.loc[trade_log["execution_status"] == "EXECUTED"].iloc[0]
        skipped = trade_log.loc[trade_log["execution_status"] == "SKIPPED_NO_NEXT_SESSION"].iloc[0]

        self.assertEqual(executed["symbol"], "AAA")
        self.assertEqual(executed["side"], "BUY")
        self.assertIsNotNone(executed["execution_date"])

        self.assertEqual(skipped["symbol"], "AAA")
        self.assertEqual(skipped["side"], "SELL")
        self.assertTrue(pd.isna(skipped["execution_date"]))


if __name__ == "__main__":
    unittest.main()
