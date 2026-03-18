from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.engine.comparison_metrics import (
    _compute_activity_metrics,
    build_aligned_drawdowns_table,
    build_aligned_equity_table,
)
from src.engine.simulator import BENCHMARK_EQUITY_FILENAME, PORTFOLIO_SNAPSHOT_FILENAME, TRADE_LOG_FILENAME


class ComparisonMetricsTests(unittest.TestCase):
    def test_activity_metrics_count_executed_status_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            trade_log = pd.DataFrame(
                [
                    {"execution_status": "EXECUTED", "quantity": 1.0, "realized_pnl": 10.0},
                    {"execution_status": "REJECTED", "quantity": 1.0, "realized_pnl": -1.0},
                    {"execution_status": "executed", "quantity": 2.0, "realized_pnl": -5.0},
                    {"execution_status": "FILLED", "quantity": 3.0, "realized_pnl": 2.0},
                ]
            )
            trade_log.to_csv(output_dir / TRADE_LOG_FILENAME, index=False)

            metrics = _compute_activity_metrics({"name": "momentum_baseline", "output_dir": str(output_dir)})

            self.assertEqual(metrics["trade_count"], 3)
            self.assertEqual(metrics["activity_metrics_status"], "computed")
            self.assertAlmostEqual(metrics["win_rate"], 2.0 / 3.0, places=8)

    def test_build_aligned_equity_and_drawdowns_preserve_run_order_and_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            buy_hold_dir = base_dir / "buy"
            momentum_dir = base_dir / "mom"
            buy_hold_dir.mkdir(parents=True, exist_ok=True)
            momentum_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"date": "2024-01-01", "benchmark_equity": 100.0},
                    {"date": "2024-01-02", "benchmark_equity": 110.0},
                    {"date": "2024-01-03", "benchmark_equity": 105.0},
                ]
            ).to_csv(buy_hold_dir / BENCHMARK_EQUITY_FILENAME, index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-01", "total_equity": 100.0},
                    {"date": "2024-01-03", "total_equity": 120.0},
                ]
            ).to_csv(momentum_dir / PORTFOLIO_SNAPSHOT_FILENAME, index=False)

            runs = [
                {"name": "buy_and_hold", "status": "completed", "output_dir": str(buy_hold_dir)},
                {"name": "momentum_baseline", "status": "completed", "output_dir": str(momentum_dir)},
            ]
            aligned = build_aligned_equity_table(runs)
            self.assertEqual(list(aligned.columns), ["date", "buy_and_hold", "momentum_baseline"])
            self.assertEqual(aligned["date"].tolist(), ["2024-01-01", "2024-01-02", "2024-01-03"])
            self.assertTrue(pd.isna(aligned.loc[1, "momentum_baseline"]))

            drawdowns = build_aligned_drawdowns_table(aligned)
            self.assertEqual(list(drawdowns.columns), ["date", "buy_and_hold", "momentum_baseline"])
            self.assertAlmostEqual(float(drawdowns.loc[0, "buy_and_hold"]), 0.0, places=8)
            self.assertAlmostEqual(float(drawdowns.loc[2, "buy_and_hold"]), 105.0 / 110.0 - 1.0, places=8)
            self.assertTrue(pd.isna(drawdowns.loc[1, "momentum_baseline"]))

    def test_build_aligned_equity_raises_if_no_comparable_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "No comparable runs exist"):
            build_aligned_equity_table([])

    def test_build_aligned_equity_raises_for_missing_required_equity_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            runs = [{"name": "momentum_baseline", "status": "completed", "output_dir": str(run_dir)}]
            with self.assertRaisesRegex(ValueError, "momentum_baseline"):
                build_aligned_equity_table(runs)


if __name__ == "__main__":
    unittest.main()
