from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.engine.comparison_metrics import _compute_activity_metrics
from src.engine.simulator import TRADE_LOG_FILENAME


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


if __name__ == "__main__":
    unittest.main()
