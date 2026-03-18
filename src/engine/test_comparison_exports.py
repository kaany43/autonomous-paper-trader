from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.engine.comparison_exports import write_aligned_equity_curves


class ComparisonExportsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _write_csv(self, run_name: str, filename: str, rows: list[dict[str, object]]) -> Path:
        output_dir = self.tmp_path / "runs" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / filename, index=False)
        return output_dir

    def test_aligned_equity_curves_uses_deterministic_sorted_order_and_ffill(self) -> None:
        momentum_dir = self._write_csv(
            "momentum_fast",
            "daily_portfolio_snapshots.csv",
            [
                {"date": "2024-01-02", "total_equity": 102.0},
                {"date": "2024-01-04", "total_equity": 108.0},
            ],
        )
        buy_hold_dir = self._write_csv(
            "buy_and_hold",
            "benchmark_equity_curve.csv",
            [
                {"date": "2024-01-01", "benchmark_equity": 100.0},
                {"date": "2024-01-03", "benchmark_equity": 103.0},
            ],
        )
        equal_weight_dir = self._write_csv(
            "equal_weight",
            "equal_weight_equity_curve.csv",
            [
                {"date": "2024-01-02", "equal_weight_equity": 101.0},
                {"date": "2024-01-04", "equal_weight_equity": 105.0},
            ],
        )

        run_records = [
            {"name": "momentum_fast", "strategy_type": "momentum", "output_dir": str(momentum_dir)},
            {"name": "buy_and_hold", "strategy_type": "baseline", "output_dir": str(buy_hold_dir)},
            {"name": "equal_weight", "strategy_type": "baseline", "output_dir": str(equal_weight_dir)},
        ]

        result = write_aligned_equity_curves(
            comparison_dir=self.tmp_path,
            comparison_run_id="cmp-001",
            created_at="2026-03-18T00:00:00+00:00",
            run_records=run_records,
        )

        output_path = result["aligned_equity_curves_csv_path"]
        self.assertTrue(output_path.exists())

        aligned = pd.read_csv(output_path)
        self.assertEqual(aligned.columns.tolist(), ["date", "buy_and_hold", "equal_weight", "momentum_fast"])
        self.assertEqual(aligned["date"].tolist(), ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])

        self.assertEqual(aligned.loc[0, "buy_and_hold"], 100.0)
        self.assertTrue(pd.isna(aligned.loc[0, "equal_weight"]))
        self.assertTrue(pd.isna(aligned.loc[0, "momentum_fast"]))
        self.assertEqual(aligned.loc[2, "equal_weight"], 101.0)
        self.assertEqual(aligned.loc[2, "momentum_fast"], 102.0)

    def test_unsupported_run_type_raises_value_error(self) -> None:
        unknown_dir = self._write_csv(
            "mystery",
            "daily_portfolio_snapshots.csv",
            [{"date": "2024-01-01", "total_equity": 100.0}],
        )

        with self.assertRaisesRegex(ValueError, "Unsupported run type"):
            write_aligned_equity_curves(
                comparison_dir=self.tmp_path,
                comparison_run_id="cmp-001",
                created_at="2026-03-18T00:00:00+00:00",
                run_records=[{"name": "mystery", "strategy_type": "baseline", "output_dir": str(unknown_dir)}],
            )


if __name__ == "__main__":
    unittest.main()
