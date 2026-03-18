from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.engine.comparison_ranking import write_comparison_ranking


def _metric_row(
    *,
    run_name: str,
    run_id: str,
    strategy_type: str = "momentum",
    variant_name: str = "baseline",
    sharpe_ratio: float = 0.0,
    cumulative_return: float = 0.0,
    max_drawdown: float = 0.0,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "run_id": run_id,
        "strategy_type": strategy_type,
        "variant_name": variant_name,
        "sharpe_ratio": sharpe_ratio,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "return_over_max_drawdown": 0.0,
        "volatility": 0.0,
        "average_daily_return": 0.0,
        "trade_count": 0,
        "win_rate": None,
        "activity_metrics_status": "computed",
    }


class ComparisonRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name)
        self.comparison_dir = self.tmp_path / "comparison"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.comparison_dir / "comparison_metrics.csv"

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _write_metrics(self, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(self.metrics_path, index=False)

    def test_ranking_summary_orders_runs_and_marks_preferred_run(self) -> None:
        self._write_metrics(
            [
                _metric_row(run_name="buy_and_hold", run_id="run-bh", strategy_type="baseline", variant_name="buy_and_hold", sharpe_ratio=0.6, cumulative_return=0.10, max_drawdown=-0.08),
                _metric_row(run_name="momentum_fast", run_id="run-fast", sharpe_ratio=0.9, cumulative_return=0.14, max_drawdown=-0.07, variant_name="fast"),
                _metric_row(run_name="equal_weight", run_id="run-ew", strategy_type="baseline", variant_name="equal_weight", sharpe_ratio=0.7, cumulative_return=0.12, max_drawdown=-0.06),
            ]
        )

        result = write_comparison_ranking(
            comparison_dir=self.comparison_dir,
            comparison_run_id="cmp-001",
            generated_at="2026-03-18T00:00:00+00:00",
            comparison_metrics_csv_path=self.metrics_path,
        )

        self.assertTrue(result["ranking_summary_json_path"].exists())
        self.assertTrue(result["strategy_ranking_csv_path"].exists())

        summary = json.loads(result["ranking_summary_json_path"].read_text(encoding="utf-8"))
        self.assertEqual(summary["comparison_run_id"], "cmp-001")
        self.assertEqual(summary["preferred_run"]["run_name"], "momentum_fast")
        self.assertEqual(summary["preferred_run"]["rank"], 1)
        self.assertEqual(summary["selection_rule"]["primary_metric"]["name"], "sharpe_ratio")
        self.assertEqual(
            [item["run_name"] for item in summary["ranked_runs"]],
            ["momentum_fast", "equal_weight", "buy_and_hold"],
        )

        ranking_df = pd.read_csv(result["strategy_ranking_csv_path"])
        self.assertEqual(ranking_df["rank"].tolist(), [1, 2, 3])
        self.assertEqual(ranking_df["run_name"].tolist(), ["momentum_fast", "equal_weight", "buy_and_hold"])

    def test_single_run_is_ranked_and_selected(self) -> None:
        self._write_metrics([_metric_row(run_name="momentum_baseline", run_id="run-1", sharpe_ratio=0.4, cumulative_return=0.08, max_drawdown=-0.03)])

        result = write_comparison_ranking(
            comparison_dir=self.comparison_dir,
            comparison_run_id="cmp-001",
            generated_at="2026-03-18T00:00:00+00:00",
            comparison_metrics_csv_path=self.metrics_path,
        )

        summary = json.loads(result["ranking_summary_json_path"].read_text(encoding="utf-8"))
        self.assertEqual(len(summary["ranked_runs"]), 1)
        self.assertEqual(summary["preferred_run"]["run_name"], "momentum_baseline")
        self.assertEqual(summary["preferred_run"]["rank"], 1)

    def test_ties_are_resolved_by_cumulative_return_then_drawdown_then_run_name(self) -> None:
        self._write_metrics(
            [
                _metric_row(run_name="momentum_beta", run_id="run-2", sharpe_ratio=1.0, cumulative_return=0.10, max_drawdown=-0.08),
                _metric_row(run_name="momentum_alpha", run_id="run-1", sharpe_ratio=1.0, cumulative_return=0.10, max_drawdown=-0.08),
                _metric_row(run_name="momentum_gamma", run_id="run-3", sharpe_ratio=1.0, cumulative_return=0.12, max_drawdown=-0.09),
                _metric_row(run_name="momentum_delta", run_id="run-4", sharpe_ratio=1.0, cumulative_return=0.10, max_drawdown=-0.05),
            ]
        )

        result = write_comparison_ranking(
            comparison_dir=self.comparison_dir,
            comparison_run_id="cmp-001",
            generated_at="2026-03-18T00:00:00+00:00",
            comparison_metrics_csv_path=self.metrics_path,
        )

        summary = json.loads(result["ranking_summary_json_path"].read_text(encoding="utf-8"))
        self.assertEqual(
            [item["run_name"] for item in summary["ranked_runs"]],
            ["momentum_gamma", "momentum_delta", "momentum_alpha", "momentum_beta"],
        )

    def test_repeated_generation_is_identical_for_identical_inputs(self) -> None:
        self._write_metrics(
            [
                _metric_row(run_name="equal_weight", run_id="run-ew", strategy_type="baseline", variant_name="equal_weight", sharpe_ratio=0.7, cumulative_return=0.12, max_drawdown=-0.06),
                _metric_row(run_name="momentum_fast", run_id="run-fast", sharpe_ratio=0.9, cumulative_return=0.14, max_drawdown=-0.07, variant_name="fast"),
            ]
        )

        first = write_comparison_ranking(
            comparison_dir=self.comparison_dir,
            comparison_run_id="cmp-001",
            generated_at="2026-03-18T00:00:00+00:00",
            comparison_metrics_csv_path=self.metrics_path,
        )
        first_summary = first["ranking_summary_json_path"].read_text(encoding="utf-8")
        first_csv = first["strategy_ranking_csv_path"].read_text(encoding="utf-8")

        second = write_comparison_ranking(
            comparison_dir=self.comparison_dir,
            comparison_run_id="cmp-001",
            generated_at="2026-03-18T00:00:00+00:00",
            comparison_metrics_csv_path=self.metrics_path,
        )
        second_summary = second["ranking_summary_json_path"].read_text(encoding="utf-8")
        second_csv = second["strategy_ranking_csv_path"].read_text(encoding="utf-8")

        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_csv, second_csv)

    def test_missing_comparison_metrics_file_fails_clearly(self) -> None:
        missing_path = self.comparison_dir / "missing.csv"
        with self.assertRaisesRegex(FileNotFoundError, "Missing comparison metrics input for ranking"):
            write_comparison_ranking(
                comparison_dir=self.comparison_dir,
                comparison_run_id="cmp-001",
                generated_at="2026-03-18T00:00:00+00:00",
                comparison_metrics_csv_path=missing_path,
            )

    def test_missing_required_ranking_metrics_fail_clearly(self) -> None:
        pd.DataFrame([{"run_name": "momentum_baseline", "run_id": "run-1"}]).to_csv(self.metrics_path, index=False)

        with self.assertRaisesRegex(ValueError, "Ranking input is missing required columns"):
            write_comparison_ranking(
                comparison_dir=self.comparison_dir,
                comparison_run_id="cmp-001",
                generated_at="2026-03-18T00:00:00+00:00",
                comparison_metrics_csv_path=self.metrics_path,
            )


if __name__ == "__main__":
    unittest.main()
