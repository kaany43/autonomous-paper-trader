from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.data.loader as loader_module
from src.engine.comparison_runner import run_m3_comparison
from src.engine.comparison_metrics import write_comparison_metrics


class ComparisonRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name)
        self.raw_dir = self.tmp_path / "data" / "raw"
        self.output_dir = self.tmp_path / "outputs" / "backtests" / "comparisons"

        self._original_raw = loader_module.RAW_DATA_DIR
        loader_module.RAW_DATA_DIR = self.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.tmp_path / "config" / "m3_protocol.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_symbol("AAA", start=100.0, drift=1.2)
        self._write_symbol("BBB", start=120.0, drift=0.8)
        self._write_symbol("SPY", start=300.0, drift=0.3)

    def tearDown(self) -> None:
        loader_module.RAW_DATA_DIR = self._original_raw
        self._tmp_dir.cleanup()

    def _write_symbol(self, symbol: str, start: float, drift: float) -> None:
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        closes = [start + i * drift for i in range(len(dates))]
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "adj_close": closes,
                "volume": [1000 + i for i in range(len(dates))],
                "dividends": [0.0] * len(dates),
                "stock_splits": [0.0] * len(dates),
            }
        )
        df.to_parquet(self.raw_dir / f"{symbol}.parquet", index=False)

    def _write_config(self, benchmark_symbol: str = "SPY") -> None:
        self.config_path.write_text(
            f"""
portfolio:
  initial_cash: 10000.0
  max_open_positions: 2
  fractional_shares: true
execution:
  commission_rate: 0.0
  slippage_rate: 0.0
strategy:
  top_k: 2
  min_score: 0.0
  min_volume_ratio: 0.8
  variants:
    - name: baseline
      params:
        top_k: 2
        min_score: 0.0
        min_volume_ratio: 0.8
    - name: fast
      params:
        top_k: 2
        min_score: -0.1
        min_volume_ratio: 0.5
benchmark:
  benchmark_symbol: "{benchmark_symbol}"
data:
  start_date: "2024-02-01"
  end_date: "2024-03-20"
universe:
  symbols: ["AAA", "BBB"]
baselines:
  buy_and_hold: true
  equal_weight: true
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def test_full_comparison_entrypoint_and_grouped_outputs(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        self.assertTrue((result["output_dir"] / "runs").exists())
        self.assertTrue(result["comparison_manifest_path"].exists())
        self.assertTrue(result["comparison_config_path"].exists())
        self.assertTrue(result["comparison_summary_path"].exists())
        self.assertTrue(result["aligned_equity_csv_path"].exists())

        run_names = [item["name"] for item in result["runs"]]
        self.assertEqual(run_names, sorted(run_names))
        self.assertIn("buy_and_hold", run_names)
        self.assertIn("equal_weight", run_names)
        self.assertIn("momentum_baseline", run_names)
        self.assertIn("momentum_fast", run_names)

        manifest = json.loads(result["comparison_manifest_path"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["comparison_run_id"], result["comparison_run_id"])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["runs"]), 4)
        self.assertEqual(manifest["comparison_summary_path"], str(result["comparison_summary_path"]))
        self.assertEqual(manifest["aligned_equity_csv_path"], str(result["aligned_equity_csv_path"]))

        summary = json.loads(result["comparison_summary_path"].read_text(encoding="utf-8"))
        self.assertEqual(summary["comparison_export_version"], 1)
        self.assertEqual(summary["comparison_run_id"], result["comparison_run_id"])
        self.assertEqual(summary["comparison_config_path"], str(result["comparison_config_path"]))
        self.assertEqual(summary["config_source"], str(self.config_path))
        self.assertEqual(len(summary["deterministic_runs"]), len(result["runs"]))
        self.assertEqual(
            sorted(item["run_name"] for item in summary["deterministic_runs"]),
            sorted(item["name"] for item in result["runs"]),
        )
        self.assertEqual(summary["artifacts"]["comparison_metrics_csv_path"], str(result["comparison_metrics_csv_path"]))
        self.assertEqual(
            summary["artifacts"]["aligned_equity_curves_csv_path"],
            str(result["aligned_equity_curves_csv_path"]),
        )
        self.assertEqual(summary["artifacts"]["aligned_drawdowns_csv_path"], str(result["aligned_drawdowns_csv_path"]))
        self.assertEqual(summary["artifacts"]["comparison_metrics_json_path"], str(result["comparison_metrics_json_path"]))
        self.assertIn("aligned_equity_curves_csv_path", manifest)
        self.assertIn("aligned_equity_csv_path", manifest)
        self.assertEqual(manifest["aligned_equity_curves_csv_path"], manifest["aligned_equity_csv_path"])

    def test_full_comparison_entrypoint_and_grouped_outputs(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        self.assertTrue((result["output_dir"] / "runs").exists())
        self.assertTrue(result["comparison_manifest_path"].exists())
        self.assertTrue(result["comparison_config_path"].exists())
        self.assertTrue(result["comparison_summary_path"].exists())
        self.assertTrue(result["aligned_equity_csv_path"].exists())

        run_names = [item["name"] for item in result["runs"]]
        self.assertEqual(run_names, sorted(run_names))
        self.assertIn("buy_and_hold", run_names)
        self.assertIn("equal_weight", run_names)
        self.assertIn("momentum_baseline", run_names)
        self.assertIn("momentum_fast", run_names)

        manifest = json.loads(result["comparison_manifest_path"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["comparison_run_id"], result["comparison_run_id"])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["runs"]), 4)
        self.assertEqual(manifest["comparison_summary_path"], str(result["comparison_summary_path"]))
        self.assertEqual(manifest["aligned_equity_csv_path"], str(result["aligned_equity_csv_path"]))


    def test_equal_weight_uses_benchmark_curve_for_metrics(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        runs_by_name = {item["name"]: item for item in result["runs"]}
        equal_weight_metrics = json.loads(Path(runs_by_name["equal_weight"]["metrics_path"]).read_text(encoding="utf-8"))
        buy_hold_metrics = json.loads(Path(runs_by_name["buy_and_hold"]["metrics_path"]).read_text(encoding="utf-8"))

        self.assertGreater(equal_weight_metrics["benchmark"]["cumulative_return"], 0.0)
        self.assertEqual(equal_weight_metrics["benchmark"], buy_hold_metrics["benchmark"])

    def test_missing_required_baseline_components_fail_clearly(self) -> None:
        self._write_config(benchmark_symbol="")
        with self.assertRaisesRegex(ValueError, "Buy-and-hold baseline requires benchmark"):
            run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

    def test_shared_comparison_metrics_outputs_include_all_runs(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        self.assertTrue(result["comparison_metrics_json_path"].exists())
        self.assertTrue(result["comparison_metrics_csv_path"].exists())
        self.assertTrue(result["aligned_equity_csv_path"].exists())
        self.assertTrue(result["aligned_drawdowns_csv_path"].exists())

        payload = json.loads(result["comparison_metrics_json_path"].read_text(encoding="utf-8"))
        self.assertIn("drawdown_formula", payload["metric_methodology"])
        rows = payload["rows"]
        self.assertEqual([row["run_name"] for row in rows], sorted([row["run_name"] for row in rows]))
        self.assertEqual(len(rows), len(result["runs"]))

        by_name = {row["run_name"]: row for row in rows}
        self.assertIn("momentum_baseline", by_name)
        self.assertIn("buy_and_hold", by_name)
        self.assertIn("equal_weight", by_name)

        required_fields = {
            "run_name",
            "strategy_type",
            "variant_name",
            "cumulative_return",
            "period_return",
            "max_drawdown",
            "volatility",
            "average_daily_return",
            "sharpe_ratio",
            "return_over_max_drawdown",
            "trade_count",
            "win_rate",
            "activity_metrics_status",
        }
        for row in rows:
            self.assertTrue(required_fields.issubset(row.keys()))
            self.assertEqual(row["period_return"], row["cumulative_return"])

        self.assertIsNone(by_name["buy_and_hold"]["trade_count"])
        self.assertIsNone(by_name["buy_and_hold"]["win_rate"])
        self.assertEqual(by_name["buy_and_hold"]["activity_metrics_status"], "not_applicable")

        self.assertIsNone(by_name["equal_weight"]["trade_count"])
        self.assertIsNone(by_name["equal_weight"]["win_rate"])
        self.assertEqual(by_name["equal_weight"]["activity_metrics_status"], "not_applicable")

        momentum_activity = by_name["momentum_baseline"]["activity_metrics_status"]
        self.assertIn(momentum_activity, {"computed", "missing_trade_log", "win_rate_unavailable"})

        csv_df = pd.read_csv(result["comparison_metrics_csv_path"])
        self.assertEqual(sorted(csv_df["run_name"].tolist()), sorted(by_name.keys()))

        equity_df = pd.read_csv(result["aligned_equity_csv_path"])
        drawdown_df = pd.read_csv(result["aligned_drawdowns_csv_path"])
        self.assertTrue(result["aligned_equity_csv_path"].exists())
        self.assertEqual(list(equity_df.columns), list(drawdown_df.columns))
        self.assertEqual(equity_df["date"].tolist(), drawdown_df["date"].tolist())

    def test_comparison_exports_include_aligned_curves_drawdowns_and_summary(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        metrics_csv = result["comparison_metrics_csv_path"]
        aligned_equity_csv = result["aligned_equity_curves_csv_path"]
        aligned_drawdowns_csv = result["aligned_drawdowns_csv_path"]
        summary_json = result["comparison_summary_json_path"]

        self.assertTrue(metrics_csv.exists())
        self.assertTrue(aligned_equity_csv.exists())
        self.assertTrue(aligned_drawdowns_csv.exists())
        self.assertTrue(summary_json.exists())

        metrics_df = pd.read_csv(metrics_csv)
        self.assertEqual(len(metrics_df), len(result["runs"]))
        self.assertEqual(sorted(metrics_df["run_name"].tolist()), sorted(item["name"] for item in result["runs"]))
        self.assertTrue(metrics_df["run_id"].notna().all())
        self.assertTrue((metrics_df["run_id"].astype(str).str.len() > 0).all())

        aligned_equity_df = pd.read_csv(aligned_equity_csv, parse_dates=["date"])
        self.assertIn("date", aligned_equity_df.columns)
        expected_columns = ["date"] + sorted(item["name"] for item in result["runs"])
        self.assertEqual(aligned_equity_df.columns.tolist(), expected_columns)
        self.assertTrue(aligned_equity_df["date"].is_monotonic_increasing)

        aligned_drawdowns_df = pd.read_csv(aligned_drawdowns_csv, parse_dates=["date"])
        self.assertEqual(aligned_drawdowns_df.columns.tolist(), aligned_equity_df.columns.tolist())
        self.assertEqual(aligned_drawdowns_df["date"].tolist(), aligned_equity_df["date"].tolist())
        self.assertEqual(aligned_drawdowns_df.shape, aligned_equity_df.shape)

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertEqual(summary["comparison_run_id"], result["comparison_run_id"])
        self.assertEqual(summary["exports"]["comparison_metrics_csv_path"], "comparison_metrics.csv")
        self.assertEqual(summary["exports"]["aligned_equity_curves_csv"], "aligned_equity_curves.csv")
        self.assertEqual(summary["exports"]["aligned_drawdowns_csv"], "aligned_drawdowns.csv")
        self.assertEqual(sorted(item["run_name"] for item in summary["runs"]), sorted(item["name"] for item in result["runs"]))

    def test_export_artifact_paths_are_stable_under_comparison_directory(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        comparison_dir = result["output_dir"]
        expected_paths = {
            "comparison_metrics_csv_path": comparison_dir / "comparison_metrics.csv",
            "aligned_equity_curves_csv_path": comparison_dir / "aligned_equity_curves.csv",
            "aligned_drawdowns_csv_path": comparison_dir / "aligned_drawdowns.csv",
            "comparison_summary_json_path": comparison_dir / "comparison_summary.json",
            "ranking_summary_json_path": comparison_dir / "ranking_summary.json",
            "strategy_ranking_csv_path": comparison_dir / "strategy_ranking.csv",
        }
        for key, expected_path in expected_paths.items():
            self.assertEqual(result[key], expected_path)
            self.assertTrue(expected_path.exists())

    def test_ranking_summary_artifact_identifies_preferred_run(self) -> None:
        self._write_config()
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        ranking_summary = json.loads(result["ranking_summary_json_path"].read_text(encoding="utf-8"))
        ranking_csv = pd.read_csv(result["strategy_ranking_csv_path"])
        manifest = json.loads(result["comparison_manifest_path"].read_text(encoding="utf-8"))
        summary = json.loads(result["comparison_summary_path"].read_text(encoding="utf-8"))

        self.assertEqual(ranking_summary["comparison_run_id"], result["comparison_run_id"])
        self.assertEqual(ranking_summary["preferred_run"]["rank"], 1)
        self.assertEqual(ranking_summary["preferred_run"]["run_name"], ranking_csv.iloc[0]["run_name"])
        self.assertEqual(ranking_csv["rank"].tolist(), list(range(1, len(ranking_csv) + 1)))
        self.assertEqual(
            [item["run_name"] for item in ranking_summary["ranked_runs"]],
            ranking_csv["run_name"].tolist(),
        )
        self.assertEqual(summary["preferred_run"]["run_name"], ranking_summary["preferred_run"]["run_name"])
        self.assertEqual(
            summary["artifacts"]["ranking_summary_json_path"],
            str(result["ranking_summary_json_path"]),
        )
        self.assertEqual(
            summary["artifacts"]["strategy_ranking_csv_path"],
            str(result["strategy_ranking_csv_path"]),
        )
        self.assertEqual(manifest["ranking_summary_json_path"], str(result["ranking_summary_json_path"]))
        self.assertEqual(manifest["strategy_ranking_csv_path"], str(result["strategy_ranking_csv_path"]))
        self.assertEqual(manifest["preferred_run"]["run_name"], ranking_summary["preferred_run"]["run_name"])

    def test_repeated_runs_preserve_export_schema_and_column_ordering(self) -> None:
        self._write_config()
        first = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)
        second = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        first_metrics = pd.read_csv(first["comparison_metrics_csv_path"])
        second_metrics = pd.read_csv(second["comparison_metrics_csv_path"])
        self.assertEqual(first_metrics.columns.tolist(), second_metrics.columns.tolist())
        self.assertEqual(first_metrics["run_name"].tolist(), second_metrics["run_name"].tolist())

        first_aligned = pd.read_csv(first["aligned_equity_curves_csv_path"])
        second_aligned = pd.read_csv(second["aligned_equity_curves_csv_path"])
        self.assertEqual(first_aligned.columns.tolist(), second_aligned.columns.tolist())

        first_drawdowns = pd.read_csv(first["aligned_drawdowns_csv_path"])
        second_drawdowns = pd.read_csv(second["aligned_drawdowns_csv_path"])
        self.assertEqual(first_drawdowns.columns.tolist(), second_drawdowns.columns.tolist())
        self.assertEqual(first_drawdowns["date"].tolist(), second_drawdowns["date"].tolist())

    def test_single_run_only_is_supported(self) -> None:
        self.config_path.write_text(
            """
portfolio:
  initial_cash: 10000.0
  max_open_positions: 2
  fractional_shares: true
execution:
  commission_rate: 0.0
  slippage_rate: 0.0
strategy:
  top_k: 2
  min_score: 0.0
  min_volume_ratio: 0.8
  variants:
    - name: baseline
      params:
        top_k: 2
        min_score: 0.0
        min_volume_ratio: 0.8
benchmark:
  benchmark_symbol: "SPY"
data:
  start_date: "2024-02-01"
  end_date: "2024-03-20"
universe:
  symbols: ["AAA", "BBB"]
baselines:
  buy_and_hold: false
  equal_weight: false
""".strip()
            + "\n",
            encoding="utf-8",
        )
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)
        self.assertEqual([run["name"] for run in result["runs"]], ["momentum_baseline"])
        single_metrics = pd.read_csv(result["comparison_metrics_csv_path"])
        self.assertEqual(single_metrics["run_name"].tolist(), ["momentum_baseline"])

    def test_missing_optional_baselines_are_respected(self) -> None:
        self.config_path.write_text(
            """
portfolio:
  initial_cash: 10000.0
  max_open_positions: 2
  fractional_shares: true
execution:
  commission_rate: 0.0
  slippage_rate: 0.0
strategy:
  variants:
    - name: baseline
      params:
        top_k: 2
        min_score: 0.0
        min_volume_ratio: 0.8
benchmark:
  benchmark_symbol: "SPY"
data:
  start_date: "2024-02-01"
  end_date: "2024-03-20"
universe:
  symbols: ["AAA", "BBB"]
baselines:
  buy_and_hold: false
""".strip()
            + "\n",
            encoding="utf-8",
        )
        result = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)
        run_names = [item["name"] for item in result["runs"]]
        self.assertEqual(run_names, ["equal_weight", "momentum_baseline"])

    def test_missing_metric_fields_are_serialized_as_null_in_csv(self) -> None:
        comparison_dir = self.tmp_path / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "total_equity": [100.0, 101.0, 102.0],
            }
        ).to_csv(run_dir / "daily_portfolio_snapshots.csv", index=False)

        paths = write_comparison_metrics(
            comparison_dir=comparison_dir,
            comparison_run_id="comparison-id",
            created_at="2026-01-01T00:00:00+00:00",
            runs=[
                {
                    "name": "momentum_baseline",
                    "run_id": "run-1",
                    "strategy_type": "momentum",
                    "variant_name": "baseline",
                    "output_dir": str(run_dir),
                }
            ],
        )
        df = pd.read_csv(paths["csv_path"])
        self.assertIn("trade_count", df.columns)
        self.assertIn("win_rate", df.columns)
        self.assertTrue(pd.isna(df.loc[0, "win_rate"]))
        self.assertEqual(int(df.loc[0, "trade_count"]), 0)

    def test_incomplete_run_artifacts_fail_clearly(self) -> None:
        comparison_dir = self.tmp_path / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        missing_run_dir = self.tmp_path / "missing-run"
        missing_run_dir.mkdir(parents=True, exist_ok=True)

        with self.assertRaisesRegex(FileNotFoundError, "Missing required run artifact"):
            write_comparison_metrics(
                comparison_dir=comparison_dir,
                comparison_run_id="comparison-id",
                created_at="2026-01-01T00:00:00+00:00",
                runs=[
                    {
                        "name": "momentum_baseline",
                        "run_id": "run-1",
                        "strategy_type": "momentum",
                        "variant_name": "baseline",
                        "output_dir": str(missing_run_dir),
                    }
                ],
            )



if __name__ == "__main__":
    unittest.main()
