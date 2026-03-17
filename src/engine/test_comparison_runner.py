from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.data.loader as loader_module
from src.engine.comparison_runner import run_m3_comparison


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

    def test_repeated_runs_do_not_overwrite_and_structure_is_reproducible(self) -> None:
        self._write_config()

        first = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)
        second = run_m3_comparison(config_path=self.config_path, output_root=self.output_dir)

        self.assertNotEqual(first["comparison_run_id"], second["comparison_run_id"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])

        first_names = [item["name"] for item in first["runs"]]
        second_names = [item["name"] for item in second["runs"]]
        self.assertEqual(first_names, second_names)

        first_metrics = json.loads(Path(first["runs"][0]["metrics_path"]).read_text(encoding="utf-8"))
        second_metrics = json.loads(Path(second["runs"][0]["metrics_path"]).read_text(encoding="utf-8"))
        self.assertEqual(first_metrics, second_metrics)

        first_comparison_metrics = json.loads(first["comparison_metrics_json_path"].read_text(encoding="utf-8"))
        second_comparison_metrics = json.loads(second["comparison_metrics_json_path"].read_text(encoding="utf-8"))
        self.assertEqual(first_comparison_metrics["rows"], second_comparison_metrics["rows"])


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

        payload = json.loads(result["comparison_metrics_json_path"].read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    unittest.main()