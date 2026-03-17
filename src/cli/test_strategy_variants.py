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

import src.data.loader as loader_module
import src.engine.simulator as simulator_module
from src.cli.backtest import _parse_momentum_variants, run_backtest


class StrategyVariantBacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name)

        self._original_raw = loader_module.RAW_DATA_DIR
        self._original_outputs = simulator_module.BACKTEST_OUTPUTS_DIR

        loader_module.RAW_DATA_DIR = self.tmp_path / "data" / "raw"
        simulator_module.BACKTEST_OUTPUTS_DIR = self.tmp_path / "outputs" / "backtests"
        loader_module.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.config_path = self.tmp_path / "config" / "m3_protocol.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_symbol("AAA", start=100.0, drift=1.0)
        self._write_symbol("BBB", start=120.0, drift=0.6)
        self._write_symbol("SPY", start=300.0, drift=0.3)

    def tearDown(self) -> None:
        loader_module.RAW_DATA_DIR = self._original_raw
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_outputs
        self._tmp_dir.cleanup()

    def _write_symbol(self, symbol: str, start: float, drift: float) -> None:
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
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
        df.to_parquet(loader_module.RAW_DATA_DIR / f"{symbol}.parquet", index=False)

    def _write_config(self, variants: str) -> None:
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
  name: momentum_v0
  top_k: 2
  min_score: 0.0
  min_volume_ratio: 0.8
  variants:
{variants}
benchmark:
  benchmark_symbol: "SPY"
data:
  start_date: "2024-02-01"
  end_date: "2024-03-20"
universe:
  symbols: ["AAA", "BBB"]
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def test_multiple_variants_run_in_one_flow_with_isolated_outputs(self) -> None:
        self._write_config(
            """
    - name: baseline
      params:
        top_k: 2
        min_score: 0.0
        min_volume_ratio: 0.8
    - name: faster_momentum
      params:
        top_k: 2
        min_score: -0.1
        min_volume_ratio: 0.0
""".rstrip()
        )

        results = run_backtest(config_path=self.config_path)
        variants = results["variant_results"]
        self.assertEqual(len(variants), 2)

        output_dirs = [item["output_dir"] for item in variants]
        self.assertEqual(len(set(output_dirs)), 2)

        for item in variants:
            manifest = json.loads(item["manifest_path"].read_text(encoding="utf-8"))
            config = json.loads(item["config_path"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["strategy_variant"], item["strategy_variant"]["name"])
            self.assertEqual(config["strategy_variant"]["name"], item["strategy_variant"]["name"])
            self.assertIn("params", config["strategy_variant"])

    def test_repeated_variant_runs_are_deterministic(self) -> None:
        self._write_config(
            """
    - name: baseline
      params:
        top_k: 2
        min_score: 0.0
        min_volume_ratio: 0.8
""".rstrip()
        )

        first = run_backtest(config_path=self.config_path)
        second = run_backtest(config_path=self.config_path)

        first_variant = first["variant_results"][0]
        second_variant = second["variant_results"][0]

        first_metrics = json.loads(first_variant["backtest_metrics_path"].read_text(encoding="utf-8"))
        second_metrics = json.loads(second_variant["backtest_metrics_path"].read_text(encoding="utf-8"))
        self.assertEqual(first_metrics, second_metrics)

        first_trades = pd.read_csv(first_variant["trade_log_path"])
        second_trades = pd.read_csv(second_variant["trade_log_path"])
        pd.testing.assert_frame_equal(first_trades, second_trades)

    def test_invalid_variant_configs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate strategy variant name"):
            _parse_momentum_variants(
                {
                    "variants": [
                        {"name": "baseline", "params": {"top_k": 2}},
                        {"name": "baseline", "params": {"top_k": 1}},
                    ]
                }
            )

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            _parse_momentum_variants({"variants": []})

        with self.assertRaisesRegex(ValueError, "at most"):
            _parse_momentum_variants(
                {
                    "variants": [
                        {"name": f"v{i}", "params": {"top_k": 1}}
                        for i in range(6)
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
