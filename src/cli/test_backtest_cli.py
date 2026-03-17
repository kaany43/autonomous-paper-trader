from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

import src.data.loader as loader_module
import src.engine.simulator as simulator_module
from src.cli.backtest import main


class BacktestCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name)

        self._original_raw = loader_module.RAW_DATA_DIR
        self._original_outputs = simulator_module.BACKTEST_OUTPUTS_DIR

        loader_module.RAW_DATA_DIR = self.tmp_path / "data" / "raw"
        simulator_module.BACKTEST_OUTPUTS_DIR = self.tmp_path / "outputs" / "backtests"
        loader_module.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.config_path = self.tmp_path / "config" / "settings.yaml"
        self.universe_path = self.tmp_path / "config" / "universe.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.config_path.write_text(
            """
portfolio:
  initial_cash: 1000.0
  max_open_positions: 2
  fractional_shares: true
execution:
  commission_rate: 0.001
  slippage_rate: 0.001
strategy:
  top_k: 2
benchmark:
  benchmark_symbol: "SPY"
data:
  start_date: "2024-01-02"
  end_date: "2024-01-05"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        self.universe_path.write_text(
            """
universe:
  symbols: ["AAA"]
""".strip()
            + "\n",
            encoding="utf-8",
        )

        self._write_symbol("AAA", [10.0, 11.0, 12.0, 11.5])
        self._write_symbol("SPY", [100.0, 101.0, 102.0, 103.0])

    def tearDown(self) -> None:
        loader_module.RAW_DATA_DIR = self._original_raw
        simulator_module.BACKTEST_OUTPUTS_DIR = self._original_outputs
        self._tmp_dir.cleanup()

    def _write_symbol(self, symbol: str, closes: list[float]) -> None:
        dates = pd.date_range("2024-01-02", periods=len(closes), freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "adj_close": closes,
                "volume": [1000] * len(closes),
                "dividends": [0.0] * len(closes),
                "stock_splits": [0.0] * len(closes),
            }
        )
        df.to_parquet(loader_module.RAW_DATA_DIR / f"{symbol}.parquet", index=False)

    def test_cli_accepts_config_and_runs_full_backtest(self) -> None:
        out = io.StringIO()
        err = io.StringIO()

        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--config", str(self.config_path)])

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        stdout = out.getvalue()
        self.assertIn("Backtest completed", stdout)
        self.assertIn("Run ID:", stdout)
        self.assertIn("Output directory:", stdout)
        self.assertIn("Metrics file:", stdout)

        run_dirs = [p for p in simulator_module.BACKTEST_OUTPUTS_DIR.iterdir() if p.is_dir()]
        self.assertEqual(len(run_dirs), 1)
        files = {p.name for p in run_dirs[0].iterdir()}
        self.assertIn("trade_log.csv", files)
        self.assertIn("backtest_metrics.json", files)
        self.assertIn("manifest.json", files)

    def test_cli_accepts_date_overrides(self) -> None:
        out = io.StringIO()
        err = io.StringIO()

        with redirect_stdout(out), redirect_stderr(err):
            code = main(
                [
                    "--config",
                    str(self.config_path),
                    "--start-date",
                    "2024-01-03",
                    "--end-date",
                    "2024-01-04",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn("Window: 2024-01-03 -> 2024-01-04", out.getvalue())

    def test_cli_supports_inline_universe_in_config(self) -> None:
        inline_config = self.tmp_path / "config" / "m3_protocol.yaml"
        inline_config.write_text(
            """
portfolio:
  initial_cash: 1000.0
  max_open_positions: 2
  fractional_shares: true
execution:
  commission_rate: 0.001
  slippage_rate: 0.001
strategy:
  top_k: 2
benchmark:
  benchmark_symbol: "SPY"
data:
  start_date: "2024-01-02"
  end_date: "2024-01-05"
universe:
  symbols: ["AAA"]
""".strip()
            + "\n",
            encoding="utf-8",
        )

        out = io.StringIO()
        err = io.StringIO()

        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--config", str(inline_config)])

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn("Backtest completed", out.getvalue())


    def test_cli_invalid_date_returns_readable_error_and_nonzero(self) -> None:
        out = io.StringIO()
        err = io.StringIO()

        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--config", str(self.config_path), "--start-date", "bad-date"])

        self.assertEqual(code, 2)
        self.assertIn("Invalid --start-date date format", err.getvalue())

    def test_cli_start_after_end_returns_readable_error_and_nonzero(self) -> None:
        out = io.StringIO()
        err = io.StringIO()

        with redirect_stdout(out), redirect_stderr(err):
            code = main(
                [
                    "--config",
                    str(self.config_path),
                    "--start-date",
                    "2024-01-05",
                    "--end-date",
                    "2024-01-03",
                ]
            )

        self.assertEqual(code, 2)
        self.assertIn("start date 2024-01-05 is after end date 2024-01-03", err.getvalue())

    def test_cli_missing_config_path_returns_nonzero(self) -> None:
        out = io.StringIO()
        err = io.StringIO()

        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--config", str(self.tmp_path / "missing.yaml")])

        self.assertEqual(code, 2)
        self.assertIn("Config file not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
