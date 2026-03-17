from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.strategy.momentum import MomentumStrategy


class MomentumEntryConditionTests(unittest.TestCase):
    def _build_valid_row(self, volume_ratio: float) -> pd.Series:
        return pd.Series(
            {
                "ret_5d": 0.02,
                "ma_20": 105.0,
                "ma_50": 100.0,
                "price_vs_ma20": 0.01,
                "ma20_vs_ma50": 0.05,
                "adj_close": 106.0,
                "volume_ratio_20": volume_ratio,
            }
        )

    def test_entry_condition_uses_configured_min_volume_ratio(self) -> None:
        strategy = MomentumStrategy(min_volume_ratio=0.6)
        row = self._build_valid_row(volume_ratio=0.7)

        self.assertTrue(strategy._entry_condition(row))

    def test_entry_condition_fails_when_below_configured_min_volume_ratio(self) -> None:
        strategy = MomentumStrategy(min_volume_ratio=0.6)
        row = self._build_valid_row(volume_ratio=0.59)

        self.assertFalse(strategy._entry_condition(row))


if __name__ == "__main__":
    unittest.main()
