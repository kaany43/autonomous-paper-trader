from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.data.splits import OFFICIAL_M4_SPLIT_METHOD, load_m4_split_definition


class M4SplitConfigTests(unittest.TestCase):
    def test_official_m4_split_config_matches_expected_contract(self) -> None:
        split_path = Path("config/modeling/m4_split.yaml")
        self.assertTrue(split_path.exists(), f"Missing split config file: {split_path}")

        data = load_yaml(split_path)
        self.assertIn("split", data)

        definition = load_m4_split_definition(split_path)

        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.method, OFFICIAL_M4_SPLIT_METHOD)
        self.assertEqual(definition.symbol_column, "symbol")
        self.assertEqual(definition.feature_timestamp_column, "date")
        self.assertEqual(definition.target_timestamp_column, "target_date")
        self.assertEqual(definition.official_target_column, "target_next_session_direction")
        self.assertEqual(definition.validation_start_date, "2025-01-01")
        self.assertEqual(definition.validation_end_date, "2025-12-31")


if __name__ == "__main__":
    unittest.main()
