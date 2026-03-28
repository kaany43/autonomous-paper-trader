from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.data.modeling_dataset import (
    get_m4_modeling_dataset_column_order,
    load_m4_modeling_dataset_definition,
)
from src.data.targets import load_m4_target_definition


class M4ModelingDatasetConfigTests(unittest.TestCase):
    def test_official_m4_modeling_dataset_config_matches_expected_contract(self) -> None:
        config_path = Path("config/modeling/m4_dataset.yaml")
        self.assertTrue(config_path.exists(), f"Missing dataset config file: {config_path}")

        data = load_yaml(config_path)
        self.assertIn("dataset", data)

        definition = load_m4_modeling_dataset_definition(config_path)
        target_definition = load_m4_target_definition()

        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.contract_name, "m4_official_modeling_dataset")
        self.assertEqual(definition.identifier_columns, ("symbol",))
        self.assertEqual(definition.feature_timestamp_column, "date")
        self.assertEqual(definition.target_timestamp_column, "target_date")
        self.assertEqual(definition.target_valid_column, "target_is_valid")
        self.assertEqual(definition.split_ready_sort_order, ("symbol", "date"))
        self.assertEqual(definition.inference_key_columns, ("symbol", "date", "target_date"))
        self.assertIn("adj_close", definition.passthrough_feature_columns)
        self.assertIn("ret_1d", definition.engineered_feature_columns)

        column_order = get_m4_modeling_dataset_column_order(definition, target_definition)
        self.assertEqual(column_order[:2], ["date", "symbol"])
        self.assertEqual(
            column_order[-4:],
            [
                "target_date",
                "target_is_valid",
                target_definition.helper_return_column,
                target_definition.official_target_column,
            ],
        )


if __name__ == "__main__":
    unittest.main()
