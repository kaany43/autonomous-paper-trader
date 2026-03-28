from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.data.prediction_logs import load_m4_prediction_log_definition


class M4PredictionLogConfigTests(unittest.TestCase):
    def test_official_m4_prediction_log_config_matches_expected_contract(self) -> None:
        config_path = Path("config/modeling/m4_prediction_logs.yaml")
        self.assertTrue(config_path.exists(), f"Missing prediction log config file: {config_path}")

        data = load_yaml(config_path)
        self.assertIn("logging", data)

        definition = load_m4_prediction_log_definition(config_path)
        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.contract_name, "m4_official_model_output_log")
        self.assertEqual(definition.output_filename, "baseline_model_predictions.parquet")
        self.assertEqual(definition.metadata_filename, "baseline_model_predictions.metadata.json")
        self.assertEqual(definition.identifier_columns, ("symbol",))
        self.assertEqual(definition.feature_timestamp_column, "date")
        self.assertEqual(definition.target_timestamp_column, "target_date")
        self.assertEqual(
            definition.join_key_columns,
            ("model_name", "symbol", "date", "target_date"),
        )
        self.assertEqual(
            definition.prediction_value_columns,
            ("predicted_class", "predicted_probability"),
        )


if __name__ == "__main__":
    unittest.main()
