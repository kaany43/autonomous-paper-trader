from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.engine.ml_vs_rule_comparison import load_m4_ml_vs_rule_comparison_definition


class M4MLVsRuleComparisonConfigTests(unittest.TestCase):
    def test_official_m4_ml_vs_rule_comparison_config_matches_expected_contract(self) -> None:
        config_path = Path("config/evaluation/m4_ml_vs_rule_comparison.yaml")
        self.assertTrue(config_path.exists(), f"Missing ML-vs-rule comparison config file: {config_path}")

        data = load_yaml(config_path)
        self.assertIn("comparison", data)

        definition = load_m4_ml_vs_rule_comparison_definition(config_path)
        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.contract_name, "m4_official_ml_vs_rule_comparison")
        self.assertEqual(definition.training_config_path, "config/modeling/m4_baselines.yaml")
        self.assertEqual(definition.prediction_config_path, "config/evaluation/m4_batch_prediction.yaml")
        self.assertEqual(definition.prediction_log_config_path, "config/modeling/m4_prediction_logs.yaml")
        self.assertEqual(definition.strategy_config_path, "config/settings.yaml")
        self.assertEqual(definition.feature_dataset_path, "data/processed/market_features.parquet")
        self.assertEqual(definition.output_dir, "outputs/reports/ml_vs_rule_comparisons")
        self.assertEqual(definition.strategy_name, "M4MLVsRuleComparison")
        self.assertEqual(definition.run_label, "m4-ml-vs-rule-comparison")
        self.assertEqual(definition.methodology_name, "validation_signal_alignment")
        self.assertEqual(definition.comparison_signal_column, "rule_entry_signal")


if __name__ == "__main__":
    unittest.main()
