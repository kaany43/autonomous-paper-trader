from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.strategy.ml_baselines import load_m4_baseline_training_definition


class M4BaselineTrainingConfigTests(unittest.TestCase):
    def test_official_m4_baseline_training_config_matches_expected_contract(self) -> None:
        config_path = Path("config/modeling/m4_baselines.yaml")
        self.assertTrue(config_path.exists(), f"Missing baseline training config file: {config_path}")

        data = load_yaml(config_path)
        self.assertIn("training", data)

        definition = load_m4_baseline_training_definition(config_path)
        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.contract_name, "m4_official_tabular_baseline_training")
        self.assertEqual(definition.modeling_dataset_path, "data/processed/m4_modeling_dataset.parquet")
        self.assertEqual(
            definition.modeling_dataset_metadata_path,
            "data/processed/m4_modeling_dataset.metadata.json",
        )
        self.assertEqual(definition.split_config_path, "config/modeling/m4_split.yaml")
        self.assertEqual(
            definition.split_metadata_path,
            "data/processed/m4_train_validation_split.metadata.json",
        )
        self.assertEqual(definition.output_dir, "outputs/models")
        self.assertEqual(definition.target_column, "target_next_session_direction")
        self.assertEqual(definition.metrics, ("accuracy", "precision", "recall", "f1"))
        self.assertEqual([model.name for model in definition.models], ["logistic_regression", "decision_tree"])
        self.assertEqual(
            [model.estimator for model in definition.models],
            ["logistic_regression", "decision_tree_classifier"],
        )


if __name__ == "__main__":
    unittest.main()
