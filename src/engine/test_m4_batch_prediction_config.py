from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.engine.prediction_pipeline import load_m4_batch_prediction_definition


class M4BatchPredictionConfigTests(unittest.TestCase):
    def test_official_m4_batch_prediction_config_matches_expected_contract(self) -> None:
        config_path = Path("config/evaluation/m4_batch_prediction.yaml")
        self.assertTrue(config_path.exists(), f"Missing prediction config file: {config_path}")

        data = load_yaml(config_path)
        self.assertIn("prediction", data)

        definition = load_m4_batch_prediction_definition(config_path)
        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.contract_name, "m4_official_baseline_batch_prediction")
        self.assertEqual(definition.training_config_path, "config/modeling/m4_baselines.yaml")
        self.assertEqual(definition.output_dir, "outputs/predictions/model_batches")
        self.assertEqual(definition.strategy_name, "M4BaselineBatchPrediction")
        self.assertEqual(definition.run_label, "m4-batch-predictions")
        self.assertEqual(definition.inference_partition, "validation")
        self.assertEqual(definition.prediction_task_type, "classification")


if __name__ == "__main__":
    unittest.main()
