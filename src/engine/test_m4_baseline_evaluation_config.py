from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml
from src.engine.model_evaluation import load_m4_baseline_evaluation_definition


class M4BaselineEvaluationConfigTests(unittest.TestCase):
    def test_official_m4_baseline_evaluation_config_matches_expected_contract(self) -> None:
        config_path = Path("config/evaluation/m4_baseline_evaluation.yaml")
        self.assertTrue(config_path.exists(), f"Missing evaluation config file: {config_path}")

        data = load_yaml(config_path)
        self.assertIn("evaluation", data)

        definition = load_m4_baseline_evaluation_definition(config_path)
        self.assertEqual(definition.milestone, "M4")
        self.assertEqual(definition.contract_name, "m4_official_baseline_model_evaluation")
        self.assertEqual(definition.training_config_path, "config/modeling/m4_baselines.yaml")
        self.assertEqual(definition.output_dir, "outputs/reports/model_evaluations")
        self.assertEqual(definition.strategy_name, "M4BaselineModelEvaluation")
        self.assertEqual(definition.run_label, "m4-baseline-evaluation")
        self.assertEqual(definition.metrics, ("accuracy", "precision", "recall", "f1"))


if __name__ == "__main__":
    unittest.main()
