from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.prediction_logs import (
    build_m4_prediction_log_metadata,
    build_m4_prediction_log_schema,
    get_m4_prediction_log_column_order,
    load_m4_prediction_log_bundle,
    load_m4_prediction_log_definition,
    normalize_m4_prediction_log,
    save_m4_prediction_log,
    validate_m4_prediction_log_contract,
)


class M4PredictionLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = load_m4_prediction_log_definition()

    def _build_prediction_log(self) -> pd.DataFrame:
        rows = [
            {
                "prediction_run_id": "m4-batch-predictions-20260328T111402Z",
                "training_run_id": "m4-baselines-20260328T101639Z",
                "inference_partition": "validation",
                "model_name": "logistic_regression",
                "estimator": "logistic_regression",
                "model_artifact_path": "D:/repo/outputs/models/run/logistic_regression.pkl",
                "model_metadata_path": "D:/repo/outputs/models/run/logistic_regression.metadata.json",
                "symbol": "aaa",
                "date": "2025-01-02",
                "target_date": "2025-01-03",
                "target_column": "target_next_session_direction",
                "task_type": "classification",
                "predicted_class": 1,
                "predicted_probability": 0.63,
            },
            {
                "prediction_run_id": "m4-batch-predictions-20260328T111402Z",
                "training_run_id": "m4-baselines-20260328T101639Z",
                "inference_partition": "validation",
                "model_name": "decision_tree",
                "estimator": "decision_tree_classifier",
                "model_artifact_path": "D:/repo/outputs/models/run/decision_tree.pkl",
                "model_metadata_path": "D:/repo/outputs/models/run/decision_tree.metadata.json",
                "symbol": "AAA",
                "date": "2025-01-02",
                "target_date": "2025-01-03",
                "target_column": "target_next_session_direction",
                "task_type": "classification",
                "predicted_class": 0,
                "predicted_probability": 0.41,
            },
        ]
        prediction_log = pd.DataFrame(rows)
        prediction_log = prediction_log.loc[:, get_m4_prediction_log_column_order(self.definition)]
        prediction_log = prediction_log.sample(frac=1.0, random_state=7).reset_index(drop=True)
        return prediction_log

    def test_prediction_log_schema_is_explicit_and_deterministic(self) -> None:
        schema = build_m4_prediction_log_schema(self.definition)
        self.assertEqual(schema["identifier_columns"], ["symbol"])
        self.assertEqual(schema["feature_timestamp_columns"], ["date"])
        self.assertEqual(schema["target_timestamp_columns"], ["target_date"])
        self.assertEqual(
            schema["model_identity_columns"],
            ["model_name", "estimator", "model_artifact_path", "model_metadata_path"],
        )
        self.assertEqual(
            schema["prediction_value_columns"],
            ["predicted_class", "predicted_probability"],
        )

    def test_save_and_reload_prediction_log_bundle(self) -> None:
        prediction_log = normalize_m4_prediction_log(self._build_prediction_log(), self.definition)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / self.definition.output_filename
            metadata_path = tmp_path / self.definition.metadata_filename
            metadata = build_m4_prediction_log_metadata(
                output_path=output_path,
                metadata_path=metadata_path,
                prediction_log_df=prediction_log,
                definition=self.definition,
                prediction_run_id="m4-batch-predictions-20260328T111402Z",
                training_run_id="m4-baselines-20260328T101639Z",
                training_summary_path=tmp_path / "baseline_training_summary.json",
                training_output_dir=tmp_path / "training_run",
                feature_schema_path=tmp_path / "feature_schema.json",
                split_summary_path=tmp_path / "split_summary.json",
                prediction_config_path=tmp_path / "m4_batch_prediction.yaml",
                source_dataset_path=tmp_path / "m4_modeling_dataset.parquet",
                source_dataset_metadata_path=tmp_path / "m4_modeling_dataset.metadata.json",
                split_config_path=tmp_path / "m4_split.yaml",
                split_metadata_path=tmp_path / "m4_train_validation_split.metadata.json",
                split_summary={"validation_row_count": 2},
                logged_output_signature="abc123",
                source_models=[
                    {"model_name": "decision_tree", "model_artifact_path": "a"},
                    {"model_name": "logistic_regression", "model_artifact_path": "b"},
                ],
                inference_partition="validation",
                target_column="target_next_session_direction",
                task_type="classification",
                inference_key_columns=["symbol", "date", "target_date"],
            )
            save_m4_prediction_log(
                prediction_log,
                output_path=output_path,
                metadata=metadata,
                metadata_path=metadata_path,
                definition=self.definition,
            )

            bundle = load_m4_prediction_log_bundle(
                dataset_path=output_path,
                metadata_path=metadata_path,
            )
            pd.testing.assert_frame_equal(bundle["dataframe"], prediction_log)
            self.assertEqual(bundle["metadata"]["output_log"]["row_count"], 2)
            self.assertEqual(bundle["schema"]["join_key_columns"], ["model_name", "symbol", "date", "target_date"])

    def test_prediction_log_rejects_missing_traceability_columns(self) -> None:
        prediction_log = self._build_prediction_log().drop(columns=["model_artifact_path"])
        with self.assertRaisesRegex(ValueError, "missing required columns: model_artifact_path"):
            normalize_m4_prediction_log(prediction_log, self.definition)

    def test_prediction_log_rejects_duplicate_model_row_keys(self) -> None:
        prediction_log = normalize_m4_prediction_log(self._build_prediction_log(), self.definition)
        duplicate_row = prediction_log.iloc[[0]].copy()
        duplicate_row["estimator"] = "different_estimator"
        duplicate_log = pd.concat([prediction_log, duplicate_row], ignore_index=True)
        duplicate_log = duplicate_log.loc[:, get_m4_prediction_log_column_order(self.definition)]
        duplicate_log = duplicate_log.sort_values(list(self.definition.sort_order)).reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "duplicate model/timestamp prediction rows"):
            validate_m4_prediction_log_contract(duplicate_log, self.definition)

    def test_prediction_log_rejects_invalid_probability_values(self) -> None:
        prediction_log = self._build_prediction_log()
        prediction_log.loc[0, "predicted_probability"] = 1.5
        with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
            normalize_m4_prediction_log(prediction_log, self.definition)


if __name__ == "__main__":
    unittest.main()
