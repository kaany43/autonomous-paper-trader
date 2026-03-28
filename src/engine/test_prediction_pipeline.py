from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.data.modeling_dataset import (
    get_m4_modeling_dataset_column_order,
    get_m4_modeling_feature_columns,
    load_m4_modeling_dataset_definition,
    normalize_m4_modeling_dataset,
)
from src.data.splits import load_m4_split_definition
from src.data.targets import load_m4_target_definition
from src.engine.prediction_pipeline import (
    load_m4_batch_prediction_definition,
    run_m4_batch_prediction,
)
from src.strategy.ml_baselines import (
    load_m4_baseline_training_definition,
    prepare_m4_baseline_training_data,
    run_m4_baseline_training,
)


class M4BatchPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_definition = load_m4_modeling_dataset_definition()
        self.target_definition = load_m4_target_definition()
        self.training_definition = load_m4_baseline_training_definition()
        self.prediction_definition = load_m4_batch_prediction_definition()
        self.split_definition = replace(
            load_m4_split_definition(),
            validation_start_date="2024-01-09",
            validation_end_date="2024-01-10",
        )
        self.feature_columns = get_m4_modeling_feature_columns(self.dataset_definition)

    def _build_modeling_dataset(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        business_dates = pd.bdate_range("2024-01-02", periods=6)
        target_map = {
            "AAA": [0, 1, 0, 1, 1, 0],
            "BBB": [1, 0, 1, 0, 0, 1],
        }

        for symbol_index, symbol in enumerate(["AAA", "BBB"]):
            for row_index, decision_date in enumerate(business_dates):
                target_value = target_map[symbol][row_index]
                base_row: dict[str, object] = {
                    "date": decision_date,
                    "symbol": symbol,
                }
                for feature_index, column in enumerate(self.feature_columns):
                    signed_signal = 1.0 if target_value == 1 else -1.0
                    base_value = float((row_index + 1) * (feature_index + 2) + (symbol_index * 3))
                    value = signed_signal * (0.05 + (base_value / 1000.0))
                    if column in {"open", "high", "low", "close", "adj_close"}:
                        value = 100.0 + base_value + (target_value * 2.0)
                    if column == "volume":
                        value = 1000000.0 + (row_index * 10000.0) + (symbol_index * 5000.0)
                    if column in {"dividends", "stock_splits"}:
                        value = 0.0
                    base_row[column] = value

                if row_index == 0:
                    base_row["ma_50"] = pd.NA
                    base_row["vol_20"] = pd.NA

                base_row["target_date"] = decision_date + pd.offsets.BDay(1)
                base_row["target_is_valid"] = True
                base_row["target_next_session_return"] = 0.01 if target_value == 1 else -0.01
                base_row["target_next_session_direction"] = target_value
                rows.append(base_row)

        modeling = pd.DataFrame(rows)
        modeling = modeling.loc[
            :,
            get_m4_modeling_dataset_column_order(self.dataset_definition, self.target_definition),
        ]
        modeling = modeling.sample(frac=1.0, random_state=17).reset_index(drop=True)
        return normalize_m4_modeling_dataset(
            modeling,
            dataset_definition=self.dataset_definition,
            target_definition=self.target_definition,
        )

    def _write_modeling_artifacts(self, tmp_path: Path, modeling_df: pd.DataFrame) -> tuple[Path, Path]:
        dataset_path = tmp_path / "m4_modeling_dataset.parquet"
        metadata_path = tmp_path / "m4_modeling_dataset.metadata.json"
        modeling_df.to_parquet(dataset_path, index=False)
        metadata_path.write_text("{}", encoding="utf-8")
        return dataset_path, metadata_path

    def _create_training_run(
        self,
        tmp_path: Path,
        modeling_df: pd.DataFrame | None = None,
    ) -> tuple[dict[str, object], object]:
        resolved_modeling_df = modeling_df if modeling_df is not None else self._build_modeling_dataset()
        dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, resolved_modeling_df)

        training_definition = replace(
            self.training_definition,
            modeling_dataset_path=str(dataset_path),
            modeling_dataset_metadata_path=str(metadata_path),
            output_dir=str(tmp_path / "outputs" / "models"),
            split_metadata_path=str(tmp_path / "m4_train_validation_split.metadata.json"),
        )

        training_result = run_m4_baseline_training(
            training_definition=training_definition,
            split_definition=self.split_definition,
            target_definition=self.target_definition,
        )
        return training_result, training_definition

    def test_batch_prediction_generates_reloadable_predictions_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            training_summary = json.loads(
                Path(str(training_result["training_summary_path"])).read_text(encoding="utf-8")
            )
            original_mtimes = {
                record["model_name"]: Path(record["artifact_path"]).stat().st_mtime_ns
                for record in training_summary["models"]
            }

            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )
            result = run_m4_batch_prediction(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                prediction_definition=prediction_definition,
            )

            self.assertTrue(result["output_dir"].exists())
            self.assertTrue(result["manifest_path"].exists())
            self.assertTrue(result["summary_json_path"].exists())
            self.assertTrue(result["predictions_path"].exists())

            prediction_df = pd.read_parquet(result["predictions_path"])
            self.assertEqual(result["prediction_row_count"], len(prediction_df))
            self.assertEqual(sorted(prediction_df["model_name"].unique().tolist()), ["decision_tree", "logistic_regression"])

            summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["model_count"], 2)
            self.assertEqual(summary["prediction_output"]["format"], "parquet")

            refreshed_mtimes = {
                record["model_name"]: Path(record["artifact_path"]).stat().st_mtime_ns
                for record in training_summary["models"]
            }
            self.assertEqual(original_mtimes, refreshed_mtimes)

    def test_predictions_preserve_validation_time_order_and_row_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, training_definition = self._create_training_run(tmp_path)
            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )

            result = run_m4_batch_prediction(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                prediction_definition=prediction_definition,
            )
            prediction_df = pd.read_parquet(result["predictions_path"])
            prepared = prepare_m4_baseline_training_data(
                training_definition=training_definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )
            expected_keys = prepared["validation_dataframe"][["symbol", "date", "target_date"]].reset_index(drop=True)

            self.assertEqual(len(prediction_df), len(expected_keys) * 2)
            for model_name in ["decision_tree", "logistic_regression"]:
                model_predictions = (
                    prediction_df.loc[prediction_df["model_name"] == model_name, ["symbol", "date", "target_date"]]
                    .reset_index(drop=True)
                )
                pd.testing.assert_frame_equal(model_predictions, expected_keys)

    def test_repeated_runs_with_same_inputs_produce_identical_prediction_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )

            first = run_m4_batch_prediction(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                prediction_definition=prediction_definition,
            )
            second = run_m4_batch_prediction(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                prediction_definition=prediction_definition,
            )

            first_df = pd.read_parquet(first["predictions_path"])
            second_df = pd.read_parquet(second["predictions_path"])
            pd.testing.assert_frame_equal(first_df, second_df)

    def test_feature_schema_mismatch_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            feature_schema_path = Path(str(training_result["feature_schema_path"]))
            feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
            feature_schema["feature_columns"] = ["ret_1d"]
            feature_schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "Training feature schema does not match the rebuilt inference feature order",
            ):
                run_m4_batch_prediction(
                    training_summary_path=Path(str(training_result["training_summary_path"])),
                    prediction_definition=prediction_definition,
                )

    def test_missing_model_artifact_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            training_summary_path = Path(str(training_result["training_summary_path"]))
            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            Path(training_summary["models"][0]["artifact_path"]).unlink()

            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing trained model artifact"):
                run_m4_batch_prediction(
                    training_summary_path=training_summary_path,
                    prediction_definition=prediction_definition,
                )

    def test_prediction_allows_older_training_runs_without_stored_validation_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            split_summary_path = Path(str(training_result["split_summary_path"]))
            split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
            split_summary.pop("validation_dataset_signature", None)
            split_summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

            training_summary_path = Path(str(training_result["training_summary_path"]))
            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            training_summary["official_split"]["summary"].pop("validation_dataset_signature", None)
            training_summary_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )
            result = run_m4_batch_prediction(
                training_summary_path=training_summary_path,
                prediction_definition=prediction_definition,
            )

            summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["official_split"]["validation_dataset_signature_check_status"],
                "not_available_in_training_artifacts",
            )
            self.assertIsNone(summary["official_split"]["stored_validation_dataset_signature"])

    def test_prediction_fails_when_validation_dataset_signature_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, training_definition = self._create_training_run(tmp_path)

            mutated_df = pd.read_parquet(training_definition.modeling_dataset_path).copy()
            validation_target_dates = mutated_df["target_date"].between(
                self.split_definition.validation_start_date,
                self.split_definition.validation_end_date,
            )
            self.assertTrue(validation_target_dates.any())
            mutated_df.loc[validation_target_dates, "open"] = (
                pd.to_numeric(mutated_df.loc[validation_target_dates, "open"], errors="coerce") + 1.0
            )
            mutated_df.to_parquet(training_definition.modeling_dataset_path, index=False)

            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "Rebuilt validation partition signature does not match the stored training artifacts",
            ):
                run_m4_batch_prediction(
                    training_summary_path=Path(str(training_result["training_summary_path"])),
                    prediction_definition=prediction_definition,
                )


if __name__ == "__main__":
    unittest.main()
