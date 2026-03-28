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
from src.strategy.ml_baselines import (
    load_m4_baseline_training_definition,
    load_trained_baseline_model,
    prepare_m4_baseline_training_data,
    run_m4_baseline_training,
)


class M4BaselineTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_definition = load_m4_modeling_dataset_definition()
        self.target_definition = load_m4_target_definition()
        self.training_definition = load_m4_baseline_training_definition()
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
        modeling = modeling.sample(frac=1.0, random_state=11).reset_index(drop=True)
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

    def test_prepare_training_data_reuses_official_time_aware_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset()
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
            )

            prepared = prepare_m4_baseline_training_data(
                training_definition=definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )

            self.assertEqual(prepared["split_definition"].method, "chronological_holdout_by_target_date_window")
            self.assertLess(
                prepared["train_dataframe"]["target_date"].max(),
                prepared["validation_dataframe"]["target_date"].min(),
            )
            self.assertEqual(prepared["split_summary"]["train_row_count"], 8)
            self.assertEqual(prepared["split_summary"]["validation_row_count"], 4)

    def test_run_training_writes_reloadable_model_artifacts_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset()
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
                split_metadata_path=str(tmp_path / "m4_train_validation_split.metadata.json"),
            )

            result = run_m4_baseline_training(
                training_definition=definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )

            self.assertTrue(result["output_dir"].exists())
            self.assertTrue(result["manifest_path"].exists())
            self.assertTrue(result["feature_schema_path"].exists())
            self.assertTrue(result["split_summary_path"].exists())
            self.assertTrue(result["training_summary_path"].exists())
            self.assertEqual(len(result["models"]), 2)

            summary = json.loads(result["training_summary_path"].read_text(encoding="utf-8"))
            self.assertEqual(summary["official_split"]["method"], self.split_definition.method)
            self.assertEqual(summary["model_count"], 2)
            self.assertEqual(summary["official_split"]["summary"]["train_row_count"], 8)
            self.assertEqual(summary["official_split"]["summary"]["validation_row_count"], 4)

            feature_schema = json.loads(result["feature_schema_path"].read_text(encoding="utf-8"))
            self.assertEqual(feature_schema["target_column"], self.training_definition.target_column)
            self.assertEqual(feature_schema["feature_columns"], self.feature_columns)

            for model_record in result["models"]:
                model_path = Path(model_record["artifact_path"])
                metadata_path = Path(model_record["metadata_path"])
                self.assertTrue(model_path.exists())
                self.assertTrue(metadata_path.exists())

                loaded_pipeline = load_trained_baseline_model(model_path)
                self.assertEqual(loaded_pipeline.__class__.__name__, "Pipeline")

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["feature_columns"], self.feature_columns)
                self.assertEqual(metadata["target_column"], self.training_definition.target_column)
                self.assertIn("accuracy", metadata["metrics"])

    def test_training_is_reproducible_for_same_data_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset()
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
            )

            first = run_m4_baseline_training(
                training_definition=definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )
            second = run_m4_baseline_training(
                training_definition=definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )

            prepared = prepare_m4_baseline_training_data(
                training_definition=definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )
            x_validation = prepared["x_validation"]

            self.assertEqual(
                [record["metrics"] for record in first["models"]],
                [record["metrics"] for record in second["models"]],
            )

            for first_record, second_record in zip(first["models"], second["models"], strict=True):
                first_model = load_trained_baseline_model(Path(first_record["artifact_path"]))
                second_model = load_trained_baseline_model(Path(second_record["artifact_path"]))
                self.assertEqual(
                    first_model.predict(x_validation).tolist(),
                    second_model.predict(x_validation).tolist(),
                )

    def test_training_fails_clearly_for_missing_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset().drop(columns=["ret_1d"])
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                run_m4_baseline_training(
                    training_definition=definition,
                    split_definition=self.split_definition,
                    target_definition=self.target_definition,
                )

    def test_training_fails_clearly_for_non_numeric_feature_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset().copy()
            modeling_df["ret_1d"] = modeling_df["ret_1d"].map(str)
            modeling_df.loc[0, "ret_1d"] = "bad-value"
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
            )

            with self.assertRaisesRegex(ValueError, "non-numeric values in 'ret_1d'"):
                run_m4_baseline_training(
                    training_definition=definition,
                    split_definition=self.split_definition,
                    target_definition=self.target_definition,
                )

    def test_training_rejects_stale_metadata_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset()
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)
            metadata_path.write_text(
                json.dumps({"schema": {"feature_columns": ["ret_1d"]}}, indent=2),
                encoding="utf-8",
            )

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
            )

            with self.assertRaisesRegex(ValueError, "feature_columns do not match the official schema"):
                run_m4_baseline_training(
                    training_definition=definition,
                    split_definition=self.split_definition,
                    target_definition=self.target_definition,
                )

    def test_training_fails_clearly_when_training_target_has_single_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            modeling_df = self._build_modeling_dataset().copy()
            training_mask = modeling_df["target_date"] < pd.Timestamp("2024-01-09")
            modeling_df.loc[training_mask, "target_next_session_direction"] = 1
            modeling_df.loc[training_mask, "target_next_session_return"] = 0.01
            dataset_path, metadata_path = self._write_modeling_artifacts(tmp_path, modeling_df)

            definition = replace(
                self.training_definition,
                modeling_dataset_path=str(dataset_path),
                modeling_dataset_metadata_path=str(metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
            )

            with self.assertRaisesRegex(ValueError, "at least two target classes"):
                run_m4_baseline_training(
                    training_definition=definition,
                    split_definition=self.split_definition,
                    target_definition=self.target_definition,
                )


if __name__ == "__main__":
    unittest.main()
