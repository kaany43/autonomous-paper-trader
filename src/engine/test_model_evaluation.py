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
from src.engine.model_evaluation import (
    load_m4_baseline_evaluation_definition,
    run_m4_baseline_evaluation,
)
from src.strategy.ml_baselines import (
    load_m4_baseline_training_definition,
    prepare_m4_baseline_training_data,
    run_m4_baseline_training,
)


class M4BaselineEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_definition = load_m4_modeling_dataset_definition()
        self.target_definition = load_m4_target_definition()
        self.training_definition = load_m4_baseline_training_definition()
        self.evaluation_definition = load_m4_baseline_evaluation_definition()
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
        modeling = modeling.sample(frac=1.0, random_state=13).reset_index(drop=True)
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

    def test_each_trained_model_produces_structured_evaluation_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            result = run_m4_baseline_evaluation(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                evaluation_definition=evaluation_definition,
            )

            self.assertTrue(result["output_dir"].exists())
            self.assertTrue(result["manifest_path"].exists())
            self.assertTrue(result["summary_json_path"].exists())
            self.assertTrue(result["summary_csv_path"].exists())
            self.assertEqual(len(result["model_reports"]), 2)

            summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["model_count"], 2)
            self.assertEqual(summary["metrics"], ["accuracy", "precision", "recall", "f1"])

            summary_csv = pd.read_csv(result["summary_csv_path"])
            self.assertEqual(summary_csv["model_name"].tolist(), ["decision_tree", "logistic_regression"])

            for model_report in result["model_reports"]:
                report_path = Path(model_report["report_path"])
                self.assertTrue(report_path.exists())
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["target_column"], "target_next_session_direction")
                self.assertEqual(report["validation_dataset"]["row_count"], 4)
                self.assertIn("accuracy", report["metrics"])

    def test_evaluation_uses_the_official_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, training_definition = self._create_training_run(tmp_path)

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            result = run_m4_baseline_evaluation(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                evaluation_definition=evaluation_definition,
            )

            prepared = prepare_m4_baseline_training_data(
                training_definition=training_definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )
            summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))

            self.assertEqual(summary["validation_dataset"]["row_count"], len(prepared["validation_dataframe"]))
            self.assertEqual(
                summary["validation_dataset"]["target_date_start"],
                prepared["split_summary"]["validation_target_date_start"],
            )
            self.assertEqual(
                summary["validation_dataset"]["target_date_end"],
                prepared["split_summary"]["validation_target_date_end"],
            )
            self.assertEqual(summary["official_split"]["method"], self.split_definition.method)

    def test_evaluation_metrics_match_training_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            training_summary = json.loads(
                Path(str(training_result["training_summary_path"])).read_text(encoding="utf-8")
            )

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            result = run_m4_baseline_evaluation(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                evaluation_definition=evaluation_definition,
            )
            evaluation_summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))

            expected_metrics = {
                row["model_name"]: row["metrics"]
                for row in training_summary["models"]
            }
            observed_metrics = {
                row["model_name"]: {metric: row[metric] for metric in ["accuracy", "precision", "recall", "f1"]}
                for row in evaluation_summary["rows"]
            }
            self.assertEqual(observed_metrics, expected_metrics)

    def test_repeated_evaluation_runs_produce_identical_metric_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            first = run_m4_baseline_evaluation(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                evaluation_definition=evaluation_definition,
            )
            second = run_m4_baseline_evaluation(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                evaluation_definition=evaluation_definition,
            )

            first_summary = json.loads(Path(first["summary_json_path"]).read_text(encoding="utf-8"))
            second_summary = json.loads(Path(second["summary_json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    {key: value for key, value in row.items() if key != "evaluation_report_path"}
                    for row in first_summary["rows"]
                ],
                [
                    {key: value for key, value in row.items() if key != "evaluation_report_path"}
                    for row in second_summary["rows"]
                ],
            )

    def test_combined_summary_matches_per_model_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            result = run_m4_baseline_evaluation(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                evaluation_definition=evaluation_definition,
            )

            summary_csv = pd.read_csv(result["summary_csv_path"]).sort_values("model_name").reset_index(drop=True)
            report_rows: list[dict[str, object]] = []
            for model_report in sorted(result["model_reports"], key=lambda item: str(item["model_name"])):
                report = json.loads(Path(model_report["report_path"]).read_text(encoding="utf-8"))
                report_rows.append(
                    {
                        "model_name": report["model_name"],
                        "accuracy": report["metrics"]["accuracy"],
                        "precision": report["metrics"]["precision"],
                        "recall": report["metrics"]["recall"],
                        "f1": report["metrics"]["f1"],
                    }
                )
            report_df = pd.DataFrame(report_rows).sort_values("model_name").reset_index(drop=True)
            pd.testing.assert_series_equal(summary_csv["model_name"], report_df["model_name"])
            for metric in ["accuracy", "precision", "recall", "f1"]:
                pd.testing.assert_series_equal(summary_csv[metric], report_df[metric], check_names=False)

    def test_missing_model_artifact_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_result, _ = self._create_training_run(tmp_path)
            training_summary_path = Path(str(training_result["training_summary_path"]))
            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            missing_model_path = Path(training_summary["models"][0]["artifact_path"])
            missing_model_path.unlink()

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing trained model artifact"):
                run_m4_baseline_evaluation(
                    training_summary_path=training_summary_path,
                    evaluation_definition=evaluation_definition,
                )

    def test_evaluation_fails_when_validation_dataset_signature_changes(self) -> None:
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
                pd.to_numeric(mutated_df.loc[validation_target_dates, "open"], errors="coerce") + 0.5
            )
            mutated_df.to_parquet(training_definition.modeling_dataset_path, index=False)

            evaluation_definition = replace(
                self.evaluation_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "model_evaluations"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "Rebuilt validation partition signature does not match the stored training artifacts",
            ):
                run_m4_baseline_evaluation(
                    training_summary_path=Path(str(training_result["training_summary_path"])),
                    evaluation_definition=evaluation_definition,
                )


if __name__ == "__main__":
    unittest.main()
