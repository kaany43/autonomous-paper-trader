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
from src.engine.ml_vs_rule_comparison import (
    load_m4_ml_vs_rule_comparison_definition,
    run_m4_ml_vs_rule_comparison,
)
from src.engine.prediction_pipeline import (
    load_m4_batch_prediction_definition,
    run_m4_batch_prediction,
)
from src.strategy.ml_baselines import (
    build_dataframe_signature,
    load_m4_baseline_training_definition,
    run_m4_baseline_training,
)


class M4MLVsRuleComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_definition = load_m4_modeling_dataset_definition()
        self.target_definition = load_m4_target_definition()
        self.training_definition = load_m4_baseline_training_definition()
        self.prediction_definition = load_m4_batch_prediction_definition()
        self.comparison_definition = load_m4_ml_vs_rule_comparison_definition()
        self.split_definition = replace(
            load_m4_split_definition(),
            validation_start_date="2024-01-09",
            validation_end_date="2024-01-10",
        )
        self.feature_columns = get_m4_modeling_feature_columns(self.dataset_definition)

    def _build_feature_and_modeling_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        business_dates = pd.bdate_range("2024-01-02", periods=6)
        plan = {
            "AAA": [
                {"adj_close": 99.0, "ma20": 100.0, "ma50": 101.0, "ret5d": -0.05, "volr": 0.70, "target": 0},
                {"adj_close": 99.0, "ma20": 100.0, "ma50": 101.0, "ret5d": 0.02, "volr": 1.00, "target": 1},
                {"adj_close": 104.0, "ma20": 100.0, "ma50": 98.0, "ret5d": 0.06, "volr": 1.20, "target": 1},
                {"adj_close": 105.0, "ma20": 101.0, "ma50": 99.0, "ret5d": 0.05, "volr": 1.10, "target": 1},
                {"adj_close": 106.0, "ma20": 102.0, "ma50": 100.0, "ret5d": 0.04, "volr": 1.00, "target": 0},
                {"adj_close": 100.0, "ma20": 103.0, "ma50": 101.0, "ret5d": -0.03, "volr": 0.90, "target": 0},
            ],
            "BBB": [
                {"adj_close": 49.0, "ma20": 50.0, "ma50": 52.0, "ret5d": -0.02, "volr": 0.90, "target": 0},
                {"adj_close": 50.0, "ma20": 51.0, "ma50": 52.0, "ret5d": -0.01, "volr": 0.90, "target": 0},
                {"adj_close": 51.0, "ma20": 52.0, "ma50": 53.0, "ret5d": 0.01, "volr": 0.90, "target": 0},
                {"adj_close": 52.0, "ma20": 53.0, "ma50": 54.0, "ret5d": 0.02, "volr": 0.90, "target": 0},
                {"adj_close": 58.0, "ma20": 54.0, "ma50": 52.0, "ret5d": 0.07, "volr": 1.30, "target": 1},
                {"adj_close": 59.0, "ma20": 55.0, "ma50": 53.0, "ret5d": 0.06, "volr": 1.10, "target": 1},
            ],
        }

        feature_rows: list[dict[str, object]] = []
        modeling_rows: list[dict[str, object]] = []
        for symbol_index, symbol in enumerate(["AAA", "BBB"]):
            for row_index, decision_date in enumerate(business_dates):
                raw = plan[symbol][row_index]
                adj_close = float(raw["adj_close"])
                ma20 = float(raw["ma20"])
                ma50 = float(raw["ma50"])
                ret5d = float(raw["ret5d"])
                volume_ratio = float(raw["volr"])
                volume = 1_000_000.0 + (symbol_index * 50_000.0) + (row_index * 1_000.0)
                volume_ma20 = volume / volume_ratio
                ma10 = ma20 + 1.0
                rolling_high = adj_close + 5.0
                rolling_low = adj_close - 5.0
                range_pos = (adj_close - rolling_low) / (rolling_high - rolling_low)

                base_row: dict[str, object] = {
                    "date": decision_date,
                    "symbol": symbol,
                    "open": adj_close - 1.0,
                    "high": adj_close + 1.0,
                    "low": adj_close - 2.0,
                    "close": adj_close - 0.5,
                    "adj_close": adj_close,
                    "volume": volume,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "ret_1d": ret5d / 5.0,
                    "ret_5d": ret5d,
                    "ret_10d": ret5d / 2.0,
                    "ma_10": ma10,
                    "ma_20": ma20,
                    "ma_50": ma50,
                    "vol_20": 0.02 + (row_index * 0.001),
                    "volume_change_1d": 0.01 if ret5d > 0 else -0.01,
                    "volume_ma_20": volume_ma20,
                    "volume_ratio_20": volume_ratio,
                    "price_vs_ma10": (adj_close / ma10) - 1.0,
                    "price_vs_ma20": (adj_close / ma20) - 1.0,
                    "price_vs_ma50": (adj_close / ma50) - 1.0,
                    "ma10_vs_ma20": (ma10 / ma20) - 1.0,
                    "ma20_vs_ma50": (ma20 / ma50) - 1.0,
                    "rolling_high_20": rolling_high,
                    "rolling_low_20": rolling_low,
                    "range_pos_20": range_pos,
                }
                feature_rows.append(dict(base_row))

                modeling_row = dict(base_row)
                target_value = int(raw["target"])
                modeling_row["target_date"] = decision_date + pd.offsets.BDay(1)
                modeling_row["target_is_valid"] = True
                modeling_row["target_next_session_return"] = 0.02 if target_value == 1 else -0.02
                modeling_row["target_next_session_direction"] = target_value
                modeling_rows.append(modeling_row)

        feature_df = pd.DataFrame(feature_rows).loc[:, ["date", "symbol", *self.feature_columns]]
        feature_df = feature_df.sample(frac=1.0, random_state=23).reset_index(drop=True)

        modeling_df = pd.DataFrame(modeling_rows)
        modeling_df = modeling_df.loc[
            :,
            get_m4_modeling_dataset_column_order(self.dataset_definition, self.target_definition),
        ]
        modeling_df = modeling_df.sample(frac=1.0, random_state=29).reset_index(drop=True)
        modeling_df = normalize_m4_modeling_dataset(
            modeling_df,
            dataset_definition=self.dataset_definition,
            target_definition=self.target_definition,
        )
        return feature_df, modeling_df

    def _write_input_artifacts(
        self,
        tmp_path: Path,
    ) -> tuple[Path, Path, Path, Path]:
        feature_df, modeling_df = self._build_feature_and_modeling_datasets()
        feature_dataset_path = tmp_path / "market_features.parquet"
        modeling_dataset_path = tmp_path / "m4_modeling_dataset.parquet"
        modeling_metadata_path = tmp_path / "m4_modeling_dataset.metadata.json"
        settings_path = tmp_path / "settings.yaml"

        feature_df.to_parquet(feature_dataset_path, index=False)
        modeling_df.to_parquet(modeling_dataset_path, index=False)
        modeling_metadata_path.write_text("{}", encoding="utf-8")
        settings_path.write_text(
            "\n".join(
                [
                    "project:",
                    "  name: ml-vs-rule-test",
                    "strategy:",
                    "  name: momentum_v0",
                    "  top_k: 2",
                    "  min_score: 0.0",
                    "  min_volume_ratio: 0.8",
                    "portfolio:",
                    "  initial_cash: 500.0",
                    "  max_open_positions: 2",
                    "  fractional_shares: true",
                    "execution:",
                    "  commission_rate: 0.001",
                    "  slippage_rate: 0.001",
                    "  execution_timing: next_open",
                    "benchmark:",
                    '  benchmark_symbol: ""',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return feature_dataset_path, modeling_dataset_path, modeling_metadata_path, settings_path

    def _run_comparison_fixture(
        self,
        tmp_path: Path,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        feature_dataset_path, modeling_dataset_path, modeling_metadata_path, settings_path = self._write_input_artifacts(
            tmp_path
        )
        training_definition = replace(
            self.training_definition,
            modeling_dataset_path=str(modeling_dataset_path),
            modeling_dataset_metadata_path=str(modeling_metadata_path),
            output_dir=str(tmp_path / "outputs" / "models"),
            split_metadata_path=str(tmp_path / "m4_train_validation_split.metadata.json"),
        )
        training_result = run_m4_baseline_training(
            training_definition=training_definition,
            split_definition=self.split_definition,
            target_definition=self.target_definition,
        )

        prediction_definition = replace(
            self.prediction_definition,
            output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
        )
        prediction_result = run_m4_batch_prediction(
            training_summary_path=Path(str(training_result["training_summary_path"])),
            prediction_definition=prediction_definition,
        )

        comparison_definition = replace(
            self.comparison_definition,
            output_dir=str(tmp_path / "outputs" / "reports" / "ml_vs_rule_comparisons"),
            strategy_config_path=str(settings_path),
            feature_dataset_path=str(feature_dataset_path),
        )
        comparison_result = run_m4_ml_vs_rule_comparison(
            predictions_path=Path(str(prediction_result["predictions_path"])),
            metadata_path=Path(str(prediction_result["prediction_log_metadata_path"])),
            comparison_definition=comparison_definition,
        )
        return training_result, prediction_result, comparison_result

    def _expected_rule_alignment(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "date": pd.Timestamp("2024-01-08"),
                    "target_date": pd.Timestamp("2024-01-09"),
                    "rule_action": "HOLD",
                    "rule_entry_signal": 0,
                    "rule_exit_signal": 0,
                    "rule_schedule_status": "NO_SIGNAL",
                },
                {
                    "symbol": "AAA",
                    "date": pd.Timestamp("2024-01-09"),
                    "target_date": pd.Timestamp("2024-01-10"),
                    "rule_action": "SELL",
                    "rule_entry_signal": 0,
                    "rule_exit_signal": 1,
                    "rule_schedule_status": "NO_NEXT_TRADING_SESSION",
                },
                {
                    "symbol": "BBB",
                    "date": pd.Timestamp("2024-01-08"),
                    "target_date": pd.Timestamp("2024-01-09"),
                    "rule_action": "BUY",
                    "rule_entry_signal": 1,
                    "rule_exit_signal": 0,
                    "rule_schedule_status": "SCHEDULED",
                },
                {
                    "symbol": "BBB",
                    "date": pd.Timestamp("2024-01-09"),
                    "target_date": pd.Timestamp("2024-01-10"),
                    "rule_action": "HOLD",
                    "rule_entry_signal": 0,
                    "rule_exit_signal": 0,
                    "rule_schedule_status": "NO_SIGNAL",
                },
            ]
        )

    def test_comparison_aligns_prediction_rows_with_rule_strategy_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, comparison_result = self._run_comparison_fixture(tmp_path)

            self.assertTrue(Path(str(comparison_result["output_dir"])).exists())
            self.assertTrue(Path(str(comparison_result["manifest_path"])).exists())
            self.assertTrue(Path(str(comparison_result["aligned_path"])).exists())
            self.assertTrue(Path(str(comparison_result["aligned_metadata_path"])).exists())
            self.assertTrue(Path(str(comparison_result["summary_json_path"])).exists())

            aligned_df = pd.read_parquet(comparison_result["aligned_path"])
            self.assertEqual(len(aligned_df), 8)
            expected_rule_rows = self._expected_rule_alignment()

            for model_name in sorted(aligned_df["model_name"].unique().tolist()):
                observed = (
                    aligned_df.loc[
                        aligned_df["model_name"] == model_name,
                        [
                            "symbol",
                            "date",
                            "target_date",
                            "rule_action",
                            "rule_entry_signal",
                            "rule_exit_signal",
                            "rule_schedule_status",
                        ],
                    ]
                    .sort_values(["symbol", "date"])
                    .reset_index(drop=True)
                )
                pd.testing.assert_frame_equal(observed, expected_rule_rows)

            summary = json.loads(Path(str(comparison_result["summary_json_path"])).read_text(encoding="utf-8"))
            self.assertEqual(summary["validation_dataset"]["row_count"], 4)
            self.assertEqual(summary["methodology"]["name"], "validation_signal_alignment")
            self.assertTrue(Path(summary["rule_strategy"]["rule_replay_manifest_path"]).exists())

    def test_summary_metrics_match_aligned_artifact_and_are_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, comparison_result = self._run_comparison_fixture(tmp_path)

            aligned_df = pd.read_parquet(comparison_result["aligned_path"])
            summary_csv = pd.read_csv(comparison_result["summary_csv_path"]).sort_values("model_name").reset_index(drop=True)
            metadata = json.loads(Path(str(comparison_result["aligned_metadata_path"])).read_text(encoding="utf-8"))

            self.assertEqual(metadata["aligned_output"]["row_count"], len(aligned_df))
            self.assertEqual(metadata["aligned_output"]["output_signature"], build_dataframe_signature(aligned_df))

            for _, row in summary_csv.iterrows():
                group = aligned_df.loc[aligned_df["model_name"] == row["model_name"]].copy()
                self.assertEqual(int(row["row_count"]), len(group))
                self.assertEqual(int(row["symbol_count"]), group["symbol"].nunique())
                self.assertAlmostEqual(float(row["agreement_rate"]), float(group["is_agreement"].mean()))
                self.assertAlmostEqual(float(row["disagreement_rate"]), float(group["is_disagreement"].mean()))
                self.assertEqual(int(row["shared_entry_count"]), int((group["comparison_outcome"] == "both_entry").sum()))
                self.assertEqual(int(row["ml_only_entry_count"]), int((group["comparison_outcome"] == "ml_only_entry").sum()))
                self.assertEqual(int(row["rule_only_entry_count"]), int((group["comparison_outcome"] == "rule_only_entry").sum()))
                self.assertEqual(int(row["shared_non_entry_count"]), int((group["comparison_outcome"] == "shared_non_entry").sum()))
                self.assertEqual(int(row["rule_buy_count"]), int((group["rule_action"] == "BUY").sum()))
                self.assertEqual(int(row["rule_sell_count"]), int((group["rule_action"] == "SELL").sum()))
                self.assertEqual(int(row["rule_hold_count"]), int((group["rule_action"] == "HOLD").sum()))

    def test_repeated_runs_with_same_inputs_produce_identical_comparison_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, prediction_result, first = self._run_comparison_fixture(tmp_path)

            comparison_definition = replace(
                self.comparison_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "ml_vs_rule_comparisons"),
                strategy_config_path=str(tmp_path / "settings.yaml"),
                feature_dataset_path=str(tmp_path / "market_features.parquet"),
            )
            second = run_m4_ml_vs_rule_comparison(
                predictions_path=Path(str(prediction_result["predictions_path"])),
                metadata_path=Path(str(prediction_result["prediction_log_metadata_path"])),
                comparison_definition=comparison_definition,
            )

            first_aligned = pd.read_parquet(first["aligned_path"])
            second_aligned = pd.read_parquet(second["aligned_path"])
            pd.testing.assert_frame_equal(first_aligned, second_aligned)

            first_summary = pd.read_csv(first["summary_csv_path"]).sort_values("model_name").reset_index(drop=True)
            second_summary = pd.read_csv(second["summary_csv_path"]).sort_values("model_name").reset_index(drop=True)
            pd.testing.assert_frame_equal(first_summary, second_summary)

            first_symbol_summary = pd.read_csv(first["per_symbol_summary_csv_path"]).sort_values(["model_name", "symbol"]).reset_index(drop=True)
            second_symbol_summary = pd.read_csv(second["per_symbol_summary_csv_path"]).sort_values(["model_name", "symbol"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(first_symbol_summary, second_symbol_summary)

    def test_prediction_alignment_mismatch_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, prediction_result, _ = self._run_comparison_fixture(tmp_path)

            prediction_path = Path(str(prediction_result["predictions_path"]))
            corrupted = pd.read_parquet(prediction_path)
            corrupted = corrupted.iloc[1:].reset_index(drop=True)
            corrupted.to_parquet(prediction_path, index=False)

            comparison_definition = replace(
                self.comparison_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "ml_vs_rule_comparisons"),
                strategy_config_path=str(tmp_path / "settings.yaml"),
                feature_dataset_path=str(tmp_path / "market_features.parquet"),
            )
            with self.assertRaisesRegex(ValueError, "do not match the official validation row count"):
                run_m4_ml_vs_rule_comparison(
                    predictions_path=prediction_path,
                    metadata_path=Path(str(prediction_result["prediction_log_metadata_path"])),
                    comparison_definition=comparison_definition,
                )

    def test_missing_feature_dataset_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, prediction_result, _ = self._run_comparison_fixture(tmp_path)

            comparison_definition = replace(
                self.comparison_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "ml_vs_rule_comparisons"),
                strategy_config_path=str(tmp_path / "settings.yaml"),
                feature_dataset_path=str(tmp_path / "missing_market_features.parquet"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing rule feature dataset"):
                run_m4_ml_vs_rule_comparison(
                    predictions_path=Path(str(prediction_result["predictions_path"])),
                    metadata_path=Path(str(prediction_result["prediction_log_metadata_path"])),
                    comparison_definition=comparison_definition,
                )

    def test_comparison_fails_when_validation_dataset_signature_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_dataset_path, modeling_dataset_path, modeling_metadata_path, settings_path = self._write_input_artifacts(
                tmp_path
            )
            training_definition = replace(
                self.training_definition,
                modeling_dataset_path=str(modeling_dataset_path),
                modeling_dataset_metadata_path=str(modeling_metadata_path),
                output_dir=str(tmp_path / "outputs" / "models"),
                split_metadata_path=str(tmp_path / "m4_train_validation_split.metadata.json"),
            )
            training_result = run_m4_baseline_training(
                training_definition=training_definition,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )
            prediction_definition = replace(
                self.prediction_definition,
                output_dir=str(tmp_path / "outputs" / "predictions" / "model_batches"),
            )
            prediction_result = run_m4_batch_prediction(
                training_summary_path=Path(str(training_result["training_summary_path"])),
                prediction_definition=prediction_definition,
            )

            mutated_modeling_df = pd.read_parquet(modeling_dataset_path)
            validation_mask = pd.to_datetime(mutated_modeling_df["target_date"]) == pd.Timestamp("2024-01-09")
            mutated_modeling_df.loc[validation_mask, "target_next_session_direction"] = 1
            mutated_modeling_df.loc[validation_mask, "target_next_session_return"] = 0.02
            mutated_modeling_df.to_parquet(modeling_dataset_path, index=False)

            comparison_definition = replace(
                self.comparison_definition,
                output_dir=str(tmp_path / "outputs" / "reports" / "ml_vs_rule_comparisons"),
                strategy_config_path=str(settings_path),
                feature_dataset_path=str(feature_dataset_path),
            )
            with self.assertRaisesRegex(
                ValueError,
                "Rebuilt validation partition signature does not match the stored training artifacts",
            ):
                run_m4_ml_vs_rule_comparison(
                    predictions_path=Path(str(prediction_result["predictions_path"])),
                    metadata_path=Path(str(prediction_result["prediction_log_metadata_path"])),
                    comparison_definition=comparison_definition,
                )

    def test_summary_captures_decision_oriented_disagreement_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, comparison_result = self._run_comparison_fixture(tmp_path)

            summary = json.loads(Path(str(comparison_result["summary_json_path"])).read_text(encoding="utf-8"))
            self.assertEqual(len(summary["rows"]), 2)
            for row in summary["rows"]:
                self.assertIn("actual_positive_rate_ml_only_entry", row)
                self.assertIn("actual_positive_rate_rule_only_entry", row)
                self.assertIn("rule_sell_count", row)
                self.assertIn("ml_accuracy_vs_actual", row)
                self.assertIn("rule_entry_accuracy_vs_actual", row)


if __name__ == "__main__":
    unittest.main()
