from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.target_pipeline import prepare_m4_modeling_dataset, run_m4_target_preparation
from src.data.targets import (
    TARGET_DATE_COLUMN,
    TARGET_VALID_COLUMN,
    load_m4_target_definition,
)


class M4TargetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = load_m4_target_definition()

    def test_prepare_modeling_dataset_is_deterministic_and_strips_legacy_target_columns(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "date": "2024-01-03",
                    "symbol": "BBB",
                    "adj_close": 205.0,
                    "ret_1d": 0.0789,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 1,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "adj_close": 100.0,
                    "ret_1d": 0.0,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 1,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "adj_close": 101.0,
                    "ret_1d": 0.01,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 1,
                },
                {
                    "date": "2024-01-04",
                    "symbol": "AAA",
                    "adj_close": 99.0,
                    "ret_1d": -0.1,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 0,
                },
                {
                    "date": "2024-01-01",
                    "symbol": "BBB",
                    "adj_close": 190.0,
                    "ret_1d": 0.0,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 1,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "AAA",
                    "adj_close": 110.0,
                    "ret_1d": 0.0891,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 1,
                },
                {
                    "date": "2024-01-04",
                    "symbol": "BBB",
                    "adj_close": 200.0,
                    "ret_1d": -0.0244,
                    "target_ret_1d": 999.0,
                    "target_up_1d": 0,
                },
            ]
        )

        first, first_summary = prepare_m4_modeling_dataset(features, self.definition)
        second, second_summary = prepare_m4_modeling_dataset(features, self.definition)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            first[["symbol", "date", TARGET_DATE_COLUMN]].to_dict("records"),
            [
                {
                    "symbol": "AAA",
                    "date": pd.Timestamp("2024-01-02"),
                    TARGET_DATE_COLUMN: pd.Timestamp("2024-01-03"),
                },
                {
                    "symbol": "AAA",
                    "date": pd.Timestamp("2024-01-03"),
                    TARGET_DATE_COLUMN: pd.Timestamp("2024-01-04"),
                },
                {
                    "symbol": "BBB",
                    "date": pd.Timestamp("2024-01-01"),
                    TARGET_DATE_COLUMN: pd.Timestamp("2024-01-03"),
                },
                {
                    "symbol": "BBB",
                    "date": pd.Timestamp("2024-01-03"),
                    TARGET_DATE_COLUMN: pd.Timestamp("2024-01-04"),
                },
            ],
        )
        self.assertTrue(bool(first[TARGET_VALID_COLUMN].all()))
        self.assertEqual(
            first[self.definition.official_target_column].astype("int64").tolist(),
            [1, 0, 1, 0],
        )
        self.assertNotIn("target_ret_1d", first.columns)
        self.assertNotIn("target_up_1d", first.columns)
        self.assertEqual(
            first_summary["dropped_input_target_columns"],
            ["target_ret_1d", "target_up_1d"],
        )
        self.assertEqual(first_summary["dropped_duplicate_row_count"], 1)
        self.assertEqual(first_summary["dropped_invalid_row_count"], 2)

    def test_prepare_modeling_dataset_drops_invalid_rows_consistently(self) -> None:
        features = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0, "ret_1d": 0.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": pd.NA, "ret_1d": pd.NA},
                {"date": "2024-01-01", "symbol": "BBB", "adj_close": 5.0, "ret_1d": 0.0},
                {"date": "2024-01-02", "symbol": "BBB", "adj_close": 6.0, "ret_1d": 0.2},
                {"date": "2024-01-03", "symbol": "BBB", "adj_close": 7.0, "ret_1d": 0.1667},
            ]
        )

        modeling_df, summary = prepare_m4_modeling_dataset(features, self.definition)

        self.assertEqual(
            modeling_df[["symbol", "date"]].to_dict("records"),
            [
                {"symbol": "BBB", "date": pd.Timestamp("2024-01-01")},
                {"symbol": "BBB", "date": pd.Timestamp("2024-01-02")},
            ],
        )
        self.assertTrue(bool(modeling_df[TARGET_VALID_COLUMN].all()))
        self.assertEqual(summary["dropped_invalid_row_count"], 3)

    def test_prepare_modeling_dataset_rejects_missing_required_columns(self) -> None:
        features = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "ret_1d": 0.0},
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing required columns: adj_close"):
            prepare_m4_modeling_dataset(features, self.definition)

    def test_prepare_modeling_dataset_rejects_invalid_dates(self) -> None:
        features = pd.DataFrame(
            [
                {"date": "not-a-date", "symbol": "AAA", "adj_close": 10.0, "ret_1d": 0.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0, "ret_1d": 0.1},
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires valid non-null date values"):
            prepare_m4_modeling_dataset(features, self.definition)

    def test_run_target_preparation_writes_loadable_outputs(self) -> None:
        features = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0, "ret_1d": 0.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 11.0, "ret_1d": 0.1},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 10.5, "ret_1d": -0.0455},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "market_features.parquet"
            output_path = tmp_path / "m4_modeling_dataset.parquet"
            metadata_path = tmp_path / "m4_modeling_dataset.metadata.json"

            features.to_parquet(input_path, index=False)
            result = run_m4_target_preparation(
                input_path=input_path,
                output_path=output_path,
                metadata_path=metadata_path,
                definition=self.definition,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(metadata_path.exists())

            loaded = pd.read_parquet(output_path)
            pd.testing.assert_frame_equal(loaded, result["dataframe"])

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["input_dataset"]["path"], str(input_path))
            self.assertEqual(metadata["output_dataset"]["path"], str(output_path))
            self.assertEqual(
                metadata["target_columns"],
                [
                    TARGET_DATE_COLUMN,
                    TARGET_VALID_COLUMN,
                    self.definition.helper_return_column,
                    self.definition.official_target_column,
                ],
            )
            self.assertEqual(
                metadata["invalid_label_handling"]["pipeline_rule"],
                "drop rows where target_is_valid is false after label construction",
            )


if __name__ == "__main__":
    unittest.main()
