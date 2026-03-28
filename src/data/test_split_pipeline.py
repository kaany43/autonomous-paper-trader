from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.data.split_pipeline import run_m4_train_validation_split
from src.data.splits import load_m4_split_definition, split_m4_modeling_dataset
from src.data.targets import load_m4_target_definition


class M4TrainValidationSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target_definition = load_m4_target_definition()
        self.split_definition = replace(
            load_m4_split_definition(),
            validation_start_date="2024-01-04",
            validation_end_date="2024-01-05",
        )

    @staticmethod
    def _build_modeling_dataset() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2024-01-05",
                    "symbol": "AAA",
                    "ret_1d": 0.01,
                    "target_date": "2024-01-08",
                    "target_is_valid": True,
                    "target_next_session_return": 0.02,
                    "target_next_session_direction": 1,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "BBB",
                    "ret_1d": -0.02,
                    "target_date": "2024-01-04",
                    "target_is_valid": True,
                    "target_next_session_return": -0.01,
                    "target_next_session_direction": 0,
                },
                {
                    "date": "2024-01-01",
                    "symbol": "AAA",
                    "ret_1d": 0.00,
                    "target_date": "2024-01-02",
                    "target_is_valid": True,
                    "target_next_session_return": 0.01,
                    "target_next_session_direction": 1,
                },
                {
                    "date": "2024-01-04",
                    "symbol": "AAA",
                    "ret_1d": 0.03,
                    "target_date": "2024-01-05",
                    "target_is_valid": True,
                    "target_next_session_return": -0.02,
                    "target_next_session_direction": 0,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "AAA",
                    "ret_1d": 0.02,
                    "target_date": "2024-01-04",
                    "target_is_valid": True,
                    "target_next_session_return": 0.04,
                    "target_next_session_direction": 1,
                },
                {
                    "date": "2024-01-01",
                    "symbol": "BBB",
                    "ret_1d": 0.00,
                    "target_date": "2024-01-02",
                    "target_is_valid": True,
                    "target_next_session_return": 0.01,
                    "target_next_session_direction": 1,
                },
                {
                    "date": "2024-01-04",
                    "symbol": "BBB",
                    "ret_1d": 0.05,
                    "target_date": "2024-01-05",
                    "target_is_valid": True,
                    "target_next_session_return": 0.01,
                    "target_next_session_direction": 1,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "ret_1d": -0.01,
                    "target_date": "2024-01-03",
                    "target_is_valid": True,
                    "target_next_session_return": -0.03,
                    "target_next_session_direction": 0,
                },
                {
                    "date": "2024-01-05",
                    "symbol": "BBB",
                    "ret_1d": -0.03,
                    "target_date": "2024-01-08",
                    "target_is_valid": True,
                    "target_next_session_return": 0.01,
                    "target_next_session_direction": 1,
                },
            ]
        )

    def test_split_is_deterministic_and_protects_boundary_leakage(self) -> None:
        modeling = self._build_modeling_dataset()

        first_train, first_validation, first_summary = split_m4_modeling_dataset(
            modeling,
            split_definition=self.split_definition,
            target_definition=self.target_definition,
        )
        second_train, second_validation, second_summary = split_m4_modeling_dataset(
            modeling.sample(frac=1.0, random_state=7).reset_index(drop=True),
            split_definition=self.split_definition,
            target_definition=self.target_definition,
        )

        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_validation, second_validation)
        self.assertEqual(first_summary, second_summary)

        self.assertTrue(
            bool(
                (
                    first_train[self.split_definition.target_timestamp_column]
                    < self.split_definition.validation_start_timestamp
                ).all()
            )
        )
        self.assertTrue(
            bool(
                first_validation[self.split_definition.target_timestamp_column].between(
                    self.split_definition.validation_start_timestamp,
                    self.split_definition.validation_end_timestamp,
                    inclusive="both",
                ).all()
            )
        )
        self.assertLess(
            first_train[self.split_definition.target_timestamp_column].max(),
            first_validation[self.split_definition.target_timestamp_column].min(),
        )
        self.assertEqual(first_summary["excluded_future_row_count"], 2)
        self.assertFalse(first_summary["input_was_sorted_by_symbol_date"])

        validation_rows = first_validation[["symbol", "date", "target_date"]].to_dict("records")
        self.assertIn(
            {
                "symbol": "BBB",
                "date": pd.Timestamp("2024-01-02"),
                "target_date": pd.Timestamp("2024-01-04"),
            },
            validation_rows,
        )
        self.assertNotIn(
            {
                "symbol": "BBB",
                "date": pd.Timestamp("2024-01-02"),
                "target_date": pd.Timestamp("2024-01-04"),
            },
            first_train[["symbol", "date", "target_date"]].to_dict("records"),
        )

    def test_split_rejects_duplicate_symbol_date_rows(self) -> None:
        modeling = pd.concat(
            [
                self._build_modeling_dataset(),
                pd.DataFrame(
                    [
                        {
                            "date": "2024-01-02",
                            "symbol": "AAA",
                            "ret_1d": 0.99,
                            "target_date": "2024-01-03",
                            "target_is_valid": True,
                            "target_next_session_return": 0.50,
                            "target_next_session_direction": 1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "duplicate symbol/date rows"):
            split_m4_modeling_dataset(
                modeling,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )

    def test_split_rejects_empty_partitions_from_bad_boundaries(self) -> None:
        modeling = self._build_modeling_dataset()

        with self.assertRaisesRegex(ValueError, "training partition empty"):
            split_m4_modeling_dataset(
                modeling,
                split_definition=replace(
                    self.split_definition,
                    validation_start_date="2024-01-02",
                    validation_end_date="2024-01-03",
                ),
                target_definition=self.target_definition,
            )

        with self.assertRaisesRegex(ValueError, "validation partition empty"):
            split_m4_modeling_dataset(
                modeling,
                split_definition=replace(
                    self.split_definition,
                    validation_start_date="2024-02-01",
                    validation_end_date="2024-02-02",
                ),
                target_definition=self.target_definition,
            )

    def test_split_requires_forward_target_timestamps(self) -> None:
        modeling = self._build_modeling_dataset()
        modeling.loc[0, "target_date"] = modeling.loc[0, "date"]

        with self.assertRaisesRegex(ValueError, "target timestamps to stay strictly after feature timestamps"):
            split_m4_modeling_dataset(
                modeling,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )

    def test_run_split_writes_loadable_outputs_and_metadata(self) -> None:
        modeling = self._build_modeling_dataset()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "m4_modeling_dataset.parquet"
            train_output_path = tmp_path / "m4_train_dataset.parquet"
            validation_output_path = tmp_path / "m4_validation_dataset.parquet"
            metadata_path = tmp_path / "m4_train_validation_split.metadata.json"

            modeling.to_parquet(input_path, index=False)

            direct_train, direct_validation, _ = split_m4_modeling_dataset(
                modeling,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )
            result = run_m4_train_validation_split(
                input_path=input_path,
                train_output_path=train_output_path,
                validation_output_path=validation_output_path,
                metadata_path=metadata_path,
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )

            self.assertTrue(train_output_path.exists())
            self.assertTrue(validation_output_path.exists())
            self.assertTrue(metadata_path.exists())

            loaded_train = pd.read_parquet(train_output_path)
            loaded_validation = pd.read_parquet(validation_output_path)
            pd.testing.assert_frame_equal(loaded_train, direct_train)
            pd.testing.assert_frame_equal(loaded_validation, direct_validation)
            pd.testing.assert_frame_equal(result["train_dataframe"], direct_train)
            pd.testing.assert_frame_equal(result["validation_dataframe"], direct_validation)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["input_dataset"]["path"], str(input_path))
            self.assertEqual(metadata["train_dataset"]["path"], str(train_output_path))
            self.assertEqual(metadata["validation_dataset"]["path"], str(validation_output_path))
            self.assertEqual(metadata["counts"]["train_row_count"], len(direct_train))
            self.assertEqual(metadata["counts"]["validation_row_count"], len(direct_validation))
            self.assertEqual(metadata["split_rule"]["boundary_anchor_column"], "target_date")
            self.assertEqual(
                metadata["split_rule"]["validation_inclusion_rule"],
                "2024-01-04 <= target_date <= 2024-01-05",
            )
            self.assertEqual(metadata["excluded_rows"]["count"], 2)
            self.assertEqual(metadata["time_safety"]["target_horizon_sessions"], 1)

    def test_run_split_requires_existing_modeling_dataset(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "M4 modeling dataset not found"):
            run_m4_train_validation_split(
                input_path=Path("does-not-exist.parquet"),
                split_definition=self.split_definition,
                target_definition=self.target_definition,
            )


if __name__ == "__main__":
    unittest.main()
