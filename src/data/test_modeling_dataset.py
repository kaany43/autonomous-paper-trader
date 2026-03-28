from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.modeling_dataset import (
    build_m4_modeling_dataset_schema,
    load_m4_modeling_dataset_bundle,
    load_m4_modeling_dataset_definition,
    normalize_m4_modeling_dataset,
    validate_m4_modeling_dataset_contract,
)
from src.data.target_pipeline import prepare_m4_modeling_dataset, run_m4_target_preparation
from src.data.targets import load_m4_target_definition


class M4ModelingDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_definition = load_m4_modeling_dataset_definition()
        self.target_definition = load_m4_target_definition()

    @staticmethod
    def _build_features() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2024-01-04",
                    "symbol": "AAA",
                    "open": 11.0,
                    "high": 12.0,
                    "low": 10.5,
                    "close": 11.5,
                    "adj_close": 11.5,
                    "volume": 1200,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "ret_1d": 0.10,
                    "ret_5d": 0.20,
                    "ret_10d": 0.30,
                    "ma_10": 10.0,
                    "ma_20": 9.5,
                    "ma_50": 9.0,
                    "vol_20": 0.15,
                    "volume_change_1d": 0.10,
                    "volume_ma_20": 1000.0,
                    "volume_ratio_20": 1.20,
                    "price_vs_ma10": 0.15,
                    "price_vs_ma20": 0.21,
                    "price_vs_ma50": 0.28,
                    "ma10_vs_ma20": 0.05,
                    "ma20_vs_ma50": 0.06,
                    "rolling_high_20": 12.0,
                    "rolling_low_20": 8.0,
                    "range_pos_20": 0.875,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "open": 9.5,
                    "high": 10.5,
                    "low": 9.0,
                    "close": 10.0,
                    "adj_close": 10.0,
                    "volume": 1000,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "ret_1d": 0.00,
                    "ret_5d": 0.10,
                    "ret_10d": 0.20,
                    "ma_10": 9.0,
                    "ma_20": 8.5,
                    "ma_50": 8.0,
                    "vol_20": 0.10,
                    "volume_change_1d": 0.00,
                    "volume_ma_20": 950.0,
                    "volume_ratio_20": 1.05,
                    "price_vs_ma10": 0.11,
                    "price_vs_ma20": 0.18,
                    "price_vs_ma50": 0.25,
                    "ma10_vs_ma20": 0.06,
                    "ma20_vs_ma50": 0.06,
                    "rolling_high_20": 10.5,
                    "rolling_low_20": 7.5,
                    "range_pos_20": 0.8333,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "AAA",
                    "open": 10.0,
                    "high": 11.5,
                    "low": 9.8,
                    "close": 11.0,
                    "adj_close": 11.0,
                    "volume": 1100,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "ret_1d": 0.10,
                    "ret_5d": 0.15,
                    "ret_10d": 0.25,
                    "ma_10": 9.5,
                    "ma_20": 9.0,
                    "ma_50": 8.5,
                    "vol_20": 0.12,
                    "volume_change_1d": 0.10,
                    "volume_ma_20": 975.0,
                    "volume_ratio_20": 1.1282,
                    "price_vs_ma10": 0.1579,
                    "price_vs_ma20": 0.2222,
                    "price_vs_ma50": 0.2941,
                    "ma10_vs_ma20": 0.0556,
                    "ma20_vs_ma50": 0.0588,
                    "rolling_high_20": 11.5,
                    "rolling_low_20": 7.8,
                    "range_pos_20": 0.8649,
                },
            ]
        )

    def test_modeling_dataset_schema_groups_are_explicit_and_deterministic(self) -> None:
        modeling_df, _ = prepare_m4_modeling_dataset(
            self._build_features(),
            definition=self.target_definition,
            dataset_definition=self.dataset_definition,
        )

        schema = build_m4_modeling_dataset_schema(
            dataset_definition=self.dataset_definition,
            target_definition=self.target_definition,
        )

        self.assertEqual(schema["identifier_columns"], ["symbol"])
        self.assertEqual(schema["feature_timestamp_columns"], ["date"])
        self.assertEqual(schema["target_timestamp_columns"], ["target_date"])
        self.assertEqual(schema["feature_columns"], list(modeling_df.columns[2:-4]))
        self.assertEqual(
            schema["target_label_columns"],
            [
                self.target_definition.helper_return_column,
                self.target_definition.official_target_column,
            ],
        )
        self.assertEqual(schema["inference_key_columns"], ["symbol", "date", "target_date"])
        self.assertEqual(list(modeling_df.columns), schema["column_order"])

    def test_run_target_preparation_writes_reloadable_bundle_with_schema_metadata(self) -> None:
        features = self._build_features()

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
                definition=self.target_definition,
                dataset_definition=self.dataset_definition,
            )

            bundle = load_m4_modeling_dataset_bundle(
                dataset_path=output_path,
                metadata_path=metadata_path,
            )

            pd.testing.assert_frame_equal(bundle["dataframe"], result["dataframe"])
            self.assertEqual(bundle["schema"]["identifier_columns"], ["symbol"])
            self.assertEqual(bundle["schema"]["official_split_boundary_column"], "target_date")
            self.assertEqual(
                bundle["metadata"]["split_ready_metadata"]["method"],
                "chronological_holdout_by_target_date_window",
            )
            self.assertEqual(
                bundle["metadata"]["inference_ready_metadata"]["inference_key_columns"],
                ["symbol", "date", "target_date"],
            )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_definition"]["contract_name"], "m4_official_modeling_dataset")
            self.assertEqual(metadata["counts"]["feature_column_count"], len(bundle["schema"]["feature_columns"]))

    def test_normalize_and_validate_modeling_dataset_reject_schema_drift(self) -> None:
        modeling_df, _ = prepare_m4_modeling_dataset(
            self._build_features(),
            definition=self.target_definition,
            dataset_definition=self.dataset_definition,
        )

        missing_feature = modeling_df.drop(columns=["ret_1d"])
        with self.assertRaisesRegex(ValueError, "missing required columns: ret_1d"):
            normalize_m4_modeling_dataset(
                missing_feature,
                dataset_definition=self.dataset_definition,
                target_definition=self.target_definition,
            )

        unexpected_feature = modeling_df.copy()
        unexpected_feature["future_magic_feature"] = 1.23
        with self.assertRaisesRegex(ValueError, "unexpected columns outside the official schema"):
            normalize_m4_modeling_dataset(
                unexpected_feature,
                dataset_definition=self.dataset_definition,
                target_definition=self.target_definition,
            )

        null_identifier = modeling_df.copy()
        null_identifier.loc[0, "symbol"] = pd.NA
        with self.assertRaisesRegex(ValueError, "valid non-null identifier values in 'symbol'"):
            normalize_m4_modeling_dataset(
                null_identifier,
                dataset_definition=self.dataset_definition,
                target_definition=self.target_definition,
            )

        bad_order = modeling_df.copy()[list(reversed(modeling_df.columns))]
        with self.assertRaisesRegex(ValueError, "official schema order"):
            validate_m4_modeling_dataset_contract(
                bad_order,
                dataset_definition=self.dataset_definition,
                target_definition=self.target_definition,
            )


if __name__ == "__main__":
    unittest.main()
