from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.modeling_dataset import (
    get_m4_modeling_dataset_column_order,
    load_m4_modeling_dataset_definition,
)
from src.data.target_pipeline import prepare_m4_modeling_dataset, run_m4_target_preparation
from src.data.targets import (
    TARGET_DATE_COLUMN,
    TARGET_VALID_COLUMN,
    load_m4_target_definition,
)


def _feature_row(
    *,
    date: str,
    symbol: str,
    adj_close: float | object,
    ret_1d: float | object,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "date": date,
        "symbol": symbol,
        "open": adj_close,
        "high": adj_close,
        "low": adj_close,
        "close": adj_close,
        "adj_close": adj_close,
        "volume": 1000,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "ret_1d": ret_1d,
        "ret_5d": 0.0,
        "ret_10d": 0.0,
        "ma_10": 1.0,
        "ma_20": 1.0,
        "ma_50": 1.0,
        "vol_20": 0.1,
        "volume_change_1d": 0.0,
        "volume_ma_20": 1000.0,
        "volume_ratio_20": 1.0,
        "price_vs_ma10": 0.0,
        "price_vs_ma20": 0.0,
        "price_vs_ma50": 0.0,
        "ma10_vs_ma20": 0.0,
        "ma20_vs_ma50": 0.0,
        "rolling_high_20": float(adj_close) if adj_close is not pd.NA else pd.NA,
        "rolling_low_20": float(adj_close) if adj_close is not pd.NA else pd.NA,
        "range_pos_20": 0.5,
    }
    if extra:
        row.update(extra)
    return row


class M4TargetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = load_m4_target_definition()
        self.dataset_definition = load_m4_modeling_dataset_definition()

    def test_prepare_modeling_dataset_is_deterministic_and_strips_legacy_target_columns(self) -> None:
        features = pd.DataFrame(
            [
                _feature_row(
                    date="2024-01-03",
                    symbol="BBB",
                    adj_close=205.0,
                    ret_1d=0.0789,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 1},
                ),
                _feature_row(
                    date="2024-01-02",
                    symbol="AAA",
                    adj_close=100.0,
                    ret_1d=0.0,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 1},
                ),
                _feature_row(
                    date="2024-01-02",
                    symbol="AAA",
                    adj_close=101.0,
                    ret_1d=0.01,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 1},
                ),
                _feature_row(
                    date="2024-01-04",
                    symbol="AAA",
                    adj_close=99.0,
                    ret_1d=-0.1,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 0},
                ),
                _feature_row(
                    date="2024-01-01",
                    symbol="BBB",
                    adj_close=190.0,
                    ret_1d=0.0,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 1},
                ),
                _feature_row(
                    date="2024-01-03",
                    symbol="AAA",
                    adj_close=110.0,
                    ret_1d=0.0891,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 1},
                ),
                _feature_row(
                    date="2024-01-04",
                    symbol="BBB",
                    adj_close=200.0,
                    ret_1d=-0.0244,
                    extra={"target_ret_1d": 999.0, "target_up_1d": 0},
                ),
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
        self.assertEqual(
            list(first.columns),
            get_m4_modeling_dataset_column_order(self.dataset_definition, self.definition),
        )

    def test_prepare_modeling_dataset_drops_invalid_rows_consistently(self) -> None:
        features = pd.DataFrame(
            [
                _feature_row(date="2024-01-01", symbol="AAA", adj_close=10.0, ret_1d=0.0),
                _feature_row(date="2024-01-02", symbol="AAA", adj_close=pd.NA, ret_1d=pd.NA),
                _feature_row(date="2024-01-01", symbol="BBB", adj_close=5.0, ret_1d=0.0),
                _feature_row(date="2024-01-02", symbol="BBB", adj_close=6.0, ret_1d=0.2),
                _feature_row(date="2024-01-03", symbol="BBB", adj_close=7.0, ret_1d=0.1667),
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

    def test_prepare_modeling_dataset_rejects_schema_drift_after_target_generation(self) -> None:
        features = pd.DataFrame(
            [
                _feature_row(date="2024-01-01", symbol="AAA", adj_close=10.0, ret_1d=0.0),
                _feature_row(date="2024-01-02", symbol="AAA", adj_close=11.0, ret_1d=0.1),
            ]
        ).drop(columns=["ma_50"])

        with self.assertRaisesRegex(ValueError, "missing required columns: ma_50"):
            prepare_m4_modeling_dataset(features, self.definition)

    def test_prepare_modeling_dataset_rejects_invalid_dates(self) -> None:
        features = pd.DataFrame(
            [
                _feature_row(date="not-a-date", symbol="AAA", adj_close=10.0, ret_1d=0.0),
                _feature_row(date="2024-01-02", symbol="AAA", adj_close=11.0, ret_1d=0.1),
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires valid non-null date values"):
            prepare_m4_modeling_dataset(features, self.definition)

    def test_run_target_preparation_writes_loadable_outputs(self) -> None:
        features = pd.DataFrame(
            [
                _feature_row(date="2024-01-01", symbol="AAA", adj_close=10.0, ret_1d=0.0),
                _feature_row(date="2024-01-02", symbol="AAA", adj_close=11.0, ret_1d=0.1),
                _feature_row(date="2024-01-03", symbol="AAA", adj_close=10.5, ret_1d=-0.0455),
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
            self.assertEqual(metadata["dataset_definition"]["contract_name"], "m4_official_modeling_dataset")
            self.assertEqual(metadata["schema"]["identifier_columns"], ["symbol"])
            self.assertEqual(metadata["schema"]["feature_timestamp_columns"], ["date"])
            self.assertEqual(metadata["schema"]["target_timestamp_columns"], ["target_date"])
            self.assertEqual(
                metadata["schema"]["target_label_columns"],
                [
                    self.definition.helper_return_column,
                    self.definition.official_target_column,
                ],
            )
            self.assertEqual(metadata["split_ready_metadata"]["boundary_column"], "target_date")
            self.assertEqual(
                metadata["invalid_label_handling"]["dataset_policy"],
                "drop_rows_where_target_is_valid_is_false",
            )


if __name__ == "__main__":
    unittest.main()
