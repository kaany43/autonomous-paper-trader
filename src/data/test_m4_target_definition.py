from __future__ import annotations

import unittest

import pandas as pd

from src.data.features import add_basic_features
from src.data.targets import add_m4_target_columns, load_m4_target_definition


class M4TargetDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = load_m4_target_definition()

    def test_official_target_config_matches_expected_contract(self) -> None:
        self.assertEqual(self.definition.milestone, "M4")
        self.assertEqual(self.definition.task_type, "classification")
        self.assertEqual(self.definition.official_target_column, "target_next_session_direction")
        self.assertEqual(self.definition.helper_return_column, "target_next_session_return")
        self.assertEqual(self.definition.forecast_horizon_sessions, 1)
        self.assertEqual(self.definition.price_column, "adj_close")
        self.assertEqual(self.definition.invalid_target_policy, "null_and_exclude_from_training")

    def test_target_generation_is_reproducible_and_normalizes_unsorted_input(self) -> None:
        market = pd.DataFrame(
            [
                {"date": "2024-01-03", "symbol": "BBB", "adj_close": 205.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 100.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 101.0},
                {"date": "2024-01-04", "symbol": "AAA", "adj_close": 99.0},
                {"date": "2024-01-01", "symbol": "BBB", "adj_close": 190.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 110.0},
                {"date": "2024-01-04", "symbol": "BBB", "adj_close": 200.0},
            ]
        )

        first = add_m4_target_columns(market, self.definition)
        second = add_m4_target_columns(market, self.definition)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            first[["symbol", "date"]].to_dict("records"),
            [
                {"symbol": "AAA", "date": pd.Timestamp("2024-01-02")},
                {"symbol": "AAA", "date": pd.Timestamp("2024-01-03")},
                {"symbol": "AAA", "date": pd.Timestamp("2024-01-04")},
                {"symbol": "BBB", "date": pd.Timestamp("2024-01-01")},
                {"symbol": "BBB", "date": pd.Timestamp("2024-01-03")},
                {"symbol": "BBB", "date": pd.Timestamp("2024-01-04")},
            ],
        )

        aaa_start = first.loc[
            (first["symbol"] == "AAA") & (first["date"] == pd.Timestamp("2024-01-02"))
        ].iloc[0]
        self.assertAlmostEqual(float(aaa_start[self.definition.helper_return_column]), (110.0 / 101.0) - 1.0)
        self.assertEqual(int(aaa_start[self.definition.official_target_column]), 1)

    def test_target_alignment_is_forward_looking_and_grouped_per_symbol(self) -> None:
        market = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 12.0},
                {"date": "2024-01-02", "symbol": "BBB", "adj_close": 50.0},
                {"date": "2024-01-03", "symbol": "BBB", "adj_close": 40.0},
            ]
        )

        result = add_m4_target_columns(market, self.definition)

        aaa_row = result.loc[
            (result["symbol"] == "AAA") & (result["date"] == pd.Timestamp("2024-01-01"))
        ].iloc[0]
        bbb_row = result.loc[
            (result["symbol"] == "BBB") & (result["date"] == pd.Timestamp("2024-01-02"))
        ].iloc[0]

        self.assertAlmostEqual(float(aaa_row[self.definition.helper_return_column]), 0.2)
        self.assertEqual(int(aaa_row[self.definition.official_target_column]), 1)
        self.assertAlmostEqual(float(bbb_row[self.definition.helper_return_column]), -0.2)
        self.assertEqual(int(bbb_row[self.definition.official_target_column]), 0)

        aaa_last = result.loc[
            (result["symbol"] == "AAA") & (result["date"] == pd.Timestamp("2024-01-03"))
        ].iloc[0]
        self.assertTrue(pd.isna(aaa_last[self.definition.helper_return_column]))
        self.assertTrue(pd.isna(aaa_last[self.definition.official_target_column]))

    def test_missing_or_invalid_future_prices_leave_targets_null(self) -> None:
        market = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "adj_close": 10.0},
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": pd.NA},
                {"date": "2024-01-01", "symbol": "BBB", "adj_close": 0.0},
                {"date": "2024-01-02", "symbol": "BBB", "adj_close": 5.0},
                {"date": "2024-01-01", "symbol": "CCC", "adj_close": 5.0},
                {"date": "2024-01-02", "symbol": "CCC", "adj_close": -1.0},
            ]
        )

        result = add_m4_target_columns(market, self.definition)

        invalid_rows = result.loc[result["symbol"].isin(["AAA", "BBB", "CCC"])]
        self.assertTrue(invalid_rows[self.definition.helper_return_column].isna().all())
        self.assertTrue(invalid_rows[self.definition.official_target_column].isna().all())

    def test_add_basic_features_includes_the_official_target_schema(self) -> None:
        market = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "symbol": "AAA",
                    "open": 9.8,
                    "high": 10.2,
                    "low": 9.7,
                    "close": 10.0,
                    "adj_close": 10.0,
                    "volume": 1000,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "open": 10.8,
                    "high": 11.2,
                    "low": 10.7,
                    "close": 11.0,
                    "adj_close": 11.0,
                    "volume": 1100,
                },
            ]
        )

        features = add_basic_features(market)

        self.assertIn(self.definition.helper_return_column, features.columns)
        self.assertIn(self.definition.official_target_column, features.columns)
        self.assertAlmostEqual(
            float(features.iloc[0][self.definition.helper_return_column]),
            0.1,
        )
        self.assertEqual(int(features.iloc[0][self.definition.official_target_column]), 1)
        self.assertTrue(pd.isna(features.iloc[1][self.definition.helper_return_column]))
        self.assertTrue(pd.isna(features.iloc[1][self.definition.official_target_column]))


if __name__ == "__main__":
    unittest.main()
