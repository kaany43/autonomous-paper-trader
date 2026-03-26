from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pandas as pd

from src.data.loader import load_market_data


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "market_features.parquet"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate leakage-safe technical features from daily OHLCV data.
    Features are computed per symbol using only historical rows.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = (
        # Keep the last ingested correction row for each symbol/date before ordering rows.
        df.drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    grouped = df.groupby("symbol", group_keys=False)

    # Price returns
    df["ret_1d"] = grouped["adj_close"].pct_change(1)
    df["ret_5d"] = grouped["adj_close"].pct_change(5)
    df["ret_10d"] = grouped["adj_close"].pct_change(10)

    # Moving averages
    df["ma_10"] = grouped["adj_close"].transform(
        lambda s: s.rolling(window=10, min_periods=10).mean()
    )
    df["ma_20"] = grouped["adj_close"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    df["ma_50"] = grouped["adj_close"].transform(
        lambda s: s.rolling(window=50, min_periods=50).mean()
    )

    # Rolling volatility from daily returns
    df["vol_20"] = grouped["ret_1d"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std()
    )

    # Volume features
    df["volume_change_1d"] = grouped["volume"].pct_change(1)
    df["volume_ma_20"] = grouped["volume"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    df["volume_ratio_20"] = df["volume"] / df["volume_ma_20"]

    # Trend features
    df["price_vs_ma10"] = df["adj_close"] / df["ma_10"] - 1.0
    df["price_vs_ma20"] = df["adj_close"] / df["ma_20"] - 1.0
    df["price_vs_ma50"] = df["adj_close"] / df["ma_50"] - 1.0
    df["ma10_vs_ma20"] = df["ma_10"] / df["ma_20"] - 1.0
    df["ma20_vs_ma50"] = df["ma_20"] / df["ma_50"] - 1.0

    # Rolling highs/lows
    df["rolling_high_20"] = grouped["high"].transform(
        lambda s: s.rolling(window=20, min_periods=20).max()
    )
    df["rolling_low_20"] = grouped["low"].transform(
        lambda s: s.rolling(window=20, min_periods=20).min()
    )

    # Position of current price in 20-day range
    range_width = df["rolling_high_20"] - df["rolling_low_20"]
    df["range_pos_20"] = (df["adj_close"] - df["rolling_low_20"]) / range_width
    df.loc[range_width == 0, "range_pos_20"] = pd.NA

    return df


def save_features(df: pd.DataFrame) -> Path:
    """Save processed feature dataframe to parquet."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    return OUTPUT_FILE


def main() -> None:
    print("Loading raw market data...")
    df = load_market_data()

    print("Generating technical features...")
    features_df = add_basic_features(df)

    output_path = save_features(features_df)

    print("-" * 60)
    print(f"Saved feature dataset to: {output_path}")
    print(f"Rows: {len(features_df)}")
    print(f"Symbols: {features_df['symbol'].nunique()}")
    print(f"Date range: {features_df['date'].min().date()} -> {features_df['date'].max().date()}")

    preview_columns = [
        "date",
        "symbol",
        "adj_close",
        "ret_1d",
        "ret_5d",
        "ma_20",
        "ma_50",
        "vol_20",
        "volume_ratio_20",
    ]
    print(features_df[preview_columns].head(10))


if __name__ == "__main__":
    main()
