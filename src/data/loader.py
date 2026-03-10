from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return it as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in: {path}")

    return data


def load_settings() -> dict[str, Any]:
    """Load settings.yaml."""
    return load_yaml(CONFIG_DIR / "settings.yaml")


def load_universe() -> list[str]:
    """Load symbol list from universe.yaml."""
    data = load_yaml(CONFIG_DIR / "universe.yaml")
    symbols = data.get("universe", {}).get("symbols", [])

    if not isinstance(symbols, list) or not symbols:
        raise ValueError("No symbols found in config/universe.yaml")

    cleaned = []
    for symbol in symbols:
        if isinstance(symbol, str) and symbol.strip():
            cleaned.append(symbol.strip().upper())

    if not cleaned:
        raise ValueError("Universe symbol list is empty after cleaning.")

    return cleaned


def get_target_symbols() -> list[str]:
    """Return universe symbols plus benchmark symbol if needed."""
    settings = load_settings()
    universe_symbols = load_universe()

    benchmark_symbol = (
        settings.get("benchmark", {}).get("symbol", "") or ""
    ).strip().upper()

    symbols = list(universe_symbols)
    if benchmark_symbol and benchmark_symbol not in symbols:
        symbols.append(benchmark_symbol)

    return symbols


def validate_schema(df: pd.DataFrame, symbol: str) -> None:
    """Validate required schema for a single symbol dataframe."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for {symbol}: {', '.join(missing_columns)}"
        )


def read_symbol_parquet(symbol: str) -> pd.DataFrame:
    """Read one symbol parquet file from data/raw/."""
    file_path = RAW_DATA_DIR / f"{symbol}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing parquet file for {symbol}: {file_path}")

    df = pd.read_parquet(file_path)
    validate_schema(df, symbol)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[REQUIRED_COLUMNS]
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    return df


def load_market_data(symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Load market data for all requested symbols and return one combined dataframe.
    If symbols is None, read symbols from config.
    """
    target_symbols = symbols or get_target_symbols()

    if not target_symbols:
        raise ValueError("No target symbols provided for loading.")

    frames: list[pd.DataFrame] = []
    failed_symbols: list[str] = []

    for symbol in target_symbols:
        try:
            df = read_symbol_parquet(symbol)
            if df.empty:
                failed_symbols.append(symbol)
                continue
            frames.append(df)
        except Exception as exc:
            print(f"[ERROR] Failed to load {symbol}: {exc}")
            failed_symbols.append(symbol)

    if not frames:
        raise ValueError("No parquet files could be loaded successfully.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )

    print("-" * 60)
    print(f"Loaded symbols: {combined['symbol'].nunique()}")
    print(f"Rows: {len(combined)}")
    print(f"Date range: {combined['date'].min().date()} -> {combined['date'].max().date()}")

    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")

    return combined


def main() -> None:
    df = load_market_data()
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()