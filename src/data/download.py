from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its content as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in: {path}")

    return data


def load_settings() -> dict[str, Any]:
    """Load project settings from config/settings.yaml."""
    return load_yaml(CONFIG_DIR / "settings.yaml")


def load_universe() -> list[str]:
    """Load stock symbols from config/universe.yaml."""
    data = load_yaml(CONFIG_DIR / "universe.yaml")

    symbols = data.get("universe", {}).get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("No symbols found in config/universe.yaml")

    clean_symbols = []
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        clean_symbols.append(symbol.strip().upper())

    if not clean_symbols:
        raise ValueError("Universe symbol list is empty after cleaning.")

    return clean_symbols


def get_all_symbols(settings: dict[str, Any], universe_symbols: list[str]) -> list[str]:
    """Return universe symbols plus benchmark if it is not already included."""
    benchmark_symbol = (
        settings.get("benchmark", {}).get("symbol", "") or ""
    ).strip().upper()

    symbols = list(universe_symbols)
    if benchmark_symbol and benchmark_symbol not in symbols:
        symbols.append(benchmark_symbol)

    return symbols


def standardize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standardize raw yfinance output into a clean daily OHLCV dataframe.
    Expected output columns:
    date, symbol, open, high, low, close, adj_close, volume, dividends, stock_splits
    """
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Normalize column names from yfinance
    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    df = df.rename(columns=rename_map)

    # Some downloads may not contain actions columns depending on source/asset
    if "dividends" not in df.columns:
        df["dividends"] = 0.0
    if "stock_splits" not in df.columns:
        df["stock_splits"] = 0.0
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    ordered_columns = [
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

    for col in ordered_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ordered_columns].sort_values("date").reset_index(drop=True)

    return df


def download_symbol_data(
    symbol: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    auto_adjust: bool,
    actions: bool,
) -> pd.DataFrame:
    """Download historical data for a single symbol using yfinance."""
    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        progress=False,
        threads=False,
    )

    return standardize_ohlcv(df, symbol)


def save_symbol_data(df: pd.DataFrame, symbol: str) -> Path:
    """Save a symbol dataframe to data/raw/<SYMBOL>.parquet."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"{symbol}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    settings = load_settings()
    universe_symbols = load_universe()
    all_symbols = get_all_symbols(settings, universe_symbols)

    data_settings = settings.get("data", {})
    start_date = data_settings.get("start_date")
    end_date = data_settings.get("end_date")
    interval = data_settings.get("interval", "1d")
    auto_adjust = bool(data_settings.get("auto_adjust", False))
    actions = bool(data_settings.get("actions", True))

    if not start_date:
        raise ValueError("data.start_date is required in config/settings.yaml")

    print("Downloading daily market data...")
    print(f"Symbols: {', '.join(all_symbols)}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date or 'latest available'}")
    print(f"Interval: {interval}")
    print("-" * 60)

    success_count = 0
    failed_symbols: list[str] = []

    for symbol in all_symbols:
        try:
            df = download_symbol_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            )

            if df.empty:
                print(f"[WARN] No data returned for {symbol}")
                failed_symbols.append(symbol)
                continue

            output_path = save_symbol_data(df, symbol)
            print(f"[OK] {symbol}: {len(df)} rows saved to {output_path}")
            success_count += 1

        except Exception as exc:
            print(f"[ERROR] Failed to download {symbol}: {exc}")
            failed_symbols.append(symbol)

    print("-" * 60)
    print(f"Completed. Success: {success_count} | Failed: {len(failed_symbols)}")

    if failed_symbols:
        print("Failed symbols:", ", ".join(failed_symbols))


if __name__ == "__main__":
    main()