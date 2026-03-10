from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.strategy.momentum import MomentumStrategy
from src.strategy.base import BaseStrategy, StrategySignal


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DATA_DIR / "market_features.parquet"


class DailySimulator:
    """
    Daily backtest simulator.

    Rules:
    - Processes one trading day at a time in chronological order
    - For each decision date, only rows up to that date are visible
    - Target/label columns are removed before passing data to strategy
    - Orders are executed using the same day's adj_close price
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        broker: Broker,
        price_column: str = "adj_close",
    ) -> None:
        self.market_data = self._prepare_market_data(market_data, price_column)
        self.strategy = strategy
        self.portfolio = portfolio
        self.broker = broker
        self.price_column = price_column

        self._portfolio_history: list[dict[str, Any]] = []
        self._positions_history: list[dict[str, Any]] = []
        self._trade_history: list[dict[str, Any]] = []
        self._signal_history: list[dict[str, Any]] = []

    @staticmethod
    def _prepare_market_data(
        market_data: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        if market_data.empty:
            raise ValueError("market_data is empty.")

        required_columns = {"date", "symbol", price_column}
        missing = required_columns - set(market_data.columns)
        if missing:
            raise ValueError(
                f"market_data is missing required columns: {', '.join(sorted(missing))}"
            )

        df = market_data.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

        df = (
            df.sort_values(["date", "symbol"])
            .drop_duplicates(subset=["date", "symbol"], keep="last")
            .reset_index(drop=True)
        )

        return df

    @staticmethod
    def _drop_label_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove forward-looking target columns before sending data to strategy.
        """
        label_columns = [col for col in df.columns if col.startswith("target_")]
        if not label_columns:
            return df
        return df.drop(columns=label_columns)

    def _all_trading_dates(self) -> list[pd.Timestamp]:
        return [pd.Timestamp(d) for d in sorted(self.market_data["date"].dropna().unique())]

    def _selected_trading_dates(
        self,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> list[pd.Timestamp]:
        all_dates = self._all_trading_dates()

        if not all_dates:
            return []

        start_ts = pd.Timestamp(start_date) if start_date is not None else all_dates[0]
        end_ts = pd.Timestamp(end_date) if end_date is not None else all_dates[-1]

        return [d for d in all_dates if start_ts <= d <= end_ts]

    def _history_until(self, decision_date: pd.Timestamp) -> pd.DataFrame:
        history = self.market_data.loc[self.market_data["date"] <= decision_date].copy()
        history = self._drop_label_columns(history)
        return history

    def _day_prices(self, trading_date: pd.Timestamp) -> dict[str, float]:
        day_df = self.market_data.loc[self.market_data["date"] == trading_date].copy()

        if day_df.empty:
            return {}

        return {
            str(row["symbol"]): float(row[self.price_column])
            for _, row in day_df[["symbol", self.price_column]].dropna().iterrows()
        }

    @staticmethod
    def _signal_to_row(signal: StrategySignal) -> dict[str, Any]:
        row = asdict(signal)
        return row

    def _record_signals(self, signals: list[StrategySignal]) -> None:
        for signal in signals:
            self._signal_history.append(self._signal_to_row(signal))

    def _record_daily_snapshots(
        self,
        trading_date: pd.Timestamp,
        price_map: dict[str, float],
    ) -> None:
        self._portfolio_history.append(
            self.portfolio.portfolio_snapshot(
                date=trading_date,
                price_map=price_map,
            )
        )
        self._positions_history.extend(
            self.portfolio.positions_snapshot(
                date=trading_date,
                price_map=price_map,
            )
        )

    @staticmethod
    def _sort_signals(signals: list[StrategySignal]) -> list[StrategySignal]:
        action_priority = {"SELL": 0, "BUY": 1, "HOLD": 2}
        return sorted(
            signals,
            key=lambda s: (
                action_priority.get(str(s.action).upper(), 99),
                -float(s.score),
                str(s.symbol),
            ),
        )

    def _execute_sell_signals(
        self,
        signals: list[StrategySignal],
        trading_date: pd.Timestamp,
        price_map: dict[str, float],
    ) -> None:
        for signal in signals:
            if str(signal.action).upper() != "SELL":
                continue

            symbol = str(signal.symbol).upper()

            if not self.portfolio.has_position(symbol):
                continue

            market_price = price_map.get(symbol)
            if market_price is None:
                continue

            position = self.portfolio.get_position(symbol)
            if position is None or position.quantity <= 0:
                continue

            result = self.broker.sell(
                portfolio=self.portfolio,
                symbol=symbol,
                quantity=float(position.quantity),
                market_price=float(market_price),
            )

            row = result.to_dict()
            row["date"] = trading_date
            self._trade_history.append(row)

    def _execute_buy_signals(
        self,
        signals: list[StrategySignal],
        trading_date: pd.Timestamp,
        price_map: dict[str, float],
    ) -> None:
        buy_signals = [s for s in signals if str(s.action).upper() == "BUY"]
        if not buy_signals:
            return

        equity_after_sells = self.portfolio.total_equity(price_map)

        for idx, signal in enumerate(buy_signals):
            symbol = str(signal.symbol).upper()

            if self.portfolio.has_position(symbol):
                continue

            market_price = price_map.get(symbol)
            if market_price is None:
                continue

            if signal.target_weight is not None and float(signal.target_weight) > 0:
                cash_to_allocate = min(
                    float(self.portfolio.cash),
                    float(signal.target_weight) * float(equity_after_sells),
                )
            else:
                remaining = len(buy_signals) - idx
                if remaining <= 0:
                    break
                cash_to_allocate = float(self.portfolio.cash) / float(remaining)

            if cash_to_allocate <= 0:
                continue

            result = self.broker.buy_with_cash_amount(
                portfolio=self.portfolio,
                symbol=symbol,
                cash_to_allocate=float(cash_to_allocate),
                market_price=float(market_price),
            )

            row = result.to_dict()
            row["date"] = trading_date
            self._trade_history.append(row)

    def _process_day(self, trading_date: pd.Timestamp) -> None:
        historical_data = self._history_until(trading_date)

        signals = self.strategy.generate_signals(
            decision_date=trading_date,
            market_data=historical_data,
            portfolio=self.portfolio,
        )
        signals = self._sort_signals(signals)

        self._record_signals(signals)

        price_map = self._day_prices(trading_date)

        self._execute_sell_signals(
            signals=signals,
            trading_date=trading_date,
            price_map=price_map,
        )
        self._execute_buy_signals(
            signals=signals,
            trading_date=trading_date,
            price_map=price_map,
        )

        self._record_daily_snapshots(
            trading_date=trading_date,
            price_map=price_map,
        )

    def run(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        self._portfolio_history.clear()
        self._positions_history.clear()
        self._trade_history.clear()
        self._signal_history.clear()

        trading_dates = self._selected_trading_dates(
            start_date=start_date,
            end_date=end_date,
        )

        if not trading_dates:
            raise ValueError("No trading dates found for the requested range.")

        for trading_date in trading_dates:
            self._process_day(trading_date)

        portfolio_history = pd.DataFrame(self._portfolio_history)
        positions_history = pd.DataFrame(self._positions_history)
        trade_history = pd.DataFrame(self._trade_history)
        signal_history = pd.DataFrame(self._signal_history)

        if not portfolio_history.empty:
            portfolio_history = portfolio_history.sort_values("date").reset_index(drop=True)

        if not positions_history.empty:
            positions_history = positions_history.sort_values(
                ["date", "symbol"]
            ).reset_index(drop=True)

        if not trade_history.empty:
            trade_history = trade_history.sort_values(
                ["date", "symbol", "side"]
            ).reset_index(drop=True)

        if not signal_history.empty:
            signal_history = signal_history.sort_values(
                ["date", "symbol", "action"]
            ).reset_index(drop=True)

        return {
            "portfolio_history": portfolio_history,
            "positions_history": positions_history,
            "trade_history": trade_history,
            "signal_history": signal_history,
        }


def load_feature_data() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_FILE}\n"
            "Run `python -m src.data.features` first."
        )

    df = pd.read_parquet(FEATURES_FILE)
    if df.empty:
        raise ValueError("Loaded feature dataset is empty.")

    return df


def main() -> None:
    print("Loading processed market features...")
    market_data = load_feature_data()

    strategy = MomentumStrategy(
        max_open_positions=2,
        top_k=2,
        min_score=0.0,
    )
    portfolio = Portfolio(initial_cash=10_000.0)
    broker = Broker(
        commission_rate=0.001,
        slippage_rate=0.001,
        fractional_shares=True,
    )

    simulator = DailySimulator(
        market_data=market_data,
        strategy=strategy,
        portfolio=portfolio,
        broker=broker,
        price_column="adj_close",
    )

    start_date = market_data["date"].min()
    end_date = market_data["date"].max()

    print(f"Running backtest from {pd.Timestamp(start_date).date()} to {pd.Timestamp(end_date).date()} ...")
    results = simulator.run(start_date=start_date, end_date=end_date)

    portfolio_history = results["portfolio_history"]
    trade_history = results["trade_history"]
    signal_history = results["signal_history"]

    print("-" * 60)
    print("Backtest complete.")
    print(f"Days processed: {len(portfolio_history)}")
    print(f"Trades executed: {len(trade_history)}")
    print(f"Signals generated: {len(signal_history)}")

    if not portfolio_history.empty:
        last_row = portfolio_history.iloc[-1]
        print(f"Final cash: {last_row['cash']:.2f}")
        print(f"Final total equity: {last_row['total_equity']:.2f}")
        print(f"Realized PnL: {last_row['realized_pnl']:.2f}")
        print(f"Unrealized PnL: {last_row['unrealized_pnl']:.2f}")

    print("-" * 60)
    print("Portfolio history preview:")
    print(portfolio_history.head(5))

    print("-" * 60)
    print("Trade history preview:")
    print(trade_history.head(10))

    print("-" * 60)
    print("Signal history preview:")
    print(signal_history.head(10))


if __name__ == "__main__":
    main()