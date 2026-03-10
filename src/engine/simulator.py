from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine.broker import Broker, ExecutionResult
from src.engine.portfolio import Portfolio
from src.strategy.base import BaseStrategy, StrategySignal


@dataclass
class SimulationResult:
    trades: pd.DataFrame
    portfolio_history: pd.DataFrame
    positions_history: pd.DataFrame
    metrics: dict[str, Any]


class DailySimulator:
    def __init__(
        self,
        strategy: BaseStrategy,
        broker: Broker,
        portfolio: Portfolio,
        max_position_weight: float,
    ) -> None:
        if max_position_weight <= 0 or max_position_weight > 1:
            raise ValueError("max_position_weight must be in (0, 1].")

        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.max_position_weight = float(max_position_weight)

    @staticmethod
    def _build_open_price_map(day_df: pd.DataFrame) -> dict[str, float]:
        rows = day_df.dropna(subset=["open"]) [["symbol", "open"]]
        return {str(row["symbol"]): float(row["open"]) for _, row in rows.iterrows()}

    @staticmethod
    def _build_close_price_map(day_df: pd.DataFrame) -> dict[str, float]:
        rows = day_df.dropna(subset=["close"]) [["symbol", "close"]]
        return {str(row["symbol"]): float(row["close"]) for _, row in rows.iterrows()}

    def _sell_all_quantity(self, signal: StrategySignal, open_prices: dict[str, float]) -> ExecutionResult | None:
        position = self.portfolio.get_position(signal.symbol)
        if position is None:
            return None

        market_price = open_prices.get(signal.symbol)
        if market_price is None or market_price <= 0:
            return None

        return self.broker.sell(
            portfolio=self.portfolio,
            symbol=signal.symbol,
            quantity=position.quantity,
            market_price=market_price,
        )

    def _buy_with_risk_limit(self, signal: StrategySignal, open_prices: dict[str, float]) -> ExecutionResult | None:
        market_price = open_prices.get(signal.symbol)
        if market_price is None or market_price <= 0:
            return None

        max_alloc = self.portfolio.total_equity(open_prices) * self.max_position_weight
        target_alloc = max_alloc

        if signal.target_weight is not None:
            target_alloc = self.portfolio.total_equity(open_prices) * float(signal.target_weight)

        cash_to_allocate = min(self.portfolio.cash, max_alloc, target_alloc)
        if cash_to_allocate <= 0:
            return None

        return self.broker.buy_with_cash_amount(
            portfolio=self.portfolio,
            symbol=signal.symbol,
            cash_to_allocate=cash_to_allocate,
            market_price=market_price,
        )

    def run(self, market_data: pd.DataFrame) -> SimulationResult:
        if market_data.empty:
            raise ValueError("market_data is empty.")

        required_columns = {"date", "symbol", "open", "close"}
        missing = sorted(required_columns - set(market_data.columns))
        if missing:
            raise ValueError(f"market_data missing required columns: {', '.join(missing)}")

        df = market_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

        trades_rows: list[dict[str, Any]] = []
        portfolio_rows: list[dict[str, Any]] = []
        positions_rows: list[dict[str, Any]] = []

        for decision_date in sorted(df["date"].dropna().unique()):
            day_df = df.loc[df["date"] == decision_date].copy()
            if day_df.empty:
                continue

            open_prices = self._build_open_price_map(day_df)
            close_prices = self._build_close_price_map(day_df)

            signals = self.strategy.generate_signals(
                decision_date=decision_date,
                market_data=df,
                portfolio=self.portfolio,
            )

            for signal in [s for s in signals if s.action == "SELL"]:
                result = self._sell_all_quantity(signal, open_prices)
                if result is not None:
                    record = result.to_dict()
                    record["date"] = decision_date
                    record["reason_code"] = signal.reason_code
                    trades_rows.append(record)

            for signal in [s for s in signals if s.action == "BUY"]:
                if self.portfolio.has_position(signal.symbol):
                    continue

                result = self._buy_with_risk_limit(signal, open_prices)
                if result is not None:
                    record = result.to_dict()
                    record["date"] = decision_date
                    record["reason_code"] = signal.reason_code
                    trades_rows.append(record)

            portfolio_rows.append(self.portfolio.portfolio_snapshot(decision_date, close_prices))
            positions_rows.extend(self.portfolio.positions_snapshot(decision_date, close_prices))

        trades_df = pd.DataFrame(trades_rows)
        portfolio_history_df = pd.DataFrame(portfolio_rows)
        positions_history_df = pd.DataFrame(positions_rows)

        metrics = self._build_metrics(portfolio_history_df, trades_df)

        return SimulationResult(
            trades=trades_df,
            portfolio_history=portfolio_history_df,
            positions_history=positions_history_df,
            metrics=metrics,
        )

    def _build_metrics(self, portfolio_history: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
        if portfolio_history.empty:
            return {
                "initial_equity": float(self.portfolio.initial_cash),
                "final_equity": float(self.portfolio.initial_cash),
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "trade_count": int(len(trades)),
                "win_rate": None,
            }

        equity = portfolio_history["total_equity"].astype(float)
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0

        initial_equity = float(equity.iloc[0])
        final_equity = float(equity.iloc[-1])
        total_return = float(final_equity / initial_equity - 1.0) if initial_equity > 0 else 0.0

        sell_trades = trades.loc[(trades.get("side") == "SELL") & (trades.get("success") == True)].copy()
        if "realized_pnl" in sell_trades.columns and not sell_trades.empty:
            wins = (sell_trades["realized_pnl"].fillna(0.0) > 0).sum()
            win_rate = float(wins / len(sell_trades))
        else:
            win_rate = None

        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "trade_count": int(len(trades)),
            "win_rate": win_rate,
        }


def save_simulation_outputs(
    result: SimulationResult,
    output_dir: Path,
    save_trades_csv: bool = True,
    save_portfolio_csv: bool = True,
    save_positions_csv: bool = True,
    save_metrics_json: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    if save_trades_csv:
        path = output_dir / "trades.csv"
        result.trades.to_csv(path, index=False)
        written["trades"] = path

    if save_portfolio_csv:
        path = output_dir / "portfolio.csv"
        result.portfolio_history.to_csv(path, index=False)
        written["portfolio"] = path

    if save_positions_csv:
        path = output_dir / "positions.csv"
        result.positions_history.to_csv(path, index=False)
        written["positions"] = path

    if save_metrics_json:
        path = output_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.metrics, f, indent=2)
        written["metrics"] = path

    return written
