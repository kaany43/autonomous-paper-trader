from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.engine.broker import Broker
from src.engine.portfolio import Portfolio
from src.engine.order_builder import OrderBuilder
from src.engine.metrics import METRICS_FILENAME, compute_backtest_metrics, write_metrics_json
from src.engine.run_artifacts import RunArtifactManager
from src.strategy.momentum import MomentumStrategy
from src.strategy.base import BaseStrategy, StrategySignal


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DATA_DIR / "market_features.parquet"
BACKTEST_OUTPUTS_DIR = REPO_ROOT / "outputs" / "backtests"
TRADE_LOG_FILENAME = "trade_log.csv"
PORTFOLIO_SNAPSHOT_FILENAME = "daily_portfolio_snapshots.csv"
POSITION_SNAPSHOT_FILENAME = "daily_position_snapshots.csv"
BENCHMARK_EQUITY_FILENAME = "benchmark_equity_curve.csv"
BACKTEST_METRICS_FILENAME = METRICS_FILENAME

TRADE_LOG_COLUMNS = [
    "order_id",
    "decision_date",
    "execution_date",
    "symbol",
    "side",
    "quantity",
    "price",
    "fees",
    "slippage",
    "execution_status",
    "reason",
    "cash_before",
    "cash_after",
    "decision_price",
    "execution_price",
    "strategy_name",
    "created_at",
]

PORTFOLIO_SNAPSHOT_COLUMNS = [
    "date",
    "cash_balance",
    "invested_value",
    "total_equity",
    "realized_pnl",
    "unrealized_pnl",
    "open_positions",
]

POSITION_SNAPSHOT_COLUMNS = [
    "date",
    "symbol",
    "quantity",
    "average_cost",
    "latest_price",
    "market_value",
    "unrealized_pnl",
    "position_weight",
    "portfolio_total_equity",
]

BENCHMARK_EQUITY_COLUMNS = [
    "date",
    "benchmark_symbol",
    "benchmark_price",
    "benchmark_return",
    "benchmark_equity",
    "cumulative_return",
]


class PortfolioSnapshotWriter:
    @staticmethod
    def from_portfolio_history(portfolio_history: pd.DataFrame) -> pd.DataFrame:
        snapshots = portfolio_history.copy()
        if snapshots.empty:
            return pd.DataFrame(columns=PORTFOLIO_SNAPSHOT_COLUMNS)

        snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.normalize()
        snapshots = snapshots.rename(
            columns={
                "cash": "cash_balance",
                "number_of_positions": "open_positions",
            }
        )

        snapshots = snapshots[PORTFOLIO_SNAPSHOT_COLUMNS]
        snapshots = snapshots.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        for column in [
            "cash_balance",
            "invested_value",
            "total_equity",
            "realized_pnl",
            "unrealized_pnl",
            "open_positions",
        ]:
            snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")

        snapshots = snapshots.reset_index(drop=True)
        return snapshots

    @staticmethod
    def write_csv(snapshots: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_csv(output_path, index=False)
        return output_path


class PositionSnapshotWriter:
    @staticmethod
    def from_positions_history(
        positions_history: pd.DataFrame,
        portfolio_snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        if positions_history.empty:
            return pd.DataFrame(columns=POSITION_SNAPSHOT_COLUMNS)

        snapshots = positions_history.copy()
        snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.normalize()
        snapshots = snapshots.rename(
            columns={
                "avg_cost": "average_cost",
                "last_price": "latest_price",
            }
        )

        equity = portfolio_snapshots[["date", "total_equity"]].rename(
            columns={"total_equity": "portfolio_total_equity"}
        )
        equity["date"] = pd.to_datetime(equity["date"]).dt.normalize()

        snapshots = snapshots.merge(equity, on="date", how="left")
        snapshots = snapshots.sort_values(["date", "symbol"]).reset_index(drop=True)

        for column in [
            "quantity",
            "average_cost",
            "latest_price",
            "market_value",
            "unrealized_pnl",
            "portfolio_total_equity",
        ]:
            snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")

        snapshots["position_weight"] = snapshots["market_value"] / snapshots["portfolio_total_equity"]
        snapshots.loc[snapshots["portfolio_total_equity"] == 0.0, "position_weight"] = 0.0

        snapshots = snapshots[POSITION_SNAPSHOT_COLUMNS]
        return snapshots

    @staticmethod
    def write_csv(snapshots: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_csv(output_path, index=False)
        return output_path


class BenchmarkComparator:
    @staticmethod
    def build_benchmark_curve(
        market_data: pd.DataFrame,
        portfolio_snapshots: pd.DataFrame,
        benchmark_symbol: str,
        initial_capital: float,
        price_column: str,
    ) -> pd.DataFrame:
        canonical_dates = pd.DataFrame({
            "date": pd.to_datetime(portfolio_snapshots.get("date", pd.Series(dtype="datetime64[ns]"))).dt.normalize()
        })
        canonical_dates = canonical_dates.dropna().drop_duplicates(subset=["date"]).sort_values("date")

        if canonical_dates.empty:
            return pd.DataFrame(columns=BENCHMARK_EQUITY_COLUMNS)

        symbol = str(benchmark_symbol or "").strip().upper()
        if not symbol:
            curve = canonical_dates.copy()
            curve["benchmark_symbol"] = ""
            curve["benchmark_price"] = pd.NA
            curve["benchmark_return"] = 0.0
            curve["benchmark_equity"] = float(initial_capital)
            curve["cumulative_return"] = 0.0
            return curve[BENCHMARK_EQUITY_COLUMNS]

        benchmark_prices = market_data.loc[
            market_data["symbol"].astype(str).str.upper().str.strip() == symbol,
            ["date", price_column],
        ].copy()

        if benchmark_prices.empty:
            curve = canonical_dates.copy()
            curve["benchmark_symbol"] = symbol
            curve["benchmark_price"] = pd.NA
            curve["benchmark_return"] = 0.0
            curve["benchmark_equity"] = float(initial_capital)
            curve["cumulative_return"] = 0.0
            return curve[BENCHMARK_EQUITY_COLUMNS]

        benchmark_prices["date"] = pd.to_datetime(benchmark_prices["date"]).dt.normalize()
        benchmark_prices[price_column] = pd.to_numeric(benchmark_prices[price_column], errors="coerce")
        benchmark_prices = benchmark_prices.drop_duplicates(subset=["date"], keep="last")
        benchmark_prices = benchmark_prices.rename(columns={price_column: "benchmark_price"})

        curve = canonical_dates.merge(benchmark_prices, on="date", how="left").sort_values("date")
        curve["benchmark_symbol"] = symbol
        curve["benchmark_price"] = curve["benchmark_price"].ffill().bfill()

        if curve["benchmark_price"].isna().all():
            curve["benchmark_return"] = 0.0
            curve["benchmark_equity"] = float(initial_capital)
            curve["cumulative_return"] = 0.0
            return curve[BENCHMARK_EQUITY_COLUMNS]

        curve["benchmark_return"] = curve["benchmark_price"].pct_change().fillna(0.0)
        curve["benchmark_return"] = curve["benchmark_return"].replace([pd.NA, float("inf"), float("-inf")], 0.0).fillna(0.0)

        starting_capital = float(initial_capital)
        curve["benchmark_equity"] = starting_capital * (1.0 + curve["benchmark_return"]).cumprod()

        if starting_capital == 0.0:
            curve["cumulative_return"] = 0.0
        else:
            curve["cumulative_return"] = curve["benchmark_equity"] / starting_capital - 1.0

        return curve[BENCHMARK_EQUITY_COLUMNS]

    @staticmethod
    def write_csv(curve: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(output_path, index=False)
        return output_path


class DailySimulator:
    """
    Daily backtest simulator.

    Rules:
    - Processes one trading day at a time in chronological order
    - For each decision date, only rows up to that date are visible
    - Target/label columns are removed before passing data to strategy
    - Orders decided on day t are executed on the next trading day
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
        self._trade_log_history: list[dict[str, Any]] = []
        self._signal_history: list[dict[str, Any]] = []
        self._pending_signals: dict[pd.Timestamp, list[StrategySignal]] = {}
        self._order_sequence: int = 0

        max_open_positions = int(getattr(strategy, "max_open_positions", 1) or 1)
        max_position_size = getattr(strategy, "max_position_size", None)
        self.order_builder = OrderBuilder(
            max_open_positions=max_open_positions,
            max_position_size=max_position_size,
            fractional_shares=self.broker.fractional_shares,
            commission_rate=self.broker.commission_rate,
            slippage_rate=self.broker.slippage_rate,
        )

    def _ensure_runtime_state(self) -> None:
        """
        Backward-compatible guard for runtime attributes.

        Some environments may execute code with stale artifacts where newly added
        attributes are missing at runtime. This method ensures required mutable
        containers exist before simulation steps proceed.
        """
        if not hasattr(self, "_pending_signals") or self._pending_signals is None:
            self._pending_signals = {}

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

    def _record_signals(
        self,
        signals: list[StrategySignal],
        scheduled_execution_date: pd.Timestamp | None,
        schedule_status: str,
    ) -> None:
        for signal in signals:
            row = self._signal_to_row(signal)
            row["decision_date"] = pd.Timestamp(signal.date)
            row["scheduled_execution_date"] = scheduled_execution_date
            row["schedule_status"] = schedule_status
            self._signal_history.append(row)

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
    def _format_datetime(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

    def _next_order_id(self) -> str:
        self._order_sequence += 1
        return f"ORD-{self._order_sequence:08d}"

    def _record_trade_log(self, row: dict[str, Any]) -> None:
        normalized = {col: row.get(col) for col in TRADE_LOG_COLUMNS}
        self._trade_log_history.append(normalized)

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

    def _execute_orders(
        self,
        orders: list[dict[str, Any]],
        execution_date: pd.Timestamp,
        decision_date: pd.Timestamp,
    ) -> None:
        for order in orders:
            side = str(order.get("side", "")).upper()
            symbol = str(order.get("symbol", "")).upper()
            quantity = float(order.get("quantity", 0.0) or 0.0)
            market_price = order.get("market_price")
            order_id = self._next_order_id()

            if not symbol or side not in {"BUY", "SELL"}:
                self._record_trade_log(
                    {
                        "order_id": order_id,
                        "decision_date": self._format_datetime(decision_date),
                        "execution_date": self._format_datetime(execution_date),
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": None,
                        "fees": 0.0,
                        "slippage": 0.0,
                        "execution_status": "SKIPPED_INVALID_ORDER",
                        "reason": "Order has invalid side or symbol.",
                        "cash_before": float(self.portfolio.cash),
                        "cash_after": float(self.portfolio.cash),
                        "decision_price": market_price,
                        "execution_price": None,
                        "strategy_name": self.strategy.__class__.__name__,
                        "created_at": self._format_datetime(execution_date),
                    }
                )
                continue
            if quantity <= 0 or market_price is None or float(market_price) <= 0:
                self._record_trade_log(
                    {
                        "order_id": order_id,
                        "decision_date": self._format_datetime(decision_date),
                        "execution_date": self._format_datetime(execution_date),
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": None,
                        "fees": 0.0,
                        "slippage": 0.0,
                        "execution_status": "SKIPPED_INVALID_ORDER",
                        "reason": "Order has invalid quantity or market price.",
                        "cash_before": float(self.portfolio.cash),
                        "cash_after": float(self.portfolio.cash),
                        "decision_price": market_price,
                        "execution_price": None,
                        "strategy_name": self.strategy.__class__.__name__,
                        "created_at": self._format_datetime(execution_date),
                    }
                )
                continue

            if side == "SELL":
                result = self.broker.sell(
                    portfolio=self.portfolio,
                    symbol=symbol,
                    quantity=quantity,
                    market_price=float(market_price),
                )
            else:
                result = self.broker.buy(
                    portfolio=self.portfolio,
                    symbol=symbol,
                    quantity=quantity,
                    market_price=float(market_price),
                )

            row = result.to_dict()
            row["date"] = execution_date
            row["decision_date"] = pd.Timestamp(decision_date)
            row["execution_date"] = execution_date
            self._trade_history.append(row)
            self._record_trade_log(
                {
                    "order_id": order_id,
                    "decision_date": self._format_datetime(decision_date),
                    "execution_date": self._format_datetime(execution_date),
                    "symbol": symbol,
                    "side": side,
                    "quantity": float(row.get("requested_quantity", 0.0) or 0.0),
                    "price": row.get("execution_price")
                    if row.get("execution_price") is not None
                    else row.get("requested_price"),
                    "fees": float(row.get("fee", 0.0) or 0.0),
                    "slippage": float(row.get("slippage_cost", 0.0) or 0.0),
                    "execution_status": "EXECUTED" if bool(row.get("success")) else "REJECTED",
                    "reason": row.get("message", ""),
                    "cash_before": float(row.get("cash_before", 0.0) or 0.0),
                    "cash_after": float(row.get("cash_after", 0.0) or 0.0),
                    "decision_price": row.get("requested_price"),
                    "execution_price": row.get("execution_price"),
                    "strategy_name": self.strategy.__class__.__name__,
                    "created_at": self._format_datetime(execution_date),
                }
            )

    @staticmethod
    def _next_trading_date_map(
        trading_dates: list[pd.Timestamp],
    ) -> dict[pd.Timestamp, pd.Timestamp | None]:
        next_map: dict[pd.Timestamp, pd.Timestamp | None] = {}
        for idx, trading_date in enumerate(trading_dates):
            next_map[trading_date] = trading_dates[idx + 1] if idx + 1 < len(trading_dates) else None
        return next_map

    def _schedule_signals(
        self,
        signals: list[StrategySignal],
        next_execution_date: pd.Timestamp | None,
    ) -> None:
        if not signals:
            return

        if next_execution_date is None:
            self._record_signals(
                signals=signals,
                scheduled_execution_date=None,
                schedule_status="NO_NEXT_TRADING_SESSION",
            )
            for signal in signals:
                side = str(signal.action).upper()
                if side not in {"BUY", "SELL"}:
                    continue
                self._record_trade_log(
                    {
                        "order_id": self._next_order_id(),
                        "decision_date": self._format_datetime(signal.date),
                        "execution_date": None,
                        "symbol": str(signal.symbol).upper(),
                        "side": side,
                        "quantity": None,
                        "price": None,
                        "fees": 0.0,
                        "slippage": 0.0,
                        "execution_status": "SKIPPED_NO_NEXT_SESSION",
                        "reason": str(signal.reason_code or "No next trading session available."),
                        "cash_before": float(self.portfolio.cash),
                        "cash_after": float(self.portfolio.cash),
                        "decision_price": None,
                        "execution_price": None,
                        "strategy_name": self.strategy.__class__.__name__,
                        "created_at": self._format_datetime(signal.date),
                    }
                )
            return

        self._record_signals(
            signals=signals,
            scheduled_execution_date=next_execution_date,
            schedule_status="SCHEDULED",
        )
        self._pending_signals.setdefault(next_execution_date, []).extend(signals)

    def _execute_pending_signals(self, execution_date: pd.Timestamp) -> None:
        pending = self._pending_signals.pop(execution_date, [])
        if not pending:
            return

        signals = self._sort_signals(pending)
        price_map = self._day_prices(execution_date)

        decision_date = pd.Timestamp(signals[0].date)

        orders = self.order_builder.build_orders(
            signals=signals,
            portfolio=self.portfolio,
            price_map=price_map,
            available_cash=float(self.portfolio.cash),
        )

        self._execute_orders(
            orders=orders,
            execution_date=execution_date,
            decision_date=decision_date,
        )

    def _process_day(
        self,
        trading_date: pd.Timestamp,
        next_execution_date: pd.Timestamp | None,
    ) -> None:
        self._execute_pending_signals(execution_date=trading_date)

        historical_data = self._history_until(trading_date)


        signals = self.strategy.generate_signals(
            decision_date=trading_date,
            market_data=historical_data,
            portfolio=self.portfolio,
        )
        signals = self._sort_signals(signals)

        self._schedule_signals(
            signals=signals,
            next_execution_date=next_execution_date,
        )

        price_map = self._day_prices(trading_date)

        self._record_daily_snapshots(
            trading_date=trading_date,
            price_map=price_map,
        )

    def run(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        benchmark_symbol: str = "",
        benchmark_output_filename: str = BENCHMARK_EQUITY_FILENAME,
        run_config: dict[str, Any] | None = None,
        config_source: str = "",
    ) -> dict[str, pd.DataFrame]:
        self._portfolio_history.clear()
        self._positions_history.clear()
        self._trade_history.clear()
        self._trade_log_history.clear()
        self._signal_history.clear()
        self._order_sequence = 0

        trading_dates = self._selected_trading_dates(
            start_date=start_date,
            end_date=end_date,
        )

        if not trading_dates:
            raise ValueError("No trading dates found for the requested range.")

        artifact_manager = RunArtifactManager(
            base_output_dir=BACKTEST_OUTPUTS_DIR,
            strategy_name=self.strategy.__class__.__name__,
            benchmark_symbol=benchmark_symbol,
            start_date=self._format_datetime(trading_dates[0]),
            end_date=self._format_datetime(trading_dates[-1]),
        )

        resolved_run_config = run_config or {
            "strategy_name": self.strategy.__class__.__name__,
            "strategy_parameters": dict(getattr(self.strategy, "__dict__", {})),
            "broker": {
                "commission_rate": float(getattr(self.broker, "commission_rate", 0.0) or 0.0),
                "slippage_rate": float(getattr(self.broker, "slippage_rate", 0.0) or 0.0),
                "fractional_shares": bool(getattr(self.broker, "fractional_shares", False)),
            },
            "portfolio": {
                "initial_cash": float(getattr(self.portfolio, "initial_cash", 0.0) or 0.0),
            },
            "run": {
                "start_date": self._format_datetime(trading_dates[0]),
                "end_date": self._format_datetime(trading_dates[-1]),
                "benchmark_symbol": str(benchmark_symbol or ""),
                "price_column": self.price_column,
            },
        }
        config_path = artifact_manager.write_config_snapshot(resolved_run_config)

        try:
            next_date_map = self._next_trading_date_map(trading_dates)

            for trading_date in trading_dates:
                self._process_day(
                    trading_date=trading_date,
                    next_execution_date=next_date_map.get(trading_date),
                )

            portfolio_history = pd.DataFrame(self._portfolio_history)
            positions_history = pd.DataFrame(self._positions_history)
            trade_history = pd.DataFrame(self._trade_history)
            signal_history = pd.DataFrame(self._signal_history)
            trade_log = pd.DataFrame(self._trade_log_history, columns=TRADE_LOG_COLUMNS)

            if not portfolio_history.empty:
                portfolio_history = portfolio_history.sort_values("date").reset_index(drop=True)

            portfolio_snapshots = PortfolioSnapshotWriter.from_portfolio_history(portfolio_history)
            position_snapshots = PositionSnapshotWriter.from_positions_history(
                positions_history=positions_history,
                portfolio_snapshots=portfolio_snapshots,
            )

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

            if not trade_log.empty:
                trade_log = trade_log.sort_values(
                    ["decision_date", "symbol", "side", "order_id"]
                ).reset_index(drop=True)

            benchmark_curve = BenchmarkComparator.build_benchmark_curve(
                market_data=self.market_data,
                portfolio_snapshots=portfolio_snapshots,
                benchmark_symbol=benchmark_symbol,
                initial_capital=float(getattr(self.portfolio, "initial_cash", 0.0)),
                price_column=self.price_column,
            )

            trade_log_path = artifact_manager.artifact_path(TRADE_LOG_FILENAME)
            portfolio_snapshots_path = artifact_manager.artifact_path(PORTFOLIO_SNAPSHOT_FILENAME)
            position_snapshots_path = artifact_manager.artifact_path(POSITION_SNAPSHOT_FILENAME)
            benchmark_curve_path = artifact_manager.artifact_path(
                str(benchmark_output_filename or BENCHMARK_EQUITY_FILENAME)
            )
            metrics_path = artifact_manager.artifact_path(BACKTEST_METRICS_FILENAME)

            trade_log.to_csv(trade_log_path, index=False)
            artifact_manager.register_artifact("trade_log", trade_log_path)
            PortfolioSnapshotWriter.write_csv(portfolio_snapshots, portfolio_snapshots_path)
            artifact_manager.register_artifact("portfolio_snapshots", portfolio_snapshots_path)
            PositionSnapshotWriter.write_csv(position_snapshots, position_snapshots_path)
            artifact_manager.register_artifact("position_snapshots", position_snapshots_path)
            BenchmarkComparator.write_csv(benchmark_curve, benchmark_curve_path)
            artifact_manager.register_artifact("benchmark_curve", benchmark_curve_path)

            backtest_metrics = compute_backtest_metrics(
                strategy_equity_curve=portfolio_snapshots,
                benchmark_equity_curve=benchmark_curve,
                trade_history=trade_history,
            )
            write_metrics_json(backtest_metrics, metrics_path)
            artifact_manager.register_artifact("backtest_metrics", metrics_path)

            manifest_path = artifact_manager.write_manifest(
                status="completed",
                config_source=config_source,
            )
        except Exception as exc:
            artifact_manager.write_manifest(
                status="failed",
                config_source=config_source,
                error_message=str(exc),
            )
            raise

        return {
            "portfolio_history": portfolio_history,
            "portfolio_snapshots": portfolio_snapshots,
            "position_snapshots": position_snapshots,
            "positions_history": positions_history,
            "trade_history": trade_history,
            "signal_history": signal_history,
            "trade_log": trade_log,
            "benchmark_curve": benchmark_curve,
            "trade_log_path": trade_log_path,
            "portfolio_snapshots_path": portfolio_snapshots_path,
            "position_snapshots_path": position_snapshots_path,
            "benchmark_curve_path": benchmark_curve_path,
            "backtest_metrics": backtest_metrics,
            "backtest_metrics_path": metrics_path,
            "run_id": artifact_manager.run_id,
            "output_dir": artifact_manager.output_dir,
            "config_path": config_path,
            "manifest_path": manifest_path,
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