from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any

import pandas as pd

from src.strategy.base import BaseStrategy, StrategySignal


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        max_open_positions: int = 2,
        top_k: int = 2,
        min_score: float = 0.0,
        min_volume_ratio: float = 0.8,
    ) -> None:
        self.max_open_positions = int(max_open_positions)
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.min_volume_ratio = float(min_volume_ratio)
 
    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        required = [
            "date",
            "symbol",
            "adj_close",
            "ret_5d",
            "ma_20",
            "ma_50",
            "price_vs_ma20",
            "ma20_vs_ma50",
            "volume_ratio_20",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required strategy columns: {', '.join(missing)}")

    def _entry_condition(self, row: pd.Series) -> bool:
        """
        Simple long entry logic.
        Uses only same-day features that are derived from historical data up to that day.
        """
        return bool(
            pd.notna(row["ret_5d"])
            and pd.notna(row["ma_20"])
            and pd.notna(row["ma_50"])
            and pd.notna(row["price_vs_ma20"])
            and pd.notna(row["ma20_vs_ma50"])
            and row["ret_5d"] > 0
            and row["adj_close"] > row["ma_20"]
            and row["ma_20"] > row["ma_50"]
            and row["volume_ratio_20"] > self.min_volume_ratio
        )

    @staticmethod
    def _exit_condition(row: pd.Series) -> bool:
        """
        Simple exit logic for existing long positions.
        """
        return bool(
            pd.notna(row["ma_20"])
            and pd.notna(row["ma_50"])
            and (
                row["adj_close"] < row["ma_20"]
                or row["ma_20"] < row["ma_50"]
                or (pd.notna(row["ret_5d"]) and row["ret_5d"] < 0)
            )
        )

    @staticmethod
    def _score(row: pd.Series) -> float:
        """
        Candidate ranking score.
        Higher is better.
        """
        price_vs_ma20 = float(row["price_vs_ma20"]) if pd.notna(row["price_vs_ma20"]) else 0.0
        ret_5d = float(row["ret_5d"]) if pd.notna(row["ret_5d"]) else 0.0
        trend_strength = float(row["ma20_vs_ma50"]) if pd.notna(row["ma20_vs_ma50"]) else 0.0
        volume_ratio = float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else 1.0

        volume_boost = max(volume_ratio - 1.0, 0.0)

        score = (
            0.45 * ret_5d
            + 0.35 * price_vs_ma20
            + 0.15 * trend_strength
            + 0.05 * volume_boost
        )
        return float(score)

    def generate_signals(
        self,
        decision_date: Any,
        market_data: pd.DataFrame,
        portfolio,
    ) -> list[StrategySignal]:
        self._validate_required_columns(market_data)

        decision_date = pd.to_datetime(decision_date)
        day_df = market_data.loc[market_data["date"] == decision_date].copy()

        if day_df.empty:
            raise ValueError(f"No market data found for decision_date={decision_date.date()}")

        signals: list[StrategySignal] = []

        # 1) Exit checks for existing positions
        held_symbols = sorted(portfolio.positions.keys())

        for symbol in held_symbols:
            symbol_row = day_df.loc[day_df["symbol"] == symbol]
            if symbol_row.empty:
                continue

            row = symbol_row.iloc[0]

            if self._exit_condition(row):
                signals.append(
                    StrategySignal(
                        date=decision_date,
                        symbol=symbol,
                        action="SELL",
                        score=self._score(row),
                        reason_code="MOMENTUM_EXIT",
                        metadata={
                            "ret_5d": row.get("ret_5d"),
                            "price_vs_ma20": row.get("price_vs_ma20"),
                            "ma20_vs_ma50": row.get("ma20_vs_ma50"),
                        },
                    )
                )

        # Available slots after potential exits
        current_open_positions = len(portfolio.positions)
        planned_exits = len([s for s in signals if s.action == "SELL"])
        available_slots = max(self.max_open_positions - (current_open_positions - planned_exits), 0)

        # 2) Entry candidate ranking
        candidates: list[StrategySignal] = []

        for _, row in day_df.iterrows():
            symbol = row["symbol"]

            if portfolio.has_position(symbol):
                continue

            if not self._entry_condition(row):
                continue

            score = self._score(row)
            if score < self.min_score:
                continue

            candidates.append(
                StrategySignal(
                    date=decision_date,
                    symbol=symbol,
                    action="BUY",
                    score=score,
                    reason_code="MOMENTUM_ENTRY",
                    metadata={
                        "ret_5d": row.get("ret_5d"),
                        "price_vs_ma20": row.get("price_vs_ma20"),
                        "ma20_vs_ma50": row.get("ma20_vs_ma50"),
                        "volume_ratio_20": row.get("volume_ratio_20"),
                    },
                )
            )

        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

        buy_signals = candidates[: min(self.top_k, available_slots)]

        # Optional equal-weight target for later broker allocation
        if buy_signals:
            target_weight = 1.0 / max(self.max_open_positions, 1)
            for signal in buy_signals:
                signal.target_weight = target_weight

        signals.extend(buy_signals)

        # Sort so sells happen before buys later in simulator logic
        action_priority = {"SELL": 0, "BUY": 1, "HOLD": 2}
        signals = sorted(signals, key=lambda s: (action_priority.get(s.action, 99), -s.score, s.symbol))

        return signals
