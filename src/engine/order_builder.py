from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.portfolio import Portfolio
from src.strategy.base import StrategySignal


@dataclass(frozen=True)
class ExecutableOrder:
    symbol: str
    side: str
    quantity: float
    market_price: float

    def to_broker_dict(self) -> dict[str, Any]:
        return {
            "symbol": str(self.symbol),
            "side": str(self.side).upper(),
            "quantity": float(self.quantity),
            "market_price": float(self.market_price),
        }


class OrderBuilder:
    def __init__(
        self,
        max_open_positions: int,
        max_position_size: float | None = None,
        allow_add_to_existing: bool = False,
        fractional_shares: bool = True,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.0,
    ) -> None:
        if max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
        if max_position_size is not None and (max_position_size <= 0 or max_position_size > 1.0):
            raise ValueError("max_position_size must be in (0, 1]")

        self.max_open_positions = int(max_open_positions)
        self.max_position_size = (
            float(max_position_size)
            if max_position_size is not None
            else 1.0 / float(self.max_open_positions)
        )
        self.allow_add_to_existing = bool(allow_add_to_existing)
        self.fractional_shares = bool(fractional_shares)
        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)

    @staticmethod
    def _normalize_signal(signal: StrategySignal) -> dict[str, Any] | None:
        symbol = str(getattr(signal, "symbol", "")).upper().strip()
        action = str(getattr(signal, "action", "")).upper().strip()
        score = float(getattr(signal, "score", 0.0) or 0.0)
        target_weight = getattr(signal, "target_weight", None)

        if not symbol or action not in {"BUY", "SELL", "HOLD"}:
            return None

        normalized_target = None
        if target_weight is not None:
            try:
                tw = float(target_weight)
                normalized_target = tw if tw > 0 else None
            except (TypeError, ValueError):
                normalized_target = None

        return {
            "symbol": symbol,
            "action": action,
            "score": score,
            "target_weight": normalized_target,
        }

    def build_orders(
        self,
        signals: list[StrategySignal],
        portfolio: Portfolio,
        price_map: dict[str, float],
        available_cash: float | None = None,
    ) -> list[dict[str, Any]]:
        if not signals:
            return []

        cash = float(portfolio.cash if available_cash is None else available_cash)
        equity = float(cash + portfolio.invested_value(price_map))
        current_positions = {
            symbol: position
            for symbol, position in portfolio.positions.items()
            if position.quantity > 0
        }

        normalized = [self._normalize_signal(s) for s in signals]
        normalized = [s for s in normalized if s is not None]

        sell_candidates = sorted(
            [s for s in normalized if s["action"] == "SELL"],
            key=lambda s: (-float(s["score"]), s["symbol"]),
        )

        sell_orders: list[ExecutableOrder] = []
        sold_symbols: set[str] = set()
        total_sell_proceeds = 0.0
        total_sell_proceeds_multiplier = (1.0 - self.slippage_rate) * (1.0 - self.commission_rate)
        for signal in sell_candidates:
            symbol = signal["symbol"]
            if symbol in sold_symbols:
                continue
            position = current_positions.get(symbol)
            market_price = price_map.get(symbol)
            if position is None or position.quantity <= 0:
                continue
            if market_price is None or float(market_price) <= 0:
                continue
            sell_orders.append(
                ExecutableOrder(
                    symbol=symbol,
                    side="SELL",
                    quantity=float(position.quantity),
                    market_price=float(market_price),
                )
            )
            total_sell_proceeds += (
                float(position.quantity)
                * float(market_price)
                * total_sell_proceeds_multiplier
            )
            sold_symbols.add(symbol)

        cash += total_sell_proceeds


        post_sell_open_positions = len(current_positions) - len(sell_orders)
        slots_left = max(self.max_open_positions - post_sell_open_positions, 0)
        if slots_left <= 0:
            return [order.to_broker_dict() for order in sell_orders]

        buy_signals_by_symbol: dict[str, dict[str, Any]] = {}
        for signal in normalized:
            if signal["action"] != "BUY":
                continue
            symbol = signal["symbol"]
            market_price = price_map.get(symbol)
            if market_price is None or float(market_price) <= 0:
                continue
            if symbol in sold_symbols:
                continue
            if not self.allow_add_to_existing and symbol in current_positions:
                continue
            existing = buy_signals_by_symbol.get(symbol)
            if existing is None or (
                float(signal["score"]) > float(existing["score"])
                or (
                    float(signal["score"]) == float(existing["score"])
                    and symbol < str(existing["symbol"])
                )
            ):
                buy_signals_by_symbol[symbol] = signal

        buy_candidates = sorted(
            buy_signals_by_symbol.values(),
            key=lambda s: (-float(s["score"]), s["symbol"]),
        )

        max_notional = float(max(equity * self.max_position_size, 0.0))
        total_buy_cost_per_share_multiplier = (1.0 + self.slippage_rate) * (1.0 + self.commission_rate)

        buy_orders: list[ExecutableOrder] = []
        for candidate in buy_candidates:
            if slots_left <= 0:
                break
            symbol = candidate["symbol"]
            market_price = float(price_map[symbol])
            if market_price <= 0:
                continue

            target_weight = candidate.get("target_weight")
            if target_weight is not None:
                desired_notional = float(equity * float(target_weight))
            else:
                desired_notional = float(cash / max(slots_left, 1))

            notional = min(cash, max_notional, desired_notional)
            if notional <= 0:
                continue

            effective_cost_per_share = market_price * total_buy_cost_per_share_multiplier
            if effective_cost_per_share <= 0:
                continue

            raw_quantity = notional / effective_cost_per_share
            quantity = float(raw_quantity if self.fractional_shares else int(raw_quantity))
            if quantity <= 0:
                fallback_notional = min(cash, max_notional)
                raw_quantity = fallback_notional / effective_cost_per_share
                quantity = float(raw_quantity if self.fractional_shares else int(raw_quantity))
                if quantity <= 0:
                    continue

            estimated_total_cost = quantity * effective_cost_per_share
            if estimated_total_cost > cash + 1e-12:
                continue

            buy_orders.append(
                ExecutableOrder(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    market_price=market_price,
                )
            )
            cash -= estimated_total_cost
            slots_left -= 1

        all_orders = [*sell_orders, *buy_orders]
        return [order.to_broker_dict() for order in all_orders]
