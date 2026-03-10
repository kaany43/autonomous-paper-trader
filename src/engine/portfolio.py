from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float

    def market_value(self, last_price: float) -> float:
        return float(self.quantity * last_price)

    def unrealized_pnl(self, last_price: float) -> float:
        return float((last_price - self.avg_cost) * self.quantity)

    def unrealized_pnl_pct(self, last_price: float) -> float | None:
        if self.avg_cost == 0:
            return None
        return float(last_price / self.avg_cost - 1.0)


class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        if initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0

    def has_position(self, symbol: str) -> bool:
        position = self.positions.get(symbol)
        return position is not None and position.quantity > 0

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def buy(self, symbol: str, quantity: float, price: float, fee: float = 0.0) -> None:
        if quantity <= 0:
            raise ValueError("Buy quantity must be positive")
        if price <= 0:
            raise ValueError("Buy price must be positive")
        if fee < 0:
            raise ValueError("Fee cannot be negative")

        gross_value = quantity * price
        total_cost = gross_value + fee

        if total_cost > self.cash + 1e-12:
            raise ValueError(
                f"Insufficient cash to buy {symbol}. Needed={total_cost:.4f}, available={self.cash:.4f}"
            )

        existing = self.positions.get(symbol)

        if existing is None:
            new_quantity = quantity
            new_avg_cost = gross_value / quantity
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=float(new_quantity),
                avg_cost=float(new_avg_cost),
            )
        else:
            new_quantity = existing.quantity + quantity
            new_cost_basis = (existing.quantity * existing.avg_cost) + gross_value
            new_avg_cost = new_cost_basis / new_quantity
            existing.quantity = float(new_quantity)
            existing.avg_cost = float(new_avg_cost)

        self.cash -= total_cost

    def sell(self, symbol: str, quantity: float, price: float, fee: float = 0.0) -> float:
        if quantity <= 0:
            raise ValueError("Sell quantity must be positive")
        if price <= 0:
            raise ValueError("Sell price must be positive")
        if fee < 0:
            raise ValueError("Fee cannot be negative")
        if symbol not in self.positions:
            raise ValueError(f"No open position for {symbol}")

        position = self.positions[symbol]

        if quantity > position.quantity + 1e-12:
            raise ValueError(
                f"Cannot sell more than current position for {symbol}. "
                f"Requested={quantity:.6f}, available={position.quantity:.6f}"
            )

        gross_value = quantity * price
        net_proceeds = gross_value - fee
        realized = (price - position.avg_cost) * quantity - fee

        position.quantity -= quantity
        self.cash += net_proceeds
        self.realized_pnl += realized

        if position.quantity <= 1e-12:
            del self.positions[symbol]

        return float(realized)

    def invested_value(self, price_map: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            last_price = price_map.get(symbol)
            if last_price is None:
                continue
            total += position.market_value(last_price)
        return float(total)

    def total_equity(self, price_map: dict[str, float]) -> float:
        return float(self.cash + self.invested_value(price_map))

    def total_unrealized_pnl(self, price_map: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            last_price = price_map.get(symbol)
            if last_price is None:
                continue
            total += position.unrealized_pnl(last_price)
        return float(total)

    def portfolio_snapshot(self, date: Any, price_map: dict[str, float]) -> dict[str, Any]:
        invested = self.invested_value(price_map)
        total_equity = self.total_equity(price_map)
        unrealized = self.total_unrealized_pnl(price_map)

        return {
            "date": date,
            "cash": float(self.cash),
            "invested_value": float(invested),
            "total_equity": float(total_equity),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(unrealized),
            "number_of_positions": int(len(self.positions)),
        }

    def positions_snapshot(self, date: Any, price_map: dict[str, float]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for symbol in sorted(self.positions.keys()):
            position = self.positions[symbol]
            last_price = price_map.get(symbol)
            market_value = position.market_value(last_price) if last_price is not None else None
            unrealized_pnl = position.unrealized_pnl(last_price) if last_price is not None else None
            unrealized_pnl_pct = (
                position.unrealized_pnl_pct(last_price) if last_price is not None else None
            )

            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "quantity": float(position.quantity),
                    "avg_cost": float(position.avg_cost),
                    "last_price": float(last_price) if last_price is not None else None,
                    "market_value": float(market_value) if market_value is not None else None,
                    "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
                    "unrealized_pnl_pct": (
                        float(unrealized_pnl_pct) if unrealized_pnl_pct is not None else None
                    ),
                }
            )

        return rows

    def summary(self) -> dict[str, Any]:
        return {
            "initial_cash": float(self.initial_cash),
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl),
            "open_positions": len(self.positions),
            "positions": {
                symbol: asdict(position) for symbol, position in self.positions.items()
            },
        }