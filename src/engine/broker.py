from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass, asdict
from typing import Any

from src.engine.portfolio import Portfolio


@dataclass
class ExecutionResult:
    success: bool
    symbol: str
    side: str
    requested_quantity: float
    executed_quantity: float
    requested_price: float
    execution_price: float | None
    gross_value: float
    fee: float
    slippage_cost: float
    cash_before: float
    cash_after: float
    message: str
    realized_pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Broker:
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
        fractional_shares: bool = True,
    ) -> None:
        if commission_rate < 0:
            raise ValueError("commission_rate must be non-negative")
        if slippage_rate < 0:
            raise ValueError("slippage_rate must be non-negative")

        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)
        self.fractional_shares = bool(fractional_shares)

    def _normalize_quantity(self, quantity: float) -> float:
        if self.fractional_shares:
            return float(quantity)
        return float(int(quantity))

    def _buy_execution_price(self, market_price: float) -> float:
        return float(market_price * (1.0 + self.slippage_rate))

    def _sell_execution_price(self, market_price: float) -> float:
        return float(market_price * (1.0 - self.slippage_rate))

    def _fee(self, gross_value: float) -> float:
        return float(gross_value * self.commission_rate)

    def buy(
        self,
        portfolio: Portfolio,
        symbol: str,
        quantity: float,
        market_price: float,
    ) -> ExecutionResult:
        cash_before = float(portfolio.cash)

        if quantity <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=quantity,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Buy quantity must be positive.",
            )

        if market_price <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=quantity,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Market price must be positive.",
            )

        quantity = self._normalize_quantity(quantity)
        if quantity <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=0.0,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Quantity became zero after normalization.",
            )

        execution_price = self._buy_execution_price(market_price)
        gross_value = float(quantity * execution_price)
        fee = self._fee(gross_value)
        slippage_cost = float(quantity * (execution_price - market_price))

        try:
            portfolio.buy(symbol=symbol, quantity=quantity, price=execution_price, fee=fee)
        except Exception as exc:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=quantity,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=execution_price,
                gross_value=gross_value,
                fee=fee,
                slippage_cost=slippage_cost,
                cash_before=cash_before,
                cash_after=float(portfolio.cash),
                message=str(exc),
            )

        return ExecutionResult(
            success=True,
            symbol=symbol,
            side="BUY",
            requested_quantity=quantity,
            executed_quantity=quantity,
            requested_price=market_price,
            execution_price=execution_price,
            gross_value=gross_value,
            fee=fee,
            slippage_cost=slippage_cost,
            cash_before=cash_before,
            cash_after=float(portfolio.cash),
            message="Buy order executed successfully.",
        )

    def sell(
        self,
        portfolio: Portfolio,
        symbol: str,
        quantity: float,
        market_price: float,
    ) -> ExecutionResult:
        cash_before = float(portfolio.cash)

        if quantity <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL",
                requested_quantity=quantity,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Sell quantity must be positive.",
            )

        if market_price <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL",
                requested_quantity=quantity,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Market price must be positive.",
            )

        quantity = self._normalize_quantity(quantity)
        if quantity <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL",
                requested_quantity=0.0,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Quantity became zero after normalization.",
            )

        execution_price = self._sell_execution_price(market_price)
        gross_value = float(quantity * execution_price)
        fee = self._fee(gross_value)
        slippage_cost = float(quantity * (market_price - execution_price))

        try:
            realized_pnl = portfolio.sell(
                symbol=symbol,
                quantity=quantity,
                price=execution_price,
                fee=fee,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="SELL",
                requested_quantity=quantity,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=execution_price,
                gross_value=gross_value,
                fee=fee,
                slippage_cost=slippage_cost,
                cash_before=cash_before,
                cash_after=float(portfolio.cash),
                message=str(exc),
                realized_pnl=None,
            )

        return ExecutionResult(
            success=True,
            symbol=symbol,
            side="SELL",
            requested_quantity=quantity,
            executed_quantity=quantity,
            requested_price=market_price,
            execution_price=execution_price,
            gross_value=gross_value,
            fee=fee,
            slippage_cost=slippage_cost,
            cash_before=cash_before,
            cash_after=float(portfolio.cash),
            message="Sell order executed successfully.",
            realized_pnl=float(realized_pnl),
        )

    def buy_with_cash_amount(
        self,
        portfolio: Portfolio,
        symbol: str,
        cash_to_allocate: float,
        market_price: float,
    ) -> ExecutionResult:
        cash_before = float(portfolio.cash)

        if cash_to_allocate <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=0.0,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Cash allocation must be positive.",
            )

        if market_price <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=0.0,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Market price must be positive.",
            )

        execution_price = self._buy_execution_price(market_price)

        total_cost_per_share = execution_price * (1.0 + self.commission_rate)
        if total_cost_per_share <= 0:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                side="BUY",
                requested_quantity=0.0,
                executed_quantity=0.0,
                requested_price=market_price,
                execution_price=None,
                gross_value=0.0,
                fee=0.0,
                slippage_cost=0.0,
                cash_before=cash_before,
                cash_after=cash_before,
                message="Invalid total cost per share.",
            )

        quantity = cash_to_allocate / total_cost_per_share
        return self.buy(
            portfolio=portfolio,
            symbol=symbol,
            quantity=quantity,
            market_price=market_price,
        )