from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategySignal:
    date: Any
    symbol: str
    action: str  # BUY / SELL / HOLD
    score: float = 0.0
    reason_code: str = ""
    target_weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(
        self,
        decision_date: Any,
        market_data,
        portfolio,
    ) -> list[StrategySignal]:
        """
        Generate trading signals for a given decision date.

        Parameters
        ----------
        decision_date:
            The date on which the decision is made.
        market_data:
            Feature-enriched dataframe containing all symbols.
        portfolio:
            Current portfolio state.

        Returns
        -------
        list[StrategySignal]
            A list of BUY / SELL / HOLD signals.
        """
        raise NotImplementedError