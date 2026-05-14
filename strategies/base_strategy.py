"""
strategies/base_strategy.py
----------------------------
Abstract base class and shared Signal dataclass for all strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Signal:
    strategy: str          # "rsmb" | "existing"
    symbol: str
    side: str              # "BUY" | "SELL"
    entry: float
    sl: float
    target1: float
    target2: float
    qty: int
    score: float
    rs_rank: Optional[float]
    rejection_reason: Optional[str]
    timestamp: pd.Timestamp
    id: Optional[str] = None


class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, symbol: str, bar: pd.Series) -> Optional[Signal]:
        """Called on each completed bar. Return Signal or None."""
        ...

    @abstractmethod
    def on_fill(self, signal: Signal, fill_price: float) -> None:
        """Called when an order is confirmed filled."""
        ...

    @abstractmethod
    def on_target_hit(self, signal: Signal, target_num: int) -> None:
        """Called when Target 1 or Target 2 is hit."""
        ...

    @abstractmethod
    def on_sl_hit(self, signal: Signal) -> None:
        """Called when the stop loss is hit."""
        ...
