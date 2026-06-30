from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Position:
    """Represent one simulated open position."""
    symbol: str
    strategy: str
    side: str
    entry: float
    sl: float
    target1: float | None
    target2: float | None
    qty: int
    qty_open: int
    status: str
    entry_ts: pd.Timestamp
    entry_i: int
    score: float
    risk_per_unit: float
    target2_mode: str = "price"
    t1_exit_pct: float = 0.5
    theta_veto_bars: int = 0
    t1_done: bool = False
    gross_realised: float = 0.0
    brokerage_legs: int = 1
    exit_price: float = 0.0
    outcome: str = ""


@dataclass
class StrategyResult:
    """Hold metrics and trades for one strategy."""
    name: str
    trades: list[dict[str, Any]]
    metrics: dict[str, Any]
    monthly: pd.DataFrame
    daily_pnl: pd.Series
    sensitivity: pd.DataFrame | None = None
    flags: dict[str, str] | None = None


@dataclass
class BacktestResults:
    """Hold all strategy results."""
    strategies: dict[str, StrategyResult]
