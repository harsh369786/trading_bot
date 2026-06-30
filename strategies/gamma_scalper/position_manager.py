"""
strategies/gamma_scalper/position_manager.py
--------------------------------------------
Thread-safe position state for Gamma Scalper.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategies.base_strategy import Signal
from strategies.gamma_scalper.indicators import ema


@dataclass
class GammaPosition:
    position_id: str
    signal: Signal
    leg: str
    fill_price: float = 0.0
    fill_time: Optional[pd.Timestamp] = None
    qty_open: int = 0
    qty_t1_booked: int = 0
    bars_since_entry: int = 0
    trailing_sl: float = 0.0
    status: str = "PENDING"
    paper_order_id: Optional[str] = None


def infer_option_leg(symbol: str) -> str:
    upper = str(symbol or "").upper().replace(" ", "")
    if upper.endswith("CE") or "CE" in upper:
        return "CE"
    if upper.endswith("PE") or "PE" in upper:
        return "PE"
    return "UNKNOWN"


class GammaPositionManager:
    """Max two option legs: one CE and one PE active at the same time."""

    MAX_OPEN_TRADES = 2

    def __init__(self, max_open_trades: int = MAX_OPEN_TRADES) -> None:
        self.max_open_trades = int(max_open_trades or self.MAX_OPEN_TRADES)
        self._positions: Dict[str, GammaPosition] = {}
        self._lock = threading.Lock()

    def can_open(self, leg: str) -> bool:
        with self._lock:
            open_positions = [
                p for p in self._positions.values()
                if p.status in {"PENDING", "ACTIVE", "PARTIAL"}
            ]
            if len(open_positions) >= self.max_open_trades:
                return False
            return not any(p.leg == leg and p.status in {"PENDING", "ACTIVE", "PARTIAL"} for p in open_positions)

    def open_position(self, signal: Signal) -> Optional[str]:
        leg = infer_option_leg(signal.symbol)
        if not self.can_open(leg):
            logger.warning(f"GammaPositionManager: cannot open {signal.symbol}; {leg} leg already active or max reached")
            return None
        pos_id = str(uuid.uuid4())[:8]
        pos = GammaPosition(
            position_id=pos_id,
            signal=signal,
            leg=leg,
            qty_open=signal.qty,
            trailing_sl=signal.sl,
            status="PENDING",
        )
        with self._lock:
            self._positions[pos_id] = pos
        return pos_id

    def on_fill(self, position_id: str, fill_price: float, fill_time: pd.Timestamp) -> None:
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                logger.warning(f"GammaPositionManager.on_fill: unknown position {position_id}")
                return
            pos.fill_price = fill_price
            pos.fill_time = fill_time
            pos.status = "ACTIVE"
            pos.trailing_sl = max(pos.trailing_sl, fill_price * 0.88)

    def bind_order_id(self, position_id: str, paper_order_id: str) -> None:
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is not None:
                pos.paper_order_id = paper_order_id

    def paper_order_id_for(self, position_id: str) -> Optional[str]:
        with self._lock:
            pos = self._positions.get(position_id)
            return pos.paper_order_id if pos else None

    def cancel_pending(self, position_id: str) -> None:
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is not None and pos.status == "PENDING":
                del self._positions[position_id]

    def on_price_update(self, symbol: str, price: float) -> List[Tuple[str, str, float]]:
        """Mirror SL/T1 state for strategy stats. PaperEngine owns P&L/journal."""
        events: List[Tuple[str, str, float]] = []
        with self._lock:
            snapshot = list(self._positions.items())

        for pos_id, pos in snapshot:
            if pos.signal.symbol != symbol or pos.status not in {"ACTIVE", "PARTIAL"}:
                continue
            if price <= pos.trailing_sl:
                events.append((pos_id, "SL_HIT", price))
            elif pos.status == "ACTIVE" and price >= pos.signal.target1:
                events.append((pos_id, "T1_HIT", price))

        for pos_id, event, event_price in events:
            self._handle_event(pos_id, event, event_price)
        return events

    def on_bar_close(self, symbol: str, option_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Gamma-specific exits:
        - target2: after T1, close remaining when 5m option close is below EMA9.
        - theta veto: if T1 not hit within expire_after_bars bars.
        """
        if option_df is None or option_df.empty or "close" not in option_df.columns:
            return []
        close = float(pd.to_numeric(option_df["close"], errors="coerce").iloc[-1])
        ema9 = float(ema(option_df["close"], 9).iloc[-1]) if len(option_df) >= 9 else close
        events: List[Tuple[str, str, float]] = []

        with self._lock:
            snapshot = list(self._positions.items())

        for pos_id, pos in snapshot:
            if pos.signal.symbol != symbol or pos.status not in {"ACTIVE", "PARTIAL"}:
                continue
            event = None
            with self._lock:
                current = self._positions.get(pos_id)
                if current is None or current.status not in {"ACTIVE", "PARTIAL"}:
                    continue
                current.bars_since_entry += 1
                if current.status == "PARTIAL" and close < ema9:
                    event = "T2_HIT"
                elif (
                    current.status == "ACTIVE"
                    and current.signal.expire_after_bars > 0
                    and current.bars_since_entry >= current.signal.expire_after_bars
                ):
                    event = "THETA_VETO"
            if event:
                events.append((pos_id, event, close))

        for pos_id, event, event_price in events:
            self._handle_event(pos_id, event, event_price)
        return events

    def _handle_event(self, pos_id: str, event: str, price: float) -> None:
        with self._lock:
            pos = self._positions.get(pos_id)
            if pos is None or pos.status == "CLOSED":
                return
            entry = pos.fill_price or pos.signal.entry
            if event == "T1_HIT":
                t1_qty = max(1, int(round(pos.signal.qty * pos.signal.t1_exit_pct)))
                t1_qty = min(t1_qty, pos.qty_open)
                pos.qty_open -= t1_qty
                pos.qty_t1_booked += t1_qty
                pos.trailing_sl = max(entry, entry * 1.20)
                pos.status = "PARTIAL" if pos.qty_open > 0 else "CLOSED"
            elif event in {"SL_HIT", "T2_HIT", "THETA_VETO"}:
                pos.qty_open = 0
                pos.status = "CLOSED"
            logger.info(f"GammaPositionManager: {pos_id} {event} @ {price:.2f}")

    def get_open_positions(self) -> List[GammaPosition]:
        with self._lock:
            return [p for p in self._positions.values() if p.status in {"PENDING", "ACTIVE", "PARTIAL"}]

    def get_stats(self) -> dict:
        with self._lock:
            closed = sum(1 for p in self._positions.values() if p.status == "CLOSED")
            open_count = sum(1 for p in self._positions.values() if p.status in {"PENDING", "ACTIVE", "PARTIAL"})
        return {
            "total_trades": closed,
            "open_trades": open_count,
            "note": "Detailed P&L handled by PaperEngine",
        }

