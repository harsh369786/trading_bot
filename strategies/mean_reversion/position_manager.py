"""
strategies/mean_reversion/position_manager.py
---------------------------------------------
Thread-safe position state for 15m 200-MA mean reversion.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategies.base_strategy import Signal


@dataclass
class MeanRevPosition:
    position_id: str
    signal: Signal
    fill_price: float = 0.0
    fill_time: Optional[pd.Timestamp] = None
    qty_open: int = 0
    qty_t1_booked: int = 0
    trailing_sl: float = 0.0
    status: str = "PENDING"
    paper_order_id: Optional[str] = None


class MeanRevPositionManager:
    MAX_OPEN_TRADES = 3

    def __init__(self, max_open_trades: int = MAX_OPEN_TRADES) -> None:
        self.max_open_trades = int(max_open_trades or self.MAX_OPEN_TRADES)
        self._positions: Dict[str, MeanRevPosition] = {}
        self._lock = threading.Lock()

    def can_open(self, symbol: str | None = None) -> bool:
        with self._lock:
            open_positions = [
                p for p in self._positions.values()
                if p.status in {"PENDING", "ACTIVE", "PARTIAL"}
            ]
            if len(open_positions) >= self.max_open_trades:
                return False
            if symbol is not None:
                return not any(p.signal.symbol == symbol for p in open_positions)
            return True

    def open_position(self, signal: Signal) -> Optional[str]:
        if not self.can_open(signal.symbol):
            logger.warning(f"MeanRevPositionManager: cannot open {signal.symbol}; active symbol or max reached")
            return None
        pos_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._positions[pos_id] = MeanRevPosition(
                position_id=pos_id,
                signal=signal,
                qty_open=signal.qty,
                trailing_sl=signal.sl,
                status="PENDING",
            )
        return pos_id

    def on_fill(self, position_id: str, fill_price: float, fill_time: pd.Timestamp) -> None:
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                logger.warning(f"MeanRevPositionManager.on_fill: unknown position {position_id}")
                return
            pos.fill_price = fill_price
            pos.fill_time = fill_time
            pos.status = "ACTIVE"

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
        events: List[Tuple[str, str, float]] = []
        with self._lock:
            snapshot = list(self._positions.items())

        for pos_id, pos in snapshot:
            if pos.signal.symbol != symbol or pos.status not in {"ACTIVE", "PARTIAL"}:
                continue
            side = pos.signal.side
            if side == "BUY":
                if price <= pos.trailing_sl:
                    events.append((pos_id, "SL_HIT", price))
                elif pos.status == "PARTIAL" and price >= pos.signal.target2:
                    events.append((pos_id, "T2_HIT", price))
                elif pos.status == "ACTIVE" and price >= pos.signal.target1:
                    events.append((pos_id, "T1_HIT", price))
            else:
                if price >= pos.trailing_sl:
                    events.append((pos_id, "SL_HIT", price))
                elif pos.status == "PARTIAL" and price <= pos.signal.target2:
                    events.append((pos_id, "T2_HIT", price))
                elif pos.status == "ACTIVE" and price <= pos.signal.target1:
                    events.append((pos_id, "T1_HIT", price))

        for pos_id, event, event_price in events:
            self._handle_event(pos_id, event, event_price)
        return events

    def _handle_event(self, pos_id: str, event: str, price: float) -> None:
        with self._lock:
            pos = self._positions.get(pos_id)
            if pos is None or pos.status == "CLOSED":
                return
            if event == "T1_HIT":
                t1_qty = max(1, int(round(pos.signal.qty * pos.signal.t1_exit_pct)))
                t1_qty = min(t1_qty, pos.qty_open)
                pos.qty_open -= t1_qty
                pos.qty_t1_booked += t1_qty
                pos.trailing_sl = pos.fill_price or pos.signal.entry
                pos.status = "PARTIAL" if pos.qty_open > 0 else "CLOSED"
            elif event in {"T2_HIT", "SL_HIT"}:
                pos.qty_open = 0
                pos.status = "CLOSED"
            logger.info(f"MeanRevPositionManager: {pos_id} {event} @ {price:.2f}")

    def get_open_positions(self) -> List[MeanRevPosition]:
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

