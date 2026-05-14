"""
strategies/rsmb/position_manager.py
--------------------------------------
RSMB position sizing, SL management, and trailing stop logic.

Responsibilities:
- Track open RSMB positions (max 3 simultaneously)
- Update trailing stops every bar (Supertrend-based)
- Trigger partial exit at T1, full exit at T2 or SL
- Thread-safe via threading.Lock
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategies.base_strategy import Signal
from strategies.rsmb.indicators import compute_supertrend


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class RSMBPosition:
    """Represents a live RSMB paper position."""
    position_id: str
    signal: Signal

    # Fill details (updated on fill)
    fill_price: float = 0.0
    fill_time: Optional[pd.Timestamp] = None

    # Qty tracking
    qty_open: int = 0
    qty_t1_booked: int = 0
    sl_moved_to_breakeven: bool = False

    # Current trailing stop (updated each bar)
    trailing_sl: float = 0.0

    # Status
    status: str = "PENDING"   # PENDING → ACTIVE → PARTIAL → CLOSED


# ---------------------------------------------------------------------------
# RSMBPositionManager
# ---------------------------------------------------------------------------

class RSMBPositionManager:
    """
    Manages all open RSMB positions.

    Thread-safety: all mutations use self._lock.
    """

    MAX_OPEN_TRADES = 3

    def __init__(self, cost_per_order_inr: float = 22.0) -> None:
        self._positions: Dict[str, RSMBPosition] = {}
        self._lock = threading.Lock()
        self._cost_per_order = cost_per_order_inr

    # ------------------------------------------------------------------
    # Position intake
    # ------------------------------------------------------------------

    def can_open(self) -> bool:
        """True if fewer than MAX_OPEN_TRADES are currently open."""
        return self.open_count() < self.MAX_OPEN_TRADES

    def open_position(self, signal: Signal) -> Optional[str]:
        """
        Register a new position. Returns position_id or None if at capacity.
        """
        if not self.can_open():
            logger.warning(
                f"RSMBPositionManager: max {self.MAX_OPEN_TRADES} open trades reached; "
                f"signal for {signal.symbol} rejected"
            )
            return None

        pos_id = str(uuid.uuid4())[:8]
        pos = RSMBPosition(
            position_id=pos_id,
            signal=signal,
            qty_open=signal.qty,
            trailing_sl=signal.sl,
            status="PENDING",
        )

        with self._lock:
            self._positions[pos_id] = pos

        logger.info(
            f"RSMBPositionManager: opened {signal.side} {signal.symbol} "
            f"id={pos_id} qty={signal.qty} sl={signal.sl:.2f}"
        )
        return pos_id

    def on_fill(self, position_id: str, fill_price: float, fill_time: pd.Timestamp) -> None:
        """Mark a position as filled at fill_price."""
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                logger.warning(f"RSMBPositionManager.on_fill: unknown position {position_id}")
                return
            pos.fill_price = fill_price
            pos.fill_time = fill_time
            pos.status = "ACTIVE"

        logger.info(
            f"RSMBPositionManager: position {position_id} filled @ {fill_price:.2f}"
        )

    # ------------------------------------------------------------------
    # Price updates — check SL, T1, T2
    # ------------------------------------------------------------------

    def on_price_update(
        self, price: float, symbol: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Check all open positions against the current price.

        Returns a list of (position_id, event, exit_price) tuples where event
        is one of: "SL_HIT", "T1_HIT", "T2_HIT".
        """
        events = []

        with self._lock:
            positions_snapshot = list(self._positions.items())

        for pos_id, pos in positions_snapshot:
            if symbol is not None and pos.signal.symbol != symbol:
                continue
            if pos.status not in ("ACTIVE", "PARTIAL"):
                continue

            side = pos.signal.side
            sl = pos.trailing_sl
            t1 = pos.signal.target1
            t2 = pos.signal.target2

            if side == "BUY":
                if price <= sl:
                    events.append((pos_id, "SL_HIT", price))
                elif price >= t2 and pos.status == "PARTIAL":
                    events.append((pos_id, "T2_HIT", price))
                elif price >= t1 and pos.status == "ACTIVE":
                    events.append((pos_id, "T1_HIT", price))
            else:  # SELL
                if price >= sl:
                    events.append((pos_id, "SL_HIT", price))
                elif price <= t2 and pos.status == "PARTIAL":
                    events.append((pos_id, "T2_HIT", price))
                elif price <= t1 and pos.status == "ACTIVE":
                    events.append((pos_id, "T1_HIT", price))

        # Process events
        for pos_id, event, exit_price in events:
            self._handle_event(pos_id, event, exit_price)

        return events

    def _handle_event(self, pos_id: str, event: str, exit_price: float) -> None:
        with self._lock:
            pos = self._positions.get(pos_id)
            if pos is None or pos.status in ("CLOSED",):
                return

            entry = pos.fill_price if pos.fill_price else pos.signal.entry
            multiplier = 1  # equity shares — cost per order is flat

            if event == "T1_HIT":
                # Book 50% qty (Module 2 fix: P&L handled by paper_engine)
                t1_qty = max(1, pos.qty_open // 2)
                pos.qty_open -= t1_qty
                pos.qty_t1_booked = t1_qty
                pos.sl_moved_to_breakeven = True
                pos.trailing_sl = entry   # SL → breakeven
                pos.status = "PARTIAL"
                logger.info(
                    f"RSMBPositionManager: {pos_id} T1 HIT @ {exit_price:.2f} "
                    f"booked {t1_qty} shares, SL moved to BE={entry:.2f}"
                )

            elif event in ("T2_HIT", "SL_HIT"):
                # Module 2 fix: P&L handled by paper_engine
                pos.qty_open = 0
                pos.status = "CLOSED"
                logger.info(
                    f"RSMBPositionManager: {pos_id} {event} @ {exit_price:.2f}"
                )

    def update_trailing_stop(
        self, position_id: str, df_15m: pd.DataFrame
    ) -> None:
        """
        Update the trailing stop for a PARTIAL position using Supertrend(3,10).

        Call this on every new 15m bar close.
        """
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None or pos.status != "PARTIAL":
                return
            current_sl = pos.trailing_sl
            side = pos.signal.side

        supertrend = compute_supertrend(df_15m, factor=3.0, period=10)
        if supertrend.empty or pd.isna(supertrend.iloc[-1]):
            return

        new_sl = float(supertrend.iloc[-1])

        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                return
            if side == "BUY":
                # Trail up: never move SL below current level
                if new_sl > pos.trailing_sl:
                    pos.trailing_sl = new_sl
                    logger.debug(
                        f"RSMBPositionManager: {position_id} trailing SL → {new_sl:.2f}"
                    )
            else:  # SELL: trail down
                if new_sl < pos.trailing_sl:
                    pos.trailing_sl = new_sl
                    logger.debug(
                        f"RSMBPositionManager: {position_id} trailing SL → {new_sl:.2f}"
                    )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_open_positions(self) -> List[RSMBPosition]:
        with self._lock:
            return [
                p for p in self._positions.values()
                if p.status in ("PENDING", "ACTIVE", "PARTIAL")
            ]

    def get_closed_positions(self) -> List[RSMBPosition]:
        with self._lock:
            return [p for p in self._positions.values() if p.status == "CLOSED"]

    def get_position(self, position_id: str) -> Optional[RSMBPosition]:
        with self._lock:
            return self._positions.get(position_id)

    def open_count(self) -> int:
        with self._lock:
            return sum(
                1 for p in self._positions.values()
                if p.status in ("PENDING", "ACTIVE", "PARTIAL")
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return performance stats for the RSMB strategy."""
        with self._lock:
            closed_count = sum(1 for p in self._positions.values() if p.status == "CLOSED")

        return {
            "total_trades": closed_count,
            "open_trades": self.open_count(),
            "note": "Detailed P&L handled by PaperEngine"
        }
