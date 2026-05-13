"""
execution/paper_engine.py
---------------------------
Thread-safe paper trading order book for ALL strategies.

Spec compliance:
- Fill price = next bar open (no look-ahead bias)
- SL, T1, T2 managed on every price update
- T1: close 50% qty, move SL to breakeven
- T2: close remaining
- P&L includes cost_per_order_inr × 2 (entry + exit) per position
- All order book mutations under threading.Lock
- get_strategy_stats() returns per-strategy performance dict
"""
from __future__ import annotations

import csv
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategies.base_strategy import Signal


# ---------------------------------------------------------------------------
# Order dataclass
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    strategy: str         # "rsmb" | "existing" | "TrendFollowing" etc.
    symbol: str
    side: str             # "BUY" | "SELL"
    entry: float          # signal entry price (fill happens at next bar open)
    sl: float
    target1: float
    target2: float
    qty: int
    qty_open: int
    qty_t1_booked: int = 0
    fill_price: float = 0.0
    status: str = "PENDING"   # PENDING → ACTIVE → PARTIAL → CLOSED
    pnl_unrealised: float = 0.0
    pnl_realised: float = 0.0
    score: float = 0.0
    rs_rank: Optional[float] = None
    sl_at_breakeven: bool = False
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    outcome: str = ""          # "TARGET_HIT" | "SL_HIT" | "T1_HIT" | "MANUAL"


# ---------------------------------------------------------------------------
# PaperEngine
# ---------------------------------------------------------------------------

class PaperEngine:
    """
    Shared paper order book for all strategies running concurrently.

    Usage
    -----
    engine = PaperEngine()
    order_id = engine.simulate_fill(signal, next_bar_open_price)
    engine.on_price_update("RELIANCE", 2551.5)
    stats = engine.get_strategy_stats("rsmb")
    """

    def __init__(
        self,
        cost_per_order_inr: float = 22.0,
        journal_path: str = "data/trade_journal.csv",
    ) -> None:
        self._orders: Dict[str, Order] = {}
        self._lock = threading.Lock()
        self._cost_per_order = cost_per_order_inr
        self._journal_path = journal_path
        self._ensure_journal_header()

    # ------------------------------------------------------------------
    # Order intake
    # ------------------------------------------------------------------

    def simulate_fill(self, signal: Signal, next_bar_open: float) -> str:
        """
        Accept a signal and fill it at next_bar_open price.

        Parameters
        ----------
        signal        : Signal dataclass from any strategy.
        next_bar_open : Next 15m bar's open price (no look-ahead bias).

        Returns
        -------
        order_id (str)
        """
        order_id = str(uuid.uuid4())[:12]
        now_ist = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        order = Order(
            order_id=order_id,
            strategy=signal.strategy,
            symbol=signal.symbol,
            side=signal.side,
            entry=signal.entry,
            sl=signal.sl,
            target1=signal.target1,
            target2=signal.target2,
            qty=signal.qty,
            qty_open=signal.qty,
            fill_price=next_bar_open,
            score=signal.score,
            rs_rank=signal.rs_rank,
            status="ACTIVE",
            entry_time=now_ist,
        )

        with self._lock:
            self._orders[order_id] = order

        logger.info(
            f"PaperEngine: [{signal.strategy}] {signal.side} {signal.symbol} "
            f"filled @ {next_bar_open:.2f} | id={order_id}"
        )
        return order_id

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def on_price_update(self, symbol: str, price: float) -> List[Tuple[str, str]]:
        """
        Update unrealised P&L and check all active orders for SL/T1/T2 hits.

        Parameters
        ----------
        symbol : NSE symbol string.
        price  : Current market price.

        Returns
        -------
        List of (order_id, event) tuples for any exits triggered.
        """
        events: List[Tuple[str, str]] = []

        with self._lock:
            relevant = [
                (oid, o) for oid, o in self._orders.items()
                if o.symbol == symbol and o.status in ("ACTIVE", "PARTIAL")
            ]

        for order_id, order in relevant:
            side = order.side

            # Update unrealised P&L
            with self._lock:
                if order.status in ("ACTIVE", "PARTIAL"):
                    if side == "BUY":
                        order.pnl_unrealised = (price - order.fill_price) * order.qty_open
                    else:
                        order.pnl_unrealised = (order.fill_price - price) * order.qty_open

            # Check exits
            if side == "BUY":
                if price <= order.sl:
                    self._close_position(order_id, price, "SL_HIT")
                    events.append((order_id, "SL_HIT"))
                elif order.status == "PARTIAL" and price >= order.target2:
                    self._close_position(order_id, price, "TARGET_HIT")
                    events.append((order_id, "T2_HIT"))
                elif order.status == "ACTIVE" and price >= order.target1:
                    self._hit_t1(order_id, price)
                    events.append((order_id, "T1_HIT"))
            else:  # SELL
                if price >= order.sl:
                    self._close_position(order_id, price, "SL_HIT")
                    events.append((order_id, "SL_HIT"))
                elif order.status == "PARTIAL" and price <= order.target2:
                    self._close_position(order_id, price, "TARGET_HIT")
                    events.append((order_id, "T2_HIT"))
                elif order.status == "ACTIVE" and price <= order.target1:
                    self._hit_t1(order_id, price)
                    events.append((order_id, "T1_HIT"))

        return events

    # ------------------------------------------------------------------
    # Exit handlers
    # ------------------------------------------------------------------

    def _hit_t1(self, order_id: str, exit_price: float) -> None:
        """Book 50% of qty at T1, move SL to breakeven."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.status != "ACTIVE":
                return

            t1_qty = max(1, order.qty_open // 2)
            gross = self._calc_gross(order.side, order.fill_price, exit_price, t1_qty)
            net = gross - (self._cost_per_order * 2)

            order.pnl_realised += net
            order.qty_open -= t1_qty
            order.qty_t1_booked = t1_qty
            order.sl = order.fill_price   # SL to breakeven
            order.sl_at_breakeven = True
            order.status = "PARTIAL"
            order.outcome = "T1_HIT"

        logger.info(
            f"PaperEngine: {order_id} T1 HIT @ {exit_price:.2f} "
            f"booked {t1_qty} shares, SL → BE={order.fill_price:.2f}"
        )

    def _close_position(self, order_id: str, exit_price: float, outcome: str) -> None:
        """Close remaining qty, compute final P&L, write to journal."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.status == "CLOSED":
                return

            gross = self._calc_gross(order.side, order.fill_price, exit_price, order.qty_open)
            net = gross - (self._cost_per_order * 2)
            order.pnl_realised += net
            order.qty_open = 0
            order.pnl_unrealised = 0.0
            order.status = "CLOSED"
            order.outcome = outcome
            order.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Take a snapshot for journal write (outside lock)
            journal_row = self._build_journal_row(order, exit_price)

        self._write_to_journal(journal_row)
        logger.info(
            f"PaperEngine: {order_id} {outcome} @ {exit_price:.2f} "
            f"net_pnl={order.pnl_realised:.2f}"
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_strategy_stats(self, strategy: str) -> dict:
        """
        Performance statistics for a specific strategy name.

        Parameters
        ----------
        strategy : "rsmb" | "existing" | "TrendFollowing" | "all"

        Returns
        -------
        dict with keys: net_pnl, win_rate, profit_factor, max_drawdown,
                        expectancy, total_trades, open_trades
        """
        with self._lock:
            if strategy == "all":
                closed = [o for o in self._orders.values() if o.status == "CLOSED"]
                open_orders = [o for o in self._orders.values() if o.status in ("ACTIVE", "PARTIAL")]
            else:
                closed = [
                    o for o in self._orders.values()
                    if o.status == "CLOSED" and o.strategy == strategy
                ]
                open_orders = [
                    o for o in self._orders.values()
                    if o.status in ("ACTIVE", "PARTIAL") and o.strategy == strategy
                ]

        if not closed:
            return {
                "net_pnl": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "expectancy": 0.0,
                "total_trades": 0,
                "open_trades": len(open_orders),
            }

        pnls = [o.pnl_realised for o in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return {
            "net_pnl": sum(pnls),
            "win_rate": len(wins) / len(pnls) * 100 if pnls else 0.0,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
            "total_trades": len(closed),
            "open_trades": len(open_orders),
        }

    def get_active_orders(self, strategy: Optional[str] = None) -> List[Order]:
        with self._lock:
            orders = [
                o for o in self._orders.values()
                if o.status in ("ACTIVE", "PARTIAL")
            ]
        if strategy:
            orders = [o for o in orders if o.strategy == strategy]
        return orders

    def get_closed_orders(self, strategy: Optional[str] = None, limit: int = 100) -> List[Order]:
        with self._lock:
            orders = sorted(
                [o for o in self._orders.values() if o.status == "CLOSED"],
                key=lambda o: o.exit_time or "",
                reverse=True,
            )
        if strategy:
            orders = [o for o in orders if o.strategy == strategy]
        return orders[:limit]

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def _ensure_journal_header(self) -> None:
        os.makedirs(os.path.dirname(self._journal_path) or ".", exist_ok=True)
        if not os.path.exists(self._journal_path) or os.path.getsize(self._journal_path) == 0:
            with open(self._journal_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._journal_fields())
                writer.writeheader()

    @staticmethod
    def _journal_fields() -> list:
        return [
            "date", "symbol", "side", "strategy",
            "entry_price", "exit_price", "qty",
            "pnl_inr", "pnl_after_costs", "outcome", "confidence",
        ]

    @staticmethod
    def _build_journal_row(order: Order, exit_price: float) -> dict:
        return {
            "date": order.exit_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": order.symbol,
            "side": order.side,
            "strategy": order.strategy,
            "entry_price": round(order.fill_price, 4),
            "exit_price": round(exit_price, 4),
            "qty": order.qty,
            "pnl_inr": round(order.pnl_realised, 2),
            "pnl_after_costs": round(order.pnl_realised, 2),  # costs already deducted
            "outcome": order.outcome,
            "confidence": round(order.score, 4),
        }

    def _write_to_journal(self, row: dict) -> None:
        for attempt in range(3):
            try:
                with open(self._journal_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._journal_fields())
                    writer.writerow(row)
                return
            except PermissionError as exc:
                if attempt == 2:
                    logger.error(f"PaperEngine: journal write failed after 3 attempts: {exc}")
                    return
                logger.warning(f"PaperEngine: journal locked, retry {attempt+1}")
                time.sleep(0.2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_gross(side: str, entry: float, exit_price: float, qty: int) -> float:
        if side == "BUY":
            return (exit_price - entry) * qty
        return (entry - exit_price) * qty
