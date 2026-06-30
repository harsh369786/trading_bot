"""
execution/paper_engine.py
---------------------------
Thread-safe paper trading order book for ALL strategies.

Spec compliance:
- Fill price = next bar open (no look-ahead bias)
- SL, T1, T2 managed on every price update
- T1: close 50% qty, move SL to breakeven
- T2: close remaining
- P&L deducts cost_per_order_inr per brokerage leg:
- Full-exit trades: 2× total (Entry + Exit)
- T1+T2 trades: 3× total (Entry + T1 + T2)
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from strategies.base_strategy import Signal
from notifications.email_notifier import EmailNotifier


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
    pnl_gross_realised: float = 0.0
    pnl_realised: float = 0.0
    score: float = 0.0
    rs_rank: Optional[float] = None
    sl_at_breakeven: bool = False
    expire_after_bars: int = 0
    t1_exit_pct: float = 0.5
    target2_mode: str = "price"
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
        active_orders_path: str = "data/paper_orders.json",
        config: dict | None = None,
    ) -> None:
        self._orders: Dict[str, Order] = {}
        self._lock = threading.RLock()
        self._cost_per_order = cost_per_order_inr
        self._journal_path = journal_path
        self._active_orders_path = active_orders_path
        self.config = config or {}
        self._load_active_orders()
        self._ensure_journal_header()
        self._write_active_snapshot()
        self.notifier = EmailNotifier()

    def _load_active_orders(self) -> None:
        """Reload ACTIVE/PARTIAL orders from disk to survive bot restarts."""
        if not os.path.exists(self._active_orders_path):
            return
        try:
            with open(self._active_orders_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                return

            with self._lock:
                for row in rows:
                    try:
                        order_id = row["order_id"]
                        order = Order(
                            order_id=order_id,
                            strategy=row["strategy"],
                            symbol=row["symbol"],
                            side=row["side"],
                            entry=float(row.get("signal_entry", row.get("entry"))),
                            sl=float(row["sl"]),
                            target1=float(row.get("target1", row.get("target", 0))),
                            target2=float(row.get("target2", row.get("target", 0))),
                            qty=int(row["qty"]),
                            qty_open=int(row["qty_open"]),
                            qty_t1_booked=int(row.get("qty_t1_booked", 0)),
                            fill_price=float(row.get("fill_price", row.get("entry"))),
                            score=row.get("confidence", 0),
                            status=row["status"],
                            pnl_unrealised=row.get("pnl_unrealised", 0),
                            pnl_gross_realised=row.get("pnl_gross_realised", 0),
                            pnl_realised=row.get("pnl_realised", 0),
                            sl_at_breakeven=bool(row.get("sl_at_breakeven", False)),
                            expire_after_bars=int(row.get("expire_after_bars", 0) or 0),
                            t1_exit_pct=float(row.get("t1_exit_pct", 0.5) or 0.5),
                            target2_mode=str(row.get("target2_mode", "price") or "price"),
                            entry_time=row.get("entry_time"),
                        )
                        self._orders[order_id] = order
                    except KeyError as e:
                        logger.warning(f"PaperEngine: skipping malformed order row during reload: {e}")
            logger.info(f"PaperEngine: reloaded {len(self._orders)} active orders from disk.")
        except Exception as exc:
            logger.error(f"PaperEngine: failed to reload active orders: {exc}")

    # ------------------------------------------------------------------
    # Order intake
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_signal_object(signal: Signal | dict) -> Signal | SimpleNamespace:
        if not isinstance(signal, dict):
            return signal
        target1 = signal.get("target1", signal.get("target", signal.get("t1")))
        target2 = signal.get("target2", target1)
        return SimpleNamespace(
            strategy=signal.get("strategy", "Unknown"),
            symbol=signal.get("symbol"),
            side=signal.get("side"),
            entry=signal.get("entry"),
            sl=signal.get("sl"),
            target1=target1,
            target2=target2,
            qty=signal.get("qty", signal.get("lots", 1)),
            score=signal.get("score", signal.get("confidence", 0.0)),
            rs_rank=signal.get("rs_rank"),
            expire_after_bars=signal.get("expire_after_bars", 0),
            t1_exit_pct=signal.get("t1_exit_pct", 0.5),
            target2_mode=signal.get("target2_mode", "price"),
        )

    def _domain_for_strategy(self, strategy: str) -> str:
        name = str(strategy or "")
        if name == "gamma_scalper":
            return "gamma"
        if name == "mean_reversion":
            return "mean_reversion"
        return "equity"

    def _max_open_for_domain(self, domain: str) -> int:
        capital_cfg = self.config.get("capital", {}) if isinstance(self.config, dict) else {}
        risk_cfg = self.config.get("risk", {}) if isinstance(self.config, dict) else {}
        if domain == "gamma":
            return int(risk_cfg.get("gamma_max_open_trades", capital_cfg.get("gamma_max_open_trades", 2)))
        if domain == "mean_reversion":
            return int(risk_cfg.get("meanrev_max_open_trades", capital_cfg.get("meanrev_max_open_trades", 3)))
        return int(capital_cfg.get("max_open_trades_equity", 2))

    def _check_entry_limits(self, signal: Signal | SimpleNamespace) -> tuple[bool, str]:
        symbol = str(getattr(signal, "symbol", "") or "")
        strategy = str(getattr(signal, "strategy", "") or "")
        domain = self._domain_for_strategy(strategy)
        max_open = self._max_open_for_domain(domain)

        with self._lock:
            active = [o for o in self._orders.values() if o.status in ("ACTIVE", "PARTIAL")]
            if any(o.symbol == symbol for o in active):
                return False, f"paper trade already active for {symbol}"
            domain_open = sum(1 for o in active if self._domain_for_strategy(o.strategy) == domain)

        if max_open > 0 and domain_open >= max_open:
            return False, f"{domain} max open trades reached: {domain_open}/{max_open}"
        return True, "allowed"

    def simulate_fill(self, signal: Signal | dict, next_bar_open: float) -> str | None:
        """
        Accept a signal and fill it at next_bar_open price.

        Parameters
        ----------
        signal        : Signal dataclass or normalized signal dict from any strategy.
        next_bar_open : Next 15m bar's open price (no look-ahead bias).

        Returns
        -------
        order_id (str)
        """
        signal = self._normalize_signal_object(signal)
        allowed, reason = self._check_entry_limits(signal)
        if not allowed:
            logger.warning(f"PaperEngine: rejected {signal.strategy} {signal.symbol}: {reason}")
            return None

        order_id = str(uuid.uuid4())[:12]
        now_ist = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = float(signal.entry)
        fill = float(next_bar_open)
        sl_dist = abs(entry - float(signal.sl))
        t1_dist = abs(float(signal.target1) - entry)
        t2_dist = abs(float(signal.target2) - entry)
        if str(signal.side).upper() == "BUY":
            sl = fill - sl_dist
            target1 = fill + t1_dist
            target2 = fill + t2_dist
        else:
            sl = fill + sl_dist
            target1 = fill - t1_dist
            target2 = fill - t2_dist

        order = Order(
            order_id=order_id,
            strategy=signal.strategy,
            symbol=signal.symbol,
            side=signal.side,
            entry=signal.entry,
            sl=sl,
            target1=target1,
            target2=target2,
            qty=signal.qty,
            qty_open=signal.qty,
            fill_price=fill,
            score=signal.score,
            rs_rank=signal.rs_rank,
            status="ACTIVE",
            entry_time=now_ist,
            pnl_realised=-self._cost_per_order,
            expire_after_bars=int(getattr(signal, "expire_after_bars", 0) or 0),
            t1_exit_pct=float(getattr(signal, "t1_exit_pct", 0.5) or 0.5),
            target2_mode=str(getattr(signal, "target2_mode", "price") or "price"),
        )

        with self._lock:
            self._orders[order_id] = order

        self._write_active_snapshot()
        logger.info(
            f"PaperEngine: [{signal.strategy}] {signal.side} {signal.symbol} "
            f"filled @ {fill:.2f} | id={order_id}"
        )

        # Send async-friendly email notification in background thread
        threading.Thread(
            target=self.notifier.send_trade_fill,
            args=(
                signal.symbol, signal.side, signal.strategy,
                fill, sl, target1, signal.score
            ),
            daemon=True
        ).start()

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
            event = None
            with self._lock:
                if order.status not in ("ACTIVE", "PARTIAL"):
                    continue

                side = order.side
                if side == "BUY":
                    order.pnl_unrealised = (price - order.fill_price) * order.qty_open
                    if price <= order.sl:
                        event = "SL_HIT"
                    elif order.status == "PARTIAL" and order.target2_mode == "price" and price >= order.target2:
                        event = "T2_HIT"
                    elif order.status == "ACTIVE" and price >= order.target1:
                        event = "T1_HIT"
                else:
                    order.pnl_unrealised = (order.fill_price - price) * order.qty_open
                    if price >= order.sl:
                        event = "SL_HIT"
                    elif order.status == "PARTIAL" and order.target2_mode == "price" and price <= order.target2:
                        event = "T2_HIT"
                    elif order.status == "ACTIVE" and price <= order.target1:
                        event = "T1_HIT"

            if event == "T1_HIT":
                if self._hit_t1(order_id, price):
                    events.append((order_id, event))
            elif event in {"SL_HIT", "T2_HIT"}:
                outcome = "TARGET_HIT" if event == "T2_HIT" else event
                if self._close_position(order_id, price, outcome):
                    events.append((order_id, event))

        return events

    # ------------------------------------------------------------------
    # Exit handlers
    # ------------------------------------------------------------------

    def _hit_t1(self, order_id: str, exit_price: float) -> bool:
        """Book 50% of qty at T1, move SL to breakeven."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.status != "ACTIVE":
                return False

            pct = min(max(float(order.t1_exit_pct or 0.5), 0.0), 1.0)
            t1_qty = max(1, int(round(order.qty * pct)))
            t1_qty = min(t1_qty, order.qty_open)
            gross = self._calc_gross(order.side, order.fill_price, exit_price, t1_qty)
            net = gross - self._cost_per_order

            order.pnl_gross_realised += gross
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

        self._write_active_snapshot()
        return True

    def close_position(self, order_id: str, exit_price: float, outcome: str = "MANUAL") -> bool:
        """Public close hook for strategy-managed exits such as theta veto."""
        return self._close_position(order_id, exit_price, outcome)

    def _close_position(self, order_id: str, exit_price: float, outcome: str) -> bool:
        """Close remaining qty, compute final P&L, write to journal."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.status == "CLOSED":
                return False

            gross = self._calc_gross(order.side, order.fill_price, exit_price, order.qty_open)
            net = gross - self._cost_per_order
            order.pnl_gross_realised += gross
            order.pnl_realised += net
            order.qty_open_before = order.qty_open  # stash for email
            order.qty_open = 0
            order.pnl_unrealised = 0.0
            order.status = "CLOSED"
            order.outcome = outcome
            order.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Take a snapshot for journal write (outside lock)
            journal_row = self._build_journal_row(order, exit_price)

        self._write_to_journal(journal_row)
        self._write_active_snapshot()
        logger.info(
            f"PaperEngine: {order_id} {outcome} @ {exit_price:.2f} "
            f"net_pnl={order.pnl_realised:.2f}"
        )

        threading.Thread(
            target=self.notifier.send_trade_close,
            args=(
                order.symbol, order.side, order.strategy,
                order.fill_price, exit_price, order.pnl_realised,
                outcome, order.qty_open_before if hasattr(order, 'qty_open_before') else order.qty
            ),
            daemon=True
        ).start()

        return True

    def square_off_all(self, current_prices: Dict[str, float]) -> List[Tuple[str, str]]:
        """Close all ACTIVE/PARTIAL positions at the provided market prices."""
        events: List[Tuple[str, str]] = []
        with self._lock:
            active_ids = [
                oid for oid, o in self._orders.items()
                if o.status in ("ACTIVE", "PARTIAL")
            ]

        if not active_ids:
            return events

        logger.warning(f"PaperEngine: Squaring off {len(active_ids)} positions at market close.")
        for oid in active_ids:
            with self._lock:
                symbol = self._orders[oid].symbol

            price = current_prices.get(symbol)
            if price is None:
                with self._lock:
                    price = self._orders[oid].fill_price
                logger.warning(
                    f"PaperEngine: No market price for {symbol} at EOD; "
                    f"using fill_price fallback={price:.2f}. Check tick/LTP feed."
                )

            if self._close_position(oid, price, "EOD_SQUAREOFF"):
                events.append((oid, "EOD_SQUAREOFF"))
        return events

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

    def get_order_snapshot(self, order_id: str) -> Optional[Order]:
        """Return the current in-memory order object for accounting callbacks."""
        with self._lock:
            return self._orders.get(order_id)

    def _active_snapshot_rows(self) -> List[dict]:
        with self._lock:
            active = [
                o for o in self._orders.values()
                if o.status in ("ACTIVE", "PARTIAL")
            ]
            return [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "strategy": o.strategy,
                    "signal_entry": round(o.entry, 4),
                    "fill_price": round(o.fill_price or o.entry, 4),
                    "entry": round(o.fill_price or o.entry, 4),
                    "sl": round(o.sl, 4),
                    "target": round(o.target1, 4),
                    "target1": round(o.target1, 4),
                    "target2": round(o.target2, 4),
                    "qty": o.qty,
                    "qty_open": o.qty_open,
                    "qty_t1_booked": o.qty_t1_booked,
                    "status": o.status,
                    "pnl_unrealised": round(o.pnl_unrealised, 2),
                    "pnl_gross_realised": round(o.pnl_gross_realised, 2),
                    "pnl_realised": round(o.pnl_realised, 2),
                    "sl_at_breakeven": o.sl_at_breakeven,
                    "expire_after_bars": o.expire_after_bars,
                    "t1_exit_pct": o.t1_exit_pct,
                    "target2_mode": o.target2_mode,
                    "confidence": round(o.score, 4),
                    "source": "paper_engine",
                    "entry_time": o.entry_time,
                }
                for o in active
            ]

    def _write_active_snapshot(self) -> None:
        os.makedirs(os.path.dirname(self._active_orders_path) or ".", exist_ok=True)
        rows = self._active_snapshot_rows()
        tmp_path = f"{self._active_orders_path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            os.replace(tmp_path, self._active_orders_path)
        except Exception as exc:
            logger.warning(f"PaperEngine: active order snapshot write failed: {exc}")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def _ensure_journal_header(self) -> None:
        os.makedirs(os.path.dirname(self._journal_path) or ".", exist_ok=True)
        fields = self._journal_fields()
        if not os.path.exists(self._journal_path) or os.path.getsize(self._journal_path) == 0:
            with open(self._journal_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
            return

        try:
            with open(self._journal_path, newline="") as f:
                rows = list(csv.reader(f))
            if not rows or rows[0] == fields:
                return

            old_fields = [
                "date", "symbol", "side", "entry_price", "exit_price",
                "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence",
            ]
            if rows[0] != old_fields:
                return

            repaired = [fields]
            for row in rows[1:]:
                if not row:
                    continue
                if len(row) == len(old_fields):
                    repaired.append(row[:3] + ["Unknown"] + row[3:])
                elif len(row) == len(fields):
                    repaired.append(row)

            tmp_path = f"{self._journal_path}.{uuid.uuid4().hex}.tmp"
            with open(tmp_path, "w", newline="") as f:
                csv.writer(f).writerows(repaired)
            os.replace(tmp_path, self._journal_path)
            logger.warning("PaperEngine: repaired trade journal header to include strategy column.")
        except Exception as exc:
            logger.warning(f"PaperEngine: trade journal schema check failed: {exc}")

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
            "pnl_inr": round(order.pnl_gross_realised, 2),
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
