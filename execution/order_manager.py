import json
import math
import os
import time
import uuid
from datetime import datetime
from typing import Dict

from loguru import logger

from .broker_api import BaseBroker, MockBroker
from risk.risk_engine import RiskEngine

# Default round-trip cost fraction (brokerage + STT + exchange + GST approximation)
# Conservative estimate: 0.06% per side = 0.12% round trip for equity intraday.
_DEFAULT_COST_FRACTION = 0.0006  # per side


class OrderManager:
    """
    Handles order lifecycle: entry, stop-loss, and targets.
    State is persisted in Redis to survive restarts.
    """
    def __init__(self, config: dict, redis_client=None, broker: BaseBroker = None, risk_engine=None):
        self.config = config
        self.redis = redis_client
        self.broker = broker or MockBroker()
        self.KEY_ACTIVE = "bot:execution:active_orders"
        self.risk_engine = risk_engine or RiskEngine(config, redis_client)
        self.paper_mode = bool(config.get("paper_mode", True))
        self.live_enabled = os.environ.get("TRADING_MODE", "paper").strip().lower() == "live"
        # Configurable cost fraction (M4 fix)
        self.cost_fraction = float(
            config.get("execution", {}).get("cost_fraction_per_side", _DEFAULT_COST_FRACTION)
        )
        # Initialize journal file with header exactly once at construction
        self._journal_path = "data/trade_journal.csv"
        self._ensure_journal_header()

    async def _get_active_orders(self) -> Dict[str, dict]:
        if not self.redis:
            return {}
        data = await self.redis.get(self.KEY_ACTIVE)
        return json.loads(data) if data else {}

    async def get_active_orders(self) -> Dict[str, dict]:
        """Public read-only accessor for active order state."""
        return await self._get_active_orders()

    async def _save_active_orders(self, orders: Dict[str, dict]):
        if not self.redis:
            return
        await self.redis.set(self.KEY_ACTIVE, json.dumps(orders, default=str))

    async def _acquire_symbol_lock(self, symbol: str) -> bool:
        if not self.redis:
            return True
        try:
            return bool(await self.redis.set(f"bot:lock:order:{symbol}", "1", nx=True, ex=10))
        except TypeError:
            # Test fakes may not implement NX; production Redis does.
            return True

    async def _release_symbol_lock(self, symbol: str) -> None:
        if not self.redis:
            return
        try:
            await self.redis.delete(f"bot:lock:order:{symbol}")
        except AttributeError:
            pass

    async def reconcile_startup_state(self):
        """
        Clean stale or incompatible persisted orders before accepting new signals.
        Paper PENDING orders should never survive a restart because paper fills are immediate.
        """
        active = await self._get_active_orders()
        if not active:
            return

        cleaned = {}
        dropped = 0
        seen_symbols = set()

        for order_id, order in active.items():
            if not isinstance(order, dict):
                dropped += 1
                continue

            status = order.get("status")
            symbol = order.get("symbol")
            if self.paper_mode and status == "PENDING":
                dropped += 1
                continue
            if not symbol or status not in {"PENDING", "PROTECTED"}:
                dropped += 1
                continue

            target = order.get("target", order.get("t1"))
            try:
                order["entry"] = float(order["entry"])
                order["sl"] = float(order["sl"])
                order["target"] = float(target)
                order["t1"] = float(target)
                order["qty"] = int(float(order.get("qty", order.get("lots", 1))))
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue

            strategy = order.get("strategy", "Unknown")
            seen_key = (symbol, strategy)
            if seen_key in seen_symbols:
                logger.warning(
                    f"Startup reconcile: dropping duplicate {symbol} order for {strategy} "
                    f"order_id={order_id} (first occurrence kept)"
                )
                dropped += 1
                continue
            seen_symbols.add(seen_key)
            order["order_id"] = order.get("order_id", order_id)
            order["domain"] = order.get("domain") or self._domain_for_symbol(symbol)
            cleaned[order_id] = order

        if dropped:
            logger.warning(f"Startup reconciliation dropped {dropped} stale/invalid active orders.")
        if len(cleaned) != len(active):
            await self._save_active_orders(cleaned)

    def _domain_for_symbol(self, symbol: str) -> str:
        currency_symbols = set(self.config.get("instruments", {}).get("currency", []))
        return "currency" if symbol in currency_symbols else "equity"

    @staticmethod
    def _is_paper_fallback_strategy(strategy: str | None) -> bool:
        strategy_name = str(strategy or "")
        return strategy_name.endswith("_PAPER_RELAXED") or strategy_name.endswith("_PAPER_TECHNICAL")

    def _normalize_signal(self, signal: dict) -> dict | None:
        """Normalize old/new signal field names into one safe order schema."""
        if not isinstance(signal, dict):
            logger.warning(f"Rejected non-dict signal: {type(signal)}")
            return None

        normalized = dict(signal)
        missing = [key for key in ["symbol", "side", "entry", "sl"] if key not in normalized]
        if missing:
            logger.warning(f"Rejected signal missing required fields {missing}: {signal}")
            return None

        normalized["side"] = str(normalized["side"]).upper()
        if normalized["side"] not in {"BUY", "SELL"}:
            logger.warning(f"Rejected signal with invalid side: {normalized.get('side')}")
            return None

        target = (
            normalized.get("target")
            or normalized.get("t1")
            or normalized.get("target_price")
            or normalized.get("take_profit")
        )
        if target is None:
            logger.warning(f"Rejected signal without target/t1: {signal}")
            return None

        qty = normalized.get("qty")
        if qty is None:
            lots = normalized.get("lots")
            qty = int(lots) if lots is not None else 1

        try:
            entry = float(normalized["entry"])
            sl = float(normalized["sl"])
            target = float(target)
            qty = int(float(qty))
        except (TypeError, ValueError):
            logger.warning(f"Rejected signal with non-numeric order fields: {signal}")
            return None

        if not all(math.isfinite(value) for value in [entry, sl, target]) or qty <= 0:
            logger.warning(f"Rejected signal with invalid price/qty values: {signal}")
            return None

        side = normalized["side"]
        if side == "BUY" and not (sl < entry < target):
            logger.warning(f"Rejected BUY signal with invalid SL/target relation: {signal}")
            return None
        if side == "SELL" and not (target < entry < sl):
            logger.warning(f"Rejected SELL signal with invalid SL/target relation: {signal}")
            return None

        normalized.update({
            "entry": entry,
            "sl": sl,
            "target": target,
            "t1": target,
            "qty": qty,
            "domain": normalized.get("domain") or self._domain_for_symbol(str(normalized["symbol"])),
        })
        return normalized

    async def execute_signal(self, signal: dict):
        """
        Main entry point for a validated signal.
        Compatible schema: {symbol, side, qty/lots, entry, sl, target/t1}.
        """
        signal = self._normalize_signal(signal)
        if signal is None:
            return

        if not self.paper_mode and not self.live_enabled:
            logger.critical("Live order blocked: set TRADING_MODE=live explicitly before disabling paper_mode.")
            return

        symbol = signal["symbol"]
        side = signal["side"]
        domain = signal["domain"]

        lock_acquired = await self._acquire_symbol_lock(symbol)
        if not lock_acquired:
            logger.warning(f"Skipping signal for {symbol}: order lock already held.")
            return

        try:
            if not await self.risk_engine.check_circuit_breakers(domain):
                logger.warning(f"Risk engine blocked {domain} signal for {symbol}.")
                return

            active_orders = await self._get_active_orders()
            for active in active_orders.values():
                if active.get("symbol") == symbol and active.get("status") in ["PENDING", "PROTECTED"]:
                    logger.warning(f"Skipping signal for {symbol}: trade already active.")
                    return

            logger.info(f"Executing {side} on {symbol} | Qty: {signal['qty']}")

            entry_res = self.broker.place_order(
                symbol=symbol,
                qty=signal["qty"],
                direction=side,
                order_type="MARKET" if self.paper_mode else "LIMIT",
                price=signal["entry"],
            )

            if entry_res.get("status") != "SUCCESS":
                logger.error(f"Entry failed for {symbol}: {entry_res.get('reason') or entry_res.get('message')}")
                return

            order_id = entry_res["order_id"]
            active = await self._get_active_orders()
            active[order_id] = {**signal, "status": "PENDING", "order_id": order_id}
            await self._save_active_orders(active)
            await self.risk_engine.update_stats(
                domain,
                trade_delta=1,
                count_daily_trade=not self._is_paper_fallback_strategy(signal.get("strategy")),
            )

            logger.info(f"Entry order {order_id} placed. Waiting for fill...")

            if self.paper_mode:
                logger.debug(f"Paper mode: simulating immediate fill for {order_id}")
                await self.handle_order_update({
                    "order_id": order_id,
                    "status": "FILLED",
                    "price": entry_res.get("fill_price", signal["entry"]),
                    "is_exit": False,
                })
        finally:
            await self._release_symbol_lock(symbol)

    def _ensure_journal_header(self):
        """Write CSV header exactly once when the file doesn't yet exist (C5 fix)."""
        import csv
        fieldnames = [
            "date", "symbol", "side", "strategy", "entry_price", "exit_price",
            "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence",
        ]
        os.makedirs(os.path.dirname(self._journal_path) or ".", exist_ok=True)
        if not os.path.exists(self._journal_path) or os.path.getsize(self._journal_path) == 0:
            with open(self._journal_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return

        try:
            with open(self._journal_path, newline="") as f:
                rows = list(csv.reader(f))
            old_fieldnames = [
                "date", "symbol", "side", "entry_price", "exit_price",
                "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence",
            ]
            if not rows or rows[0] == fieldnames or rows[0] != old_fieldnames:
                return

            repaired = [fieldnames]
            for row in rows[1:]:
                if not row:
                    continue
                if len(row) == len(old_fieldnames):
                    repaired.append(row[:3] + ["Unknown"] + row[3:])
                elif len(row) == len(fieldnames):
                    repaired.append(row)

            tmp_path = f"{self._journal_path}.{uuid.uuid4().hex}.tmp"
            with open(tmp_path, "w", newline="") as f:
                csv.writer(f).writerows(repaired)
            os.replace(tmp_path, self._journal_path)
            logger.warning("Repaired trade journal header to include strategy column.")
        except Exception as exc:
            logger.warning(f"Trade journal schema check failed: {exc}")

    async def _place_protection_orders(self, parent_id: str, signal: dict):
        """Place SL-M and target limit orders. H3: results are now checked."""
        symbol = signal["symbol"]
        qty = int(signal["qty"])
        exit_side = "SELL" if signal["side"] == "BUY" else "BUY"

        sl_res = self.broker.place_order(
            symbol=symbol,
            qty=qty,
            direction=exit_side,
            order_type="SL-M",
            price=signal["sl"],
        )
        if sl_res.get("status") != "SUCCESS":
            logger.error(
                f"SL-M order FAILED for {symbol} — position has NO stop loss! "
                f"Reason: {sl_res.get('reason') or sl_res.get('message')}"
            )

        target_qty = max(1, int(qty * 0.6)) if qty > 0 else 0
        if target_qty > 0:
            tgt_res = self.broker.place_order(
                symbol=symbol,
                qty=target_qty,
                direction=exit_side,
                order_type="LIMIT",
                price=signal["target"],
            )
            if tgt_res.get("status") != "SUCCESS":
                logger.warning(f"Target order failed for {symbol}: {tgt_res.get('reason') or tgt_res.get('message')}")

        logger.info(f"Protection orders active for {symbol} | SL: {signal['sl']} | Target: {signal['target']}")


    async def handle_order_update(self, update: dict):
        """
        Handle broker or tracker order update events.
        Exit updates may arrive as SL_HIT/TARGET_HIT from the lifecycle tracker.
        """
        order_id = update.get("order_id")
        status = update.get("status")
        active = await self._get_active_orders()

        if order_id not in active:
            return

        trade = active[order_id]
        if status == "FILLED" and trade.get("status") == "PENDING":
            logger.info(f"Entry {order_id} filled. Placing protection orders...")
            await self._place_protection_orders(order_id, trade)
            trade["status"] = "PROTECTED"
            await self._save_active_orders(active)
            return

        if update.get("is_exit"):
            outcome = status or "CLOSED"
            exit_price = update.get("price")
            logger.info(f"Exit update {outcome} for {order_id}. Trade closed.")
            self._log_to_journal(trade, exit_price, outcome)
            # Calculate pnl_r for RiskEngine circuit breakers (Module 3 fix)
            pnl_inr = self._calculate_pnl(trade, exit_price)
            risk_amount = self.config.get("capital", {}).get("equity_total", 50000) * \
                          self.config.get("capital", {}).get("risk_per_trade_pct", 1.0) / 100
            pnl_r = pnl_inr / risk_amount if risk_amount > 0 else 0

            await self.risk_engine.update_stats(
                trade.get("domain", self._domain_for_symbol(trade["symbol"])),
                pnl_r=pnl_r,
                pnl_inr=pnl_inr,
                trade_delta=-1,
            )
            del active[order_id]
            await self._save_active_orders(active)

    def _calculate_pnl(self, trade: dict, exit_price: float | None) -> float:
        """Gross P&L before costs."""
        if exit_price is None:
            return 0.0
        qty = int(trade.get("qty", 1))
        multiplier = 1000 if "INR" in str(trade.get("symbol", "")).upper() else 1
        
        if trade["side"] == "BUY":
            return (float(exit_price) - float(trade["entry"])) * qty * multiplier
        return (float(trade["entry"]) - float(exit_price)) * qty * multiplier

    def _calculate_pnl_after_costs(self, trade: dict, exit_price: float | None) -> float:
        gross = self._calculate_pnl(trade, exit_price)
        if exit_price is None:
            return gross
        
        qty = int(trade.get("qty", 1))
        domain = trade.get("domain", self._domain_for_symbol(trade["symbol"]))
        
        if domain == "currency":
            # Flat ~15 INR per order leg (30 INR round trip) regardless of notional, 
            # plus minor exchange fees (approx 5 INR per lot round trip)
            estimated_cost = 30 + (qty * 5) 
            return gross - estimated_cost
        else:
            # Standard Equity execution cost logic
            entry = float(trade["entry"])
            exit_p = float(exit_price)
            cost = (entry * qty * self.cost_fraction) + (exit_p * qty * self.cost_fraction)
            return gross - cost

    def _log_to_journal(self, trade: dict, exit_price: float, outcome: str):
        """Append closed trade to CSV for dashboard. (C5: no header race, M4: costs included.)"""
        import csv

        log_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": trade["symbol"],
            "side": trade["side"],
            "strategy": trade.get("strategy", "Unknown"),
            "entry_price": trade["entry"],
            "exit_price": exit_price,
            "qty": trade.get("qty", 1),
            "pnl_inr": round(self._calculate_pnl(trade, exit_price), 2),
            "pnl_after_costs": round(self._calculate_pnl_after_costs(trade, exit_price), 2),
            "outcome": outcome,
            "confidence": trade.get("confidence", trade.get("quant_score", 0)),
        }

        fieldnames = [
            "date", "symbol", "side", "strategy", "entry_price", "exit_price",
            "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence",
        ]
        for attempt in range(3):
            try:
                with open(self._journal_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(log_data)
                break
            except PermissionError as e:
                if attempt == 2:
                    raise
                logger.warning(f"Trade journal locked, retrying write: {e}")
                time.sleep(0.2)

        logger.info(f"Trade journal updated: {trade['symbol']} | PnL: {log_data['pnl_inr']:.2f} | Net: {log_data['pnl_after_costs']:.2f}")

    async def square_off_all(self, domain: str | None = None):
        """Close active PROTECTED/PENDING orders at market price.

        Optional domain filtering keeps equity and currency EOD square-offs on
        their correct market schedules.
        """
        active = await self._get_active_orders()
        if not active:
            return

        active_items = []
        for oid, trade in list(active.items()):
            trade_domain = trade.get("domain", self._domain_for_symbol(trade.get("symbol", "")))
            if domain is None or trade_domain == domain:
                active_items.append((oid, trade))

        if not active_items:
            return

        label = f"{domain} " if domain else ""
        logger.warning(f"OrderManager: Squaring off {len(active_items)} active {label}positions (EOD).")
        for oid, trade in active_items:
            symbol = trade.get("symbol", "UNKNOWN")
            exit_price = trade.get("last_price") or trade.get("fill_price") or trade.get("entry")
            if self.redis is not None:
                try:
                    raw = await self.redis.get(f"bot:ltp:{symbol}")
                    if raw is not None:
                        exit_price = float(raw)
                    else:
                        logger.warning(
                            f"No Redis LTP for {symbol} at EOD; using fallback exit_price={exit_price}"
                        )
                except (TypeError, ValueError, KeyError) as exc:
                    logger.warning(f"Could not read latest LTP for {symbol}: {exc}")
            await self.handle_order_update({
                "order_id": oid,
                "status": "EOD_SQUAREOFF",
                "price": exit_price,
                "is_exit": True,
            })
