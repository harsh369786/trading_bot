import asyncio

from loguru import logger


class TradeLifecycleTracker:
    """
    Module 7: tick-by-tick monitoring of active trades.
    Tracks SL and target exits for active protected trades.
    """
    def __init__(self, config: dict, order_manager=None, redis_queue=None):
        self.config = config
        self.order_manager = order_manager
        self.redis_queue = redis_queue
        self.symbols = config.get("instruments", {}).get("equity", []) + config.get("instruments", {}).get("currency", [])

    async def _monitor_symbol(self, symbol: str):
        """Monitor ticks for a specific symbol if active trades exist."""
        last_id = "$"
        while True:
            if not self.order_manager or not self.redis_queue:
                await asyncio.sleep(1)
                continue

            if hasattr(self.order_manager, "get_active_orders"):
                active_orders = await self.order_manager.get_active_orders()
            else:
                active_orders = await self.order_manager._get_active_orders()
            my_trades = {
                oid: trade
                for oid, trade in active_orders.items()
                if trade.get("symbol") == symbol and trade.get("status") == "PROTECTED"
            }

            if not my_trades:
                await asyncio.sleep(0.5)
                continue

            ticks, new_id = await self.redis_queue.read_ticks(symbol, last_id)
            for tick_wrapper in ticks:
                tick = tick_wrapper.get("data", {})
                try:
                    price = float(tick["ltp"])
                except (KeyError, TypeError, ValueError):
                    logger.warning(f"Skipping malformed tick for {symbol}: {tick}")
                    continue

                for oid, trade in list(my_trades.items()):
                    try:
                        side = trade["side"]
                        entry = float(trade["entry"])
                        sl = float(trade["sl"])
                        target_raw = trade.get("target") or trade.get("t1")
                        if target_raw is None:
                            logger.error(f"Active trade {oid} has no target/t1 field: {trade}")
                            continue
                        target = float(target_raw)
                    except (KeyError, TypeError, ValueError):
                        logger.error(f"Active trade {oid} has invalid SL/target fields: {trade}")
                        continue

                    if (side == "BUY" and price <= sl) or (side == "SELL" and price >= sl):
                        logger.warning(f"SL HIT for {symbol} at {price} (Entry: {entry}, SL: {sl})")
                        await self.order_manager.handle_order_update({
                            "order_id": oid,
                            "status": "SL_HIT",
                            "price": price,
                            "is_exit": True,
                        })
                        del my_trades[oid]

                    elif (side == "BUY" and price >= target) or (side == "SELL" and price <= target):
                        logger.success(f"TARGET HIT for {symbol} at {price} (Entry: {entry}, Target: {target})")
                        await self.order_manager.handle_order_update({
                            "order_id": oid,
                            "status": "TARGET_HIT",
                            "price": price,
                            "is_exit": True,
                        })
                        del my_trades[oid]

            last_id = new_id
            await asyncio.sleep(0.05)

    async def run(self):
        """Main monitoring loop."""
        logger.info("Trade Lifecycle Tracker active. Monitoring SL/targets.")
        tasks = [asyncio.create_task(self._monitor_symbol(symbol)) for symbol in self.symbols]
        await asyncio.gather(*tasks)
