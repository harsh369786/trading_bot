import asyncio
import yaml
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger
from dotenv import load_dotenv

# Pipeline
from pipeline.websocket_feed import WebSocketFeed
from pipeline.live_price_streamer import LivePriceStreamer
from pipeline.candle_builder import CandleBuilder

# Intelligence & Execution
from strategies.equity_signal_engine import EquitySignalEngine
from agents.pipeline import CurrencyAgentPipeline
from execution.order_manager import OrderManager
from tracking.trade_lifecycle_tracker import TradeLifecycleTracker

# RSMB Strategy
from strategies.rsmb.strategy import RSMBStrategy
from strategies.gamma_scalper.strategy import GammaScalperStrategy
from strategies.mean_reversion.strategy import MeanReversionStrategy
from execution.paper_engine import PaperEngine
from tracking.signal_logger import SignalLogger

# Post-Trade
from tracking.accuracy_analyzer import AccuracyAnalyzer
from learning.adaptive_learning_engine import AdaptiveLearningEngine
from reporting.daily_report_publisher import DailyReportPublisher

load_dotenv()

from config.logging_config import setup_logging

IST = ZoneInfo("Asia/Kolkata")

async def heartbeat():
    """Simple task to show the bot is still alive in the console."""
    while True:
        logger.success("💓 System Heartbeat: All modules operational.")
        await asyncio.sleep(300) # Every 5 minutes

async def main():
    # 1. Setup
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)
    
    logger.info("🇮🇳 UNIFIED NSE TRADING BOT — INITIALIZING")

    import os
    trading_mode = os.environ.get("TRADING_MODE", "paper").strip().lower()
    paper_mode = bool(config.get("paper_mode", True))
    if trading_mode == "live" and not paper_mode:
        logger.critical("⚠️  LIVE TRADING MODE ACTIVE — real orders will be sent to broker!")
    else:
        logger.success("✅ Paper trading mode active — no real orders will be placed.")
        if trading_mode == "live" and paper_mode:
            logger.warning("TRADING_MODE=live but paper_mode=True in config. Config wins: paper mode enforced.")

    # 2. Redis & Infrastructure
    from pipeline.redis_queue import RedisQueue
    redis_queue = RedisQueue()
    await redis_queue.connect()
    redis_client = redis_queue.client
    
    # 3. Instantiate Modules with Redis
    from risk.risk_engine import RiskEngine
    risk_engine = RiskEngine(config, redis_client)

    order_manager = OrderManager(config, redis_client, risk_engine=risk_engine)
    # Pass risk_engine to equity engine so position sizing is ATR-based (C2 fix)
    equity_engine = EquitySignalEngine(config, risk_engine=risk_engine)
    currency_engine = CurrencyAgentPipeline(config)
    
    ws_feed = WebSocketFeed(config, redis_queue=redis_queue)
    streamer = LivePriceStreamer(config, redis_queue)

    # --- RSMB: shared paper engine + strategy (independent of existing OrderManager) ---
    signal_logger = SignalLogger()
    paper_engine = PaperEngine(
        cost_per_order_inr=config.get("execution", {}).get("cost_per_order_inr", 22.0),
        config=config,
    )
    rsmb_strategy = RSMBStrategy(
        config=config,
        broker_client=getattr(equity_engine, '_broker', None),
        signal_logger=signal_logger,
    )
    gamma_strategy = GammaScalperStrategy(config=config, signal_logger=signal_logger)
    meanrev_strategy = MeanReversionStrategy(config=config, signal_logger=signal_logger)

    candles = CandleBuilder(
        config, equity_engine, currency_engine, order_manager, redis_queue,
        rsmb_strategy=rsmb_strategy,
        gamma_strategy=gamma_strategy,
        meanrev_strategy=meanrev_strategy,
        paper_engine=paper_engine,
    )
    
    tracker = TradeLifecycleTracker(config, order_manager, redis_queue)
    
    # --- STARTUP RECOVERY (Orphan Trades) ---
    logger.info("🛠️ Running Startup Recovery...")
    active_positions = order_manager.broker.get_positions()
    if active_positions:
        logger.warning(f"Found {len(active_positions)} open positions. Syncing with Redis...")
        # Logic to adopt or close unmanaged trades would go here
    await order_manager.reconcile_startup_state()
    active_orders = await order_manager.get_active_orders()
    equity_open = sum(
        1 for trade in active_orders.values()
        if trade.get("status") in {"PENDING", "PROTECTED"}
        and trade.get("domain", order_manager._domain_for_symbol(trade.get("symbol", ""))) == "equity"
    )
    currency_open = sum(
        1 for trade in active_orders.values()
        if trade.get("status") in {"PENDING", "PROTECTED"}
        and trade.get("domain", order_manager._domain_for_symbol(trade.get("symbol", ""))) == "currency"
    )
    paper_active = paper_engine.get_active_orders()
    equity_open += sum(1 for order in paper_active if order.strategy not in {"gamma_scalper", "mean_reversion"})
    gamma_open = sum(1 for order in paper_active if order.strategy == "gamma_scalper")
    meanrev_open = sum(1 for order in paper_active if order.strategy == "mean_reversion")
    await risk_engine.reconcile_open_counts(equity_open, currency_open, gamma_open, meanrev_open)
    
    analyzer = AccuracyAnalyzer()
    learner = AdaptiveLearningEngine(config=config)
    reporter = DailyReportPublisher(config)

    # 3. Execution (The Swarm)
    logger.info("Bot components starting...")
    
    async def poll_vix():
        """Feed live VIX to RSMB strategy every 5 minutes."""
        while True:
            try:
                vix_raw = await redis_client.get("market:vix:latest")
                if vix_raw:
                    vix_value = float(vix_raw)
                    rsmb_strategy.update_vix(vix_value)
                    gamma_strategy.update_vix(vix_value)
            except Exception as exc:
                logger.debug(f"VIX poll: {exc}")
            await asyncio.sleep(300)  # 5 minutes

    def paper_domain_for_strategy(strategy_name: str) -> str:
        if strategy_name == "gamma_scalper":
            return "gamma"
        if strategy_name == "mean_reversion":
            return "mean_reversion"
        return "equity"

    def risk_amount_for_domain(domain: str) -> float:
        capital_cfg = config.get("capital", {})
        risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100.0
        if domain == "gamma":
            capital = float(capital_cfg.get("gamma_total", 30000))
        elif domain == "mean_reversion":
            capital = float(capital_cfg.get("meanrev_total", 40000))
        else:
            capital = float(capital_cfg.get("equity_total", 50000))
        return max(capital * risk_pct, 1.0)

    async def record_paper_engine_events(events):
        """Record paper closes in shared RiskEngine counters."""
        if not events:
            return
        for order_id, event in events:
            if event == "T1_HIT":
                continue
            order = paper_engine.get_order_snapshot(order_id)
            if order is None or order.status != "CLOSED":
                continue
            domain = paper_domain_for_strategy(order.strategy)
            risk_amount = risk_amount_for_domain(domain)
            pnl_inr = float(order.pnl_realised)
            await risk_engine.update_stats(
                domain,
                pnl_r=pnl_inr / risk_amount,
                pnl_inr=pnl_inr,
                trade_delta=-1,
            )

    async def market_monitor():
        """Monitor IST market close windows for EOD square-off."""
        equity_square_off_done = False
        currency_square_off_done = False
        async def latest_prices_for_open_paper_orders() -> dict:
            prices = {}
            try:
                active_orders = paper_engine.get_active_orders()
            except Exception as exc:
                logger.debug(f"Could not inspect paper orders for EOD pricing: {exc}")
                return prices

            for order in active_orders:
                try:
                    raw = await redis_client.get(f"bot:ltp:{order.symbol}")
                    if raw is not None:
                        prices[order.symbol] = float(raw)
                    else:
                        logger.warning(
                            f"EOD: No Redis LTP for paper {order.symbol}; "
                            "PaperEngine will use fill_price fallback."
                        )
                except Exception as exc:
                    logger.warning(f"Could not read latest LTP for paper {order.symbol}: {exc}")
            return prices

        while True:
            now = datetime.now(IST)
            # Square off at 15:20 IST
            if now.hour == 15 and now.minute >= 20 and not equity_square_off_done:
                logger.warning("🏁 Market close approaching (15:20 IST). Squaring off all positions...")
                # 1. Square off RSMB Paper Engine
                latest_prices = await latest_prices_for_open_paper_orders()
                events = paper_engine.square_off_all(latest_prices)
                await record_paper_engine_events(events)
                # 2. Square off Existing Strategy OrderManager
                await order_manager.square_off_all(domain="equity")
                equity_square_off_done = True
                logger.success("✅ EOD Square-off complete.")
            
            if now.hour >= 17 and not currency_square_off_done:
                logger.warning("Currency market close reached (17:00 IST). Squaring off currency positions...")
                await order_manager.square_off_all(domain="currency")
                currency_square_off_done = True
                logger.success("Currency EOD square-off complete.")

            if now.hour < 15: 
                equity_square_off_done = False
            if now.hour < 9: # Reset currency only after midnight/before market open
                currency_square_off_done = False
            
            await asyncio.sleep(60)

    try:
        # Run live services
        await asyncio.gather(
            ws_feed.run(),
            streamer.run(),
            candles.run(),
            tracker.run(),
            heartbeat(),
            poll_vix(),
            market_monitor(),
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except Exception as e:
        logger.critical(f"System crash: {e}")
    finally:
        # Final safety square-off
        logger.info("Performing final safety square-off...")
        final_prices = {}
        try:
            for order in paper_engine.get_active_orders():
                raw = await redis_client.get(f"bot:ltp:{order.symbol}")
                if raw is not None:
                    final_prices[order.symbol] = float(raw)
                else:
                    logger.warning(
                        f"Shutdown square-off: No Redis LTP for paper {order.symbol}; "
                        "PaperEngine will use fill_price fallback."
                    )
        except Exception as exc:
            logger.warning(f"Could not load final paper square-off prices: {exc}")
        events = paper_engine.square_off_all(final_prices)
        await record_paper_engine_events(events)
        await order_manager.square_off_all()
        now = datetime.now(IST)
        if now.hour >= 17:
            stats = analyzer.get_stats(today_only=True)
            learner.tune_parameters(stats)
        else:
            logger.info("Skipping adaptive learning before EOD close window; shutdown was not nightly maintenance.")
        await redis_queue.close()
        logger.info("Nightly maintenance complete. Goodbye.")
        import os
        os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())
