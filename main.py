import asyncio
import yaml
import os
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
from execution.paper_engine import PaperEngine
from tracking.signal_logger import SignalLogger

# Post-Trade
from tracking.accuracy_analyzer import AccuracyAnalyzer
from learning.adaptive_learning_engine import AdaptiveLearningEngine
from reporting.daily_report_publisher import DailyReportPublisher

load_dotenv()

from config.logging_config import setup_logging

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

    order_manager = OrderManager(config, redis_client)
    # Pass risk_engine to equity engine so position sizing is ATR-based (C2 fix)
    equity_engine = EquitySignalEngine(config, risk_engine=risk_engine)
    currency_engine = CurrencyAgentPipeline(config)
    
    ws_feed = WebSocketFeed(config)
    streamer = LivePriceStreamer(config, redis_queue)

    # --- RSMB: shared paper engine + strategy (independent of existing OrderManager) ---
    signal_logger = SignalLogger()
    paper_engine = PaperEngine(
        cost_per_order_inr=config.get("execution", {}).get("cost_per_order_inr", 22.0)
    )
    rsmb_strategy = RSMBStrategy(
        config=config,
        broker_client=getattr(equity_engine, '_broker', None),
        signal_logger=signal_logger,
    )

    candles = CandleBuilder(
        config, equity_engine, currency_engine, order_manager, redis_queue,
        rsmb_strategy=rsmb_strategy,
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
    
    analyzer = AccuracyAnalyzer()
    learner = AdaptiveLearningEngine()
    reporter = DailyReportPublisher(config)

    # 3. Execution (The Swarm)
    logger.info("Bot components starting...")
    
    async def poll_vix():
        """Feed live VIX to RSMB strategy every 5 minutes."""
        import redis.asyncio as aioredis
        while True:
            try:
                rc = aioredis.from_url(
                    os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True
                )
                vix_raw = await rc.get("market:vix:latest")
                await rc.aclose()
                if vix_raw:
                    rsmb_strategy.update_vix(float(vix_raw))
            except Exception as exc:
                logger.debug(f"VIX poll: {exc}")
            await asyncio.sleep(300)  # 5 minutes

    try:
        # Run live services
        await asyncio.gather(
            ws_feed.run(),
            streamer.run(),
            candles.run(),
            tracker.run(),
            heartbeat(),
            poll_vix(),
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except Exception as e:
        logger.critical(f"System crash: {e}")
    finally:
        # 4. Nightly Cleanup & Learning (Runs at 15:30 in real scenario)
        stats = analyzer.get_stats()
        learner.tune_parameters(stats)
        report = reporter.format_daily_summary(stats)
        reporter.send_report(report)
        logger.info("Nightly maintenance complete. Goodbye.")

if __name__ == "__main__":
    asyncio.run(main())
