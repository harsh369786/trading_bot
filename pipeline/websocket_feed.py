import asyncio
import json
import os
import threading
from datetime import datetime, time

import pytz
from loguru import logger

from .redis_queue import RedisQueue
from utils.broker_utils import AngelOneMaster


class WebSocketFeed:
    """
    Connects to the broker WebSocket or mock simulator and publishes ticks to Redis.
    """
    def __init__(self, config: dict):
        self.config = config
        self.redis_queue = RedisQueue()
        self.max_retries = 3
        self.ist = pytz.timezone("Asia/Kolkata")
        self.symbols = config.get("instruments", {}).get("equity", []) + config.get("instruments", {}).get("currency", [])
        self.ws_url = os.environ.get("BROKER_WS_URL", "wss://example.broker.com/stream")
        self.api_key = os.environ.get("BROKER_API_KEY", "")
        self.exchange_type_map = {
            "NSE": 1,
            "NFO": 2,
            "CDS": 13,
            "CDE_FO": 13,
        }
        self.resolved_token_map = {}
        self.symbol_exchange_map = {}

    async def send_telegram_alert(self, message: str):
        logger.error(f"TELEGRAM ALERT: {message}")

    def current_time_ist(self) -> datetime.time:
        return datetime.now(self.ist).time()

    def _is_symbol_market_open(self, symbol: str) -> bool:
        now = self.current_time_ist()
        is_currency = symbol in self.config.get("instruments", {}).get("currency", [])
        close_time = time(17, 0) if is_currency else time(15, 30)
        return time(9, 15) <= now <= close_time

    def _normalize_price(self, raw_price, exchange: str | None) -> float:
        try:
            value = float(raw_price or 0)
        except (TypeError, ValueError):
            return 0.0
        if value <= 0:
            return 0.0
        divisor = 10000000.0 if str(exchange).upper() == "CDS" else 100.0
        return value / divisor

    async def connect_and_stream(self):
        if self.ws_url.startswith("mock://"):
            logger.info("Starting mock market simulator...")
            await self._run_simulator()
            return

        if self.ws_url.startswith("angelone://"):
            logger.info("Connecting to Angel One real-time feed...")
            await self._run_angel_one_ws()
            return

        logger.warning("No valid WebSocket URL provided.")

    async def _run_angel_one_ws(self):
        """Streams real market data from Angel One SmartAPI."""
        from SmartApi import SmartConnect
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        import pyotp

        client_id = os.environ.get("ANGEL_CLIENT_ID")
        password = os.environ.get("ANGEL_PASSWORD")
        totp_secret = os.environ.get("ANGEL_TOTP_SECRET")
        if not all([self.api_key, client_id, password, totp_secret]):
            logger.error("Angel One credentials missing. WebSocket feed will not start.")
            return

        smart_api = SmartConnect(api_key=self.api_key)
        logger.info(f"Angel One: authenticating {client_id}...")
        session = smart_api.generateSession(client_id, password, pyotp.TOTP(totp_secret).now())
        if not session.get("status"):
            logger.error(f"Angel One auth failed: {session.get('message')}")
            return

        jwt_token = session["data"]["jwtToken"]
        feed_token = smart_api.getfeedToken()
        tokens_to_sub = []
        resolved_map = {}
        symbol_exchange_map = {}

        logger.info("Resolving live tokens from master contract...")
        equity_symbols = set(self.config.get("instruments", {}).get("equity", []))
        for sym in self.symbols:
            exchange = "NSE" if sym in equity_symbols else "CDE_FO"
            token, exch, full_name = AngelOneMaster.get_token(sym, exchange)
            if token:
                logger.info(f"Resolved {sym} -> {full_name} (Token: {token})")
                exchange_type = self.exchange_type_map.get(str(exch).upper())
                if exchange_type is None:
                    logger.warning(f"Unsupported Angel One exchange segment {exch} for {sym}")
                    continue
                tokens_to_sub.append({"exchangeType": exchange_type, "tokens": [token]})
                resolved_map[str(token)] = sym
                symbol_exchange_map[sym] = str(exch).upper()
            else:
                logger.warning(f"Could not resolve token for {sym}")

        if not tokens_to_sub:
            logger.error("No Angel One instruments resolved. WebSocket feed will not start.")
            return

        self.resolved_token_map = resolved_map
        self.symbol_exchange_map = symbol_exchange_map
        correlation_id = "nse_bot_feed"
        mode = 3
        main_loop = asyncio.get_event_loop()
        sws = SmartWebSocketV2(jwt_token, self.api_key, client_id, feed_token)

        def on_data(wsapp, msg):
            try:
                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
                        return

                if not msg:
                    return

                ticks = msg if isinstance(msg, list) else msg.get("data", [msg] if "token" in msg else [])
                for tick in ticks:
                    if not isinstance(tick, dict):
                        continue
                    symbol = self.resolved_token_map.get(str(tick.get("token")))
                    if not symbol or not self._is_symbol_market_open(symbol):
                        continue

                    exchange = self.symbol_exchange_map.get(symbol)
                    ltp = self._normalize_price(tick.get("last_traded_price", 0), exchange)
                    if ltp <= 0:
                        logger.debug(f"Skipping invalid zero/negative tick for {symbol}: {tick}")
                        continue

                    formatted_tick = {
                        "symbol": symbol,
                        "timestamp": datetime.now(self.ist).isoformat(),
                        "ltp": ltp,
                        "volume": int(tick.get("volume_traded", 0)),
                        "bid": self._normalize_price(tick.get("best_bid_price", 0), exchange),
                        "ask": self._normalize_price(tick.get("best_ask_price", 0), exchange),
                        "oi": int(tick.get("open_interest", 0)),
                    }
                    if main_loop and main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.redis_queue.publish_tick(symbol, formatted_tick), main_loop)
            except Exception as e:
                logger.error(f"Tick processing error: {e}")

        def on_open(wsapp):
            logger.info("Angel One WebSocket opened.")
            sws.subscribe(correlation_id, mode, tokens_to_sub)

        def on_error(wsapp, error):
            logger.error(f"Angel One WS error: {error}")

        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error

        # Use a daemon thread so the blocking sws.connect() runs independently
        # of asyncio's ThreadPoolExecutor. This prevents "Executor shutdown has
        # been called" errors when asyncio cleans up while ticks are still arriving.
        stop_event = asyncio.Event()

        def _run_ws():
            try:
                sws.connect()
            except Exception as exc:
                logger.error(f"Angel One WS thread exited with error: {exc}")
            finally:
                # Signal the asyncio coroutine that the WS thread has stopped
                if main_loop.is_running():
                    main_loop.call_soon_threadsafe(stop_event.set)

        ws_thread = threading.Thread(target=_run_ws, daemon=True, name="angelone-ws")
        ws_thread.start()
        logger.info("Angel One WebSocket thread started. Waiting for feed...")

        # Await until WS thread exits (or bot is shut down)
        await stop_event.wait()
        logger.warning("Angel One WebSocket thread has stopped.")

    async def _run_simulator(self):
        """Simulates live market ticks for paper trading."""
        import random

        base_prices = {
            "NIFTY": 22450.0,
            "BANKNIFTY": 48200.0,
            "RELIANCE": 2950.0,
            "HDFCBANK": 1520.0,
            "USDINR": 83.45,
            "EURINR": 90.12,
            "GBPINR": 105.30,
            "JPYINR": 0.54,
        }

        while True:
            for symbol in self.symbols:
                if not self._is_symbol_market_open(symbol):
                    continue

                base = base_prices.get(symbol, 100.0)
                change = random.uniform(-0.0005, 0.0005) * base
                base_prices[symbol] += change
                tick = {
                    "symbol": symbol,
                    "timestamp": datetime.now(self.ist).isoformat(),
                    "ltp": round(base_prices[symbol], 2 if "INR" not in symbol else 4),
                    "volume": random.randint(100, 5000),
                    "bid": round(base_prices[symbol] - 0.05, 2),
                    "ask": round(base_prices[symbol] + 0.05, 2),
                    "oi": random.randint(10000, 1000000),
                }
                await self.redis_queue.publish_tick(symbol, tick)

            await asyncio.sleep(0.5)

    async def run(self):
        """Main entry point for WebSocket feed."""
        logger.info("Initializing WebSocket feed...")
        await self.redis_queue.connect()

        if not any(self._is_symbol_market_open(symbol) for symbol in self.symbols):
            logger.warning("Markets are closed for all configured symbols. Shutting down feed.")
            await self.redis_queue.close()
            return

        await self.connect_and_stream()
        await self.redis_queue.close()
