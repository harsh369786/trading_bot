import asyncio
import json
import os
import threading
from datetime import datetime, time

import pytz
import pandas as pd
from loguru import logger

from .redis_queue import RedisQueue
from utils.broker_utils import AngelOneMaster


class WebSocketFeed:
    """
    Connects to the broker WebSocket or mock simulator and publishes ticks to Redis.
    """
    def __init__(self, config: dict, redis_queue=None):
        self.config = config
        self.redis_queue = redis_queue or RedisQueue()
        self._owns_redis_queue = redis_queue is None
        self.max_retries = 3
        self.ist = pytz.timezone("Asia/Kolkata")
        instruments = config.get("instruments", {})
        gamma_cfg = config.get("gamma_scalper", {})
        gamma_symbols = list(gamma_cfg.get("symbols", []) or [])
        gamma_spot = gamma_cfg.get("spot_symbol", "SENSEX")
        configured_symbols = (
            instruments.get("equity", [])
            + instruments.get("currency", [])
            + gamma_symbols
            + ([gamma_spot] if gamma_symbols and gamma_spot else [])
        )
        self.symbols = list(dict.fromkeys(configured_symbols))
        self.ws_url = os.environ.get("BROKER_WS_URL", "wss://example.broker.com/stream")
        self.api_key = os.environ.get("BROKER_API_KEY", "")
        self.exchange_type_map = {
            "NSE": 1,
            "NFO": 2,
            "BSE": 3,
            "BFO": 4,
            "CDS": 13,
            "CDE_FO": 13,
        }
        self.resolved_token_map = {}
        self.symbol_exchange_map = {}
        self._last_cumulative_volume = {}

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

    @staticmethod
    def _first_positive_int(tick: dict, keys: list[str]) -> int:
        """Return the first positive integer-like field from a broker tick."""
        for key in keys:
            try:
                value = int(float(tick.get(key, 0) or 0))
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
        return 0

    def _extract_volume_delta(self, symbol: str, tick: dict) -> int:
        """
        Extract per-tick volume from Angel One ticks.

        SmartAPI commonly sends cumulative day volume as
        `volume_trade_for_the_day`, not `volume_traded`. Candles need interval
        volume, so convert cumulative values to deltas and fall back to LTQ.
        """
        cumulative = self._first_positive_int(
            tick,
            [
                "volume_trade_for_the_day",
                "volume_traded_today",
                "total_traded_volume",
                "total_trade_quantity",
            ],
        )
        if cumulative > 0:
            previous = self._last_cumulative_volume.get(symbol)
            self._last_cumulative_volume[symbol] = cumulative
            if previous is None or cumulative < previous:
                return 0
            return max(0, cumulative - previous)

        return self._first_positive_int(
            tick,
            [
                "volume_traded",
                "last_traded_quantity",
                "last_traded_qty",
                "ltq",
                "trade_quantity",
            ],
        )

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
        gamma_symbols = set(self.config.get("gamma_scalper", {}).get("symbols", []) or [])
        gamma_spot = self.config.get("gamma_scalper", {}).get("spot_symbol", "SENSEX")
        for sym in self.symbols:
            if sym in gamma_symbols:
                exchange = "BFO"
            elif sym == gamma_spot:
                exchange = "BSE"
            else:
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
                        "volume": self._extract_volume_delta(symbol, tick),
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

        def on_close(wsapp, close_status_code, close_msg):
            logger.warning(f"Angel One WS closed: {close_status_code} - {close_msg}")

        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error
        sws.on_close = on_close

        # Monkey patch to fix SmartApi library bug (takes 2 positional arguments but 4 were given)
        def _patched_on_close(wsapp, close_status_code, close_msg):
            if sws.on_close:
                sws.on_close(wsapp, close_status_code, close_msg)
        sws._on_close = _patched_on_close

        # Use a daemon thread so the blocking sws.connect() runs independently
        # of asyncio's ThreadPoolExecutor. This prevents "Executor shutdown has
        # been called" errors when asyncio cleans up while ticks are still arriving.
        stop_event = asyncio.Event()

        def _run_ws():
            import time
            consecutive_errors = 0
            while not stop_event.is_set():
                try:
                    sws.connect()
                    # If connect() returns, connection was closed
                    logger.warning("Angel One WS disconnected. Reconnecting in 5s...")
                    time.sleep(5)
                except Exception as exc:
                    consecutive_errors += 1
                    backoff = min(60, 2 ** consecutive_errors)
                    logger.error(f"Angel One WS thread error: {exc}. Retrying in {backoff}s...")
                    time.sleep(backoff)

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
            "INFY": 1450.0,
            "TCS": 3900.0,
            "ICICIBANK": 1100.0,
            "SBIN": 820.0,
            "AXISBANK": 1150.0,
            "KOTAKBANK": 1750.0,
            "LT": 3600.0,
            "WIPRO": 450.0,
            "BAJFINANCE": 7200.0,
            "TMPV": 950.0,
            "TATASTEEL": 150.0,
            "ADANIPORTS": 1350.0,
            "USDINR": 83.45,
            "EURINR": 90.12,
            "GBPINR": 105.30,
            "JPYINR": 0.54,
        }
        for symbol in self.symbols:
            path = os.path.join("data", "historical", f"{symbol}_6m.parquet")
            if not os.path.exists(path):
                continue
            try:
                hist = pd.read_parquet(path, columns=["close"])
                if not hist.empty:
                    base_prices[symbol] = float(hist["close"].dropna().iloc[-1])
            except Exception as exc:
                logger.debug(f"Simulator could not seed {symbol} from history: {exc}")

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
            if self._owns_redis_queue:
                await self.redis_queue.close()
            return

        await self.connect_and_stream()
        if self._owns_redis_queue:
            await self.redis_queue.close()
