import asyncio
import json
import os
import time as time_module
import uuid
from datetime import datetime, time, timedelta
from typing import Dict, List

import pandas as pd
import pytz
from loguru import logger
from features.price_features import PriceFeatures


class CandleBuilder:
    """
    Builds multi-timeframe candles (1m, 3m, 5m, 15m, 1h) from live tick data.
    """
    def __init__(self, config: dict, equity_engine=None, currency_engine=None, order_manager=None, redis_queue=None, rsmb_strategy=None, paper_engine=None):
        self.config = config
        self.symbols = config.get("instruments", {}).get("equity", []) + config.get("instruments", {}).get("currency", [])
        self.timeframes = ["1min", "3min", "5min", "15min", "1h"]
        self.redis_queue = redis_queue
        self.equity_engine = equity_engine
        self.currency_engine = currency_engine
        self.order_manager = order_manager
        self.rsmb_strategy = rsmb_strategy
        self.paper_engine = paper_engine
        self.candle_snapshot_path = config.get("dashboard", {}).get(
            "candle_snapshot_path", "data/candle_snapshot.json"
        )
        self._last_snapshot_ts = 0.0
        self.tick_data: Dict[str, List[dict]] = {sym: [] for sym in self.symbols}
        self.candles: Dict[str, Dict[str, pd.DataFrame]] = {
            sym: {tf: pd.DataFrame() for tf in self.timeframes} for sym in self.symbols
        }
        self._load_history()
        self._write_candle_snapshot()

    def _load_history(self):
        """Preload recent candles from parquet so indicators have warm-up data."""
        import os

        logger.info(f"Preloading historical candles from {os.path.abspath('data/historical')}...")
        for sym in self.symbols:
            file_path = os.path.abspath(f"data/historical/{sym}_6m.parquet")
            if not os.path.exists(file_path):
                logger.warning(f"No history file found for {sym} at {file_path}")
                continue

            try:
                df_hist = pd.read_parquet(file_path)
                logger.info(f"Found {len(df_hist)} rows for {sym}")
                for tf in self.timeframes:
                    agg_dict = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                    if "oi" in df_hist.columns:
                        agg_dict["oi"] = "last"

                    resampled = df_hist.resample(tf).agg(agg_dict).dropna(subset=["close"]).tail(500)
                    if "oi" not in resampled.columns:
                        resampled["oi"] = 0
                    self.candles[sym][tf] = resampled
                logger.info(f"Loaded history for {sym}")
            except Exception as e:
                logger.error(f"Failed to load history for {sym}: {e}")

    def _get_candle_timestamp(self, dt: datetime, timeframe: str) -> datetime:
        """Floor a datetime to the start of the timeframe interval."""
        minutes = 60 if timeframe == "1h" else int(timeframe.replace("min", ""))
        return dt.replace(second=0, microsecond=0) - timedelta(minutes=dt.minute % minutes)

    def _write_candle_snapshot(self):
        """Persist latest candles for dashboard/runtime inspection."""
        snapshot = {
            "updated_at": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "symbols": {},
        }
        for symbol, by_tf in self.candles.items():
            snapshot["symbols"][symbol] = {}
            for tf, df in by_tf.items():
                if df.empty:
                    snapshot["symbols"][symbol][tf] = []
                    continue
                out = df.tail(20).copy()
                if out.index.tz is None:
                    out.index = out.index.tz_localize("Asia/Kolkata")
                else:
                    out.index = out.index.tz_convert("Asia/Kolkata")
                out = out.reset_index(names="timestamp")
                out["timestamp"] = out["timestamp"].astype(str)
                snapshot["symbols"][symbol][tf] = out.to_dict("records")

        snapshot_path = getattr(self, "candle_snapshot_path", "data/candle_snapshot.json")
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        tmp_path = f"{snapshot_path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
            for attempt in range(3):
                try:
                    os.replace(tmp_path, snapshot_path)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time_module.sleep(0.05)
        except Exception as exc:
            logger.warning(f"Failed to write candle snapshot: {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def _is_live_session_close(self, symbol: str, closed_ts, latest_ts) -> bool:
        """
        Return True only when a candle close is from the same live session as the
        newest tick being processed. Historical preload can otherwise make a
        restart look like a candle close from yesterday.
        """
        if closed_ts is None or latest_ts is None:
            return False

        ist = pytz.timezone("Asia/Kolkata")
        closed_ts = pd.Timestamp(closed_ts)
        latest_ts = pd.Timestamp(latest_ts)
        if closed_ts.tzinfo is None:
            closed_ts = closed_ts.tz_localize(ist)
        else:
            closed_ts = closed_ts.tz_convert(ist)
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize(ist)
        else:
            latest_ts = latest_ts.tz_convert(ist)

        if closed_ts.date() != latest_ts.date():
            logger.debug(
                f"Skipping stale candle trigger for {symbol}: "
                f"closed={closed_ts}, latest_tick={latest_ts}"
            )
            return False

        currency_symbols = set(self.config.get("instruments", {}).get("currency", []))
        market_close = time(17, 0) if symbol in currency_symbols else time(15, 30)
        market_open = time(9, 15)
        closed_time = closed_ts.time()
        return market_open <= closed_time < market_close

    def on_tick(self, symbol: str, tick: dict):
        """Process a live tick and update the current minute's candle."""
        logger.debug(f"CandleBuilder: Received tick for {symbol} | ltp={tick.get('ltp')}")
        self.tick_data[symbol].append(tick)

    async def _process_ticks_to_candles(self, symbol: str, ticks: List[dict]):
        """Aggregate ticks into OHLCV candles across multiple timeframes."""
        if not ticks:
            return
            
        logger.debug(f"CandleBuilder: Processing {len(ticks)} ticks for {symbol}")
        
        # Keep track of recent ticks for real-time OHLC estimation if needed
        self.tick_data[symbol].extend(ticks)
        if len(self.tick_data[symbol]) > 100:
            self.tick_data[symbol] = self.tick_data[symbol][-100:]

        df_ticks = pd.DataFrame([t.get("data", {}) for t in ticks])
        required = {"timestamp", "ltp", "volume"}
        missing = required - set(df_ticks.columns)
        if missing:
            logger.warning(f"Skipping malformed ticks for {symbol}; missing {sorted(missing)}")
            return

        df_ticks["timestamp"] = pd.to_datetime(df_ticks["timestamp"], errors="coerce")
        df_ticks = df_ticks.dropna(subset=["timestamp", "ltp"])
        # C6 fix: sort by timestamp before resampling to guard against out-of-order ticks
        df_ticks = df_ticks.sort_values("timestamp")
        if df_ticks.empty:
            return

        latest_ltp = pd.to_numeric(df_ticks["ltp"], errors="coerce").dropna()
        if not latest_ltp.empty:
            latest_price = float(latest_ltp.iloc[-1])
            if self.redis_queue is not None and getattr(self.redis_queue, "client", None) is not None:
                try:
                    await self.redis_queue.client.set(f"bot:ltp:{symbol}", latest_price, ex=3600)
                except Exception as exc:
                    logger.debug(f"Could not persist latest LTP for {symbol}: {exc}")
            paper_engine = getattr(self, "paper_engine", None)
            rsmb_strategy = getattr(self, "rsmb_strategy", None)
            if paper_engine is not None:
                paper_engine.on_price_update(symbol, latest_price)
            if rsmb_strategy is not None:
                rsmb_strategy.on_price_update(symbol, latest_price)

        for tf in self.timeframes:
            agg_dict = {"ltp": ["first", "max", "min", "last"], "volume": "sum"}
            if "oi" in df_ticks.columns:
                agg_dict["oi"] = "last"

            resampled = df_ticks.set_index("timestamp").resample(tf).agg(agg_dict)
            if "oi" in df_ticks.columns:
                resampled.columns = ["open", "high", "low", "close", "volume", "oi"]
            else:
                resampled.columns = ["open", "high", "low", "close", "volume"]
                resampled["oi"] = 0
            resampled = resampled.dropna(subset=["open", "high", "low", "close"])

            incoming_df = resampled
            existing_df = self.candles[symbol][tf]
            ist = pytz.timezone("Asia/Kolkata")
            if incoming_df.index.tz is None:
                incoming_df.index = incoming_df.index.tz_localize(ist)
            else:
                incoming_df.index = incoming_df.index.tz_convert(ist)

            if not existing_df.empty:
                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize(ist)
                else:
                    existing_df.index = existing_df.index.tz_convert(ist)

            updated_df = (
                pd.concat([existing_df, incoming_df])
                .loc[lambda frame: ~frame.index.duplicated(keep="last")]
                .sort_index()
                .tail(1000)
            )
            old_last_ts = existing_df.index[-1] if not existing_df.empty else None
            new_last_ts = updated_df.index[-1] if not updated_df.empty else None
            self.candles[symbol][tf] = updated_df

            if (
                tf == "15min"
                and old_last_ts is not None
                and new_last_ts is not None
                and new_last_ts > old_last_ts
                and self._is_live_session_close(symbol, old_last_ts, new_last_ts)
            ):
                # Signal engines only need the recent tail for indicators/AI (Module 4 fix)
                closed_15m = self.candles[symbol]["15min"].loc[:old_last_ts].tail(200).copy()
                closed_5m = self.candles[symbol]["5min"].loc[:old_last_ts].tail(200).copy()
                logger.success(f"Candle closed (15min) for {symbol} at {old_last_ts}. Triggering engines.")

                if symbol in self.config.get("instruments", {}).get("equity", []):
                    signal = await self.equity_engine.process_symbol(symbol, closed_15m, closed_5m)
                else:
                    signal = await self.currency_engine.process_symbol(symbol, closed_5m, closed_15m)

                if signal and self.order_manager:
                    await self.order_manager.execute_signal(signal)

                # --- RSMB 15m candle hook (independent of existing strategies) ---
                if self.rsmb_strategy is not None:
                    closed_15m_rsmb = closed_15m.copy()
                    
                    # Calculate features required by XGBoost AI filter
                    try:
                        from ta.trend import MACD
                        macd = MACD(close=closed_15m_rsmb['close'], window_slow=26, window_fast=12, window_sign=9)
                        closed_15m_rsmb['macd_hist'] = macd.macd_diff()
                    except Exception:
                        pass
                    closed_15m_rsmb = PriceFeatures.add_indicators(closed_15m_rsmb)
                    
                    # Compute and push daily closes for RS_Rank calculation (lookback=20 days)
                    if not closed_15m_rsmb.empty:
                        daily_closes = closed_15m_rsmb['close'].resample("D").last().dropna()
                        self.rsmb_strategy.push_daily(symbol, daily_closes)
                    
                    self.rsmb_strategy.push_bar(symbol, closed_15m_rsmb)
                    self.rsmb_strategy.update_trailing_stops(symbol)
                    bar = closed_15m_rsmb.iloc[-1] if not closed_15m_rsmb.empty else None
                    if bar is not None:
                        rsmb_signal = self.rsmb_strategy.on_bar(symbol, bar)
                        if rsmb_signal and self.paper_engine:
                            # Conservative paper fill: use signal entry until a true next-bar open is available.
                            next_open = float(rsmb_signal.entry)
                            order_id = self.paper_engine.simulate_fill(rsmb_signal, next_open)
                            self.rsmb_strategy.on_fill(rsmb_signal, next_open)
                            logger.success(f"RSMB order {order_id}: {rsmb_signal.side} {symbol} filled @ {next_open:.2f}")

        now = time_module.monotonic()
        if now - getattr(self, "_last_snapshot_ts", 0.0) >= 5.0:
            self._write_candle_snapshot()
            self._last_snapshot_ts = now

    async def run(self):
        """Consume ticks and build candles."""
        logger.info("Initializing Candle Builder...")
        last_ids = {sym: "$" for sym in self.symbols}

        while True:
            for symbol in self.symbols:
                new_ticks, new_id = await self.redis_queue.read_ticks(symbol, last_ids[symbol])
                if new_ticks:
                    await self._process_ticks_to_candles(symbol, new_ticks)
                    last_ids[symbol] = new_id

            await asyncio.sleep(0.1)
