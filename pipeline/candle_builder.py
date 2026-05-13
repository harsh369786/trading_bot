import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import pytz
from loguru import logger


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
        self.tick_data: Dict[str, List[dict]] = {sym: [] for sym in self.symbols}
        self.candles: Dict[str, Dict[str, pd.DataFrame]] = {
            sym: {tf: pd.DataFrame() for tf in self.timeframes} for sym in self.symbols
        }
        self._load_history()

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

    async def _process_ticks_to_candles(self, symbol: str, ticks: List[dict]):
        """Aggregate ticks into OHLCV candles across multiple timeframes."""
        if not ticks:
            return

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

            updated_df = incoming_df.combine_first(existing_df).sort_index().tail(1000)
            old_last_ts = existing_df.index[-1] if not existing_df.empty else None
            new_last_ts = updated_df.index[-1] if not updated_df.empty else None
            self.candles[symbol][tf] = updated_df

            if tf == "5min" and old_last_ts is not None and new_last_ts is not None and new_last_ts > old_last_ts:
                closed_candle_ts = old_last_ts
                closed_5m = self.candles[symbol][tf].loc[:closed_candle_ts].copy()
                closed_15m = self.candles[symbol]["15min"].loc[:closed_candle_ts].copy()
                logger.success(f"Candle closed (5min) for {symbol} at {closed_candle_ts}. Triggering engines.")

                if symbol in self.config.get("instruments", {}).get("equity", []):
                    signal = await self.equity_engine.process_symbol(symbol, closed_5m)
                else:
                    signal = await self.currency_engine.process_symbol(symbol, closed_5m, closed_15m)

                if signal and self.order_manager:
                    await self.order_manager.execute_signal(signal)

            # --- RSMB 15m candle hook (independent of existing strategies) ---
            if tf == "15min" and old_last_ts is not None and new_last_ts is not None and new_last_ts > old_last_ts:
                if self.rsmb_strategy is not None:
                    closed_15m = self.candles[symbol]["15min"].loc[:old_last_ts].copy()
                    self.rsmb_strategy.push_bar(symbol, closed_15m)
                    self.rsmb_strategy.update_trailing_stops(symbol)
                    bar = closed_15m.iloc[-1] if not closed_15m.empty else None
                    if bar is not None:
                        rsmb_signal = self.rsmb_strategy.on_bar(symbol, bar)
                        if rsmb_signal and self.paper_engine:
                            # Fill at next bar open (use current last close as proxy in paper mode)
                            next_open = float(self.candles[symbol]["15min"].iloc[-1].get("open", rsmb_signal.entry))
                            order_id = self.paper_engine.simulate_fill(rsmb_signal, next_open)
                            self.rsmb_strategy.on_fill(rsmb_signal, next_open)
                            logger.success(f"RSMB order {order_id}: {rsmb_signal.side} {symbol} filled @ {next_open:.2f}")

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
