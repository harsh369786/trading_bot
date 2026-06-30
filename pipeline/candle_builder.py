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
    def __init__(
        self,
        config: dict,
        equity_engine=None,
        currency_engine=None,
        order_manager=None,
        redis_queue=None,
        rsmb_strategy=None,
        gamma_strategy=None,
        meanrev_strategy=None,
        paper_engine=None,
    ):
        self.config = config
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
        self.gamma_symbols = set(gamma_symbols)
        self.gamma_spot_symbol = gamma_spot
        self.timeframes = ["1min", "3min", "5min", "15min", "1h"]
        self.redis_queue = redis_queue
        self.equity_engine = equity_engine
        self.currency_engine = currency_engine
        self.order_manager = order_manager
        self.rsmb_strategy = rsmb_strategy
        self.gamma_strategy = gamma_strategy
        self.meanrev_strategy = meanrev_strategy
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

    def _get_risk_engine(self):
        return getattr(self.order_manager, "risk_engine", None)

    def _equity_risk_amount(self) -> float:
        capital = float(self.config.get("capital", {}).get("equity_total", 50000))
        risk_pct = float(self.config.get("capital", {}).get("risk_per_trade_pct", 1.0)) / 100.0
        return max(capital * risk_pct, 1.0)

    def _risk_amount_for_domain(self, domain: str) -> float:
        capital_cfg = self.config.get("capital", {})
        risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100.0
        if domain == "gamma":
            capital = float(capital_cfg.get("gamma_total", 30000))
        elif domain == "mean_reversion":
            capital = float(capital_cfg.get("meanrev_total", 40000))
        else:
            capital = float(capital_cfg.get("equity_total", 50000))
        return max(capital * risk_pct, 1.0)

    @staticmethod
    def _domain_for_strategy(strategy_name: str) -> str:
        if strategy_name == "gamma_scalper":
            return "gamma"
        if strategy_name == "mean_reversion":
            return "mean_reversion"
        return "equity"

    @staticmethod
    def _paper_fill_price(signal, slippage: float = 0.001) -> float:
        """Approximate executable paper fill when next-bar open is not available yet."""
        entry = float(signal.entry)
        side = str(getattr(signal, "side", "BUY")).upper()
        return entry * (1 + slippage if side == "BUY" else 1 - slippage)

    def _paper_symbol_active(self, symbol: str, strategy: str | None = None) -> bool:
        if self.paper_engine is None:
            return False
        try:
            return any(order.symbol == symbol for order in self.paper_engine.get_active_orders(strategy))
        except Exception as exc:
            logger.debug(f"Could not inspect active paper orders for {symbol}: {exc}")
            return False

    async def _record_paper_engine_events(self, events):
        """Push paper exits into the shared RiskEngine circuit counters."""
        if not events or self.paper_engine is None:
            return
        risk_engine = self._get_risk_engine()
        if risk_engine is None:
            return

        for order_id, event in events:
            if event == "T1_HIT":
                continue
            order = self.paper_engine.get_order_snapshot(order_id)
            if order is None or order.status != "CLOSED":
                continue
            domain = self._domain_for_strategy(order.strategy)
            risk_amount = self._risk_amount_for_domain(domain)
            pnl_inr = float(order.pnl_realised)
            await risk_engine.update_stats(
                domain,
                pnl_r=pnl_inr / risk_amount,
                pnl_inr=pnl_inr,
                trade_delta=-1,
            )

    async def _close_manual_strategy_events(self, strategy, events, outcome_map: dict[str, str]) -> None:
        if not events or self.paper_engine is None:
            return
        closed_events = []
        for pos_id, event, price in events:
            if event not in outcome_map:
                continue
            order_id = strategy.paper_order_id_for(pos_id) if hasattr(strategy, "paper_order_id_for") else None
            if not order_id:
                logger.warning(f"Strategy manual event {event} for {pos_id} has no paper order id")
                continue
            if self.paper_engine.close_position(order_id, float(price), outcome_map[event]):
                closed_events.append((order_id, event))
        await self._record_paper_engine_events(closed_events)

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

                    history_tail = 2500 if tf in {"15min", "1h"} else 1000
                    resampled = df_hist.resample(tf).agg(agg_dict).dropna(subset=["close"]).tail(history_tail)
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
        market_close = time(17, 0) if symbol in currency_symbols else time(15, 15)
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

        # Note: individual ticks are already stored via on_tick().
        # Do not re-extend here — that would cause duplicate entries.

        df_ticks = pd.DataFrame([t.get("data", {}) for t in ticks])
        required = {"timestamp", "ltp", "volume"}
        missing = required - set(df_ticks.columns)
        if missing:
            logger.warning(f"Skipping malformed ticks for {symbol}; missing {sorted(missing)}")
            return

        df_ticks["timestamp"] = pd.to_datetime(df_ticks["timestamp"], errors="coerce")
        df_ticks = df_ticks.dropna(subset=["timestamp", "ltp"])
        df_ticks["ltp"] = pd.to_numeric(df_ticks["ltp"], errors="coerce")
        df_ticks["volume"] = pd.to_numeric(df_ticks["volume"], errors="coerce").fillna(0).clip(lower=0)
        if "oi" in df_ticks.columns:
            df_ticks["oi"] = pd.to_numeric(df_ticks["oi"], errors="coerce").fillna(0)
        df_ticks = df_ticks.dropna(subset=["ltp"])
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
            gamma_strategy = getattr(self, "gamma_strategy", None)
            meanrev_strategy = getattr(self, "meanrev_strategy", None)
            if paper_engine is not None:
                events = paper_engine.on_price_update(symbol, latest_price)
                await self._record_paper_engine_events(events)
            if rsmb_strategy is not None and hasattr(rsmb_strategy, "on_price_update"):
                rsmb_strategy.on_price_update(symbol, latest_price)
            if gamma_strategy is not None and hasattr(gamma_strategy, "on_price_update"):
                gamma_strategy.on_price_update(symbol, latest_price)
            if meanrev_strategy is not None and hasattr(meanrev_strategy, "on_price_update"):
                meanrev_strategy.on_price_update(symbol, latest_price)

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

            retention = 2500 if tf in {"15min", "1h"} else 1000
            updated_df = (
                pd.concat([existing_df, incoming_df])
                .loc[lambda frame: ~frame.index.duplicated(keep="last")]
                .sort_index()
                .tail(retention)
            )
            old_last_ts = existing_df.index[-1] if not existing_df.empty else None
            new_last_ts = updated_df.index[-1] if not updated_df.empty else None
            self.candles[symbol][tf] = updated_df

            if (
                tf == "5min"
                and old_last_ts is not None
                and new_last_ts is not None
                and new_last_ts > old_last_ts
                and self._is_live_session_close(symbol, old_last_ts, new_last_ts)
            ):
                gamma_enabled = bool(self.config.get("gamma_scalper", {}).get("enabled", False))
                if gamma_enabled and self.gamma_strategy is not None and symbol in self.gamma_symbols:
                    closed_5m = self.candles[symbol]["5min"].loc[:old_last_ts].tail(120).copy()
                    spot_df = None
                    if self.gamma_spot_symbol in self.candles:
                        spot_df = self.candles[self.gamma_spot_symbol]["5min"].loc[:old_last_ts].tail(120).copy()
                    self.gamma_strategy.push_bars(symbol, closed_5m, spot_df)

                    manual_events = self.gamma_strategy.on_bar_close(symbol)
                    await self._close_manual_strategy_events(
                        self.gamma_strategy,
                        manual_events,
                        {"T2_HIT": "TARGET_HIT", "THETA_VETO": "theta_decay_veto"},
                    )

                    if not closed_5m.empty:
                        gamma_signal = self.gamma_strategy.on_bar(
                            symbol,
                            closed_5m.iloc[-1],
                            spot_bar=spot_df.iloc[-1] if spot_df is not None and not spot_df.empty else None,
                        )
                        if gamma_signal and self.paper_engine:
                            risk_engine = self._get_risk_engine()
                            risk_allowed = True
                            if risk_engine is not None:
                                risk_allowed = await risk_engine.check_circuit_breakers("gamma")
                            if not risk_allowed:
                                logger.warning(f"GammaScalper skipped for {symbol}: gamma risk circuit blocked entries.")
                                self.gamma_strategy.cancel_pending(gamma_signal)
                            else:
                                fill_price = self._paper_fill_price(gamma_signal)
                                order_id = self.paper_engine.simulate_fill(gamma_signal, fill_price)
                                if order_id is None:
                                    self.gamma_strategy.cancel_pending(gamma_signal)
                                else:
                                    if hasattr(self.gamma_strategy, "bind_order_id"):
                                        self.gamma_strategy.bind_order_id(gamma_signal, order_id)
                                    if risk_engine is not None:
                                        await risk_engine.update_stats("gamma", trade_delta=1)
                                    self.gamma_strategy.on_fill(gamma_signal, fill_price)
                                    logger.success(f"GammaScalper order {order_id}: {symbol} filled @ {fill_price:.2f}")

            if (
                tf == "15min"
                and old_last_ts is not None
                and new_last_ts is not None
                and new_last_ts > old_last_ts
                and self._is_live_session_close(symbol, old_last_ts, new_last_ts)
            ):
                # Signal engines only need the recent tail for indicators/AI (Module 4 fix)
                closed_15m = self.candles[symbol]["15min"].loc[:old_last_ts].tail(200).copy()
                closed_5m = self.candles[symbol]["5min"].loc[
                    self.candles[symbol]["5min"].index < new_last_ts
                ].tail(200).copy()
                logger.success(f"Candle closed (15min) for {symbol} at {old_last_ts}. Triggering engines.")

                signal = None
                if symbol in self.config.get("instruments", {}).get("equity", []):
                    signal = await self.equity_engine.process_symbol(symbol, closed_15m, closed_5m)
                elif symbol in self.config.get("instruments", {}).get("currency", []):
                    signal = await self.currency_engine.process_symbol(symbol, closed_5m, closed_15m)

                if signal and self.order_manager:
                    await self.order_manager.execute_signal(signal)

                # --- MeanReversion 15m candle hook (shared PaperEngine execution) ---
                meanrev_enabled = bool(self.config.get("mean_reversion", {}).get("enabled", False))
                if (
                    meanrev_enabled
                    and self.meanrev_strategy is not None
                    and symbol in self.config.get("instruments", {}).get("equity", [])
                ):
                    if self._paper_symbol_active(symbol):
                        logger.debug(f"MeanReversion skipped for {symbol}: paper trade already active.")
                    else:
                        closed_15m_meanrev = closed_15m.copy()
                        try:
                            closed_15m_meanrev = PriceFeatures.add_indicators(closed_15m_meanrev)
                        except Exception as exc:
                            logger.debug(f"MeanReversion feature add failed for {symbol}: {exc}")
                        df_1h = self.candles[symbol]["1h"].loc[:old_last_ts].tail(50).copy()
                        self.meanrev_strategy.push_bars(symbol, closed_15m_meanrev, df_1h)
                        meanrev_signal = self.meanrev_strategy.on_bar(
                            symbol,
                            closed_15m_meanrev.iloc[-1] if not closed_15m_meanrev.empty else pd.Series(dtype=float),
                            df_1h=df_1h,
                        )
                        if meanrev_signal and self.paper_engine:
                            risk_engine = self._get_risk_engine()
                            risk_allowed = True
                            if risk_engine is not None:
                                risk_allowed = await risk_engine.check_circuit_breakers("mean_reversion")
                            if not risk_allowed:
                                logger.warning(f"MeanReversion skipped for {symbol}: risk circuit blocked entries.")
                                self.meanrev_strategy.cancel_pending(meanrev_signal)
                            else:
                                fill_price = self._paper_fill_price(meanrev_signal)
                                order_id = self.paper_engine.simulate_fill(meanrev_signal, fill_price)
                                if order_id is None:
                                    self.meanrev_strategy.cancel_pending(meanrev_signal)
                                else:
                                    if hasattr(self.meanrev_strategy, "bind_order_id"):
                                        self.meanrev_strategy.bind_order_id(meanrev_signal, order_id)
                                    if risk_engine is not None:
                                        await risk_engine.update_stats("mean_reversion", trade_delta=1)
                                    self.meanrev_strategy.on_fill(meanrev_signal, fill_price)
                                    logger.success(f"MeanReversion order {order_id}: {meanrev_signal.side} {symbol} filled @ {fill_price:.2f}")

                # --- RSMB 15m candle hook (independent of existing strategies) ---
                if self.rsmb_strategy is not None:
                    closed_15m_rsmb = self.candles[symbol]["15min"].loc[:old_last_ts].tail(2500).copy()

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
                        if symbol == "NIFTY" and hasattr(self.rsmb_strategy, "push_nifty_daily"):
                            self.rsmb_strategy.push_nifty_daily(daily_closes)
                        self.rsmb_strategy.push_daily(symbol, daily_closes)

                    self.rsmb_strategy.push_bar(symbol, closed_15m_rsmb)
                    self.rsmb_strategy.update_trailing_stops(symbol)
                    bar = closed_15m_rsmb.iloc[-1] if not closed_15m_rsmb.empty else None
                    if bar is not None:
                        skip_rsmb = False
                        if self.paper_engine is not None:
                            active_rsmb_symbols = {
                                order.symbol for order in self.paper_engine.get_active_orders("rsmb")
                            }
                            if symbol in active_rsmb_symbols:
                                logger.debug(f"RSMB skipped for {symbol}: paper trade already active.")
                                skip_rsmb = True
                        risk_engine = self._get_risk_engine()
                        if (
                            not skip_rsmb
                            and risk_engine is not None
                            and not await risk_engine.check_circuit_breakers("equity")
                        ):
                            logger.warning(f"RSMB skipped for {symbol}: shared equity risk circuit blocked entries.")
                            skip_rsmb = True
                        if not skip_rsmb:
                            rsmb_signal = self.rsmb_strategy.on_bar(symbol, bar)
                            if rsmb_signal and self.paper_engine:
                                # Conservative paper fill: use signal entry until a true next-bar open is available.
                                next_open = self._paper_fill_price(rsmb_signal)
                                order_id = self.paper_engine.simulate_fill(rsmb_signal, next_open)
                                if order_id is not None:
                                    if risk_engine is not None:
                                        await risk_engine.update_stats("equity", trade_delta=1)
                                    self.rsmb_strategy.on_fill(rsmb_signal, next_open)
                                    logger.success(f"RSMB order {order_id}: {rsmb_signal.side} {symbol} filled @ {next_open:.2f}")

        now = time_module.monotonic()
        if now - getattr(self, "_last_snapshot_ts", 0.0) >= 30.0:
            self._write_candle_snapshot()
            self._last_snapshot_ts = now

    async def run(self):
        """Consume ticks and build candles."""
        logger.info("Initializing Candle Builder...")
        last_ids = {sym: "$" for sym in self.symbols}
        consecutive_errors = 0

        while True:
            reads = [
                self.redis_queue.read_ticks(symbol, last_ids[symbol])
                for symbol in self.symbols
            ]
            results = await asyncio.gather(*reads, return_exceptions=True)
            had_error = False

            for symbol, result in zip(self.symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"CandleBuilder read failed for {symbol}: {result}")
                    had_error = True
                    continue
                new_ticks, new_id = result
                if new_ticks:
                    await self._process_ticks_to_candles(symbol, new_ticks)
                    last_ids[symbol] = new_id

            if had_error:
                consecutive_errors += 1
                backoff = min(60, 2 ** consecutive_errors)
                logger.warning(f"CandleBuilder: Redis error. Backing off for {backoff} seconds.")
                await asyncio.sleep(backoff)
            else:
                consecutive_errors = 0
                await asyncio.sleep(0.1)
