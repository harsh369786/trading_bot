"""
strategies/rsmb/strategy.py
------------------------------
The main RSMB strategy class — implements BaseStrategy and wires together
all RSMB components into a single on_bar() hook.

This is the entry point the existing bot's main.py calls on every 15m candle.
It is completely independent of the existing TrendFollowing/MeanReversion/Ensemble_AI
strategies — shared only at the risk/execution layer (PaperEngine).
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal
from strategies.rsmb.ai_filter import RSMBAIFilter
from strategies.rsmb.indicators import (
    compute_atr,
    compute_ema,
    compute_rs_rank,
    compute_supertrend,
    compute_vwap,
    compute_volume_ratio,
)
from strategies.rsmb.position_manager import RSMBPositionManager
from strategies.rsmb.signals import evaluate_buy_signal, evaluate_sell_signal
from strategies.rsmb.vix_monitor import VIXMonitor
from data.nifty_index import load_nifty_daily
from tracking.signal_logger import SignalLogger


RSMB_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "WIPRO",
    "BAJFINANCE", "TMPV", "TATASTEEL", "ADANIPORTS",
]


class RSMBStrategy(BaseStrategy):
    """
    Relative Strength Momentum Breakout Strategy.

    On each 15m bar close the existing bot calls:
        signal = rsmb_strategy.on_bar(symbol, bar_series)

    If a signal is returned, the caller passes it to PaperEngine.simulate_fill().
    """

    def __init__(
        self,
        config: dict,
        broker_client=None,
        signal_logger: Optional[SignalLogger] = None,
    ) -> None:
        self.config = config
        self._broker = broker_client
        self._signal_logger = signal_logger
        self._position_manager = RSMBPositionManager(
            cost_per_order_inr=config.get("execution", {}).get("cost_per_order_inr", 22.0)
        )
        self._ai_filter = RSMBAIFilter()
        self._vix_monitor = VIXMonitor(spike_threshold_pct=5.0, window_size=12)

        # Cache of 15m DataFrames per symbol (populated by feed.py)
        self._bars: dict[str, pd.DataFrame] = {}

        # Daily closes cache (refreshed at session start)
        self._daily_closes: dict[str, pd.Series] = {}
        self._nifty_daily: pd.Series = pd.Series(dtype=float)

        # Risk params from config
        capital_cfg = config.get("capital", {})
        self._equity_total = capital_cfg.get("equity_total", 50000)
        self._risk_pct = capital_cfg.get("risk_per_trade_pct", 1.0)
        self._risk_capital = self._equity_total * self._risk_pct / 100

        logger.info(
            f"RSMBStrategy: initialized | universe={len(RSMB_UNIVERSE)} symbols | "
            f"risk_capital=₹{self._risk_capital:.0f}"
        )

    # ------------------------------------------------------------------
    # Data ingestion (called by feed.py on each new bar)
    # ------------------------------------------------------------------

    def push_bar(self, symbol: str, df: pd.DataFrame) -> None:
        """Receive updated 15m OHLCV DataFrame for a symbol."""
        self._bars[symbol] = df

    def push_daily(self, symbol: str, daily_closes: pd.Series) -> None:
        """Receive updated daily close Series for a symbol."""
        self._daily_closes[symbol] = daily_closes

    def push_nifty_daily(self, nifty_closes: pd.Series) -> None:
        """Receive Nifty 50 daily closes (for RS_Rank calculation)."""
        self._nifty_daily = nifty_closes

    def update_vix(self, vix_value: float) -> None:
        """Feed latest VIX reading (called every 5 min by scheduler)."""
        self._vix_monitor.update(vix_value)

    # ------------------------------------------------------------------
    # Core strategy hook — called by main.py on every completed 15m bar
    # ------------------------------------------------------------------

    def on_bar(self, symbol: str, bar: pd.Series) -> Optional[Signal]:
        """
        Evaluate RSMB entry conditions for a symbol on the latest completed bar.

        Parameters
        ----------
        symbol : NSE symbol (must be in RSMB_UNIVERSE).
        bar    : Latest completed 15m bar (pd.Series).

        Returns
        -------
        Signal if all conditions pass, None otherwise.
        """
        if symbol not in RSMB_UNIVERSE:
            return None

        df_15m = self._bars.get(symbol)
        if df_15m is None or len(df_15m) < 21:
            logger.debug(f"RSMB {symbol}: insufficient bars ({len(df_15m) if df_15m is not None else 0})")
            return None

        # VIX veto check
        vix_veto, vix_reason = self._vix_monitor.is_veto()

        # Nifty daily data check
        if self._nifty_daily.empty:
            self._nifty_daily = load_nifty_daily(self._broker)

        # RS_Rank
        stock_daily = self._daily_closes.get(symbol, pd.Series(dtype=float))
        rs_rank = compute_rs_rank(stock_daily, self._nifty_daily)

        # Daily DataFrame for EMA 50
        df_daily_df = pd.DataFrame({"close": stock_daily}) if not stock_daily.empty else pd.DataFrame()

        # AI score
        ai_score = self._compute_ai_score(symbol, df_15m, rs_rank)

        # Attempt BUY
        signal = evaluate_buy_signal(
            symbol=symbol,
            df_15m=df_15m,
            df_daily=df_daily_df,
            rs_rank=rs_rank,
            ai_score=ai_score,
            vix_veto=vix_veto,
            risk_capital=self._risk_capital,
        )

        if signal is None:
            # Attempt SELL
            signal = evaluate_sell_signal(
                symbol=symbol,
                df_15m=df_15m,
                df_daily=df_daily_df,
                rs_rank=rs_rank,
                ai_score=ai_score,
                vix_veto=vix_veto,
                risk_capital=self._risk_capital,
            )

        # Check capacity
        if signal is not None:
            if not self._position_manager.can_open():
                logger.warning(f"RSMB: max 3 open trades — {symbol} signal skipped")
                self._log_signal(symbol, signal.side, 0, 0, 0, ai_score, "NO_TRADE", "max_open_trades", rs_rank)
                return None

            take_reason = f"RS Rank: {rs_rank:.2f} | AI: {ai_score:.2f} | Setup Valid"
            self._log_signal(
                symbol, signal.side,
                signal.entry, signal.sl, signal.target1,
                ai_score, "TRADE", take_reason, rs_rank
            )
            
            # Reservation Fix: register PENDING position immediately to reserve capacity (Module 3 fix)
            pos_id = self._position_manager.open_position(signal)
            signal.id = pos_id
            
            return signal

        return None

    # ------------------------------------------------------------------
    # Position lifecycle callbacks (called by PaperEngine)
    # ------------------------------------------------------------------

    def on_fill(self, signal: Signal, fill_price: float) -> None:
        """Called when PaperEngine fills the order."""
        # Note: open_position was already called in on_bar to reserve capacity
        if signal.id:
            ts = pd.Timestamp.now(tz="Asia/Kolkata")
            self._position_manager.on_fill(signal.id, fill_price, ts)

    def on_target_hit(self, signal: Signal, target_num: int) -> None:
        logger.info(f"RSMBStrategy: {signal.symbol} target {target_num} hit")

    def on_sl_hit(self, signal: Signal) -> None:
        logger.info(f"RSMBStrategy: {signal.symbol} SL hit")

    # ------------------------------------------------------------------
    # Price update hook (called by feed.py on every tick)
    # ------------------------------------------------------------------

    def on_price_update(self, symbol: str, price: float) -> None:
        """Forward price to position manager for SL/T1/T2 monitoring."""
        events = self._position_manager.on_price_update(price, symbol=symbol)
        if events:
            for pos_id, event, exit_price in events:
                logger.info(f"RSMB position {pos_id}: {event} @ {exit_price:.2f}")

    # ------------------------------------------------------------------
    # Trailing stop update (call on each new 15m bar for partial positions)
    # ------------------------------------------------------------------

    def update_trailing_stops(self, symbol: str) -> None:
        df_15m = self._bars.get(symbol)
        if df_15m is None:
            return
        for pos in self._position_manager.get_open_positions():
            if pos.signal.symbol == symbol and pos.status == "PARTIAL":
                self._position_manager.update_trailing_stop(pos.position_id, df_15m)

    # ------------------------------------------------------------------
    # Stats (used by dashboard)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return self._position_manager.get_stats()

    # ------------------------------------------------------------------
    # VIX access (used by dashboard)
    # ------------------------------------------------------------------

    @property
    def vix_info(self) -> dict:
        return {
            "current": self._vix_monitor.current_vix,
            "spike_pct_60m": self._vix_monitor.spike_pct_60m,
            "veto_active": self._vix_monitor.is_veto()[0],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ai_score(
        self, symbol: str, df_15m: pd.DataFrame, rs_rank: float
    ) -> float:
        """Extract features and call AI filter."""
        try:
            bar = df_15m.iloc[-1]
            vwap_s = compute_vwap(df_15m)
            ema21_s = compute_ema(df_15m["close"], 21)
            atr_s = compute_atr(df_15m, 14)
            vol_ratio_s = compute_volume_ratio(df_15m["volume"], 5)

            vwap = float(vwap_s.iloc[-1]) if not vwap_s.empty else 0.0
            ema21 = float(ema21_s.iloc[-1]) if not ema21_s.empty else 0.0
            atr = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0
            vol_ratio = float(vol_ratio_s.iloc[-1]) if not vol_ratio_s.empty else 1.0

            # Bollinger and ADX for features (Module 1 optimization)
            bb_upper = float(df_15m["BBU_20_2.0"].iloc[-1]) if "BBU_20_2.0" in df_15m.columns else 0.0
            bb_lower = float(df_15m["BBL_20_2.0"].iloc[-1]) if "BBL_20_2.0" in df_15m.columns else 0.0
            adx = float(df_15m["ADX_14"].iloc[-1]) if "ADX_14" in df_15m.columns else 0.0

            features = RSMBAIFilter.extract_features(
                bar=bar,
                rs_rank=rs_rank if not math.isnan(rs_rank) else 1.0,
                vwap=vwap,
                ema21=ema21,
                atr=atr,
                volume_ratio=vol_ratio,
                adx=adx,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
            )
            return self._ai_filter.predict(features)

        except Exception as exc:
            logger.warning(f"RSMB {symbol}: AI score failed ({exc}); using fallback 0.5")
            return 0.5

    def _log_signal(
        self, symbol: str, side: str, entry: float, sl: float,
        target: float, score: float, status: str, reason: str,
        rs_rank: float,
    ) -> None:
        if self._signal_logger is None:
            return
        try:
            self._signal_logger.log_signal(
                symbol=symbol,
                side=side,
                strategy="rsmb",
                entry=entry,
                sl=sl,
                target=target,
                score=score,
                status=status,
                reason=reason,
            )
        except Exception as exc:
            logger.warning(f"RSMBStrategy._log_signal: {exc}")
