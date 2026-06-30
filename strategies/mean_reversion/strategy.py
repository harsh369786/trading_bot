"""
strategies/mean_reversion/strategy.py
-------------------------------------
15m 200-SMA Mean Reversion strategy.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal
from strategies.mean_reversion.ai_filter import MeanRevAIFilter
from strategies.mean_reversion.position_manager import MeanRevPositionManager
from strategies.mean_reversion.signals import evaluate_buy_signal, evaluate_sell_signal
from tracking.signal_logger import SignalLogger


class MeanReversionStrategy(BaseStrategy):
    """Equity/F&O 15m 200-SMA mean reversion strategy using shared PaperEngine."""

    def __init__(self, config: dict, signal_logger: Optional[SignalLogger] = None) -> None:
        self.config = config
        self.settings = config.get("mean_reversion", {})
        self.enabled = bool(self.settings.get("enabled", False))
        self.symbols = set(config.get("instruments", {}).get("equity", []))
        self._signal_logger = signal_logger
        self._ai_filter = MeanRevAIFilter()
        self._position_manager = MeanRevPositionManager(
            max_open_trades=int(self.settings.get("max_open_trades", 3))
        )
        capital_cfg = config.get("capital", {})
        self._capital = float(capital_cfg.get("meanrev_total", 40000))
        self._risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100.0
        self._risk_capital = max(self._capital * self._risk_pct, 1.0)
        self._max_notional = float(capital_cfg.get("meanrev_total", 40000))
        self._bars: dict[str, pd.DataFrame] = {}
        self._bars_1h: dict[str, pd.DataFrame] = {}
        logger.info(f"MeanReversionStrategy initialized | enabled={self.enabled} universe={len(self.symbols)}")

    def push_bars(self, symbol: str, df_15m: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> None:
        self._bars[symbol] = df_15m
        if df_1h is not None:
            self._bars_1h[symbol] = df_1h

    def on_bar(self, symbol: str, bar: pd.Series, df_1h: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        if not self.enabled or symbol not in self.symbols:
            return None
        df_15m = self._bars.get(symbol)
        if df_15m is None or df_15m.empty:
            return None
        if df_1h is None:
            df_1h = self._bars_1h.get(symbol)
        if df_1h is None or df_1h.empty:
            logger.warning(f"MeanReversion {symbol}: 1h confirmation unavailable; not vetoing")

        if not self._position_manager.can_open(symbol):
            self._log_signal(symbol, "NA", 0, 0, 0, 0, "NO_TRADE", "active_symbol_or_max_open")
            return None

        ai_score = self._compute_ai_score(symbol, df_15m, df_1h)
        common = {
            "df_15m": df_15m,
            "df_1h": df_1h,
            "ai_score": ai_score,
            "risk_capital": self._risk_capital,
            "max_notional": self._max_notional,
            "min_ai_score": float(self.settings.get("min_ai_score", 0.60)) if self._ai_filter.is_loaded else 0.0,
            "max_adx": float(self.settings.get("max_adx", 35.0)),
            "min_distance_pct": float(self.settings.get("min_distance_pct", 3.0)),
            "wick_ratio_min": float(self.settings.get("wick_ratio_min", 2.0)),
        }
        signal = evaluate_buy_signal(symbol=symbol, **common)
        if signal is None:
            signal = evaluate_sell_signal(symbol=symbol, **common)

        if signal is None:
            self._log_signal(symbol, "NA", 0, 0, 0, ai_score, "NO_TRADE", "conditions_not_met")
            return None

        pos_id = self._position_manager.open_position(signal)
        if pos_id is None:
            self._log_signal(symbol, signal.side, 0, 0, 0, ai_score, "NO_TRADE", "max_open_trades")
            return None
        signal.id = pos_id
        self._log_signal(
            symbol,
            signal.side,
            signal.entry,
            signal.sl,
            signal.target1,
            ai_score,
            "TRADE",
            "mean_reversion_setup_valid",
        )
        return signal

    def on_fill(self, signal: Signal, fill_price: float) -> None:
        if signal.id:
            self._position_manager.on_fill(signal.id, fill_price, pd.Timestamp.now(tz="Asia/Kolkata"))

    def bind_order_id(self, signal: Signal, order_id: str) -> None:
        if signal.id:
            self._position_manager.bind_order_id(signal.id, order_id)

    def paper_order_id_for(self, position_id: str) -> Optional[str]:
        return self._position_manager.paper_order_id_for(position_id)

    def cancel_pending(self, signal: Signal) -> None:
        if signal.id:
            self._position_manager.cancel_pending(signal.id)

    def on_target_hit(self, signal: Signal, target_num: int) -> None:
        logger.info(f"MeanReversion: {signal.symbol} target {target_num} hit")

    def on_sl_hit(self, signal: Signal) -> None:
        logger.info(f"MeanReversion: {signal.symbol} SL hit")

    def on_price_update(self, symbol: str, price: float):
        return self._position_manager.on_price_update(symbol, price)

    def get_stats(self) -> dict:
        return self._position_manager.get_stats()

    def _compute_ai_score(self, symbol: str, df_15m: pd.DataFrame, df_1h: Optional[pd.DataFrame]) -> float:
        try:
            from strategies.mean_reversion.signals import _features, _prepare

            prepared = _prepare(df_15m, df_1h)
            if not prepared:
                return 0.5
            wick_value = max(prepared["upper_wick_ratio"], prepared["lower_wick_ratio"])
            return self._ai_filter.predict(_features(prepared, wick_value))
        except Exception as exc:
            logger.warning(f"MeanReversion {symbol}: AI score failed ({exc}); using fallback 0.5")
            return 0.5

    def _log_signal(self, symbol: str, side: str, entry: float, sl: float, target: float, score: float, status: str, reason: str) -> None:
        if self._signal_logger is None:
            return
        try:
            self._signal_logger.log_signal(
                symbol=symbol,
                side=side,
                strategy="mean_reversion",
                entry=entry,
                sl=sl,
                target=target,
                score=score,
                status=status,
                reason=reason,
            )
        except Exception as exc:
            logger.warning(f"MeanReversion signal log failed: {exc}")
