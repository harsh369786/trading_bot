"""
strategies/gamma_scalper/strategy.py
------------------------------------
Sensex 5m Bi-Directional Gamma Scalper.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal
from strategies.gamma_scalper.ai_filter import GammaAIFilter
from strategies.gamma_scalper.indicators import compute_pcr_delta
from strategies.gamma_scalper.position_manager import GammaPositionManager, infer_option_leg
from strategies.gamma_scalper.signals import evaluate_ce_signal, evaluate_pe_signal
from tracking.signal_logger import SignalLogger


class GammaScalperStrategy(BaseStrategy):
    """5m Sensex ATM options gamma scalper using shared PaperEngine execution."""

    def __init__(self, config: dict, signal_logger: Optional[SignalLogger] = None) -> None:
        self.config = config
        self.settings = config.get("gamma_scalper", {})
        self.enabled = bool(self.settings.get("enabled", False))
        self.symbols = set(self.settings.get("symbols", []))
        self.spot_symbol = self.settings.get("spot_symbol", "SENSEX")
        self._signal_logger = signal_logger
        self._ai_filter = GammaAIFilter()
        self._position_manager = GammaPositionManager(
            max_open_trades=int(self.settings.get("max_open_trades", 2))
        )
        capital_cfg = config.get("capital", {})
        self._capital = float(capital_cfg.get("gamma_total", 30000))
        self._risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100.0
        self._risk_capital = max(self._capital * self._risk_pct, 1.0)
        self._max_notional = float(capital_cfg.get("gamma_total", 30000))
        self._bars: dict[str, pd.DataFrame] = {}
        self._spot_bars: pd.DataFrame = pd.DataFrame()
        self._vix: Optional[float] = None
        logger.info(f"GammaScalperStrategy initialized | enabled={self.enabled} symbols={len(self.symbols)}")

    def push_bars(self, symbol: str, df_5m: pd.DataFrame, spot_df_5m: Optional[pd.DataFrame] = None) -> None:
        self._bars[symbol] = df_5m
        if spot_df_5m is not None:
            self._spot_bars = spot_df_5m

    def update_vix(self, vix_value: float) -> None:
        self._vix = float(vix_value)

    def on_bar(self, symbol: str, bar: pd.Series, spot_bar=None) -> Optional[Signal]:
        if not self.enabled or (self.symbols and symbol not in self.symbols):
            return None
        option_df = self._bars.get(symbol)
        if option_df is None or option_df.empty or len(option_df) < 21:
            return None

        spot_df = self._spot_bars if not self._spot_bars.empty else None
        if spot_df is None:
            logger.debug(f"GammaScalper {symbol}: spot 5m data unavailable; skipping")
            return None

        # Attach neutral PCR delta when upstream PCR is unavailable.
        option_df = option_df.copy()
        if "pcr_delta" not in option_df.columns:
            pcr = option_df["pcr"] if "pcr" in option_df.columns else None
            option_df["pcr_delta"] = compute_pcr_delta(pcr)

        leg = infer_option_leg(symbol)
        if not self._position_manager.can_open(leg):
            self._log_signal(symbol, "BUY", 0, 0, 0, 0, "NO_TRADE", f"{leg}_leg_or_capacity_block")
            return None

        ai_score = self._compute_ai_score(symbol, option_df, spot_df)
        common = {
            "option_df": option_df,
            "spot_df": spot_df,
            "ai_score": ai_score,
            "risk_capital": self._risk_capital,
            "max_notional": self._max_notional,
            "min_candle_strength": float(self.settings.get("min_candle_strength", 10.0)),
            "min_ai_score": float(self.settings.get("min_ai_score", 0.70)) if self._ai_filter.is_loaded else 0.0,
            "min_adx": float(self.settings.get("min_adx", 20.0)),
            "theta_veto_bars": int(self.settings.get("theta_veto_bars", 3)),
        }
        if leg == "CE":
            signal = evaluate_ce_signal(symbol=symbol, **common)
        elif leg == "PE":
            signal = evaluate_pe_signal(symbol=symbol, **common)
        else:
            return None

        if signal is None:
            self._log_signal(symbol, "BUY", 0, 0, 0, ai_score, "NO_TRADE", "conditions_not_met")
            return None

        pos_id = self._position_manager.open_position(signal)
        if pos_id is None:
            self._log_signal(symbol, signal.side, 0, 0, 0, ai_score, "NO_TRADE", "max_open_trades")
            return None
        signal.id = pos_id
        self._log_signal(symbol, signal.side, signal.entry, signal.sl, signal.target1, ai_score, "TRADE", "gamma_setup_valid")
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
        logger.info(f"GammaScalper: {signal.symbol} target {target_num} hit")

    def on_sl_hit(self, signal: Signal) -> None:
        logger.info(f"GammaScalper: {signal.symbol} SL hit")

    def on_price_update(self, symbol: str, price: float):
        return self._position_manager.on_price_update(symbol, price)

    def on_bar_close(self, symbol: str):
        df = self._bars.get(symbol)
        return self._position_manager.on_bar_close(symbol, df) if df is not None else []

    def get_stats(self) -> dict:
        return self._position_manager.get_stats()

    def _compute_ai_score(self, symbol: str, option_df: pd.DataFrame, spot_df: pd.DataFrame) -> float:
        try:
            from strategies.gamma_scalper.signals import _prepare

            prepared = _prepare(option_df, spot_df)
            if not prepared:
                return 0.5
            return self._ai_filter.predict(prepared["features"])
        except Exception as exc:
            logger.warning(f"GammaScalper {symbol}: AI score failed ({exc}); using fallback 0.5")
            return 0.5

    def _log_signal(self, symbol: str, side: str, entry: float, sl: float, target: float, score: float, status: str, reason: str) -> None:
        if self._signal_logger is None:
            return
        try:
            self._signal_logger.log_signal(
                symbol=symbol,
                side=side,
                strategy="gamma_scalper",
                entry=entry,
                sl=sl,
                target=target,
                score=score,
                status=status,
                reason=reason,
            )
        except Exception as exc:
            logger.warning(f"GammaScalper signal log failed: {exc}")
