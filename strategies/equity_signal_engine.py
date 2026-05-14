import pandas as pd
from typing import Dict, Any
from models.ensemble import AIEnsemble
from features.price_features import PriceFeatures
from features.volume_features import VolumeFeatures
from features.time_features import TimeFeatures
from loguru import logger
from tracking.signal_logger import SignalLogger

class EquitySignalEngine:
    """
    Core engine for Equity/F&O signals.
    Combines AI predictions with rule-based validation.
    """
    def __init__(self, config: dict, risk_engine=None):
        self.config = config
        self.ensemble = AIEnsemble(config)
        self.ensemble.load_models()
        self.signal_logger = SignalLogger()
        self.risk_engine = risk_engine  # Optional: used for position sizing

        equity_cfg = config.get("equity_signal", {})
        self.min_buy_conf = self._float_config(equity_cfg, "min_buy_confidence", 0.62)
        self.min_sell_conf = self._float_config(equity_cfg, "min_sell_confidence", 0.62)
        self.min_rel_vol = self._float_config(equity_cfg, "min_relative_volume", 1.0)
        self.time_features = TimeFeatures()
        self._last_signal_bar = {}  # symbol -> timestamp
        logger.info(
            "EquitySignalEngine thresholds loaded | "
            f"buy={self.min_buy_conf:.2f}, sell={self.min_sell_conf:.2f}, rel_vol={self.min_rel_vol:.2f}"
        )

    @staticmethod
    def _float_config(section: dict, key: str, default: float) -> float:
        try:
            return float(section.get(key, default))
        except (TypeError, ValueError):
            logger.warning(f"Invalid equity_signal.{key}; using default {default}")
            return default

    async def process_symbol(self, symbol: str, df_15m: pd.DataFrame, df_5m: pd.DataFrame = None):
        """
        Dual-Timeframe Engine:
        - Brain: 15m candles (indicators, AI inference).
        - Feet: 5m candles (entry timing and VWAP validation).
        """
        if df_15m is None or df_15m.empty:
            return None

        # 0. Deduplication check (Hoist to prevent double logging)
        last_ts = df_15m.index[-1]
        if last_ts is not None and self._last_signal_bar.get(symbol) == last_ts:
            return None 

        # 1. Brain: 15m Indicator Calculation
        df_15m = PriceFeatures.add_indicators(df_15m)
        df_15m = VolumeFeatures.add_volume_analysis(df_15m)
        
        if len(df_15m) < 50: 
            return None

        # 2. Prepare Features for AI (Align with Training)
        feature_cols = [
            'ema_9', 'ema_21', 'ema_50', 'rsi_14', 'atr_14', 
            'vwap', 'ADX_14', 'DMP_14', 'DMN_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
        ]
        
        missing_cols = [col for col in feature_cols if col not in df_15m.columns]
        if missing_cols:
            logger.warning(f"NO TRADE {symbol}: missing feature columns {missing_cols}")
            return None

        current_df = df_15m.tail(1)[feature_cols]
        if current_df.isna().any(axis=None):
            return None
        current_features = current_df.values 
        
        sequence_features = df_15m.tail(30)[feature_cols].values 
        returns = df_15m['close'].pct_change().tail(50).values
        
        # Get AI Score from Ensemble (Trained on 15m)
        ai_score = self.ensemble.get_combined_score(current_features, sequence_features, returns)
        
        # 3. Feet: Validate Signal using 5m data (Timing Layer)
        side = "BUY" if ai_score > 0 else "SELL"
        
        # If no 5m data provided, fallback to 15m for validation
        if df_5m is not None and not df_5m.empty:
            timing_df = PriceFeatures.add_indicators(df_5m)
        else:
            timing_df = df_15m   # indicators already computed in Brain layer
        
        validation = self.validate_setup(symbol, timing_df, abs(ai_score), side)
        
        # 4. LOG TO CSV (For Dashboard)
        status = "TRADE" if validation['valid'] else "NO_TRADE"
        if validation['valid']:
            self._last_signal_bar[symbol] = last_ts
            reason = f"AI: {abs(ai_score):.2f} | 5m-ADX: {validation['metrics']['adx']:.1f} | 15m-Brain Valid"
        else:
            reason = validation['reason']
        
        entry = float(timing_df.iloc[-1]['close'])
        sl = validation.get('sl', entry * 0.99)
        target = validation.get('target', entry * 1.02)
        
        self.signal_logger.log_signal(
            symbol=symbol,
            side=side,
            strategy="Ensemble_AI",
            entry=entry,
            sl=sl,
            target=target,
            score=abs(ai_score),
            status=status,
            reason=reason
        )

        if validation['valid']:
            logger.info(f"🚀 ENSEMBLE SIGNAL: {side} {symbol} | Brain(15m): {abs(ai_score):.2f} | Feet(5m) Valid")

            qty = 1 
            if self.risk_engine is not None:
                sized_qty = self.risk_engine.get_equity_position_size(entry, sl)
                if sized_qty > 0:
                    qty = sized_qty

            return {
                "symbol": symbol,
                "side": side,
                "strategy": "Ensemble_AI",
                "qty": qty,
                "entry": entry,
                "sl": sl,
                "target": target,
                "confidence": abs(ai_score),
            }
            
        return None

    def validate_setup(self, symbol: str, df: pd.DataFrame, ai_score: float, side: str) -> Dict[str, Any]:
        """
        Validates the entry timing on the provided dataframe (usually 5m).
        """
        if df.empty or len(df) < 20:
            return {"valid": False, "reason": "Insufficient timing data"}

        last_row = df.iloc[-1]
        
        # 1. AI Score (Passed from 15m Brain)
        target_conf = self.min_buy_conf if side == "BUY" else self.min_sell_conf
        if ai_score < target_conf:
            return {"valid": False, "reason": f"Low AI confidence (15m): {ai_score:.2f}"}

        # 2. Session Filters
        df_flags = self.time_features.add_session_flags(df.tail(1))
        if df_flags['is_noise_window'].iloc[0]:
            return {"valid": False, "reason": "Inside noise window"}

        # 3. Timing: EMA Alignment (using 5m EMA)
        ema_aligned = False
        if side == "BUY":
            ema_aligned = last_row['ema_9'] > last_row['ema_21']
        else:
            ema_aligned = last_row['ema_9'] < last_row['ema_21']
            
        if not ema_aligned:
            return {"valid": False, "reason": "5m EMAs not aligned (timing)"}

        # 4. Timing: VWAP Position (5m price relative to 5m VWAP)
        if side == "BUY" and last_row['close'] < last_row['vwap']:
            return {"valid": False, "reason": "5m Price below VWAP (timing)"}
        if side == "SELL" and last_row['close'] > last_row['vwap']:
            return {"valid": False, "reason": "5m Price above VWAP (timing)"}

        # 5. Trend Strength (ADX on timing timeframe)
        import math
        adx = last_row.get('ADX_14', float('nan'))
        if math.isnan(adx) or adx < 15: # Lowered to 15 for timing layer
            return {"valid": False, "reason": f"Weak or missing 5m momentum (ADX: {adx})"}

        # 6. Volume filter (H6 fix: reject zero-volume/low-volume candles)
        rel_vol = last_row.get("rel_vol", 0)
        if rel_vol < self.min_rel_vol:
            return {"valid": False, "reason": f"Low relative volume: {rel_vol:.2f} < {self.min_rel_vol}"}

        entry = float(last_row['close'])
        atr = float(last_row.get('atr_14', entry * 0.01))
        risk = max(atr * self.config.get("risk", {}).get("atr_sl_multiplier", 1.5), entry * 0.002)
        rr = self.config.get("risk", {}).get("rr_ratio", 1.5)
        sl = entry - risk if side == "BUY" else entry + risk
        target = entry + (risk * rr) if side == "BUY" else entry - (risk * rr)

        return {
            "valid": True,
            "side": side,
            "entry": entry,
            "sl": sl,
            "target": target,
            "confidence": ai_score,
            "metrics": {
                "adx": last_row['ADX_14'],
                "rsi": last_row['rsi_14'],
                "rel_vol": rel_vol,
            }
        }

    async def run(self):
        """Main loop for signal generation (placeholder for main.py integration)."""
        logger.info("Equity Signal Engine active.")
        # This would be called by the scheduler on candle close
