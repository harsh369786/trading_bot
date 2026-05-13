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

        self.min_buy_conf = config.get("equity_signal", {}).get("min_buy_confidence", 0.72)
        self.min_sell_conf = config.get("equity_signal", {}).get("min_sell_confidence", 0.70)
        self.min_rel_vol = config.get("equity_signal", {}).get("min_relative_volume", 1.0)
        self.time_features = TimeFeatures()
        self._last_signal_bar = {}  # symbol -> timestamp

    async def process_symbol(self, symbol: str, df: pd.DataFrame):
        """
        Called when a new candle closes.
        Computes features -> AI Inference -> Signal Validation.
        """
        # 1. Add all technical indicators
        df = PriceFeatures.add_indicators(df)
        df = VolumeFeatures.add_volume_analysis(df)
        
        if len(df) < 50: 
            return

        # 2. Prepare Features for AI (Align with Training)
        feature_cols = [
            'ema_9', 'ema_21', 'ema_50', 'rsi_14', 'atr_14', 
            'vwap', 'ADX_14', 'DMP_14', 'DMN_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
        ]
        
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"NO TRADE {symbol}: missing feature columns {missing_cols}")
            return None

        current_df = df.tail(1)[feature_cols]
        if current_df.isna().any(axis=None):
            logger.debug(f"NO TRADE {symbol}: latest feature row contains NaN values")
            return None
        current_features = current_df.values # Shape (1, 12)
        
        # Sequence features for LSTM (Placeholder/Internal)
        sequence_features = df.tail(30).values 
        returns = df['close'].pct_change().tail(50).values
        
        # 3. Get AI Score from Ensemble
        ai_score = self.ensemble.get_combined_score(current_features, sequence_features, returns)
        
        # 4. Validate Signal
        side = "BUY" if ai_score > 0 else "SELL"
        validation = self.validate_setup(symbol, df, abs(ai_score), side)
        
        # Deduplication check
        last_ts = df.index[-1] if not df.empty and isinstance(df.index, pd.DatetimeIndex) else None
        if last_ts and self._last_signal_bar.get(symbol) == last_ts:
            return None  # Already fired a signal for this candle
            
        if validation['valid']:
            self._last_signal_bar[symbol] = last_ts
        
        # 5. LOG TO CSV (For Dashboard)
        status = "TRADE" if validation['valid'] else "NO_TRADE"
        reason = "" if validation['valid'] else validation['reason']
        
        # Calculate dummy SL/Target for log if not valid (for visualization)
        entry = df.iloc[-1]['close']
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
            logger.info(f"🚀 EQUITY SIGNAL: {side} {symbol} | Conf: {abs(ai_score):.2f}")

            # Position sizing via RiskEngine if available (C2 fix)
            qty = 1  # safe fallback
            if self.risk_engine is not None:
                sized_qty = self.risk_engine.get_equity_position_size(
                    validation.get("entry", entry),
                    validation.get("sl", sl),
                )
                if sized_qty > 0:
                    qty = sized_qty
                else:
                    logger.warning(f"RiskEngine returned qty=0 for {symbol}; using qty=1 fallback.")

            # Construct full signal for OrderManager
            full_signal = {
                "symbol": symbol,
                "side": side,
                "strategy": "Ensemble_AI",
                "qty": qty,
                "entry": validation.get("entry", entry),
                "sl": validation.get("sl", sl),
                "target": validation.get("target", target),
                "confidence": abs(ai_score),
            }
            return full_signal
            
        return None

    def validate_setup(self, symbol: str, df: pd.DataFrame, ai_score: float, side: str) -> Dict[str, Any]:
        """
        Validates a signal using the strict requirements from Module 3.
        """
        if df.empty or len(df) < 50:
            return {"valid": False, "reason": "Insufficient data"}

        last_row = df.iloc[-1]
        
        # 1. Check AI Confidence
        target_conf = self.min_buy_conf if side == "BUY" else self.min_sell_conf
        if ai_score < target_conf:
            return {"valid": False, "reason": f"Low AI confidence: {ai_score:.2f}"}

        # 2. Check Time Filters
        df_flags = self.time_features.add_session_flags(df.tail(1))
        if df_flags['is_noise_window'].iloc[0]:
            return {"valid": False, "reason": "Inside noise window"}
        if df_flags['is_chop_zone'].iloc[0]:
            # Optional: apply higher threshold in chop zone
            pass

        # 3. Trend Alignment (EMA 9 > 21 > 50 for BUY)
        ema_aligned = False
        if side == "BUY":
            ema_aligned = last_row['ema_9'] > last_row['ema_21'] > last_row['ema_50']
        else:
            ema_aligned = last_row['ema_9'] < last_row['ema_21'] < last_row['ema_50']
            
        if not ema_aligned:
            return {"valid": False, "reason": f"EMA not aligned for {side}"}

        # 4. VWAP Position
        if side == "BUY" and last_row['close'] < last_row['vwap']:
            return {"valid": False, "reason": "Price below VWAP"}
        if side == "SELL" and last_row['close'] > last_row['vwap']:
            return {"valid": False, "reason": "Price above VWAP"}

        # 5. Trend Strength (ADX > 20)
        if last_row['ADX_14'] < 20:
            return {"valid": False, "reason": "Weak trend (ADX < 20)"}

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
