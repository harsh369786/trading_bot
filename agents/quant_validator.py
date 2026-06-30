import pandas as pd
from loguru import logger
from features.currency_features import CurrencyFeatures
from models.xgboost.model import XGBoostModel

class QuantValidator:
    """
    Agent 3: QuantValidator
    Performs multi-timeframe alignment, hard-reject filter checks, and XGBoost ML scoring.
    """
    def __init__(self, config: dict):
        self.config = config
        self.xgb = XGBoostModel()
        self.xgb.load()
        
        self.feature_cols = [
            "ema_9", "ema_21", "ema_50", "rsi_14", "atr_14",
            "vwap", "ADX_14", "DMP_14", "DMN_14",
            "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0",
        ]

    def validate(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, side: str) -> dict:
        """
        Returns {valid: bool, reason: str, quant_score: int}
        """
        last_5m = df_5m.iloc[-1]
        last_15m = df_15m.iloc[-1]
        
        # 1. Multi-Timeframe Alignment
        if side == "BUY" and last_15m['close'] < last_15m['ema_21']:
            return {"valid": False, "reason": "MTF: 15m trend is bearish"}
        if side == "SELL" and last_15m['close'] > last_15m['ema_21']:
            return {"valid": False, "reason": "MTF: 15m trend is bullish"}
            
        # 2. ATR Range (Reject if too quiet or too volatile)
        atr = last_5m.get('atr_14', 0)
        if atr < 0.03: return {"valid": False, "reason": "Low volatility (ATR < 0.03)"}
        if atr > 0.25: return {"valid": False, "reason": "Extreme volatility (ATR > 0.25)"}
        
        # 3. Pivot Proximity (Using CurrencyFeatures)
        if CurrencyFeatures.calculate_pivot_proximity(df_5m.tail(1)).iloc[0]:
            return {"valid": False, "reason": "Price too close to Pivot level"}
            
        # 4. RSI Extremes
        if side == "BUY" and last_5m['rsi_14'] > 75: return {"valid": False, "reason": "RSI Overbought"}
        if side == "SELL" and last_5m['rsi_14'] < 25: return {"valid": False, "reason": "RSI Oversold"}

        # 5. XGBoost ML Scoring
        missing_features = [c for c in self.feature_cols if c not in df_5m.columns]
        if missing_features:
            return {"valid": False, "reason": f"Missing features for ML: {missing_features}"}
            
        features = df_5m.tail(1)[self.feature_cols].astype(float)
        try:
            probs = self.xgb.predict(features)
            if getattr(probs, "ndim", 1) == 2:
                probs = probs[0]
            
            if side == "BUY":
                confidence = probs[1]
            else:
                confidence = probs[2]
            threshold = float(self.config.get("currency_signal", {}).get("min_quant_score", 70)) / 100.0
                
            quant_score = int(confidence * 100)
            
            if confidence < threshold:
                return {"valid": False, "reason": f"Low ML confidence: {quant_score}%"}
                
        except Exception as e:
            logger.error(f"XGBoost inference failed: {e}")
            return {"valid": False, "reason": f"ML Error: {str(e)}"}

        return {"valid": True, "reason": "All checks passed", "quant_score": quant_score}
