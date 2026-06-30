"""
strategies/mean_reversion/ai_filter.py
--------------------------------------
XGBoost/joblib AI filter for 15m 200-SMA mean reversion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger


MODEL_PATH = Path("models/meanrev_xgb.pkl")
FALLBACK_SCORE = 0.5

FEATURE_COLS = [
    "rsi_14",
    "adx_14",
    "distance_pct",
    "wick_ratio",
    "candle_pattern_encoded",
    "bb_width",
    "volume_ratio",
    "sma200_slope",
    "hour_of_day",
    "day_of_week",
    "ema20_1h_dist_pct",
    "atr_pct",
]

PATTERN_ENCODING = {
    "none": 0.0,
    "hammer": 1.0,
    "bullish_engulfing": 2.0,
    "shooting_star": -1.0,
    "bearish_engulfing": -2.0,
}


class MeanRevAIFilter:
    """Read-only inference wrapper. Missing/corrupt model returns fallback 0.5."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model: Optional[object] = None
        self._loaded = False
        self._warned_missing = False
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            if not self._warned_missing:
                logger.warning(f"MeanRevAIFilter: model missing at {self.model_path}; using fallback score=0.5")
                self._warned_missing = True
            return
        try:
            self._model = joblib.load(self.model_path)
            self._loaded = True
            logger.info(f"MeanRevAIFilter: model loaded from {self.model_path}")
        except Exception as exc:
            logger.warning(f"MeanRevAIFilter: failed to load model ({exc}); using fallback score=0.5")
            self._model = None
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: dict[str, float]) -> float:
        if not self._loaded or self._model is None:
            return FALLBACK_SCORE
        try:
            row = [float(features.get(col, 0.0) or 0.0) for col in FEATURE_COLS]
            arr = np.asarray(row, dtype=float).reshape(1, -1)
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(arr)
                return float(proba[0, 1])
            return float(self._model.predict(arr)[0])
        except Exception as exc:
            logger.warning(f"MeanRevAIFilter.predict failed ({exc}); using fallback score=0.5")
            return FALLBACK_SCORE

    @staticmethod
    def extract_features(
        bar: pd.Series,
        rsi_14: float,
        adx_14: float,
        distance_pct: float,
        wick_ratio_value: float,
        pattern: str,
        bb_upper: float,
        bb_lower: float,
        volume_ratio: float,
        sma200_slope: float,
        ema20_1h_dist_pct: float,
        atr_pct_value: float,
    ) -> dict[str, float]:
        close = float(bar.get("close", 0.0) or 0.0)
        bb_width = (bb_upper - bb_lower) / close if close and bb_upper and bb_lower else 0.0
        ts = bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp.now(tz="Asia/Kolkata")
        return {
            "rsi_14": rsi_14,
            "adx_14": adx_14,
            "distance_pct": distance_pct,
            "wick_ratio": wick_ratio_value,
            "candle_pattern_encoded": PATTERN_ENCODING.get(pattern, 0.0),
            "bb_width": bb_width,
            "volume_ratio": volume_ratio,
            "sma200_slope": sma200_slope,
            "hour_of_day": float(ts.hour + ts.minute / 60.0),
            "day_of_week": float(ts.dayofweek),
            "ema20_1h_dist_pct": ema20_1h_dist_pct,
            "atr_pct": atr_pct_value,
        }

    def retrain(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Scheduler-only retrain hook. Do not call intraday."""
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            logger.error(f"MeanRevAIFilter.retrain: xgboost unavailable: {exc}")
            return False
        if len(X) < 30 or any(col not in X.columns for col in FEATURE_COLS):
            logger.warning("MeanRevAIFilter.retrain: insufficient rows or missing feature columns")
            return False
        try:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X[FEATURE_COLS], y)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            self._model = model
            self._loaded = True
            return True
        except Exception as exc:
            logger.error(f"MeanRevAIFilter.retrain failed: {exc}")
            return False

