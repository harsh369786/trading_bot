"""
strategies/gamma_scalper/ai_filter.py
-------------------------------------
XGBoost/joblib AI filter for Gamma Scalper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger


MODEL_PATH = Path("models/gamma_xgb.pkl")
FALLBACK_SCORE = 0.5

FEATURE_COLS = [
    "candle_strength",
    "rsi_14",
    "adx_14",
    "pcr_delta",
    "volume_ratio",
    "vwap_dist_pct",
    "ema9_dist_pct",
    "bb_width",
    "hour_of_day",
    "day_of_week",
    "option_oi_change_pct",
]


class GammaAIFilter:
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
                logger.warning(f"GammaAIFilter: model missing at {self.model_path}; using fallback score=0.5")
                self._warned_missing = True
            return
        try:
            self._model = joblib.load(self.model_path)
            self._loaded = True
            logger.info(f"GammaAIFilter: model loaded from {self.model_path}")
        except Exception as exc:
            logger.warning(f"GammaAIFilter: failed to load model ({exc}); using fallback score=0.5")
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
            logger.warning(f"GammaAIFilter.predict failed ({exc}); using fallback score=0.5")
            return FALLBACK_SCORE

    @staticmethod
    def extract_features(
        bar: pd.Series,
        candle_strength_value: float,
        rsi_14: float,
        adx_14: float,
        pcr_delta: float,
        volume_ratio: float,
        option_vwap: float,
        ema_9_spot: float,
        bb_upper: float = 0.0,
        bb_lower: float = 0.0,
    ) -> dict[str, float]:
        close = float(bar.get("close", 0.0) or 0.0)
        vwap_dist_pct = (close - option_vwap) / option_vwap * 100.0 if option_vwap else 0.0
        ema9_dist_pct = (close - ema_9_spot) / ema_9_spot * 100.0 if ema_9_spot else 0.0
        bb_width = (bb_upper - bb_lower) / close if close and bb_upper and bb_lower else 0.0
        ts = bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp.now(tz="Asia/Kolkata")
        oi = float(bar.get("oi", 0.0) or 0.0)
        prev_oi = float(bar.get("prev_oi", oi) or oi)
        oi_change = (oi - prev_oi) / prev_oi * 100.0 if prev_oi else 0.0
        return {
            "candle_strength": candle_strength_value,
            "rsi_14": rsi_14,
            "adx_14": adx_14,
            "pcr_delta": pcr_delta,
            "volume_ratio": volume_ratio,
            "vwap_dist_pct": vwap_dist_pct,
            "ema9_dist_pct": ema9_dist_pct,
            "bb_width": bb_width,
            "hour_of_day": float(ts.hour + ts.minute / 60.0),
            "day_of_week": float(ts.dayofweek),
            "option_oi_change_pct": oi_change,
        }

    def retrain(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Scheduler-only retrain hook. Do not call intraday."""
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            logger.error(f"GammaAIFilter.retrain: xgboost unavailable: {exc}")
            return False
        if len(X) < 30 or any(col not in X.columns for col in FEATURE_COLS):
            logger.warning("GammaAIFilter.retrain: insufficient rows or missing feature columns")
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
            logger.error(f"GammaAIFilter.retrain failed: {exc}")
            return False

