"""
strategies/rsmb/ai_filter.py
------------------------------
XGBoost AI filter for the RSMB strategy.

Spec compliance:
- Model type: XGBoost (sklearn-compatible via xgboost.XGBClassifier)
- Persistence: models/rsmb_xgb.pkl via joblib
- If model file missing or corrupted: log WARNING, return score=0.5 (do NOT block)
- Prediction returns float probability of class=1 (profitable trade)
- Retrain: only on Sunday 06:00 IST or manual trigger — NEVER intraday
- Features: [rsi_14, macd_hist, atr_pct, volume_ratio, rs_rank, vwap_dist_pct,
             ema21_dist_pct, adx_14, bb_width, hour_of_day, day_of_week]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — RSMB AI filter will use fallback score=0.5")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/rsmb_xgb.pkl")
FALLBACK_SCORE = 0.5

FEATURE_COLS = [
    "rsi_14",
    "macd_hist",
    "atr_pct",
    "volume_ratio",
    "rs_rank",
    "vwap_dist_pct",
    "ema21_dist_pct",
    "adx_14",
    "bb_width",
    "hour_of_day",
    "day_of_week",
]


# ---------------------------------------------------------------------------
# RSMBAIFilter
# ---------------------------------------------------------------------------

class RSMBAIFilter:
    """
    Wraps the XGBoost model for RSMB signal scoring.

    Thread-safety: predict() is read-only and thread-safe after load().
    retrain() must be called from a single scheduler thread (never intraday).
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._model: Optional[object] = None
        self._loaded: bool = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load model from disk. On any failure, fall back to score=0.5."""
        if not XGB_AVAILABLE:
            logger.warning("RSMBAIFilter: xgboost not available — using fallback score")
            return

        if not self.model_path.exists():
            logger.warning(
                f"RSMBAIFilter: model file not found at {self.model_path}. "
                "Will use fallback score=0.5 until retrain() is called."
            )
            return

        try:
            self._model = joblib.load(self.model_path)
            self._loaded = True
            logger.info(f"RSMBAIFilter: model loaded from {self.model_path}")
        except Exception as exc:
            logger.warning(
                f"RSMBAIFilter: failed to load model ({exc}). "
                "Using fallback score=0.5."
            )
            self._model = None
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: dict[str, float]) -> float:
        """
        Return probability of class=1 (profitable trade) in [0, 1].

        Parameters
        ----------
        features : dict with keys matching FEATURE_COLS.
                   Any missing key will be filled with 0.0.

        Returns
        -------
        float — probability [0, 1]. Returns FALLBACK_SCORE on any error.
        """
        if not self._loaded or self._model is None:
            return FALLBACK_SCORE

        try:
            row = [features.get(col, 0.0) for col in FEATURE_COLS]
            arr = np.array(row, dtype=float).reshape(1, -1)

            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(arr)
                # Binary classifier: class 0 = loss, class 1 = profit
                return float(proba[0, 1])
            else:
                # Direct predict returns 0 or 1
                return float(self._model.predict(arr)[0])

        except Exception as exc:
            logger.warning(f"RSMBAIFilter.predict: inference failed ({exc}); using fallback")
            return FALLBACK_SCORE

    # ------------------------------------------------------------------
    # Feature extraction from a bar
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(
        bar: pd.Series,
        rs_rank: float,
        vwap: float,
        ema21: float,
        atr: float,
        volume_ratio: float,
        adx: float,
        bb_upper: float,
        bb_lower: float,
    ) -> dict[str, float]:
        """
        Build the feature dict from a completed 15m bar and pre-computed indicators.

        Parameters
        ----------
        bar          : Last completed 15m bar (pd.Series with close, rsi_14, macd_hist, etc.)
        rs_rank      : Pre-computed RS_Rank (float, may be NaN)
        vwap         : Session VWAP at bar close
        ema21        : EMA 21 value at bar close
        atr          : ATR 14 value at bar close
        volume_ratio : volume / mean(last 5 bars volume)
        adx          : ADX 14 at bar close
        bb_upper     : Bollinger Band upper
        bb_lower     : Bollinger Band lower

        Returns
        -------
        dict[str, float] — feature dict matching FEATURE_COLS.
        """
        close = float(bar.get("close", 0.0))
        bb_width = (
            (bb_upper - bb_lower) / close if close != 0 else 0.0
        )
        vwap_dist_pct = (
            (close - vwap) / vwap * 100 if vwap != 0 else 0.0
        )
        ema21_dist_pct = (
            (close - ema21) / ema21 * 100 if ema21 != 0 else 0.0
        )
        atr_pct = (atr / close * 100) if close != 0 else 0.0

        # Timestamp-based features
        ts = bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp.now(tz="Asia/Kolkata")
        hour_of_day = float(ts.hour + ts.minute / 60.0)
        day_of_week = float(ts.dayofweek)  # 0=Mon, 4=Fri

        return {
            "rsi_14": float(bar.get("rsi_14", 50.0)),
            "macd_hist": float(bar.get("macd_hist", 0.0)),
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio if not np.isnan(volume_ratio) else 1.0,
            "rs_rank": rs_rank if not np.isnan(rs_rank) else 1.0,
            "vwap_dist_pct": vwap_dist_pct,
            "ema21_dist_pct": ema21_dist_pct,
            "adx_14": adx,
            "bb_width": bb_width,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
        }

    # ------------------------------------------------------------------
    # Training (Sunday 06:00 IST only — scheduler calls this)
    # ------------------------------------------------------------------

    def retrain(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Retrain the XGBoost model on historical signal data.

        MUST only be called from the Sunday 06:00 IST scheduler job.
        Never call intraday.

        Parameters
        ----------
        X : Feature DataFrame with columns matching FEATURE_COLS.
        y : Label Series (1 = profitable trade, 0 = loss).

        Returns
        -------
        bool — True if retrain succeeded and model was saved.
        """
        if not XGB_AVAILABLE:
            logger.error("RSMBAIFilter.retrain: xgboost not installed; cannot retrain")
            return False

        if len(X) < 30:
            logger.warning(
                f"RSMBAIFilter.retrain: only {len(X)} samples available "
                "(need ≥30 for reliable training); skipping retrain."
            )
            return False

        # Ensure correct feature order
        missing = set(FEATURE_COLS) - set(X.columns)
        if missing:
            logger.error(f"RSMBAIFilter.retrain: missing feature columns {missing}")
            return False

        X_ordered = X[FEATURE_COLS].copy()

        try:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_ordered, y)

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)

            self._model = model
            self._loaded = True
            logger.success(
                f"RSMBAIFilter: retrain complete — {len(X)} samples, "
                f"saved to {self.model_path}"
            )
            return True

        except Exception as exc:
            logger.error(f"RSMBAIFilter.retrain: training failed — {exc}")
            return False

    def build_training_data(
        self, signal_log_path: str = "data/signal_log.csv"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build (X, y) training pairs from the historical signal log.

        Signals with status='TRADE' and pnl_after_costs > 0 are labelled 1 (win).
        Signals with status='TRADE' and pnl_after_costs <= 0 are labelled 0 (loss).
        NO_TRADE signals are excluded.

        Returns
        -------
        (X, y) — feature DataFrame and label Series.
        Empty if insufficient data.
        """
        if not os.path.exists(signal_log_path):
            logger.warning(f"RSMBAIFilter.build_training_data: {signal_log_path} not found")
            return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

        try:
            df = pd.read_csv(signal_log_path, on_bad_lines="skip")
        except Exception as exc:
            logger.error(f"RSMBAIFilter.build_training_data: read failed — {exc}")
            return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

        # Filter RSMB trades only
        if "strategy" in df.columns:
            df = df[df["strategy"] == "rsmb"]

        trades = df[df.get("status", pd.Series()) == "TRADE"].copy()

        # All FEATURE_COLS must be present to build training data
        available = [col for col in FEATURE_COLS if col in trades.columns]
        if len(available) < len(FEATURE_COLS):
            missing = set(FEATURE_COLS) - set(available)
            logger.warning(
                f"RSMBAIFilter.build_training_data: missing feature columns {missing} "
                "in signal log; cannot build training set."
            )
            return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

        pnl_col = "pnl_after_costs" if "pnl_after_costs" in trades.columns else None
        if pnl_col is None:
            logger.warning("RSMBAIFilter.build_training_data: no pnl_after_costs column found")
            return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

        X = trades[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = (trades[pnl_col].apply(pd.to_numeric, errors="coerce").fillna(0.0) > 0).astype(int)
        return X, y
