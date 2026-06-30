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
- Features: stationary price-distance features plus normalized indicators:
  [dist_ema_9, dist_ema_21, dist_ema_50, rsi_14, atr_pct, dist_vwap,
   ADX_14, DMP_14, DMN_14, bb_pct, rs_rank]
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
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "rsi_14", "atr_pct",
    "dist_vwap", "ADX_14", "DMP_14", "DMN_14", "bb_pct", "rs_rank",
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
        **_,
    ) -> dict[str, float]:
        """Build stationary feature dict from a completed 15m bar.

        Extra keyword arguments are accepted for compatibility with older training
        scripts, but the model feature contract is derived from the bar itself.
        """
        def safe_float(value, default: float = 0.0) -> float:
            try:
                parsed = float(value)
                if np.isnan(parsed) or np.isinf(parsed):
                    return default
                return parsed
            except (TypeError, ValueError):
                return default

        close = safe_float(bar.get("close"), 0.0)
        vwap = safe_float(bar.get("vwap"), close)
        lower = safe_float(bar.get("BBL_20_2.0"), close)
        upper = safe_float(bar.get("BBU_20_2.0"), close)
        bb_width = upper - lower

        features = {
            "dist_ema_9": safe_float(bar.get("dist_ema_9"), (close - safe_float(bar.get("ema_9"), close)) / close if close else 0.0),
            "dist_ema_21": safe_float(bar.get("dist_ema_21"), (close - safe_float(bar.get("ema_21"), close)) / close if close else 0.0),
            "dist_ema_50": safe_float(bar.get("dist_ema_50"), (close - safe_float(bar.get("ema_50"), close)) / close if close else 0.0),
            "rsi_14": safe_float(bar.get("rsi_14"), 50.0),
            "atr_pct": safe_float(bar.get("atr_pct"), safe_float(bar.get("atr_14"), 0.0) / close if close else 0.0),
            "dist_vwap": safe_float(bar.get("dist_vwap"), (close - vwap) / vwap if vwap else 0.0),
            "ADX_14": safe_float(bar.get("ADX_14"), 0.0),
            "DMP_14": safe_float(bar.get("DMP_14"), 0.0),
            "DMN_14": safe_float(bar.get("DMN_14"), 0.0),
            "bb_pct": safe_float(bar.get("bb_pct"), (close - lower) / bb_width if bb_width else 0.5),
            "rs_rank": safe_float(rs_rank, 1.0),
        }
        return features

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
