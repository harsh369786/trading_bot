import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.price_features import PriceFeatures
from models.xgboost.model import XGBoostModel


class RetrainPipeline:
    """
    Weekly ML retraining pipeline.
    Trains XGBoost on all available historical parquet files.
    """
    def __init__(self, data_path: str = "data/historical/*.parquet"):
        self.data_path = data_path
        self.xgb_model = XGBoostModel()

    def _load_files(self) -> pd.DataFrame:
        files = sorted(glob.glob(self.data_path)) if any(ch in self.data_path for ch in "*?[]") else [self.data_path]
        frames = []
        for path in files:
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            if df.empty:
                continue
            symbol = os.path.basename(path).replace("_6m.parquet", "")
            df = df.copy()
            df["symbol"] = symbol
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from {path}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    def _prepare_symbol_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = PriceFeatures.add_indicators(df.copy())
        df.dropna(inplace=True)

        lookahead = 3
        df["future_close"] = df["close"].shift(-lookahead)
        # Target must be aligned with RiskManager (e.g., minimum 1.5x of the 0.5 ATR stop loss)
        # So we require a minimum move of 0.75 * ATR to consider it a valid signal
        threshold = df["atr_14"] * 0.75
        diff = df["future_close"] - df["close"]
        df["label"] = np.select([diff > threshold, diff < -threshold], [1, 2], default=0)
        df.loc[df["future_close"].isna(), "label"] = np.nan
        df.dropna(subset=["label"], inplace=True)
        return df

    def run(self):
        logger.info("Starting retraining pipeline...")
        df_all = self._load_files()
        if df_all.empty:
            logger.error(f"No training data found at {self.data_path}")
            return

        prepared = []
        for symbol, df_symbol in df_all.groupby("symbol", sort=False):
            if len(df_symbol) < 80:
                logger.warning(f"Skipping {symbol}: only {len(df_symbol)} rows")
                continue
            symbol_prepared = self._prepare_symbol_frame(df_symbol.drop(columns=["symbol"]))
            symbol_prepared["symbol"] = symbol
            prepared.append(symbol_prepared)
            logger.info(f"{symbol}: prepared {len(symbol_prepared)} labeled rows")

        if not prepared:
            logger.error("No symbols produced enough labeled rows for training.")
            return

        df = pd.concat(prepared).sort_index()
        feature_cols = [
            "ema_9", "ema_21", "ema_50", "rsi_14", "atr_14",
            "vwap", "ADX_14", "DMP_14", "DMN_14",
            "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0",
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df["label"].astype(int)
        valid = ~X.isna().any(axis=1)
        X = X[valid]
        y = y[valid]

        if len(X) < 200:
            logger.error(f"Not enough clean training rows: {len(X)}")
            return

        logger.info(f"Training rows: {len(X)} | Label distribution: {y.value_counts().to_dict()}")

        # --- C3 FIX: Time-based split (no shuffle, strict chronological boundary) ---
        # Use the last 20% of time as the test set.
        cutoff_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:cutoff_idx], X.iloc[cutoff_idx:]
        y_train, y_test = y.iloc[:cutoff_idx], y.iloc[cutoff_idx:]
        logger.info(
            f"Train: {len(X_train)} rows (up to {X_train.index[-1]}), "
            f"Test: {len(X_test)} rows (from {X_test.index[0]})"
        )

        self.xgb_model.train(X_train, y_train)

        # --- C4 FIX: Batch prediction (no Python loop) ---
        y_pred_probs = self.xgb_model.predict(X_test.values)  # returns (N, 3) array
        if y_pred_probs.ndim == 2:
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            # Fallback for single-row edge case
            y_pred = np.array([np.argmax(y_pred_probs)])

        # --- H1 FIX: Full metrics report ---
        label_names = ["NEUTRAL", "BUY", "SELL"]
        accuracy = (y_pred == y_test.values).mean()
        report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Baseline: always predict NEUTRAL (no-trade baseline)
        baseline_majority = y_train.mode()[0]
        baseline_pred = np.full(len(y_test), baseline_majority)
        baseline_accuracy = (baseline_pred == y_test.values).mean()
        baseline_label = label_names[baseline_majority]

        logger.info(f"\n{'='*60}")
        logger.info(f"RETRAINING RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
        logger.info(f"Label distribution (test): {dict(zip(*np.unique(y_test, return_counts=True)))}")
        logger.info(f"Model accuracy:   {accuracy:.2%}")
        logger.info(f"Baseline ({baseline_label}): {baseline_accuracy:.2%}  ← model must beat this")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"{'='*60}")

        if accuracy <= baseline_accuracy:
            logger.warning(
                "Model does NOT beat the no-trade baseline! "
                "Do not deploy — inspect feature quality and label balance."
            )

        # --- L1 FIX: Save model metadata sidecar ---
        meta = {
            "trained_at": datetime.now().isoformat(),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "model_accuracy": round(float(accuracy), 4),
            "baseline_accuracy": round(float(baseline_accuracy), 4),
            "baseline_class": baseline_label,
            "beats_baseline": bool(accuracy > baseline_accuracy),
            "classification_report": report,
            "label_distribution_train": y_train.value_counts().to_dict(),
            "label_distribution_test": y_test.value_counts().to_dict(),
        }
        meta_path = self.xgb_model.model_path.replace(".json", "_metadata.json")
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.success(f"Model metadata saved to {meta_path}")


if __name__ == "__main__":
    RetrainPipeline().run()
