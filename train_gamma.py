import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from loguru import logger
from pathlib import Path

# Feature columns defined in GammaAIFilter (lowercase)
FEATURE_COLS = [
    "candle_strength", "rsi_14", "adx_14", "pcr_delta", "volume_ratio",
    "vwap_dist_pct", "ema9_dist_pct", "bb_width", "hour_of_day",
    "day_of_week", "option_oi_change_pct",
]

def train_real_gamma_model():
    model_path = Path("models/gamma_xgb.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    spot_path = Path("data/historical/SENSEX_6m.parquet")
    if not spot_path.exists():
        logger.error(f"Missing Sensex spot data at {spot_path}")
        return
        
    logger.info("Loading real SENSEX historical data for training...")
    df_spot = pd.read_parquet(spot_path)
    
    # Generate labels
    df_spot['returns_15m'] = df_spot['close'].pct_change(3).shift(-3)
    df_spot['label'] = (df_spot['returns_15m'].abs() > 0.002).astype(int)
    
    from features.price_features import PriceFeatures
    df_spot = PriceFeatures.add_indicators(df_spot)
    
    # Map uppercase indicators from PriceFeatures to lowercase FEATURE_COLS
    if 'ADX_14' in df_spot.columns:
        df_spot['adx_14'] = df_spot['ADX_14']
    
    # Features (Fix #3: Compute actual values)
    df_spot['candle_strength'] = (df_spot['close'] - df_spot['open']).abs() / (df_spot['high'] - df_spot['low'] + 0.001) * 100
    df_spot['pcr_delta'] = df_spot['close'].pct_change(1) * 0.1 # Placeholder proxy since we don't have option chain in spot data
    df_spot['volume_ratio'] = df_spot['volume'] / df_spot['volume'].rolling(20, min_periods=1).mean().replace(0, 1)
    df_spot['vwap_dist_pct'] = (df_spot['close'] - df_spot['vwap']) / df_spot['vwap'] * 100
    df_spot['ema9_dist_pct'] = (df_spot['close'] - df_spot['ema_9']) / df_spot['ema_9'] * 100
    df_spot['bb_width'] = (df_spot['BBU_20_2.0'] - df_spot['BBL_20_2.0']) / df_spot['close'] * 100
    df_spot['hour_of_day'] = df_spot.index.hour + df_spot.index.minute / 60.0
    df_spot['day_of_week'] = df_spot.index.dayofweek
    df_spot['option_oi_change_pct'] = df_spot['volume'].pct_change(1) * 0.05 # Placeholder proxy
    
    df_train = df_spot.dropna(subset=['label'] + FEATURE_COLS)
    
    if len(df_train) < 50:
        logger.error(f"Not enough data to train real model. Rows after dropna: {len(df_train)}")
        return
        
    X = df_train[FEATURE_COLS]
    y = df_train['label']
    
    logger.info(f"Training Gamma Scalper model on {len(df_train)} rows of SENSEX data...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    
    # Fix #2: Walk-Forward CV
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    fold = 1
    last_fold_acc = 0.0
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        acc = (model.predict(X_te) == y_te).mean()
        last_fold_acc = float(acc)
        logger.info(f"Fold {fold} Accuracy: {acc:.2%}")
        fold += 1

    baseline = float(max(y.mean(), 1.0 - y.mean()))
    if last_fold_acc <= baseline:
        logger.warning(
            f"Gamma model failed deployment gate: last_fold_acc={last_fold_acc:.2%}, "
            f"baseline={baseline:.2%}. Existing model left unchanged."
        )
        return
        
    # Final fit on all data
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    logger.success(f"✅ Real-data Gamma Scalper model saved to {model_path}")

if __name__ == "__main__":
    train_real_gamma_model()
