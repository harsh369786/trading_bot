import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from loguru import logger
from pathlib import Path

# Feature columns defined in MeanRevAIFilter
FEATURE_COLS = [
    "rsi_14", "adx_14", "distance_pct", "wick_ratio",
    "candle_pattern_encoded", "bb_width", "volume_ratio",
    "sma200_slope", "hour_of_day", "day_of_week",
    "ema20_1h_dist_pct", "atr_pct",
]

SYMBOLS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "WIPRO", "BAJFINANCE", "TMPV",
    "TATASTEEL", "ADANIPORTS"
]

def train_real_meanrev_model():
    model_path = Path("models/meanrev_xgb.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    from features.price_features import PriceFeatures
    
    for sym in SYMBOLS:
        path = Path(f"data/historical/{sym}_6m.parquet")
        if not path.exists():
            continue
            
        logger.info(f"Loading {sym} for Mean Reversion training...")
        df = pd.read_parquet(path)
        if len(df) < 500:
            continue
            
        # Label: If price returns to 200 SMA (or crosses it) within 20 bars
        df['sma_200'] = df['close'].rolling(200).mean()
        df['dist_sma'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # Simple label: If it was far (>2%) and returned towards SMA in next 20 bars
        df['target_pnl'] = df['close'].shift(-20) - df['close']
        # For BUY (oversold): PnL > 0 if price was below SMA
        # For SELL (overbought): PnL < 0 if price was above SMA
        df['label'] = 0
        df.loc[(df['dist_sma'] < -0.02) & (df['target_pnl'] > 0), 'label'] = 1
        df.loc[(df['dist_sma'] > 0.02) & (df['target_pnl'] < 0), 'label'] = 1
        
        df = PriceFeatures.add_indicators(df)
        
        # Features (Fix #3)
        df['adx_14'] = df['ADX_14']
        df['distance_pct'] = df['dist_sma'] * 100
        df['wick_ratio'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 0.001)
        df['candle_pattern_encoded'] = np.where(df['close'] > df['open'], 1.0, -1.0) # Proxy
        df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['close'] * 100
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean().replace(0, 1)
        df['sma200_slope'] = df['sma_200'].diff(5) / df['sma_200'] * 100
        df['hour_of_day'] = df.index.hour + df.index.minute / 60.0
        df['day_of_week'] = df.index.dayofweek
        df['ema20_1h_dist_pct'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100 # Proxy using 21 EMA
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
        
        df_clean = df.dropna(subset=['label'] + FEATURE_COLS)
        all_data.append(df_clean)
        
    if not all_data:
        logger.error("No data found for Mean Reversion training.")
        return
        
    df_train = pd.concat(all_data).sort_index()
    X = df_train[FEATURE_COLS]
    y = df_train['label']
    
    logger.info(f"Training Mean Reversion model on {len(df_train)} total rows...")
    
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
            f"MeanReversion model failed deployment gate: last_fold_acc={last_fold_acc:.2%}, "
            f"baseline={baseline:.2%}. Existing model left unchanged."
        )
        return
        
    # Final fit on all data
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    logger.success(f"✅ Real-data Mean Reversion model saved to {model_path}")

if __name__ == "__main__":
    train_real_meanrev_model()
