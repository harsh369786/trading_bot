import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report, confusion_matrix
from features.price_features import PriceFeatures

INFERENCE_FEATURES = [
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "rsi_14", "atr_pct",
    "dist_vwap", "ADX_14", "DMP_14", "DMN_14", "bb_pct", "rs_rank",
]


ALL_SYMBOLS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", 
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "WIPRO", "BAJFINANCE", "TATASTEEL", 
    "ADANIPORTS", "SENSEX"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train RSMB XGBoost Binary Model")
    parser.add_argument("--dry-run", action="store_true", help="Do not save model files")
    parser.add_argument("--symbols", nargs="+", default=ALL_SYMBOLS, help="Symbols to train on")
    parser.add_argument("--lookback-bars", type=int, default=10, help="Forward simulation bars")
    return parser.parse_args()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    import ta
    from ta.trend import EMAIndicator, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    
    if len(df) < 50: return df
    df = df.copy()
    
    df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["atr_14"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    
    adx_ind = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["ADX_14"] = adx_ind.adx()
    df["DMP_14"] = adx_ind.adx_pos()
    df["DMN_14"] = adx_ind.adx_neg()
    
    bb_ind = BollingerBands(close=df["close"], window=20, window_dev=2.0)
    df["BBL_20_2.0"] = bb_ind.bollinger_lband()
    df["BBM_20_2.0"] = bb_ind.bollinger_mavg()
    df["BBU_20_2.0"] = bb_ind.bollinger_hband()
    
    df["date"] = df.index.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"] = df["tp"] * df["volume"]
    cum_tpv = df.groupby("date")["tpv"].cumsum()
    cum_vol = df.groupby("date")["volume"].cumsum()
    df["vwap"] = (cum_tpv / cum_vol.replace(0, np.nan)).fillna(df["close"])
    df.drop(columns=["date", "tp", "tpv"], inplace=True)

    df = PriceFeatures.add_relative_price_features(df)
    return df

def generate_binary_labels(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    df = df.copy()
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    atr_arr = df["atr_14"].values
    
    labels = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - n_bars):
        c = close_arr[i]
        a = atr_arr[i]
        if np.isnan(c) or np.isnan(a): continue
            
        risk = a * 1.5          # tighter SL → more positive labels
        sl_buy = c - risk
        tgt_buy = c + (risk * 2.0)  # 2R target — reachable in 10 bars
        
        hit_tgt = -1
        hit_sl = -1
        
        for j in range(1, n_bars + 1):
            idx = i + j
            h = high_arr[idx]
            l = low_arr[idx]
            
            if hit_sl == -1 and l <= sl_buy:
                hit_sl = j
            if hit_tgt == -1 and h >= tgt_buy:
                hit_tgt = j
                
            if hit_tgt != -1 and hit_sl != -1:
                break
                
        if hit_tgt != -1 and (hit_sl == -1 or hit_tgt <= hit_sl):
            labels[i] = 1
            
    df["label"] = labels
    return df

def main():
    args = parse_args()
    np.random.seed(42)
    
    logger.info("=" * 60)
    logger.info("Script: train_rsmb_xgb.py")
    logger.info(f"Data Path: data/historical/")
    logger.info(f"Model Path: models/rsmb_xgb.pkl")
    logger.info("=" * 60)
    
    import xgboost as xgb
    
    # Load NIFTY Daily
    nifty_daily_path = Path("data/historical/NIFTY_daily.parquet")
    nifty_daily = None
    if nifty_daily_path.exists():
        nifty_daily = pd.read_parquet(nifty_daily_path)
        # Ensure it's daily
        if not nifty_daily.empty:
            nifty_daily.index = pd.to_datetime(nifty_daily.index).normalize()
            nifty_daily = nifty_daily.groupby(nifty_daily.index).last()
    
    if nifty_daily is None or nifty_daily.empty:
        logger.error("NIFTY_daily.parquet missing or empty. Cannot compute RS Rank.")
        return
        
    frames = []
    
    for sym in args.symbols:
        path = Path(f"data/historical/{sym}_6m.parquet")
        if not path.exists(): continue
            
        df = pd.read_parquet(path)
        if df.empty: continue
        
        # RS Rank Calculation
        daily = df.resample("D").agg({"close": "last"}).dropna()
        if len(daily) > 0:
            merged = pd.merge(daily, nifty_daily[["close"]], left_index=True, right_index=True, suffixes=("", "_nifty"))
            rs_raw = merged["close"] / merged["close_nifty"]
            # 60-day rolling pct rank
            rs_rank = rs_raw.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            daily["rs_rank"] = rs_rank
        else:
            daily["rs_rank"] = np.nan
            
        # 15m resampling
        df_15m = df.resample("15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna(subset=["close"])
        
        df_15m = compute_indicators(df_15m)
        df_15m = generate_binary_labels(df_15m, n_bars=args.lookback_bars)
        
        # Merge RS Rank
        df_15m["date_only"] = df_15m.index.normalize()
        daily["date_only"] = daily.index
        df_15m = pd.merge(df_15m.reset_index(), daily[["date_only", "rs_rank"]], on="date_only", how="left").set_index("timestamp")
        df_15m.drop(columns=["date_only"], inplace=True)
        
        # RSMB filter
        df_15m.loc[df_15m["rs_rank"].fillna(0) <= 0.6, "label"] = 0
        
        df_15m["symbol"] = sym
        frames.append(df_15m)
        
    if not frames:
        logger.error("No data loaded. Exiting.")
        return
    master_df = pd.concat(frames)
    master_df = master_df.sort_index()
    
    master_df = master_df.dropna(subset=INFERENCE_FEATURES + ["label"])
    
    times = master_df.index.unique().sort_values()
    t_start = times.min()
    t_end = times.max()
    total_days = (t_end - t_start).days
    
    t_train_end = t_start + pd.Timedelta(days=int(total_days * 4/6))
    t_val_end = t_start + pd.Timedelta(days=int(total_days * 5/6))
    
    df_train = master_df[master_df.index <= t_train_end]
    df_val = master_df[(master_df.index > t_train_end) & (master_df.index <= t_val_end)]
    df_test = master_df[master_df.index > t_val_end]
    
    X_train = df_train[INFERENCE_FEATURES]
    y_train = df_train["label"].astype(int)
    X_val = df_val[INFERENCE_FEATURES]
    y_val = df_val["label"].astype(int)
    X_test = df_test[INFERENCE_FEATURES]
    y_test = df_test["label"].astype(int)
    
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [3, 5],
    }
    
    fixed_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "seed": 42
    }
    
    best_aucpr = -1.0
    best_params = None
    
    import itertools
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Starting grid search over {len(combinations)} combinations...")
    for params in combinations:
        p = {**fixed_params, **params}
        # xgboost aucpr uses maximization
        bst = xgb.train(p, dtrain, num_boost_round=300, evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
        preds = bst.predict(dval)
        precision, recall, _ = precision_recall_curve(y_val, preds)
        aucpr = auc(recall, precision)
        if aucpr > best_aucpr:
            best_aucpr = aucpr
            best_params = p
            
    logger.info(f"Best params: {best_params} with val AUCPR: {best_aucpr:.4f}")
    
    # Final train
    df_tv = pd.concat([df_train, df_val])
    y_tv = df_tv["label"]
    scale_tv = (y_tv == 0).sum() / max(1, (y_tv == 1).sum())
    
    # We use XGBClassifier directly
    best_params["scale_pos_weight"] = scale_tv
    model = XGBClassifier(**best_params)
    model.fit(df_tv[INFERENCE_FEATURES], y_tv)
    
    # Eval
    preds_val = model.predict_proba(df_val[INFERENCE_FEATURES])[:, 1]
    precision_val, recall_val, thresholds = precision_recall_curve(y_val, preds_val)
    
    # Find optimal threshold on validation set
    f1_scores = 2 * recall_val * precision_val / (recall_val + precision_val + 1e-9)
    opt_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
    logger.info(f"Optimal Threshold (Val): {optimal_thresh:.4f}")
    
    preds_test = model.predict_proba(df_test[INFERENCE_FEATURES])[:, 1]
    prec_t, rec_t, _ = precision_recall_curve(y_test, preds_test)
    test_aucpr = auc(rec_t, prec_t)
    logger.info(f"Test AUCPR: {test_aucpr:.4f}")
    
    y_pred_05 = (preds_test > 0.5).astype(int)
    print("\nMetrics at default threshold 0.5:")
    print(classification_report(y_test, y_pred_05))
    
    y_pred_opt = (preds_test > optimal_thresh).astype(int)
    print(f"\nMetrics at optimal threshold {optimal_thresh:.4f}:")
    print(classification_report(y_test, y_pred_opt))
    
    # Calibration check
    calib_mask = preds_test > 0.62
    if calib_mask.sum() > 0:
        prec_62 = y_test[calib_mask].mean()
        logger.info(f"Calibration check: at prob > 0.62, precision is {prec_62:.2%} ({calib_mask.sum()} samples)")
    else:
        logger.info("Calibration check: no test samples > 0.62")
        
    if not args.dry_run:
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / "rsmb_xgb.pkl"
        meta_file = out_dir / "rsmb_xgb_meta.json"
        
        import joblib
        joblib.dump(model, model_file)
        
        meta = {
            "trained_at": datetime.utcnow().isoformat(),
            "train_period": f"{t_start} to {t_val_end}",
            "test_period": f"{t_val_end} to {t_end}",
            "n_features": len(INFERENCE_FEATURES),
            "feature_cols": INFERENCE_FEATURES,
            "best_params": best_params,
            "test_aucpr": float(test_aucpr),
            "optimal_threshold": float(optimal_thresh),
            "label_distribution": {"0": int((y_train==0).sum()), "1": int((y_train==1).sum())}
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        logger.success(f"Model saved to {model_file}")
        logger.success(f"Meta saved to {meta_file}")

if __name__ == "__main__":
    main()
