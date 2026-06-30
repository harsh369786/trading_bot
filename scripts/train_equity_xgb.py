import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from features.price_features import PriceFeatures

# Global feature columns for inference
INFERENCE_FEATURES = [
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "rsi_14", "atr_pct",
    "dist_vwap", "ADX_14", "DMP_14", "DMN_14", "bb_pct",
]


ALL_SYMBOLS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", 
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "WIPRO", "BAJFINANCE", "TATASTEEL", 
    "ADANIPORTS", "SENSEX"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train Equity XGBoost 3-class Model")
    parser.add_argument("--dry-run", action="store_true", help="Do not save model files")
    parser.add_argument("--symbols", nargs="+", default=ALL_SYMBOLS, help="Symbols to train on")
    parser.add_argument("--lookback-bars", type=int, default=10, help="Forward simulation bars")
    return parser.parse_args()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    import ta
    from ta.trend import EMAIndicator, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    
    if len(df) < 50:
        return df
        
    df = df.copy()
    
    # 1. EMA
    df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    
    # 2. RSI
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    
    # 3. ATR
    df["atr_14"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    
    # 4. ADX, DMP, DMN
    adx_ind = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["ADX_14"] = adx_ind.adx()
    df["DMP_14"] = adx_ind.adx_pos()
    df["DMN_14"] = adx_ind.adx_neg()
    
    # 5. Bollinger Bands
    bb_ind = BollingerBands(close=df["close"], window=20, window_dev=2.0)
    df["BBL_20_2.0"] = bb_ind.bollinger_lband()
    df["BBM_20_2.0"] = bb_ind.bollinger_mavg()
    df["BBU_20_2.0"] = bb_ind.bollinger_hband()
    
    # 6. VWAP (rolling session)
    df["date"] = df.index.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"] = df["tp"] * df["volume"]
    cum_tpv = df.groupby("date")["tpv"].cumsum()
    cum_vol = df.groupby("date")["volume"].cumsum()
    df["vwap"] = (cum_tpv / cum_vol.replace(0, np.nan)).fillna(df["close"])
    df.drop(columns=["date", "tp", "tpv"], inplace=True)
    
    # --- Training Features ---
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df = PriceFeatures.add_relative_price_features(df)
    
    # Session: 0=open_drive (09:15–11:30), 1=chop_zone (11:30–13:30), 2=trend_window (13:30–15:15)
    def get_session(dt):
        t = dt.time()
        m = t.hour * 60 + t.minute
        if m < 11*60 + 30:
            return 0
        elif m < 13*60 + 30:
            return 1
        return 2
        
    df["session"] = df.index.map(get_session)
    return df

def generate_labels(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    df = df.copy()
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    atr_arr = df["atr_14"].values
    
    labels = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - n_bars):
        c = close_arr[i]
        a = atr_arr[i]
        if np.isnan(c) or np.isnan(a):
            continue
            
        risk = a * 1.5          # tighter SL → more positive labels
        sl_buy = c - risk
        tgt_buy = c + (risk * 2.0)  # 2R target — reachable in 10 bars
        sl_sell = c + risk
        tgt_sell = c - (risk * 2.0)
        
        hit_buy_tgt = -1
        hit_buy_sl = -1
        hit_sell_tgt = -1
        hit_sell_sl = -1
        
        for j in range(1, n_bars + 1):
            idx = i + j
            h = high_arr[idx]
            l = low_arr[idx]
            
            # BUY checks
            if hit_buy_sl == -1 and l <= sl_buy:
                hit_buy_sl = j
            if hit_buy_tgt == -1 and h >= tgt_buy:
                hit_buy_tgt = j
                
            # SELL checks
            if hit_sell_sl == -1 and h >= sl_sell:
                hit_sell_sl = j
            if hit_sell_tgt == -1 and l <= tgt_sell:
                hit_sell_tgt = j
                
            # If all hit, we can stop early
            if hit_buy_tgt != -1 and hit_buy_sl != -1 and hit_sell_tgt != -1 and hit_sell_sl != -1:
                break
                
        # Resolve BUY label
        buy_win = False
        if hit_buy_tgt != -1:
            if hit_buy_sl == -1 or hit_buy_tgt <= hit_buy_sl:
                buy_win = True
                
        # Resolve SELL label
        sell_win = False
        if hit_sell_tgt != -1:
            if hit_sell_sl == -1 or hit_sell_tgt <= hit_sell_sl:
                sell_win = True
                
        if buy_win and not sell_win:
            labels[i] = 1
        elif sell_win and not buy_win:
            labels[i] = 2
        else:
            labels[i] = 0
            
    df["label"] = labels
    
    # Filter constraints
    mask_adx = df["ADX_14"].fillna(0) < 20
    mask_vol = df["vol_ratio"].fillna(0) < 0.7
    mask_chop = df["session"] == 1
    df.loc[mask_adx | mask_vol | mask_chop, "label"] = 0
    
    return df

def simulate_equity_curve(df_test: pd.DataFrame, preds: np.ndarray) -> dict:
    df = df_test.copy()
    df["pred"] = preds
    
    pnl = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    peak = 0.0
    max_dd = 0.0
    
    for i in range(len(df) - 6):
        pred = df["pred"].iloc[i]
        if pred == 0:
            continue
            
        c = df["close"].iloc[i]
        a = df["atr_14"].iloc[i]
        risk = a * 1.5
        
        # Standard position size where 1R risk = 1000 INR
        qty = max(1, int(1000 / risk)) if risk > 0 else 0
        if qty == 0:
            continue
            
        sl_buy = c - risk
        tgt_buy = c + (risk * 2.0)
        sl_sell = c + risk
        tgt_sell = c - (risk * 2.0)
        
        trade_pnl = 0.0
        
        for j in range(1, 11):  # match lookback_bars=10
            idx = i + j
            h = df["high"].iloc[idx]
            l = df["low"].iloc[idx]
            
            if pred == 1: # BUY
                if l <= sl_buy:
                    trade_pnl = (sl_buy - c) * qty
                    break
                if h >= tgt_buy:
                    trade_pnl = (tgt_buy - c) * qty
                    break
            elif pred == 2: # SELL
                if h >= sl_sell:
                    trade_pnl = (c - sl_sell) * qty
                    break
                if l <= tgt_sell:
                    trade_pnl = (c - tgt_sell) * qty
                    break
                    
        # Apply costs
        trade_pnl -= 44.0
        
        pnl += trade_pnl
        if pnl > peak:
            peak = pnl
        dd = peak - pnl
        if dd > max_dd:
            max_dd = dd
            
        if trade_pnl > 0:
            wins += 1
            gross_profit += trade_pnl
        else:
            losses += 1
            gross_loss += abs(trade_pnl)
            
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    
    return {
        "total_pnl": pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "total_trades": total_trades
    }

def main():
    args = parse_args()
    
    np.random.seed(42)
    
    logger.info("=" * 60)
    logger.info("Script: train_equity_xgb.py")
    logger.info(f"Data Path: data/historical/")
    logger.info(f"Model Path: models/xgboost/model.json")
    logger.info(f"Symbol Count: {len(args.symbols)}")
    logger.info("=" * 60)
    
    import xgboost as xgb
    
    frames = []
    
    for sym in args.symbols:
        path = Path(f"data/historical/{sym}_6m.parquet")
        if not path.exists():
            logger.warning(f"File missing for {sym}, skipping.")
            continue
            
        df = pd.read_parquet(path)
        if df.empty:
            continue
            
        # Resample to 15m
        df_15m = df.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna(subset=["close"])
        
        # Indicators
        df_15m = compute_indicators(df_15m)
        
        # Labels
        df_15m = generate_labels(df_15m, n_bars=args.lookback_bars)
        
        df_15m["symbol"] = sym
        frames.append(df_15m)
        
    if not frames:
        logger.error("No data loaded. Exiting.")
        return
        
    master_df = pd.concat(frames)
    master_df = master_df.sort_index()
    
    logger.info(f"Total rows before NaN drop: {len(master_df)}")
    
    # Drop NaNs in features
    required_cols = INFERENCE_FEATURES + ["ret_1", "ret_3", "vol_ratio", "session"]
    master_df = master_df.dropna(subset=required_cols + ["label"])
    
    logger.info(f"Total rows after NaN drop: {len(master_df)}")
    
    # Temporal split (4 months train, 1 month val, 1 month test)
    times = master_df.index.unique().sort_values()
    t_start = times.min()
    t_end = times.max()
    total_days = (t_end - t_start).days
    
    t_train_end = t_start + pd.Timedelta(days=int(total_days * 4/6))
    t_val_end = t_start + pd.Timedelta(days=int(total_days * 5/6))
    
    train_mask = master_df.index <= t_train_end
    val_mask = (master_df.index > t_train_end) & (master_df.index <= t_val_end)
    test_mask = master_df.index > t_val_end
    
    df_train = master_df[train_mask]
    df_val = master_df[val_mask]
    df_test = master_df[test_mask]
    
    logger.info(f"Train rows: {len(df_train)}, Val rows: {len(df_val)}, Test rows: {len(df_test)}")
    
    dist = df_train["label"].value_counts(normalize=True).to_dict()
    logger.info(f"Train Label Distribution: {dist}")
    
    X_train = df_train[INFERENCE_FEATURES]
    y_train = df_train["label"].astype(int)
    X_val = df_val[INFERENCE_FEATURES]
    y_val = df_val["label"].astype(int)
    X_test = df_test[INFERENCE_FEATURES]
    y_test = df_test["label"].astype(int)
    
    # Sample weights
    n_no_trade = (y_train == 0).sum()
    n_buy = (y_train == 1).sum()
    n_sell = (y_train == 2).sum()
    
    weights = np.ones(len(y_train))
    if n_buy > 0:
        weights[y_train == 1] = n_no_trade / n_buy
    if n_sell > 0:
        weights[y_train == 2] = n_no_trade / n_sell
        
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Grid Search
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [3, 5],
    }
    
    fixed_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "seed": 42
    }
    
    best_logloss = float("inf")
    best_params = None
    best_n_rounds = 300  # BUG-02: track optimal rounds from early stopping
    
    import itertools
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Starting grid search over {len(combinations)} combinations...")
    
    for params in combinations:
        p = {**fixed_params, **params}
        bst = xgb.train(p, dtrain, num_boost_round=300, evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
        preds = bst.predict(dval)
        loss = log_loss(y_val, preds)
        if loss < best_logloss:
            best_logloss = loss
            best_params = p
            # Save the round count where early stopping triggered
            if hasattr(bst, "best_iteration") and bst.best_iteration > 0:
                best_n_rounds = bst.best_iteration + 1
            
    logger.info(f"Best params: {best_params} with val logloss: {best_logloss:.4f}")
    
    # Train final model on train+val
    df_trainval = pd.concat([df_train, df_val])
    X_tv = df_trainval[INFERENCE_FEATURES]
    y_tv = df_trainval["label"].astype(int)
    
    n_no_trade_tv = (y_tv == 0).sum()
    n_buy_tv = (y_tv == 1).sum()
    n_sell_tv = (y_tv == 2).sum()
    
    weights_tv = np.ones(len(y_tv))
    if n_buy_tv > 0: weights_tv[y_tv == 1] = n_no_trade_tv / n_buy_tv
    if n_sell_tv > 0: weights_tv[y_tv == 2] = n_no_trade_tv / n_sell_tv
    
    dtv = xgb.DMatrix(X_tv, label=y_tv, weight=weights_tv)
    
    logger.info("Training final model...")
    final_bst = xgb.train(best_params, dtv, num_boost_round=best_n_rounds)
    final_bst.set_attr(feature_names_json="|".join(INFERENCE_FEATURES))
    
    # Evaluation
    preds_proba = final_bst.predict(dtest)
    preds_class = np.argmax(preds_proba, axis=1)
    
    test_loss = log_loss(y_test, preds_proba)
    logger.info(f"Test Log-Loss: {test_loss:.4f}")
    
    print("\nClassification Report:")
    report_str = classification_report(y_test, preds_class)
    print(report_str)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds_class))
    
    # Feature Importance
    print("\nFeature Importance (Gain):")
    scores = final_bst.get_score(importance_type="gain")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
    for k, v in sorted_scores:
        print(f"{k:20s}: {v:.4f}")
        
    report = classification_report(y_test, preds_class, output_dict=True, zero_division=0)
    f1_buy = report.get("1", {}).get("f1-score", 0.0)
    f1_sell = report.get("2", {}).get("f1-score", 0.0)
    
    if f1_buy < 0.35:
        logger.warning(f"BUY F1-score is below 0.35 ({f1_buy:.2f}) - Model is not better than random for BUY.")
    if f1_sell < 0.35:
        logger.warning(f"SELL F1-score is below 0.35 ({f1_sell:.2f}) - Model is not better than random for SELL.")
        
    # Equity Curve
    logger.info("Simulating Equity Curve on Test Set...")
    eq_stats = simulate_equity_curve(df_test, preds_class)
    print("\nEquity Simulation:")
    for k, v in eq_stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        
    # Save Model
    if not args.dry_run:
        out_dir = Path("models/xgboost")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / "model.json"
        meta_file = out_dir / "model_meta.json"
        
        final_bst.save_model(model_file)
        
        meta = {
            "trained_at": datetime.utcnow().isoformat(),
            "train_period": f"{t_start} to {t_val_end}",
            "test_period": f"{t_val_end} to {t_end}",
            "n_features": len(INFERENCE_FEATURES),
            "feature_cols": INFERENCE_FEATURES,
            "best_params": best_params,
            "test_logloss": float(test_loss),
            "test_f1_buy": float(f1_buy),
            "test_f1_sell": float(f1_sell),
            "label_distribution": {str(k): float(v) for k, v in dist.items()}
        }
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        logger.success(f"Model saved to {model_file}")
        logger.success(f"Meta saved to {meta_file}")
    else:
        logger.info("Dry run complete. No files saved.")

if __name__ == "__main__":
    main()
