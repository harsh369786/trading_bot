import os
import math
import warnings
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# Set up logging
from loguru import logger
from strategies.rsmb.ai_filter import RSMBAIFilter, FEATURE_COLS
from strategies.rsmb.indicators import compute_rs_rank, compute_vwap, compute_ema, compute_atr, compute_volume_ratio
from features.price_features import PriceFeatures

warnings.filterwarnings("ignore")

RSMB_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "WIPRO",
    "BAJFINANCE", "TMPV", "TATASTEEL", "ADANIPORTS",
]

_CHOP_START = dtime(11, 30)
_CHOP_END = dtime(13, 30)
_EQUITY_CUTOFF = dtime(15, 15)

def backtest_symbol(symbol: str, df_5m: pd.DataFrame, nifty_daily: pd.Series, risk_capital: float):
    logger.info(f"Processing {symbol} ({len(df_5m)} 5m rows)...")
    
    # Resample to 15m
    df_15m = df_5m.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["close"]).copy()
    
    # Resample to daily
    df_daily = df_5m.resample("D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["close"]).copy()
    
    # Pre-compute indicators on full DF
    df_15m['date'] = pd.Series(df_15m.index).dt.date.values
    df_daily['date'] = pd.Series(df_daily.index).dt.date.values
    
    # VWAP (session based)
    df_15m['vwap'] = df_15m.groupby('date', group_keys=False).apply(compute_vwap)
    
    # EMAs and ATR
    df_15m['ema21'] = compute_ema(df_15m['close'], 21)
    df_15m['atr14'] = compute_atr(df_15m, 14)
    df_15m['volume_ratio'] = compute_volume_ratio(df_15m['volume'], 5)
    
    df_daily['daily_ema50'] = compute_ema(df_daily['close'], 50)
    
    # RS Rank
    rs_rank_series = pd.Series(index=df_daily.index, dtype=float)
    for i in range(len(df_daily)):
        if i < 20: continue
        stock_slice = df_daily["close"].iloc[:i+1]
        nifty_slice = nifty_daily.loc[:stock_slice.index[-1]]
        if not nifty_slice.empty:
            rs = compute_rs_rank(stock_slice, nifty_slice, lookback=20)
            rs_rank_series.iloc[i] = rs
            
    df_daily['rs_rank'] = rs_rank_series
    
    # Merge daily indicators to 15m
    daily_map = df_daily.set_index('date')[['close', 'daily_ema50', 'rs_rank']]
    df_15m = df_15m.join(daily_map, on='date', rsuffix='_daily')
    df_15m.rename(columns={'close_daily': 'daily_close'}, inplace=True)
    df_15m['rs_rank'] = df_15m['rs_rank'].fillna(1.0)
    
    # Technical indicators for features
    try:
        from ta.trend import MACD
        macd = MACD(close=df_15m['close'], window_slow=26, window_fast=12, window_sign=9)
        df_15m['macd_hist'] = macd.macd_diff()
    except:
        df_15m['macd_hist'] = 0.0
        
    df_15m = PriceFeatures.add_indicators(df_15m)
    
    signals_list = []
    
    # We can iterate fast now since all indicators are precomputed
    for i in range(50, len(df_15m)):
        row = df_15m.iloc[i]
        ts = row.name
        t = ts.time()
        
        if _CHOP_START <= t <= _CHOP_END or t >= _EQUITY_CUTOFF:
            continue
            
        rs_rank = row['rs_rank']
        close = row['close']
        vwap = row['vwap']
        ema21 = row['ema21']
        daily_close = row['daily_close']
        daily_ema50 = row['daily_ema50']
        atr14 = row['atr14']
        volume_ratio = row['volume_ratio']
        
        if pd.isna(atr14) or atr14 == 0:
            continue
            
        side = None
        # Evaluate BUY
        if rs_rank >= 1.05 and close > vwap and close > ema21 and daily_close > daily_ema50 and volume_ratio > 1.5:
            side = "BUY"
            sl = close - (1.5 * atr14)
            target1 = close + (1.5 * (close - sl))
        # Evaluate SELL
        elif rs_rank < 0.95 and close < vwap and daily_close < daily_ema50:
            side = "SELL"
            sl = close + (1.5 * atr14)
            target1 = close - (1.5 * (sl - close))
            
        if not side:
            continue
            
        qty = max(0, math.floor(risk_capital / abs(close - sl))) if abs(close - sl) > 0 else 0
        if qty <= 0:
            continue
            
        entry_price = close
        future_5m = df_5m.loc[ts:]
        if len(future_5m) > 1:
            future_5m = future_5m.iloc[1:]
            
        outcome = "UNKNOWN"
        exit_price = 0.0
        for f_idx, f_row in future_5m.iterrows():
            high = f_row["high"]
            low = f_row["low"]
            if side == "BUY":
                if low <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif high >= target1:
                    outcome = "WIN"
                    exit_price = target1
                    break
            else:
                if high >= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif low <= target1:
                    outcome = "WIN"
                    exit_price = target1
                    break
                    
        if outcome == "UNKNOWN":
            continue
            
        gross_pnl = (exit_price - entry_price) * qty if side == "BUY" else (entry_price - exit_price) * qty
        net_pnl = gross_pnl - 44.0
        
        bb_upper = row.get("BBU_20_2.0", 0.0)
        bb_lower = row.get("BBL_20_2.0", 0.0)
        adx = row.get("ADX_14", 0.0)
        
        features = RSMBAIFilter.extract_features(
            bar=row, rs_rank=rs_rank, vwap=vwap, ema21=ema21, atr=atr14,
            volume_ratio=volume_ratio, adx=adx, bb_upper=bb_upper, bb_lower=bb_lower
        )
        
        features["label"] = 1 if net_pnl > 0 else 0
        features["net_pnl"] = net_pnl
        features["side"] = side
        features["symbol"] = symbol
        features["timestamp"] = ts
        
        signals_list.append(features)
        
    return signals_list

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f) or {}

    capital_cfg = config.get("capital", {})
    equity_total = float(capital_cfg.get("equity_total", 50000))
    risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100
    risk_capital = equity_total * risk_pct

    logger.info("Loading NIFTY daily data for RS_Rank...")
    nifty_5m = pd.read_parquet("data/historical/NIFTY_6m.parquet")
    nifty_daily = nifty_5m.resample("D").agg({"close": "last"}).dropna()["close"]
    
    all_signals = []
    
    for sym in RSMB_UNIVERSE:
        file_path = f"data/historical/{sym}_6m.parquet"
        if os.path.exists(file_path):
            df_5m = pd.read_parquet(file_path)
            sym_sigs = backtest_symbol(sym, df_5m, nifty_daily, risk_capital)
            all_signals.extend(sym_sigs)
            
    df_results = pd.DataFrame(all_signals)
    if df_results.empty:
        logger.error("No signals found across any symbols.")
        return
        
    logger.info(f"Total signals generated: {len(df_results)}")
    
    out_path = "data/rsmb_training_data.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"Training data saved to {out_path}")
    
    df_results = df_results.sort_values("timestamp").reset_index(drop=True)
    X = df_results[FEATURE_COLS]
    y = df_results["label"]

    split_idx = int(len(df_results) * 0.8)
    if split_idx < 30 or len(df_results) - split_idx < 10:
        logger.error("Not enough samples for time-based validation; refusing to overwrite RSMB model.")
        return

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    validation_model_path = Path("models/rsmb_xgb.validation.tmp.pkl")
    validator = RSMBAIFilter(model_path=validation_model_path)
    if not validator.retrain(X_train, y_train):
        logger.error("Validation training failed; refusing to overwrite RSMB model.")
        return

    preds = [1 if validator.predict(row.to_dict()) > 0.5 else 0 for _, row in X_val.iterrows()]
    accuracy = float((np.asarray(preds) == y_val.to_numpy()).mean())
    baseline = float(max(y_val.mean(), 1 - y_val.mean()))
    logger.info(
        f"Time-based validation | samples={len(y_val)} | "
        f"accuracy={accuracy:.3f} | majority_baseline={baseline:.3f}"
    )
    try:
        validation_model_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Could not remove temporary validation model: {exc}")

    if accuracy <= baseline:
        logger.error("Validation did not beat majority baseline; refusing to overwrite RSMB model.")
        return
    
    logger.info("Training XGBoost AI Filter...")
    ai_filter = RSMBAIFilter()
    success = ai_filter.retrain(X, y)
    
    if success:
        logger.info("✅ RSMB Model successfully trained and saved!")
        win_rate = (y.sum() / len(y)) * 100
        logger.info(f"Training Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    main()
