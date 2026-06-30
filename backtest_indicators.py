from __future__ import annotations

import math
from datetime import time

import numpy as np
import pandas as pd


def sigmoid(x: float) -> float:
    """Return a bounded logistic score."""
    x = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def trading_times(freq: str) -> list[time]:
    """Return NSE intraday bar timestamps."""
    end = "15:25" if freq == "5min" else "15:15"
    return [t.time() for t in pd.date_range("09:15", end, freq=freq)]


def volume_profile(n: int) -> np.ndarray:
    """Return a normalized U-shaped intraday volume curve."""
    raw = np.array([8, 6, 5, 4.5, 4, 3.5, 3, 3, 3, 3, 3, 3.5, 4, 4, 4.5, 5, 5, 5.5, 6, 6.5, 7, 8, 8], dtype=float)
    prof = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(raw)), raw)
    return prof / prof.sum()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate average true range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    trur = atr(df, period).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / trur
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / trur
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(20)


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend value and direction (Vectorized)."""
    a = atr(df, period).bfill()
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    
    st = np.zeros(len(df))
    direction = np.ones(len(df), dtype=int)
    close_arr = df["close"].values
    upper_arr = upper.values
    lower_arr = lower.values
    
    for i in range(1, len(df)):
        if close_arr[i] > upper_arr[i-1]:
            direction[i] = 1
        elif close_arr[i] < lower_arr[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            st[i] = max(lower_arr[i], st[i-1])
        else:
            st[i] = min(upper_arr[i], st[i-1]) if st[i-1] != 0 else upper_arr[i]
            
    return pd.Series(st, index=df.index).replace(0, np.nan).ffill(), pd.Series(direction, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators required by the four strategies."""
    if df.empty:
        return df
    out = df.copy()
    out["date"] = out.index.date
    out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["sma200"] = out["close"].rolling(200, min_periods=50).mean()
    out["rsi_14"] = rsi(out["close"])
    out["atr_14"] = atr(out).fillna(0)
    
    # ADX Components
    up = out["high"].diff()
    down = -out["low"].diff()
    period = 14
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    trur = out["atr_14"].replace(0, np.nan)
    out["DMP_14"] = 100 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / period, adjust=False).mean() / trur
    out["DMN_14"] = 100 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / period, adjust=False).mean() / trur
    dx = 100 * (out["DMP_14"] - out["DMN_14"]).abs() / (out["DMP_14"] + out["DMN_14"]).replace(0, np.nan)
    out["ADX_14"] = dx.ewm(alpha=1 / period, adjust=False).mean().fillna(20)
    
    pv = out["close"] * out["volume"]
    out["vwap"] = pv.groupby(out["date"]).cumsum() / out["volume"].groupby(out["date"]).cumsum().replace(0, np.nan)
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=5).mean()
    out["volume_ratio"] = out["volume"] / out["vol_ma20"].replace(0, np.nan)
    
    # Bollinger Bands
    out["BBM_20_2.0"] = out["close"].rolling(20, min_periods=10).mean()
    bb_std = out["close"].rolling(20, min_periods=10).std()
    out["BBU_20_2.0"] = out["BBM_20_2.0"] + 2 * bb_std
    out["BBL_20_2.0"] = out["BBM_20_2.0"] - 2 * bb_std
    out["bb_width"] = (out["BBU_20_2.0"] - out["BBL_20_2.0"]) / out["BBM_20_2.0"].replace(0, np.nan) * 100
    
    out["supertrend"], out["supertrend_dir"] = supertrend(out)
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    return out.replace([np.inf, -np.inf], np.nan)


def candle_pattern(df: pd.DataFrame, i: int) -> str:
    """Classify reversal candles."""
    if i <= 0:
        return "none"
    r, p = df.iloc[i], df.iloc[i - 1]
    body = abs(r.close - r.open)
    rng = max(r.high - r.low, 1e-9)
    upper = r.high - max(r.close, r.open)
    lower = min(r.close, r.open) - r.low
    if body < rng * 0.35 and lower >= 2 * max(body, 1e-9) and r.close > r.open:
        return "hammer"
    if r.close > p.open and r.open < p.close and r.close > r.open:
        return "bullish_engulfing"
    if body < rng * 0.35 and upper >= 2 * max(body, 1e-9) and r.close < r.open:
        return "shooting_star"
    if r.open > p.close and r.close < p.open and r.close < r.open:
        return "bearish_engulfing"
    return "none"
