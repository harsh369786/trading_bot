"""
strategies/mean_reversion/indicators.py
---------------------------------------
Pure indicator helpers for the 15m 200-MA Mean Reversion strategy.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sma_200(close: pd.Series) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").rolling(200, min_periods=200).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=period, adjust=False).mean()


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0.0)


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return (atr / close.replace(0, np.nan) * 100.0).fillna(0.0)


def bollinger_bands(close: pd.Series, period: int = 20, dev: float = 2.0) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(close, errors="coerce")
    mid = values.rolling(period, min_periods=period).mean()
    std = values.rolling(period, min_periods=period).std()
    return mid + dev * std, mid - dev * std


def session_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float, index=df.index)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    typical = (high + low + close) / 3.0
    pv = typical.fillna(close) * volume
    if isinstance(df.index, pd.DatetimeIndex):
        session = df.index.date
        cum_pv = pv.groupby(session).cumsum()
        cum_volume = volume.groupby(session).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_volume = volume.cumsum()
    return (cum_pv / cum_volume.replace(0, np.nan)).fillna(close)


def wick_ratio(bar: pd.Series) -> tuple[float, float]:
    open_ = safe_float(bar.get("open"), 0.0)
    high = safe_float(bar.get("high"), open_)
    low = safe_float(bar.get("low"), open_)
    close = safe_float(bar.get("close"), open_)
    body = abs(close - open_)
    if body <= 0:
        body = 0.01
    upper = max(high - max(close, open_), 0.0)
    lower = max(min(close, open_) - low, 0.0)
    return upper / body, lower / body


def candle_pattern(current: pd.Series, previous: pd.Series | None = None) -> str:
    open_ = safe_float(current.get("open"), 0.0)
    high = safe_float(current.get("high"), open_)
    low = safe_float(current.get("low"), open_)
    close = safe_float(current.get("close"), open_)
    rng = high - low
    body = abs(close - open_)
    if rng <= 0:
        return "none"
    upper = max(high - max(close, open_), 0.0)
    lower = max(min(close, open_) - low, 0.0)

    if body < rng * 0.35 and lower >= 2.0 * max(body, 0.01) and close > open_:
        return "hammer"
    if body < rng * 0.35 and upper >= 2.0 * max(body, 0.01) and close < open_:
        return "shooting_star"

    if previous is not None:
        prev_open = safe_float(previous.get("open"), 0.0)
        prev_close = safe_float(previous.get("close"), 0.0)
        if close > prev_open and open_ < prev_close and close > open_:
            return "bullish_engulfing"
        if open_ > prev_close and close < prev_open and close < open_:
            return "bearish_engulfing"
    return "none"


def linear_slope(series: pd.Series, period: int = 10) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(period)
    if len(values) < period:
        return 0.0
    x = np.arange(len(values), dtype=float)
    try:
        return float(np.polyfit(x, values.to_numpy(dtype=float), 1)[0])
    except Exception:
        return 0.0


def safe_float(value, default: float | None = None) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(val):
        return default
    return val

