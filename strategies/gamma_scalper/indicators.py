"""
strategies/gamma_scalper/indicators.py
--------------------------------------
Indicator helpers for the Sensex 5m Gamma Scalper.

These functions are pure: no I/O, no shared state, no mutation of caller data.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def candle_strength(open_price: float, close_price: float) -> float:
    """Return candle body strength as percentage move from open to close."""
    try:
        open_f = float(open_price)
        close_f = float(close_price)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(open_f) or open_f <= 0 or not math.isfinite(close_f):
        return 0.0
    return (close_f - open_f) / open_f * 100.0


def option_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-reset VWAP for option premium candles.

    Uses typical price and volume. If cumulative volume is zero, falls back to
    close to avoid NaN propagation in live signals.
    """
    if df.empty or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float, index=df.index)

    work = df[["high", "low", "close", "volume"]].copy()
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    volume = work["volume"].fillna(0.0).clip(lower=0.0)
    pv = typical.fillna(work["close"]) * volume

    if isinstance(df.index, pd.DatetimeIndex):
        session = df.index.date
        cum_pv = pv.groupby(session).cumsum()
        cum_volume = volume.groupby(session).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_volume = volume.cumsum()

    vwap = cum_pv / cum_volume.replace(0, np.nan)
    return vwap.fillna(work["close"]).replace([np.inf, -np.inf], np.nan).ffill()


def compute_pcr_delta(pcr: pd.Series | float | int | None, periods: int = 3) -> float:
    """Return current PCR minus PCR N bars ago. Missing PCR is neutral (0.0)."""
    if pcr is None:
        return 0.0
    if isinstance(pcr, pd.Series):
        clean = pd.to_numeric(pcr, errors="coerce").dropna()
        if len(clean) <= periods:
            return 0.0
        return float(clean.iloc[-1] - clean.iloc[-1 - periods])
    try:
        value = float(pcr)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def ema(series: pd.Series, period: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.ewm(span=period, adjust=False).mean()


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ADX implementation used as a fallback when ta columns are absent."""
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0.0)


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(val):
        return default
    return val

