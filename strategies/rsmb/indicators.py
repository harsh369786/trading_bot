"""
strategies/rsmb/indicators.py
-------------------------------
All technical indicator computations for the RSMB strategy.

Rules (non-negotiable per spec):
- No look-ahead bias: functions are pure, operate on whatever slice the caller provides.
- All functions handle edge cases (fewer bars than period) by returning NaN-filled Series.
- VWAP resets at session start — caller must slice df to current session before calling.
- RS_Rank uses daily closes only.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# VWAP — intraday, must be called on a session-sliced DataFrame
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price.

    Parameters
    ----------
    df : DataFrame with columns [high, low, close, volume].
         Must be pre-sliced to the current trading session (VWAP resets daily).

    Returns
    -------
    pd.Series of VWAP values, same index as df.
    """
    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(f"compute_vwap: missing columns {missing}; returning NaN series")
        return pd.Series(np.nan, index=df.index)

    typical_price: pd.Series = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol: pd.Series = (typical_price * df["volume"]).cumsum()
    cumulative_vol: pd.Series = df["volume"].cumsum()

    # Avoid division by zero on zero-volume bars
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    return vwap


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average using pandas ewm (adjust=False, standard EMA).

    Parameters
    ----------
    series : Price series (close, etc.)
    period : EMA period (e.g. 21, 50)

    Returns
    -------
    pd.Series of EMA values. Returns NaN-filled series if len(series) < period.
    """
    if series.empty:
        return pd.Series(dtype=float)

    if len(series) < period:
        logger.debug(
            f"compute_ema: only {len(series)} bars for period={period}; "
            "first values will be NaN until enough bars accumulate"
        )

    ema = series.ewm(span=period, adjust=False).mean()
    return ema


# ---------------------------------------------------------------------------
# ATR — True Range based
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's smoothing via ewm).

    Parameters
    ----------
    df     : DataFrame with columns [high, low, close].
    period : ATR period (default 14).

    Returns
    -------
    pd.Series of ATR values.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(f"compute_atr: missing columns {missing}; returning NaN series")
        return pd.Series(np.nan, index=df.index)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# RS_Rank
# ---------------------------------------------------------------------------

def compute_rs_rank(
    stock_daily: pd.Series,
    nifty_daily: pd.Series,
    lookback: int = 20,
) -> float:
    """
    Relative Strength Rank vs Nifty 50.

    Formula:
        RS_Rank = (stock[-1] / stock[-21]) / (nifty[-1] / nifty[-21])

    Parameters
    ----------
    stock_daily : Daily closing prices for the stock (pd.Series, DatetimeIndex).
    nifty_daily : Daily closing prices for NIFTY 50 (pd.Series, DatetimeIndex).
    lookback    : Number of trading days to look back (default 20).

    Returns
    -------
    float — RS_Rank value.
             Returns NaN if there are insufficient bars or any price is zero/NaN.
    """
    required_len = lookback + 1  # need [current] and [lookback-ago]

    if len(stock_daily) < required_len:
        logger.debug(
            f"compute_rs_rank: stock has only {len(stock_daily)} bars, "
            f"need {required_len}; returning NaN"
        )
        return float("nan")

    if len(nifty_daily) < required_len:
        logger.debug(
            f"compute_rs_rank: nifty has only {len(nifty_daily)} bars, "
            f"need {required_len}; returning NaN"
        )
        return float("nan")

    stock_now = float(stock_daily.iloc[-1])
    stock_20d = float(stock_daily.iloc[-required_len])
    nifty_now = float(nifty_daily.iloc[-1])
    nifty_20d = float(nifty_daily.iloc[-required_len])

    for val, name in [
        (stock_now, "stock_now"),
        (stock_20d, "stock_20d"),
        (nifty_now, "nifty_now"),
        (nifty_20d, "nifty_20d"),
    ]:
        if math.isnan(val) or val == 0:
            logger.warning(f"compute_rs_rank: {name} is {val}; returning NaN")
            return float("nan")

    rs_rank = (stock_now / stock_20d) / (nifty_now / nifty_20d)
    return rs_rank


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

def compute_donchian(
    df: pd.DataFrame, period: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Donchian Channel (highest high / lowest low over rolling window).

    Parameters
    ----------
    df     : DataFrame with columns [high, low].
    period : Lookback period (default 20).

    Returns
    -------
    Tuple of (upper: pd.Series, lower: pd.Series).
    Values are NaN for the first `period - 1` rows.
    """
    required = {"high", "low"}
    if not required.issubset(df.columns):
        nan_series = pd.Series(np.nan, index=df.index)
        return nan_series, nan_series

    upper = df["high"].rolling(window=period, min_periods=period).max()
    lower = df["low"].rolling(window=period, min_periods=period).min()
    return upper, lower


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

def compute_supertrend(
    df: pd.DataFrame, factor: float = 3.0, period: int = 10
) -> pd.Series:
    """
    Standard Supertrend indicator.

    Uses ATR(period) and a factor multiplier to determine the stop line.
    Returns the current trailing-stop level (positive = long trail, works in both directions).

    Parameters
    ----------
    df     : DataFrame with columns [high, low, close].
    factor : Multiplier for ATR bands (default 3.0).
    period : ATR period (default 10).

    Returns
    -------
    pd.Series of Supertrend values (the stop line).
    NaN where insufficient data exists.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        logger.warning("compute_supertrend: missing columns; returning NaN series")
        return pd.Series(np.nan, index=df.index)

    atr = compute_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr

    # Final bands and trend direction
    n = len(df)
    final_upper = pd.Series(np.nan, index=df.index)
    final_lower = pd.Series(np.nan, index=df.index)
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index)  # 1 = uptrend, -1 = downtrend

    # Need at least period + 1 bars for a meaningful first value
    if n < period + 1:
        return supertrend

    idx = df.index

    for i in range(period, n):
        # Final upper band: do not raise if previous close was already below prior upper
        if i == period:
            final_upper.iloc[i] = upper_band.iloc[i]
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_upper.iloc[i] = (
                upper_band.iloc[i]
                if upper_band.iloc[i] < final_upper.iloc[i - 1]
                   or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
                else final_upper.iloc[i - 1]
            )
            final_lower.iloc[i] = (
                lower_band.iloc[i]
                if lower_band.iloc[i] > final_lower.iloc[i - 1]
                   or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
                else final_lower.iloc[i - 1]
            )

        # Determine trend direction
        if i == period:
            direction.iloc[i] = 1
        else:
            prev_dir = direction.iloc[i - 1]
            close = df["close"].iloc[i]
            if prev_dir == 1:
                direction.iloc[i] = -1 if close < final_lower.iloc[i] else 1
            else:
                direction.iloc[i] = 1 if close > final_upper.iloc[i] else -1

        supertrend.iloc[i] = (
            final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]
        )

    return supertrend


# ---------------------------------------------------------------------------
# Volume Ratio
# ---------------------------------------------------------------------------

def compute_volume_ratio(volume: pd.Series, window: int = 5) -> pd.Series:
    """
    Ratio of current volume to rolling mean of the previous `window` bars.

    Spec: uses last 5 completed bars, excludes the current forming bar.
    Callers should pass volume[:-1] (or the completed bar slice) then re-attach.

    Parameters
    ----------
    volume : Volume series.
    window : Lookback window (default 5).

    Returns
    -------
    pd.Series of volume ratios. NaN where insufficient data exists.
    """
    if volume.empty:
        return pd.Series(dtype=float)

    rolling_mean = volume.shift(1).rolling(window=window, min_periods=window).mean()
    ratio = volume / rolling_mean.replace(0, np.nan)
    return ratio
