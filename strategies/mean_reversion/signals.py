"""
strategies/mean_reversion/signals.py
------------------------------------
Pure signal evaluation for the 15m 200-SMA Mean Reversion strategy.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from strategies.base_strategy import Signal
from strategies.mean_reversion.ai_filter import MeanRevAIFilter
from strategies.mean_reversion.indicators import (
    adx,
    atr_pct,
    bollinger_bands,
    candle_pattern,
    ema,
    linear_slope,
    rsi_wilder,
    safe_float,
    session_vwap,
    sma_200,
    wick_ratio,
)


def _position_size(entry: float, sl: float, risk_capital: float, max_notional: float | None) -> int:
    distance = abs(entry - sl)
    if distance <= 0 or not math.isfinite(distance):
        return 0
    qty = math.floor(max(float(risk_capital), 0.0) / distance)
    if max_notional and max_notional > 0 and entry > 0:
        qty = min(qty, math.floor(max_notional / entry))
    return max(0, qty)


def _prepare(df_15m: pd.DataFrame, df_1h: pd.DataFrame | None) -> dict:
    required = {"open", "high", "low", "close", "volume"}
    if df_15m is None or df_15m.empty or len(df_15m) < 200 or not required.issubset(df_15m.columns):
        return {}
    df = df_15m.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["open", "high", "low", "close"]].tail(1).isna().any(axis=None):
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    sma = sma_200(df["close"])
    rsi = rsi_wilder(df["close"], 14)
    adx_s = pd.to_numeric(df["ADX_14"], errors="coerce") if "ADX_14" in df.columns else adx(df["high"], df["low"], df["close"], 14)
    ema20 = ema(df["close"], 20)
    vwap = pd.to_numeric(df["vwap"], errors="coerce") if "vwap" in df.columns else session_vwap(df)
    upper, lower = bollinger_bands(df["close"], 20, 2.0)
    atrp = atr_pct(df, 14)
    upper_wick_ratio, lower_wick_ratio = wick_ratio(last)
    pattern = candle_pattern(last, prev)

    close = safe_float(last.get("close"))
    low = safe_float(last.get("low"))
    high = safe_float(last.get("high"))
    sma_last = safe_float(sma.iloc[-1])
    if None in (close, low, high, sma_last):
        return {}

    prior_distance = (
        (df["close"].iloc[:-1] - sma.iloc[:-1]).abs() / sma.iloc[:-1].replace(0, pd.NA) * 100.0
    ).tail(50)
    distance_pct = safe_float(prior_distance.max(), 0.0) or 0.0
    volume_ma20 = df["volume"].rolling(20, min_periods=20).mean()
    vol_ma = safe_float(volume_ma20.iloc[-1], 0.0) or 0.0
    volume_ratio = (safe_float(last.get("volume"), 0.0) or 0.0) / vol_ma if vol_ma > 0 else 0.0

    ema20_1h = None
    close_1h = None
    if df_1h is not None and not df_1h.empty and "close" in df_1h.columns and len(df_1h) >= 20:
        close_1h_series = pd.to_numeric(df_1h["close"], errors="coerce")
        close_1h = safe_float(close_1h_series.iloc[-1])
        ema20_1h = safe_float(ema(close_1h_series, 20).iloc[-1])
    ema20_1h_dist = (
        (close_1h - ema20_1h) / ema20_1h * 100.0
        if close_1h is not None and ema20_1h not in (None, 0)
        else 0.0
    )

    return {
        "last": last,
        "timestamp": df.index[-1],
        "close": close,
        "low": low,
        "high": high,
        "sma200": sma_last,
        "rsi14": safe_float(rsi.iloc[-1], 50.0) or 50.0,
        "adx14": safe_float(adx_s.iloc[-1], 0.0) or 0.0,
        "ema20": safe_float(ema20.iloc[-1]),
        "vwap": safe_float(vwap.iloc[-1]),
        "bb_upper": safe_float(upper.iloc[-1], 0.0) or 0.0,
        "bb_lower": safe_float(lower.iloc[-1], 0.0) or 0.0,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "pattern": pattern,
        "distance_pct": distance_pct,
        "volume_ratio": volume_ratio,
        "sma200_slope": linear_slope(sma.dropna(), 10),
        "ema20_1h": ema20_1h,
        "close_1h": close_1h,
        "ema20_1h_dist_pct": ema20_1h_dist,
        "atr_pct": safe_float(atrp.iloc[-1], 0.0) or 0.0,
    }


def _features(data: dict, wick_ratio_value: float) -> dict[str, float]:
    return MeanRevAIFilter.extract_features(
        bar=data["last"],
        rsi_14=data["rsi14"],
        adx_14=data["adx14"],
        distance_pct=data["distance_pct"],
        wick_ratio_value=wick_ratio_value,
        pattern=data["pattern"],
        bb_upper=data["bb_upper"],
        bb_lower=data["bb_lower"],
        volume_ratio=data["volume_ratio"],
        sma200_slope=data["sma200_slope"],
        ema20_1h_dist_pct=data["ema20_1h_dist_pct"],
        atr_pct_value=data["atr_pct"],
    )


def evaluate_buy_signal(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    ai_score: float,
    risk_capital: float,
    max_notional: float | None,
    *,
    min_ai_score: float = 0.60,
    max_adx: float = 35.0,
    min_distance_pct: float = 3.0,
    wick_ratio_min: float = 2.0,
) -> Optional[Signal]:
    data = _prepare(df_15m, df_1h)
    if not data:
        return None
    if data["close_1h"] is not None and data["ema20_1h"] is not None and data["close_1h"] <= data["ema20_1h"]:
        return None

    conditions = [
        data["low"] <= data["sma200"] <= data["high"],
        data["rsi14"] < 35.0,
        data["pattern"] in {"hammer", "bullish_engulfing"},
        data["lower_wick_ratio"] >= wick_ratio_min,
        data["distance_pct"] >= min_distance_pct,
        data["adx14"] < max_adx,
        ai_score >= min_ai_score,
    ]
    if not all(conditions):
        return None

    entry = data["close"]
    sl = min(data["sma200"] - 0.10, data["low"] - 0.10)
    target1 = data["ema20"]
    target2 = data["vwap"]
    if None in (target1, target2) or sl >= entry or target1 <= entry or target2 <= entry:
        return None
    qty = _position_size(entry, sl, risk_capital, max_notional)
    if qty <= 0:
        return None
    return Signal(
        strategy="mean_reversion",
        symbol=symbol,
        side="BUY",
        entry=entry,
        sl=sl,
        target1=target1,
        target2=target2,
        qty=qty,
        score=ai_score,
        rs_rank=None,
        rejection_reason=None,
        timestamp=pd.Timestamp(data["timestamp"]),
        t1_exit_pct=0.5,
        target2_mode="price",
    )


def evaluate_sell_signal(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    ai_score: float,
    risk_capital: float,
    max_notional: float | None,
    *,
    min_ai_score: float = 0.60,
    max_adx: float = 35.0,
    min_distance_pct: float = 3.0,
    wick_ratio_min: float = 2.0,
) -> Optional[Signal]:
    data = _prepare(df_15m, df_1h)
    if not data:
        return None
    if data["close_1h"] is not None and data["ema20_1h"] is not None and data["close_1h"] >= data["ema20_1h"]:
        return None

    conditions = [
        data["low"] <= data["sma200"] <= data["high"],
        data["rsi14"] > 65.0,
        data["pattern"] in {"shooting_star", "bearish_engulfing"},
        data["upper_wick_ratio"] >= wick_ratio_min,
        data["distance_pct"] >= min_distance_pct,
        data["adx14"] < max_adx,
        ai_score >= min_ai_score,
    ]
    if not all(conditions):
        return None

    entry = data["close"]
    sl = max(data["sma200"] + 0.10, data["high"] + 0.10)
    target1 = data["ema20"]
    target2 = data["vwap"]
    if None in (target1, target2) or sl <= entry or target1 >= entry or target2 >= entry:
        return None
    qty = _position_size(entry, sl, risk_capital, max_notional)
    if qty <= 0:
        return None
    return Signal(
        strategy="mean_reversion",
        symbol=symbol,
        side="SELL",
        entry=entry,
        sl=sl,
        target1=target1,
        target2=target2,
        qty=qty,
        score=ai_score,
        rs_rank=None,
        rejection_reason=None,
        timestamp=pd.Timestamp(data["timestamp"]),
        t1_exit_pct=0.5,
        target2_mode="price",
    )

