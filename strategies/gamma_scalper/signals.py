"""
strategies/gamma_scalper/signals.py
-----------------------------------
Pure signal evaluation for Sensex ATM option gamma scalping.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from strategies.base_strategy import Signal
from strategies.gamma_scalper.ai_filter import GammaAIFilter
from strategies.gamma_scalper.indicators import (
    adx,
    candle_strength,
    ema,
    option_vwap,
    rsi_wilder,
    safe_float,
)


def _safe_last(series: pd.Series, default: Optional[float] = None) -> Optional[float]:
    if series is None or series.empty:
        return default
    value = safe_float(series.iloc[-1], default)
    return value


def _position_size(entry: float, sl: float, risk_capital: float, max_notional: float | None) -> int:
    distance = abs(entry - sl)
    if distance <= 0 or not math.isfinite(distance):
        return 0
    qty = math.floor(max(float(risk_capital), 0.0) / distance)
    if max_notional and max_notional > 0 and entry > 0:
        qty = min(qty, math.floor(max_notional / entry))
    return max(0, qty)


def _prepare(option_df: pd.DataFrame, spot_df: pd.DataFrame | None) -> dict:
    if option_df is None or option_df.empty or len(option_df) < 21:
        return {}
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(option_df.columns):
        return {}

    opt = option_df.copy()
    for col in required | {"oi"}:
        if col in opt.columns:
            opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt_vwap = option_vwap(opt)
    rsi = rsi_wilder(opt["close"], 14)
    volume_ma20 = opt["volume"].rolling(20, min_periods=20).mean()
    last = opt.iloc[-1]

    if spot_df is None or spot_df.empty or not required.issubset(spot_df.columns):
        spot_close = ema9 = spot_vwap = adx14 = None
    else:
        spot = spot_df.copy()
        for col in required:
            spot[col] = pd.to_numeric(spot[col], errors="coerce")
        spot_close = safe_float(spot["close"].iloc[-1])
        ema9 = _safe_last(ema(spot["close"], 9))
        if "vwap" in spot.columns:
            spot_vwap = safe_float(spot["vwap"].iloc[-1])
        else:
            spot_vwap = _safe_last(option_vwap(spot))
        adx14 = safe_float(spot["ADX_14"].iloc[-1]) if "ADX_14" in spot.columns else _safe_last(adx(spot["high"], spot["low"], spot["close"], 14), 0.0)

    current_vwap = _safe_last(opt_vwap)
    current_rsi = _safe_last(rsi)
    current_vol_ma20 = _safe_last(volume_ma20)
    close = safe_float(last.get("close"))
    open_ = safe_float(last.get("open"))
    low = safe_float(last.get("low"))
    volume = safe_float(last.get("volume"), 0.0)

    if None in (current_vwap, current_rsi, current_vol_ma20, close, open_, low, volume):
        return {}

    pcr_delta = safe_float(last.get("pcr_delta"), 0.0)
    strength = candle_strength(open_, close)
    volume_ratio = volume / current_vol_ma20 if current_vol_ma20 and current_vol_ma20 > 0 else 0.0
    features = GammaAIFilter.extract_features(
        bar=last,
        candle_strength_value=strength,
        rsi_14=current_rsi,
        adx_14=adx14 or 0.0,
        pcr_delta=pcr_delta or 0.0,
        volume_ratio=volume_ratio,
        option_vwap=current_vwap or 0.0,
        ema_9_spot=ema9 or 0.0,
    )
    return {
        "last": last,
        "timestamp": opt.index[-1],
        "entry": close,
        "low": low,
        "strength": strength,
        "spot_close": spot_close,
        "ema9": ema9,
        "spot_vwap": spot_vwap,
        "option_vwap": current_vwap,
        "rsi": current_rsi,
        "volume": volume,
        "volume_ma20": current_vol_ma20,
        "volume_ratio": volume_ratio,
        "adx14": adx14,
        "pcr_delta": pcr_delta or 0.0,
        "features": features,
    }


def evaluate_ce_signal(
    symbol: str,
    option_df: pd.DataFrame,
    spot_df: pd.DataFrame | None,
    ai_score: float,
    risk_capital: float,
    max_notional: float | None,
    *,
    min_candle_strength: float = 10.0,
    min_ai_score: float = 0.70,
    min_adx: float = 20.0,
    theta_veto_bars: int = 3,
) -> Optional[Signal]:
    data = _prepare(option_df, spot_df)
    if not data:
        return None

    checks = [
        data["strength"] >= min_candle_strength,
        data["spot_close"] is not None and data["ema9"] is not None and data["spot_close"] > data["ema9"],
        data["spot_close"] is not None and data["spot_vwap"] is not None and data["spot_close"] > data["spot_vwap"],
        data["entry"] > data["option_vwap"],
        data["rsi"] > 60.0,
        data["volume"] > 1.2 * data["volume_ma20"],
        (data["adx14"] or 0.0) >= min_adx,
        data["pcr_delta"] <= 0.0,
        ai_score >= min_ai_score,
    ]
    if not all(checks):
        return None

    entry = data["entry"]
    sl = max(data["low"], entry * 0.88)
    target1 = entry * 1.30
    target2 = entry * 10.0
    qty = _position_size(entry, sl, risk_capital, max_notional)
    if qty <= 0:
        return None

    return Signal(
        strategy="gamma_scalper",
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
        expire_after_bars=theta_veto_bars,
        t1_exit_pct=0.60,
        target2_mode="manual",
    )


def evaluate_pe_signal(
    symbol: str,
    option_df: pd.DataFrame,
    spot_df: pd.DataFrame | None,
    ai_score: float,
    risk_capital: float,
    max_notional: float | None,
    *,
    min_candle_strength: float = 10.0,
    min_ai_score: float = 0.70,
    min_adx: float = 20.0,
    theta_veto_bars: int = 3,
) -> Optional[Signal]:
    data = _prepare(option_df, spot_df)
    if not data:
        return None

    checks = [
        data["strength"] >= min_candle_strength,
        data["spot_close"] is not None and data["ema9"] is not None and data["spot_close"] < data["ema9"],
        data["spot_close"] is not None and data["spot_vwap"] is not None and data["spot_close"] < data["spot_vwap"],
        data["entry"] > data["option_vwap"],
        data["rsi"] > 60.0,
        data["volume"] > 1.2 * data["volume_ma20"],
        (data["adx14"] or 0.0) >= min_adx,
        data["pcr_delta"] >= 0.0,
        ai_score >= min_ai_score,
    ]
    if not all(checks):
        return None

    entry = data["entry"]
    sl = max(data["low"], entry * 0.88)
    target1 = entry * 1.30
    target2 = entry * 10.0
    qty = _position_size(entry, sl, risk_capital, max_notional)
    if qty <= 0:
        return None

    return Signal(
        strategy="gamma_scalper",
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
        expire_after_bars=theta_veto_bars,
        t1_exit_pct=0.60,
        target2_mode="manual",
    )

