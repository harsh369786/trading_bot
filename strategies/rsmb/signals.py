"""
strategies/rsmb/signals.py
----------------------------
Pure entry/exit signal logic for the RSMB strategy.

Design principles:
- Pure functions only — no side effects, no I/O, no global state.
- No look-ahead bias: every function operates on a completed bar (iloc[-1] of the slice
  the caller provides). The caller must not include the forming/live bar.
- Returns a typed Signal dataclass or None — never raises on bad/insufficient data.
- All 9 BUY conditions and 6 SELL conditions checked independently.
"""
from __future__ import annotations

import math
from datetime import time as dtime
from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import Signal
from strategies.rsmb.indicators import (
    compute_atr,
    compute_ema,
    compute_supertrend,
    compute_vwap,
    compute_volume_ratio,
)


# ---------------------------------------------------------------------------
# Session time constants (IST)
# ---------------------------------------------------------------------------

_CHOP_START = dtime(11, 30)
_CHOP_END = dtime(13, 30)
_EQUITY_CUTOFF = dtime(15, 15)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_chop_zone(ts: pd.Timestamp) -> bool:
    """Return True if the timestamp falls within the 11:30–13:30 chop zone."""
    t = ts.time()
    return _CHOP_START <= t <= _CHOP_END


def _is_after_cutoff(ts: pd.Timestamp) -> bool:
    """Return True if trading is closed for the day (>= 15:15 IST)."""
    return ts.time() >= _EQUITY_CUTOFF


def _safe_last(series: pd.Series, name: str) -> Optional[float]:
    """Extract the last value of a Series; return None if NaN or empty."""
    if series.empty:
        logger.debug(f"_safe_last: {name} series is empty")
        return None
    val = float(series.iloc[-1])
    if math.isnan(val):
        logger.debug(f"_safe_last: {name} is NaN at last bar")
        return None
    return val


def _compute_signal_indicators(
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame,
) -> dict:
    """
    Compute all indicators needed for signal evaluation.
    Returns a flat dict with keys:
        close, vwap, ema21, daily_close, daily_ema50, atr14, volume_ratio,
        supertrend
    Any value may be None if the underlying indicator returned NaN.
    """
    if df_15m.empty or len(df_15m) < 2:
        return {}

    # Use all completed bars (exclude the forming bar — caller's responsibility,
    # but we enforce at least 2 bars required)
    bar = df_15m.iloc[-1]

    vwap_series = compute_vwap(df_15m)
    ema21_series = compute_ema(df_15m["close"], 21)
    atr_series = compute_atr(df_15m, 14)
    vol_ratio_series = compute_volume_ratio(df_15m["volume"], window=5)
    supertrend_series = compute_supertrend(df_15m, factor=3.0, period=10)

    daily_ema50_series = (
        compute_ema(df_daily["close"], 50)
        if not df_daily.empty and "close" in df_daily.columns
        else pd.Series(dtype=float)
    )

    return {
        "close": _safe_last(df_15m["close"], "close"),
        "vwap": _safe_last(vwap_series, "vwap"),
        "ema21": _safe_last(ema21_series, "ema21"),
        "daily_close": _safe_last(df_daily["close"], "daily_close") if not df_daily.empty else None,
        "daily_ema50": _safe_last(daily_ema50_series, "daily_ema50"),
        "atr14": _safe_last(atr_series, "atr14"),
        "volume_ratio": _safe_last(vol_ratio_series, "volume_ratio"),
        "supertrend": _safe_last(supertrend_series, "supertrend"),
    }


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_position_size(
    entry: float,
    stop_loss: float,
    risk_capital: float,
) -> int:
    """
    floor(risk_capital / abs(entry - stop_loss))

    Returns 0 if the calculation would produce zero or infinite qty.
    """
    distance = abs(entry - stop_loss)
    if distance == 0:
        logger.warning("compute_position_size: SL == entry, cannot size position")
        return 0
    qty = math.floor(risk_capital / distance)
    return max(0, qty)


# ---------------------------------------------------------------------------
# Core signal evaluation
# ---------------------------------------------------------------------------

def evaluate_buy_signal(
    symbol: str,
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame,
    rs_rank: float,
    ai_score: float,
    vix_veto: bool,
    risk_capital: float,
    cost_per_order_inr: float = 22.0,
) -> Optional[Signal]:
    """
    Evaluate all 9 BUY conditions for RSMB.

    Parameters
    ----------
    symbol          : NSE symbol string.
    df_15m          : 15-minute OHLCV DataFrame (completed bars only, IST index).
    df_daily        : Daily OHLCV DataFrame (completed bars only).
    rs_rank         : Pre-computed RS_Rank float (NaN is treated as veto).
    ai_score        : XGBoost probability [0, 1]. Must be > 0.65 to pass.
    vix_veto        : True = VIX spike detected, block all trades.
    risk_capital    : equity_total × risk_per_trade_pct / 100 in INR.
    cost_per_order_inr : flat cost per order leg.

    Returns
    -------
    Signal if all conditions pass, None otherwise (with rejection reason logged).
    """
    if df_15m.empty:
        return None

    last_ts = df_15m.index[-1]
    if not isinstance(last_ts, pd.Timestamp):
        logger.warning(f"evaluate_buy_signal: index is not DatetimeIndex for {symbol}")
        return None

    # --- Time gate checks (conditions 8 & 9) ---
    if _is_chop_zone(last_ts):
        logger.debug(f"RSMB {symbol}: NO_TRADE — chop zone {last_ts.time()}")
        return _rejected(symbol, "BUY", rs_rank, "chop_zone")

    if _is_after_cutoff(last_ts):
        logger.debug(f"RSMB {symbol}: NO_TRADE — after equity cutoff {last_ts.time()}")
        return _rejected(symbol, "BUY", rs_rank, "after_equity_cutoff")

    # --- VIX veto (condition 7) ---
    if vix_veto:
        logger.debug(f"RSMB {symbol}: NO_TRADE — VIX veto active")
        return _rejected(symbol, "BUY", rs_rank, "vix_veto")

    # --- AI score (condition 6) ---
    if ai_score <= 0.65:
        logger.debug(f"RSMB {symbol}: NO_TRADE — AI score {ai_score:.3f} <= 0.65")
        return _rejected(symbol, "BUY", rs_rank, f"ai_score_low:{ai_score:.3f}")

    # --- RS_Rank (condition 4) ---
    if math.isnan(rs_rank) or rs_rank < 1.05:
        logger.debug(f"RSMB {symbol}: NO_TRADE — RS_Rank {rs_rank:.3f} < 1.05")
        return _rejected(symbol, "BUY", rs_rank, f"rs_rank_low:{rs_rank:.3f}")

    # --- Compute indicators ---
    ind = _compute_signal_indicators(df_15m, df_daily)
    if not ind:
        return _rejected(symbol, "BUY", rs_rank, "insufficient_data")

    close = ind.get("close")
    vwap = ind.get("vwap")
    ema21 = ind.get("ema21")
    daily_close = ind.get("daily_close")
    daily_ema50 = ind.get("daily_ema50")
    atr14 = ind.get("atr14")
    volume_ratio = ind.get("volume_ratio")

    # --- Condition 1: close > VWAP ---
    if close is None or vwap is None or close <= vwap:
        return _rejected(symbol, "BUY", rs_rank, f"close_below_vwap:{close}:{vwap}")

    # --- Condition 2: close > EMA 21 ---
    if ema21 is None or close <= ema21:
        return _rejected(symbol, "BUY", rs_rank, f"close_below_ema21:{close}:{ema21}")

    # --- Condition 3: daily_close > daily EMA 50 ---
    if daily_close is None or daily_ema50 is None or daily_close <= daily_ema50:
        return _rejected(
            symbol, "BUY", rs_rank,
            f"daily_close_below_ema50:{daily_close}:{daily_ema50}"
        )

    # --- Condition 5: volume > 1.5× mean last 5 bars ---
    if volume_ratio is None or volume_ratio <= 1.5:
        return _rejected(
            symbol, "BUY", rs_rank,
            f"volume_ratio_low:{volume_ratio}"
        )

    # --- All conditions passed — build Signal ---
    if atr14 is None or atr14 == 0:
        return _rejected(symbol, "BUY", rs_rank, "atr_unavailable")

    entry = close  # signal bar close — fill at next bar open by paper engine
    sl = entry - (1.5 * atr14)
    risk = entry - sl
    target1 = entry + (1.5 * risk)
    target2 = entry + (3.0 * risk)
    qty = compute_position_size(entry, sl, risk_capital)

    if qty <= 0:
        return _rejected(symbol, "BUY", rs_rank, "qty_zero_risk_too_large")

    logger.info(
        f"RSMB BUY SIGNAL: {symbol} | entry={entry:.2f} sl={sl:.2f} "
        f"T1={target1:.2f} T2={target2:.2f} qty={qty} rs={rs_rank:.3f} ai={ai_score:.3f}"
    )

    return Signal(
        strategy="rsmb",
        symbol=symbol,
        side="BUY",
        entry=entry,
        sl=sl,
        target1=target1,
        target2=target2,
        qty=qty,
        score=ai_score,
        rs_rank=rs_rank,
        rejection_reason=None,
        timestamp=last_ts,
    )


def evaluate_sell_signal(
    symbol: str,
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame,
    rs_rank: float,
    ai_score: float,
    vix_veto: bool,
    risk_capital: float,
    cost_per_order_inr: float = 22.0,
) -> Optional[Signal]:
    """
    Evaluate all 6 SELL conditions for RSMB (short via F&O).
    """
    if df_15m.empty:
        return None

    last_ts = df_15m.index[-1]

    # --- Time gate ---
    if _is_chop_zone(last_ts):
        return _rejected(symbol, "SELL", rs_rank, "chop_zone")

    if _is_after_cutoff(last_ts):
        return _rejected(symbol, "SELL", rs_rank, "after_equity_cutoff")

    # --- VIX veto ---
    if vix_veto:
        return _rejected(symbol, "SELL", rs_rank, "vix_veto")

    # --- AI score ---
    if ai_score <= 0.65:
        return _rejected(symbol, "SELL", rs_rank, f"ai_score_low:{ai_score:.3f}")

    # --- RS_Rank (condition 2: must be < 0.95 for SELL) ---
    if math.isnan(rs_rank) or rs_rank >= 0.95:
        return _rejected(symbol, "SELL", rs_rank, f"rs_rank_not_weak:{rs_rank:.3f}")

    # --- Indicators ---
    ind = _compute_signal_indicators(df_15m, df_daily)
    if not ind:
        return _rejected(symbol, "SELL", rs_rank, "insufficient_data")

    close = ind.get("close")
    vwap = ind.get("vwap")
    ema21 = ind.get("ema21")
    volume_ratio = ind.get("volume_ratio")
    daily_close = ind.get("daily_close")
    daily_ema50 = ind.get("daily_ema50")
    atr14 = ind.get("atr14")

    # --- Condition 1: close < VWAP ---
    if close is None or vwap is None or close >= vwap:
        return _rejected(symbol, "SELL", rs_rank, f"close_above_vwap:{close}:{vwap}")

    # --- Condition 2: close < EMA21 (Bearish Momentum) ---
    if ema21 is None or close >= ema21:
        return _rejected(symbol, "SELL", rs_rank, f"close_above_ema21:{close}:{ema21}")

    # --- Condition 5: Volume Confirmation ---
    if volume_ratio is None or volume_ratio <= 1.5:
        return _rejected(symbol, "SELL", rs_rank, f"volume_ratio_low:{volume_ratio}")

    # --- Condition 3: daily_close < daily EMA 50 ---
    if daily_close is None or daily_ema50 is None or daily_close >= daily_ema50:
        return _rejected(
            symbol, "SELL", rs_rank,
            f"daily_close_above_ema50:{daily_close}:{daily_ema50}"
        )

    if atr14 is None or atr14 == 0:
        return _rejected(symbol, "SELL", rs_rank, "atr_unavailable")

    entry = close
    sl = entry + (1.5 * atr14)
    risk = sl - entry
    target1 = entry - (1.5 * risk)
    target2 = entry - (3.0 * risk)
    qty = compute_position_size(entry, sl, risk_capital)

    if qty <= 0:
        return _rejected(symbol, "SELL", rs_rank, "qty_zero_risk_too_large")

    logger.info(
        f"RSMB SELL SIGNAL: {symbol} | entry={entry:.2f} sl={sl:.2f} "
        f"T1={target1:.2f} T2={target2:.2f} qty={qty} rs={rs_rank:.3f} ai={ai_score:.3f}"
    )

    return Signal(
        strategy="rsmb",
        symbol=symbol,
        side="SELL",
        entry=entry,
        sl=sl,
        target1=target1,
        target2=target2,
        qty=qty,
        score=ai_score,
        rs_rank=rs_rank,
        rejection_reason=None,
        timestamp=last_ts,
    )


# ---------------------------------------------------------------------------
# Private helper — construct a rejected Signal (for logging in signal feed)
# ---------------------------------------------------------------------------

def _rejected(
    symbol: str,
    side: str,
    rs_rank: float,
    reason: str,
) -> None:
    """
    Log a rejection and return None.
    Returns None so callers can do: return _rejected(...).
    """
    logger.debug(f"RSMB {side} {symbol}: REJECTED [{reason}] rs={rs_rank:.3f}")
    return None
