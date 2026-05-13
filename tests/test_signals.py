"""
tests/test_signals.py
-----------------------
Unit tests for strategies/rsmb/signals.py entry/exit logic.
All tests use synthetic DataFrames — no I/O, no mocking of broker APIs.
"""
from __future__ import annotations

import math
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.rsmb.signals import (
    _is_chop_zone,
    _is_after_cutoff,
    compute_position_size,
    evaluate_buy_signal,
    evaluate_sell_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IST = "Asia/Kolkata"

def _ts(time_str: str) -> pd.Timestamp:
    """Create an IST timestamp for today at a given HH:MM."""
    return pd.Timestamp(f"2026-01-05 {time_str}:00", tz=IST)


def _make_15m(n: int = 20, close_trend: str = "up", base: float = 2500.0) -> pd.DataFrame:
    """
    Generate a synthetic 15m OHLCV bar DataFrame.
    Default n=20 → last bar at 14:00 IST (safely before 15:15 cutoff).

    close_trend: "up" | "down" | "flat"
    Starts at 09:15 on a market day.
    """
    if close_trend == "up":
        close = np.linspace(base, base * 1.05, n)
    elif close_trend == "down":
        close = np.linspace(base, base * 0.95, n)
    else:
        close = np.full(n, base)

    high = close + 2.0
    low = close - 2.0
    volume = np.full(n, 2000.0)
    # Give the last bar a volume spike (3× the previous 5-bar mean)
    volume[-1] = 6000.0

    idx = pd.date_range("2026-01-05 09:15", periods=n, freq="15min", tz=IST)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _make_daily(n: int = 60, close_trend: str = "up", base: float = 2500.0) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame."""
    if close_trend == "up":
        close = np.linspace(base * 0.90, base, n)
    else:
        close = np.linspace(base, base * 0.90, n)
    high = close + 5.0
    low = close - 5.0
    volume = np.full(n, 500000.0)
    idx = pd.bdate_range("2025-09-01", periods=n, tz=IST)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _good_buy_params(close_15m: float = 2550.0) -> dict:
    """
    Return a complete set of parameters that should trigger a BUY signal.
    
    Strategy: build the whole 20-bar session at `close_15m` with flat volume.
    VWAP = typical_price = close_15m, so close == VWAP — then make the last bar
    slightly higher to guarantee close > VWAP strictly.
    """
    n = 20
    idx = pd.date_range("2026-01-05 09:15", periods=n, freq="15min", tz=IST)
    # First 17 bars at base, last 3 at close_15m (above base) → VWAP pulled down
    base = close_15m - 5.0
    close = np.full(n, base)
    close[-3:] = close_15m
    high = close + 2.0
    low = close - 1.0
    volume = np.full(n, 2000.0)
    volume[-1] = 6000.0  # volume spike for condition 5
    df_15m = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=idx)

    # Daily: last close above EMA 50 (uptrend)
    df_daily = _make_daily(n=60, close_trend="up", base=close_15m * 0.97)
    return {
        "symbol": "RELIANCE",
        "df_15m": df_15m,
        "df_daily": df_daily,
        "rs_rank": 1.10,
        "ai_score": 0.75,
        "vix_veto": False,
        "risk_capital": 500.0,
    }


def _good_sell_params(close_15m: float = 2450.0) -> dict:
    """
    Return a complete set of parameters that should trigger a SELL signal.
    Last bar is below VWAP (session average higher due to earlier bars at higher price).
    """
    n = 20
    idx = pd.date_range("2026-01-05 09:15", periods=n, freq="15min", tz=IST)
    base = close_15m + 5.0
    close = np.full(n, base)
    close[-3:] = close_15m  # last 3 bars dip below session VWAP
    high = close + 1.0
    low = close - 2.0
    volume = np.full(n, 2000.0)
    df_15m = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=idx)

    # Daily: last close below EMA 50 (downtrend)
    df_daily = _make_daily(n=60, close_trend="down", base=close_15m * 1.04)
    return {
        "symbol": "HDFCBANK",
        "df_15m": df_15m,
        "df_daily": df_daily,
        "rs_rank": 0.90,
        "ai_score": 0.80,
        "vix_veto": False,
        "risk_capital": 500.0,
    }


# ---------------------------------------------------------------------------
# Time gate helpers
# ---------------------------------------------------------------------------

class TestTimeGates:
    def test_chop_zone_11_30(self):
        assert _is_chop_zone(_ts("11:30"))

    def test_chop_zone_12_00(self):
        assert _is_chop_zone(_ts("12:00"))

    def test_chop_zone_13_30(self):
        assert _is_chop_zone(_ts("13:30"))

    def test_not_chop_zone_09_30(self):
        assert not _is_chop_zone(_ts("09:30"))

    def test_not_chop_zone_14_00(self):
        assert not _is_chop_zone(_ts("14:00"))

    def test_after_cutoff_15_15(self):
        assert _is_after_cutoff(_ts("15:15"))

    def test_after_cutoff_15_30(self):
        assert _is_after_cutoff(_ts("15:30"))

    def test_before_cutoff_15_14(self):
        assert not _is_after_cutoff(_ts("15:14"))

    def test_before_cutoff_09_15(self):
        assert not _is_after_cutoff(_ts("09:15"))


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestComputePositionSize:
    def test_basic_sizing(self):
        # risk 500, entry 100, sl 98 → distance 2 → qty = floor(500/2) = 250
        qty = compute_position_size(100.0, 98.0, 500.0)
        assert qty == 250

    def test_sl_equals_entry_returns_zero(self):
        qty = compute_position_size(100.0, 100.0, 500.0)
        assert qty == 0

    def test_large_risk_gives_smaller_qty(self):
        qty_tight_sl = compute_position_size(100.0, 99.0, 500.0)  # dist=1 → 500
        qty_wide_sl = compute_position_size(100.0, 95.0, 500.0)   # dist=5 → 100
        assert qty_tight_sl > qty_wide_sl

    def test_zero_risk_capital_gives_zero(self):
        qty = compute_position_size(100.0, 98.0, 0.0)
        assert qty == 0

    def test_sell_side_sizing_uses_abs_distance(self):
        # For SELL: entry 100, sl 102 → distance 2
        qty = compute_position_size(100.0, 102.0, 500.0)
        assert qty == 250


# ---------------------------------------------------------------------------
# BUY signal evaluation
# ---------------------------------------------------------------------------

class TestEvaluateBuySignal:
    def test_all_conditions_met_returns_signal(self):
        params = _good_buy_params()
        result = evaluate_buy_signal(**params)
        assert result is not None
        assert result.side == "BUY"
        assert result.strategy == "rsmb"
        assert result.qty > 0

    def test_chop_zone_blocks_buy(self):
        params = _good_buy_params()
        # Replace index to fall in chop zone
        df = params["df_15m"]
        new_idx = pd.date_range("2026-01-05 11:15", periods=len(df), freq="15min", tz=IST)
        params["df_15m"] = df.set_index(new_idx)
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_after_cutoff_blocks_buy(self):
        params = _good_buy_params()
        df = params["df_15m"]
        new_idx = pd.date_range("2026-01-05 15:15", periods=len(df), freq="15min", tz=IST)
        params["df_15m"] = df.set_index(new_idx)
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_vix_veto_blocks_buy(self):
        params = _good_buy_params()
        params["vix_veto"] = True
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_low_ai_score_blocks_buy(self):
        params = _good_buy_params()
        params["ai_score"] = 0.60  # below 0.65 threshold
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_borderline_ai_score_0_65_blocks(self):
        params = _good_buy_params()
        params["ai_score"] = 0.65  # exactly 0.65 — must be STRICTLY greater
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_rs_rank_below_threshold_blocks_buy(self):
        params = _good_buy_params()
        params["rs_rank"] = 1.04  # below 1.05
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_rs_rank_nan_blocks_buy(self):
        params = _good_buy_params()
        params["rs_rank"] = float("nan")
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_empty_15m_df_returns_none(self):
        params = _good_buy_params()
        params["df_15m"] = pd.DataFrame()
        result = evaluate_buy_signal(**params)
        assert result is None

    def test_signal_targets_are_correct_multiples(self):
        params = _good_buy_params()
        result = evaluate_buy_signal(**params)
        if result is None:
            pytest.skip("Signal not generated with synthetic data")
        risk = result.entry - result.sl
        assert result.target1 > result.entry
        assert result.target2 > result.target1
        assert pytest.approx(result.target1 - result.entry, rel=0.05) == 1.5 * risk
        assert pytest.approx(result.target2 - result.entry, rel=0.05) == 3.0 * risk

    def test_sl_is_below_entry_for_buy(self):
        params = _good_buy_params()
        result = evaluate_buy_signal(**params)
        if result is None:
            pytest.skip("Signal not generated with synthetic data")
        assert result.sl < result.entry


# ---------------------------------------------------------------------------
# SELL signal evaluation
# ---------------------------------------------------------------------------

class TestEvaluateSellSignal:
    def test_all_conditions_met_returns_signal(self):
        params = _good_sell_params()
        result = evaluate_sell_signal(**params)
        assert result is not None
        assert result.side == "SELL"
        assert result.strategy == "rsmb"

    def test_vix_veto_blocks_sell(self):
        params = _good_sell_params()
        params["vix_veto"] = True
        result = evaluate_sell_signal(**params)
        assert result is None

    def test_ai_score_too_low_blocks_sell(self):
        params = _good_sell_params()
        params["ai_score"] = 0.50
        result = evaluate_sell_signal(**params)
        assert result is None

    def test_rs_rank_not_weak_enough_blocks_sell(self):
        params = _good_sell_params()
        params["rs_rank"] = 0.96  # >= 0.95 — does NOT qualify as underperforming
        result = evaluate_sell_signal(**params)
        assert result is None

    def test_chop_zone_blocks_sell(self):
        params = _good_sell_params()
        df = params["df_15m"]
        new_idx = pd.date_range("2026-01-05 12:00", periods=len(df), freq="15min", tz=IST)
        params["df_15m"] = df.set_index(new_idx)
        result = evaluate_sell_signal(**params)
        assert result is None

    def test_sell_signal_sl_above_entry(self):
        params = _good_sell_params()
        result = evaluate_sell_signal(**params)
        if result is None:
            pytest.skip("Signal not generated with synthetic data")
        assert result.sl > result.entry

    def test_sell_signal_targets_below_entry(self):
        params = _good_sell_params()
        result = evaluate_sell_signal(**params)
        if result is None:
            pytest.skip("Signal not generated with synthetic data")
        assert result.target1 < result.entry
        assert result.target2 < result.target1

    def test_empty_df_returns_none(self):
        params = _good_sell_params()
        params["df_15m"] = pd.DataFrame()
        result = evaluate_sell_signal(**params)
        assert result is None
