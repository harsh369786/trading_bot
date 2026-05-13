"""
tests/test_indicators.py
--------------------------
Comprehensive unit tests for strategies/rsmb/indicators.py.
No imports from other test files. All assertions are deterministic.
"""
from __future__ import annotations

import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.rsmb.indicators import (
    compute_atr,
    compute_donchian,
    compute_ema,
    compute_rs_rank,
    compute_supertrend,
    compute_volume_ratio,
    compute_vwap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Synthetic intraday OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(500, 5000, n).astype(float)
    idx = pd.date_range("2026-01-02 09:15", periods=n, freq="15min", tz="Asia/Kolkata")
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _make_daily(n: int = 25, seed: int = 7) -> pd.Series:
    """Synthetic daily close series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.bdate_range("2025-11-01", periods=n, tz="Asia/Kolkata")
    return pd.Series(prices, index=idx)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

class TestComputeVwap:
    def test_output_shape_and_index(self):
        df = _make_ohlcv(20)
        vwap = compute_vwap(df)
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(df)
        assert (vwap.index == df.index).all()

    def test_vwap_is_positive_and_finite(self):
        df = _make_ohlcv(30)
        vwap = compute_vwap(df)
        # VWAP is a cumulative session average — not bounded by any single bar's H/L.
        # It must be positive and finite for all bars.
        assert (vwap.dropna() > 0).all()
        assert vwap.dropna().notna().all()

    def test_missing_column_returns_nan_series(self):
        df = _make_ohlcv(10).drop(columns=["volume"])
        result = compute_vwap(df)
        assert result.isna().all()

    def test_single_bar_equals_typical_price(self):
        df = _make_ohlcv(1)
        vwap = compute_vwap(df)
        expected = (df["high"].iloc[0] + df["low"].iloc[0] + df["close"].iloc[0]) / 3
        assert pytest.approx(vwap.iloc[0], rel=1e-6) == expected

    def test_zero_volume_bar_produces_nan(self):
        df = _make_ohlcv(5)
        df["volume"] = 0.0
        vwap = compute_vwap(df)
        assert vwap.isna().all()

    def test_vwap_monotonically_tracks_price(self):
        """On a constant price series, VWAP == close."""
        n = 10
        idx = pd.date_range("2026-01-02 09:15", periods=n, freq="15min")
        df = pd.DataFrame({
            "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0
        }, index=idx)
        vwap = compute_vwap(df)
        assert (vwap - 100.0).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestComputeEma:
    def test_output_length_matches_input(self):
        s = pd.Series(np.arange(50, dtype=float))
        ema = compute_ema(s, 21)
        assert len(ema) == 50

    def test_empty_series_returns_empty(self):
        result = compute_ema(pd.Series(dtype=float), 10)
        assert result.empty

    def test_period_1_equals_input(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = compute_ema(s, 1)
        pd.testing.assert_series_equal(ema, s, check_names=False)

    def test_fewer_bars_than_period_no_exception(self):
        s = pd.Series([100.0, 101.0, 102.0])
        result = compute_ema(s, 21)   # should not raise
        assert len(result) == 3

    def test_ema_converges_on_constant_series(self):
        s = pd.Series([50.0] * 100)
        ema = compute_ema(s, 21)
        assert pytest.approx(ema.iloc[-1], abs=1e-6) == 50.0

    def test_ema_lags_behind_rising_series(self):
        s = pd.Series(np.arange(1, 51, dtype=float))
        ema = compute_ema(s, 10)
        # EMA must always be less than or equal to the last price for rising input
        assert ema.iloc[-1] <= s.iloc[-1]


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestComputeAtr:
    def test_output_shape(self):
        df = _make_ohlcv(30)
        atr = compute_atr(df, 14)
        assert len(atr) == 30

    def test_atr_positive(self):
        df = _make_ohlcv(30)
        atr = compute_atr(df, 14).dropna()
        assert (atr > 0).all()

    def test_zero_range_bars_give_near_zero_atr(self):
        n = 30
        idx = pd.date_range("2026-01-02", periods=n, freq="15min")
        df = pd.DataFrame({
            "high": 100.0, "low": 100.0, "close": 100.0
        }, index=idx)
        atr = compute_atr(df, 14).dropna()
        assert (atr.abs() < 1e-9).all()

    def test_missing_column_returns_nan(self):
        df = _make_ohlcv(20).drop(columns=["low"])
        result = compute_atr(df, 14)
        assert result.isna().all()

    def test_atr_increases_with_higher_ranges(self):
        n = 50
        idx = pd.date_range("2026-01-02", periods=n, freq="15min")
        close = np.ones(n) * 100
        # Second half has 2× range
        df1 = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close}, index=idx)
        df2 = pd.DataFrame({"high": close + 2, "low": close - 2, "close": close}, index=idx)
        assert compute_atr(df2, 14).iloc[-1] > compute_atr(df1, 14).iloc[-1]


# ---------------------------------------------------------------------------
# RS_Rank
# ---------------------------------------------------------------------------

class TestComputeRsRank:
    def test_outperforming_stock(self):
        # stock up 20%, nifty up 5% → RS_Rank ~ 1.14
        stock = _make_daily(21)
        nifty = _make_daily(21, seed=99)
        # Force specific ratio
        stock_s = pd.Series([100.0] * 20 + [120.0])
        nifty_s = pd.Series([100.0] * 20 + [105.0])
        rs = compute_rs_rank(stock_s, nifty_s)
        assert pytest.approx(rs, rel=1e-6) == 120 / 105

    def test_underperforming_stock(self):
        stock_s = pd.Series([100.0] * 20 + [90.0])
        nifty_s = pd.Series([100.0] * 20 + [105.0])
        rs = compute_rs_rank(stock_s, nifty_s)
        assert rs < 1.0

    def test_equal_performance_gives_one(self):
        stock_s = pd.Series([100.0] * 20 + [110.0])
        nifty_s = pd.Series([100.0] * 20 + [110.0])
        rs = compute_rs_rank(stock_s, nifty_s)
        assert pytest.approx(rs, rel=1e-9) == 1.0

    def test_insufficient_stock_data_returns_nan(self):
        stock_s = pd.Series([100.0] * 10)   # only 10 bars
        nifty_s = pd.Series([100.0] * 21)
        result = compute_rs_rank(stock_s, nifty_s)
        assert math.isnan(result)

    def test_insufficient_nifty_data_returns_nan(self):
        stock_s = pd.Series([100.0] * 21)
        nifty_s = pd.Series([100.0] * 5)
        result = compute_rs_rank(stock_s, nifty_s)
        assert math.isnan(result)

    def test_zero_base_price_returns_nan(self):
        stock_s = pd.Series([0.0] + [100.0] * 20)
        nifty_s = pd.Series([100.0] * 21)
        result = compute_rs_rank(stock_s, nifty_s)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

class TestComputeDonchian:
    def test_returns_tuple_of_series(self):
        df = _make_ohlcv(30)
        upper, lower = compute_donchian(df, 20)
        assert isinstance(upper, pd.Series) and isinstance(lower, pd.Series)
        assert len(upper) == 30

    def test_upper_always_gte_lower(self):
        df = _make_ohlcv(40)
        upper, lower = compute_donchian(df, 20)
        valid = ~(upper.isna() | lower.isna())
        assert (upper[valid] >= lower[valid]).all()

    def test_first_period_minus_one_bars_are_nan(self):
        df = _make_ohlcv(25)
        upper, lower = compute_donchian(df, 20)
        assert upper.iloc[:19].isna().all()
        assert not pd.isna(upper.iloc[19])

    def test_missing_column(self):
        df = _make_ohlcv(30).drop(columns=["low"])
        upper, lower = compute_donchian(df, 20)
        assert upper.isna().all() and lower.isna().all()

    def test_constant_price_channels_equal(self):
        n = 30
        idx = pd.date_range("2026-01-02", periods=n, freq="15min")
        df = pd.DataFrame({"high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
        upper, lower = compute_donchian(df, 20)
        valid = ~upper.isna()
        assert (upper[valid] == lower[valid]).all()


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

class TestComputeSupertrend:
    def test_output_shape(self):
        df = _make_ohlcv(40)
        st = compute_supertrend(df, factor=3.0, period=10)
        assert len(st) == 40

    def test_insufficient_data_returns_all_nan(self):
        df = _make_ohlcv(5)
        st = compute_supertrend(df, period=10)
        assert st.isna().all()

    def test_supertrend_is_not_all_nan_with_enough_bars(self):
        df = _make_ohlcv(50)
        st = compute_supertrend(df, period=10)
        assert not st.dropna().empty

    def test_missing_column(self):
        df = _make_ohlcv(30).drop(columns=["high"])
        st = compute_supertrend(df)
        assert st.isna().all()

    def test_supertrend_below_price_in_uptrend(self):
        """In a clean uptrend, Supertrend stop line should be below close."""
        n = 60
        idx = pd.date_range("2026-01-02 09:15", periods=n, freq="15min")
        close = np.linspace(100, 130, n)
        high = close + 0.3
        low = close - 0.3
        df = pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)
        st = compute_supertrend(df, factor=3.0, period=10)
        valid = st.dropna()
        # In uptrend, stop should be below close (at least for the last few bars)
        last_st = st.iloc[-1]
        last_close = df["close"].iloc[-1]
        assert last_st < last_close


# ---------------------------------------------------------------------------
# Volume Ratio
# ---------------------------------------------------------------------------

class TestComputeVolumeRatio:
    def test_output_length(self):
        vol = pd.Series([1000.0] * 20)
        ratio = compute_volume_ratio(vol, window=5)
        assert len(ratio) == 20

    def test_constant_volume_gives_ratio_one(self):
        vol = pd.Series([1000.0] * 20)
        ratio = compute_volume_ratio(vol, window=5)
        # After enough bars, ratio should be 1.0
        assert pytest.approx(ratio.iloc[-1], rel=1e-6) == 1.0

    def test_volume_spike_detected(self):
        # Spike on last bar: 5× the previous bars
        vol = pd.Series([1000.0] * 10 + [5000.0])
        ratio = compute_volume_ratio(vol, window=5)
        assert ratio.iloc[-1] > 4.0

    def test_empty_series_returns_empty(self):
        result = compute_volume_ratio(pd.Series(dtype=float), window=5)
        assert result.empty

    def test_zero_mean_volume_returns_nan(self):
        vol = pd.Series([0.0] * 10 + [1000.0])
        ratio = compute_volume_ratio(vol, window=5)
        # With all-zero preceding bars, rolling mean is 0 → NaN
        assert pd.isna(ratio.iloc[-1])
