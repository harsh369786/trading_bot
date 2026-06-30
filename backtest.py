# Install if needed:
#   pip install pandas numpy yfinance matplotlib scipy
from __future__ import annotations

import math
import os
import random
import warnings
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_indicators import add_indicators, candle_pattern, sigmoid, trading_times, volume_profile
from backtest_report import BacktestReporter
from backtest_types import BacktestResults, Position, StrategyResult

import joblib
import xgboost as xgb
from models.xgboost.model import XGBoostModel
from strategies.rsmb.ai_filter import RSMBAIFilter

warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG = {
    "backtest_period_months": 6,
    "warmup_bars": 200,
    "random_seed": 42,
    "risk_free_rate_annual": 0.06,
    "brokerage_per_leg_inr": 22.0,
    "slippage_pct": 0.001,
    "risk_per_trade_pct": 0.01,
    "use_yfinance": True,
    "ensemble_ai": {
        "enabled": True,
        "capital": 50000,
        "min_confidence": 0.62,
        "min_adx": 15,
        "min_rel_vol": 1.0,
        "atr_sl_multiplier": 1.5,
        "rr_ratio": 1.5,
        "max_open_trades": 2,
        "daily_loss_limit_inr": 3000,
    },
    "rsmb": {
        "enabled": True,
        "capital": 50000,
        "min_rs_rank": 1.05,
        "min_ai_score": 0.65,
        "min_vol_ratio": 1.5,
        "atr_sl_multiplier": 1.5,
        "t1_rr": 1.5,
        "t2_rr": 3.0,
        "max_open_trades": 3,
        "daily_loss_limit_inr": 3000,
    },
    "gamma_scalper": {
        "enabled": True,
        "capital": 30000,
        "min_candle_strength_pct": 10.0,
        "min_ai_score": 0.70,
        "min_adx": 20,
        "min_vol_ratio": 1.2,
        "sl_max_pct": 0.12,
        "t1_pct": 0.30,
        "theta_veto_bars": 3,
        "max_open_trades": 2,
        "daily_loss_limit_inr": 3000,
    },
    "mean_reversion": {
        "enabled": True,
        "capital": 40000,
        "min_ai_score": 0.60,
        "max_adx": 35,
        "min_distance_pct": 3.0,
        "min_wick_ratio": 2.0,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "sl_buffer_inr": 0.10,
        "max_open_trades": 3,
        "daily_loss_limit_inr": 2000,
    },
}

EQUITY_SYMBOLS = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "LT", "WIPRO", "NIFTY"]
START_PRICES = {"RELIANCE": 2800, "HDFCBANK": 1650, "INFY": 1800, "TCS": 4200, "ICICIBANK": 1250, "SBIN": 820, "AXISBANK": 1150, "KOTAKBANK": 1800, "LT": 3600, "WIPRO": 550, "SENSEX": 75000, "NIFTY": 22000}
BASE_VOLUME = {"RELIANCE": 4_000_000, "HDFCBANK": 5_000_000, "INFY": 3_500_000, "TCS": 1_800_000, "ICICIBANK": 6_000_000, "SBIN": 10_000_000, "AXISBANK": 4_500_000, "KOTAKBANK": 2_000_000, "LT": 2_200_000, "WIPRO": 6_000_000, "SENSEX": 1_000_000, "NIFTY": 500_000}
SEED_OFFSETS = {"ensemble_ai": 11, "rsmb": 23, "gamma_scalper": 37, "mean_reversion": 41}


class EventDrivenSimulator:
    """Run one strategy through synchronized bar events."""

    def __init__(self, engine: "BacktestEngine", strategy: str, symbols: list[str], overrides: dict[str, Any] | None = None):
        """Initialize simulator state."""
        self.engine = engine
        self.strategy = strategy
        self.symbols = symbols
        self.cfg = {**engine.config[strategy], **(overrides or {})}
        self.rng = np.random.default_rng(engine.config["random_seed"] + SEED_OFFSETS[strategy])
        self.equity = float(self.cfg["capital"])
        self.positions: list[Position] = []
        self.pending: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.signals_generated = 0
        self.daily_pnl: dict[pd.Timestamp, float] = {}
        self.loss_block_days: set[pd.Timestamp] = set()

    def run(self) -> StrategyResult:
        """Run the full strategy backtest."""
        if not self.cfg.get("enabled", True):
            return self.engine.empty_result(self.strategy)
        timestamps = self._timestamps()
        print(f"Running {self.strategy} on {len(timestamps):,} synchronized bars...")
        for n, ts in enumerate(timestamps, start=1):
            if n % 1000 == 0:
                print(f"  {self.strategy}: {n:,}/{len(timestamps):,} bars")
            self._drop_stale_pending(ts)
            self._fill_pending(ts)
            self._monitor_positions(ts)
            self._square_off_eod(ts)
            self._generate_signals(ts)
        self._force_close_remaining()
        metrics, monthly, daily = self.engine.calculate_metrics(self.strategy, self.trades, self.signals_generated, self.loss_block_days)
        return StrategyResult(self.strategy, self.trades, metrics, monthly, daily)

    def _timestamps(self) -> list[pd.Timestamp]:
        """Return all timestamps relevant to this strategy."""
        keys = ["SENSEX_CE_5M", "SENSEX_PE_5M"] if self.strategy == "gamma_scalper" else self.symbols
        idx = sorted(set().union(*[set(self.engine.data[k].index) for k in keys if k in self.engine.data]))
        return idx

    def _df(self, symbol: str) -> pd.DataFrame:
        """Return a symbol dataframe."""
        return self.engine.data[symbol]

    def _row(self, symbol: str, ts: pd.Timestamp) -> pd.Series | None:
        """Return a row at timestamp if available."""
        return self.engine.row_by_ts.get(symbol, {}).get(ts)

    def _eval_start(self, df: pd.DataFrame) -> pd.Timestamp:
        """Return the train/test split timestamp."""
        return self.engine.eval_start_by_id[id(df)]

    def _drop_stale_pending(self, ts: pd.Timestamp) -> None:
        """Drop signals that did not receive a same-day next-bar fill."""
        self.pending = [p for p in self.pending if p["ts"].date() == ts.date()]

    def _fill_pending(self, ts: pd.Timestamp) -> None:
        """Fill eligible pending signals at next bar open."""
        remaining = []
        for sig in self.pending:
            row = self._row(sig["symbol"], ts)
            if row is None or ts <= sig["ts"]:
                remaining.append(sig)
                continue
            fill = self._slipped(float(row.open), sig["side"], is_entry=True)
            pos = self._make_position(sig, fill, ts)
            if pos:
                self.positions.append(pos)
        self.pending = remaining

    def _make_position(self, sig: dict[str, Any], entry: float, ts: pd.Timestamp) -> Position | None:
        """Create a sized position from a filled signal."""
        if self.strategy in ("ensemble_ai", "rsmb"):
            risk = max(float(sig["atr"]) * self.cfg["atr_sl_multiplier"], entry * 0.002)
            sl = entry - risk if sig["side"] == "BUY" else entry + risk
            if self.strategy == "ensemble_ai":
                target1, target2, pct, mode = (entry + risk * self.cfg["rr_ratio"] if sig["side"] == "BUY" else entry - risk * self.cfg["rr_ratio"]), None, 1.0, "price"
            else:
                target1 = entry + risk * self.cfg["t1_rr"] if sig["side"] == "BUY" else entry - risk * self.cfg["t1_rr"]
                target2 = entry + risk * self.cfg["t2_rr"] if sig["side"] == "BUY" else entry - risk * self.cfg["t2_rr"]
                pct, mode = 0.5, "price"
        elif self.strategy == "gamma_scalper":
            sl = max(sig["trigger_low"], entry * (1 - self.cfg["sl_max_pct"]))
            risk = max(entry - sl, entry * 0.001)
            target1, target2, pct, mode = entry * (1 + self.cfg["t1_pct"]), None, 0.6, "ema9"
        elif self.strategy == "mean_reversion":
            sl, target1, target2 = sig["sl"], sig["target1"], sig["target2"]
            risk = abs(entry - sl)
            pct, mode = 0.5, "price"
        else:
            # Fallback
            risk = entry * 0.01
            sl = entry - risk if sig["side"] == "BUY" else entry + risk
            target1, target2, pct, mode = entry + risk, None, 1.0, "price"
        qty = self._position_qty(entry, risk)
        if qty <= 0 or not all(np.isfinite([entry, sl, risk])):
            return None
        return Position(sig["symbol"], self.strategy, sig["side"], entry, sl, target1, target2, qty, qty, "OPEN", ts, sig["i"], sig["score"], risk, mode, pct, self.cfg.get("theta_veto_bars", 0))

    def _position_qty(self, entry: float, risk: float) -> int:
        """Size by 1% running equity risk and cap by cash."""
        capital = self.equity
        risk_cash = capital * float(self.engine.config.get("risk_per_trade_pct", 0.01))
        by_risk = int(risk_cash / max(risk, 1e-9))
        by_cash = int(capital / max(entry, 1e-9))
        return max(1, min(by_risk, max(1, by_cash)))

    def _monitor_positions(self, ts: pd.Timestamp) -> None:
        """Update open positions using current bar high/low."""
        for pos in list(self.positions):
            row = self._row(pos.symbol, ts)
            if row is None or ts < pos.entry_ts or pos.status != "OPEN":
                continue
            if self.strategy == "gamma_scalper" and not pos.t1_done and ts > pos.entry_ts and (self._bar_i(pos.symbol, ts) - pos.entry_i) >= pos.theta_veto_bars:
                self._final_close(pos, float(row.close), ts, "THETA_VETO")
                continue
            if not pos.t1_done and pos.target1 is not None:
                first = self._first_hit(pos.side, row, pos.sl, pos.target1)
                if first == "SL":
                    self._final_close(pos, pos.sl, ts, "SL_HIT")
                    continue
                if first == "T1":
                    if pos.t1_exit_pct >= 1.0:
                        self._final_close(pos, pos.target1, ts, "TARGET_HIT")
                        continue
                    self._partial_close(pos, pos.target1)
            if pos.status != "OPEN":
                continue
            if pos.t1_done:
                pos.sl = pos.entry if self.strategy in ("rsmb", "mean_reversion") else max(pos.sl, pos.entry * 1.20)
                if self.strategy == "rsmb" and pd.notna(getattr(row, "supertrend", np.nan)):
                    pos.sl = max(pos.sl, float(row.supertrend)) if pos.side == "BUY" else min(pos.sl, float(row.supertrend))
                if self.strategy == "gamma_scalper" and pd.notna(getattr(row, "ema9", np.nan)) and float(row.close) < float(row.ema9):
                    self._final_close(pos, float(row.close), ts, "T2_TRAIL_HIT")
                    continue
                if pos.target2 is not None:
                    first = self._first_hit(pos.side, row, pos.sl, pos.target2)
                    if first == "SL":
                        self._final_close(pos, pos.sl, ts, "TRAIL_SL_HIT")
                    elif first == "T1":
                        self._final_close(pos, pos.target2, ts, "TARGET2_HIT")
                elif self._sl_hit(pos.side, row, pos.sl):
                    self._final_close(pos, pos.sl, ts, "TRAIL_SL_HIT")

    def _first_hit(self, side: str, row: pd.Series, sl: float, target: float) -> str | None:
        """Resolve whether SL or target hits first inside a bar."""
        sl_hit = self._sl_hit(side, row, sl)
        t_hit = self._target_hit(side, row, target)
        if not sl_hit and not t_hit:
            return None
        if sl_hit and not t_hit:
            return "SL"
        if t_hit and not sl_hit:
            return "T1"
        open_px = float(row.open)
        d_sl, d_t = abs(open_px - sl), abs(open_px - target)
        if d_sl < d_t * 0.8:
            return "SL"
        if d_t < d_sl * 0.8:
            return "T1"
        return "SL"  # Pessimistic tie-break to avoid look-ahead or luck bias

    def _sl_hit(self, side: str, row: pd.Series, sl: float) -> bool:
        """Return whether stop-loss was touched."""
        return float(row.low) <= sl if side == "BUY" else float(row.high) >= sl

    def _target_hit(self, side: str, row: pd.Series, target: float) -> bool:
        """Return whether target was touched."""
        return float(row.high) >= target if side == "BUY" else float(row.low) <= target

    def _partial_close(self, pos: Position, price: float) -> None:
        """Close the T1 portion of a position."""
        qty = max(1, int(round(pos.qty * pos.t1_exit_pct)))
        qty = min(qty, pos.qty_open)
        exit_px = self._slipped(price, pos.side, is_entry=False)
        pos.gross_realised += self._pnl(pos.side, pos.entry, exit_px, qty)
        pos.qty_open -= qty
        pos.brokerage_legs += 1
        pos.t1_done = True

    def _final_close(self, pos: Position, price: float, ts: pd.Timestamp, outcome: str) -> None:
        """Close the remaining position and journal the trade."""
        exit_px = self._slipped(price, pos.side, is_entry=False)
        pos.gross_realised += self._pnl(pos.side, pos.entry, exit_px, pos.qty_open)
        pos.brokerage_legs += 1
        net = pos.gross_realised - pos.brokerage_legs * self.engine.config["brokerage_per_leg_inr"]
        pos.exit_price, pos.outcome, pos.status = exit_px, outcome, "CLOSED"
        day = ts.normalize()
        self.daily_pnl[day] = self.daily_pnl.get(day, 0.0) + net
        self.equity += net
        if self.daily_pnl[day] <= -float(self.cfg["daily_loss_limit_inr"]):
            self.loss_block_days.add(day)
        self.trades.append({
            "date": ts.date().isoformat(), "time": ts.time().strftime("%H:%M:%S"), "strategy": self.strategy, "symbol": pos.symbol,
            "side": pos.side, "entry_price": round(pos.entry, 4), "exit_price": round(exit_px, 4), "qty": pos.qty,
            "gross_pnl": round(pos.gross_realised, 2), "net_pnl": round(net, 2), "outcome": outcome,
            "holding_minutes": int((ts - pos.entry_ts).total_seconds() // 60), "signal_score": round(pos.score, 4),
            "rejection_reason": "", "r_multiple": round(net / max(pos.risk_per_unit * pos.qty, 1e-9), 4),
        })
        self.positions.remove(pos)

    def _slipped(self, price: float, side: str, is_entry: bool) -> float:
        """Apply fixed slippage to an entry or exit."""
        slip = self.engine.config["slippage_pct"]
        if is_entry:
            return price * (1 + slip) if side == "BUY" else price * (1 - slip)
        return price * (1 - slip) if side == "BUY" else price * (1 + slip)

    def _pnl(self, side: str, entry: float, exit_px: float, qty: int) -> float:
        """Calculate gross P&L."""
        return (exit_px - entry) * qty if side == "BUY" else (entry - exit_px) * qty

    def _square_off_eod(self, ts: pd.Timestamp) -> None:
        """Force-close positions at the end-of-day bar."""
        cutoff = time(15, 20) if self.strategy == "gamma_scalper" else time(15, 15)
        if ts.time() < cutoff:
            return
        for pos in list(self.positions):
            row = self._row(pos.symbol, ts)
            if row is not None:
                self._final_close(pos, float(row.close), ts, "EOD_SQUARE_OFF")
        self.pending = [p for p in self.pending if p["ts"].date() != ts.date()]

    def _force_close_remaining(self) -> None:
        """Close any residual positions at final available close."""
        for pos in list(self.positions):
            df = self._df(pos.symbol)
            row = df.iloc[-1]
            self._final_close(pos, float(row.close), df.index[-1], "FINAL_CLOSE")

    def _generate_signals(self, ts: pd.Timestamp) -> None:
        """Generate new signals after bar close."""
        if self.strategy == "gamma_scalper":
            self._generate_gamma(ts)
            return
        if ts.time() >= time(15, 0):
            return
        for symbol in self.symbols:
            df = self._df(symbol)
            i = self.engine.index_pos[symbol].get(ts)
            if i is None:
                continue
            if i < self.engine.config["warmup_bars"] or ts < self._eval_start(df) or self._blocked(ts) or not self._has_next_same_day(df, i):
                continue
            if self._open_count() >= self.cfg["max_open_trades"]:
                continue
            sig = getattr(self, f"_signal_{self.strategy}")(symbol, df, i)
            if sig:
                self.signals_generated += 1
                self.pending.append(sig)

    def _generate_gamma(self, ts: pd.Timestamp) -> None:
        """Generate Gamma Scalper CE and PE signals."""
        if ts.time() >= time(15, 20) or self._blocked(ts):
            return
        spot = self.engine.data["SENSEX_5M"]
        if ts not in spot.index:
            return
        for symbol in ["SENSEX_CE_5M", "SENSEX_PE_5M"]:
            df = self._df(symbol)
            i = self.engine.index_pos[symbol].get(ts)
            if i is None:
                continue
            if i < self.engine.config["warmup_bars"] or ts < self._eval_start(df) or not self._has_next_same_day(df, i):
                continue
            leg = "CE" if "CE" in symbol else "PE"
            if self._open_count() >= self.cfg["max_open_trades"] or self._gamma_side_open(leg):
                continue
            sig = self._signal_gamma_scalper(symbol, df, spot, i)
            if sig:
                self.signals_generated += 1
                self.pending.append(sig)

    def _signal_ensemble_ai(self, symbol: str, df: pd.DataFrame, i: int) -> dict[str, Any] | None:
        """Evaluate Ensemble AI rules using real XGBoost model."""
        r = self.engine.rows[symbol][i]
        if r.Index.time() >= time(11, 30) and r.Index.time() <= time(13, 30):
            return None
        
        # Feature names expected by models/xgboost/model.json
        cols = ["ema_9", "ema_21", "ema_50", "rsi_14", "atr_14", "vwap", "ADX_14", "DMP_14", "DMN_14", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]
        features = [getattr(r, c) for c in cols]
        
        if not np.isfinite(features).all():
            return None
            
        if not (r.ema_9 > r.ema_21 and r.close > r.vwap and r.ADX_14 > self.cfg["min_adx"] and r.volume_ratio > self.cfg["min_rel_vol"]):
            return None
            
        if self.engine.equity_model_loaded:
            probs = self.engine.equity_model.predict(features)
            # Index 1 is BUY (3-class: 0=No, 1=Buy, 2=Sell)
            score = float(probs[1])
        else:
            # Fallback to heuristic if model not loaded
            score = sigmoid(0.3 * ((r.rsi_14 - 50) / 50) + 0.4 * ((r.ADX_14 - 20) / 30) + 0.3 * ((r.volume_ratio - 1) / 1)) + self.rng.normal(0, 0.08)
            score = float(np.clip(score, 0, 1))
            
        return {"symbol": symbol, "side": "BUY", "ts": r.Index, "i": i, "atr": r.atr_14, "score": score} if score > self.cfg["min_confidence"] else None

    def _signal_rsmb(self, symbol: str, df: pd.DataFrame, i: int) -> dict[str, Any] | None:
        """Evaluate RSMB rules using real XGBoost model."""
        r = self.engine.rows[symbol][i]
        if not np.isfinite([r.vwap, r.ema_21, r.volume_ratio, r.atr_14, r.ADX_14]).all() or r.volume_ratio <= self.cfg["min_vol_ratio"]:
            return None
        rs = self.engine.rs_rank(symbol, r.Index)
        
        # Feature names expected by models/rsmb_xgb.pkl
        cols = ["ema_9", "ema_21", "ema_50", "rsi_14", "atr_14", "vwap", "ADX_14", "DMP_14", "DMN_14", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]
        features_dict = {c: float(getattr(r, c)) for c in cols}
        features_dict["rs_rank"] = float(rs)
        
        if self.engine.rsmb_model_loaded:
            score = self.engine.rsmb_model.predict(features_dict)
        else:
            # Fallback to heuristic
            score = sigmoid(0.5 * ((rs - 1) / 0.5) + 0.3 * ((r.ADX_14 - 20) / 30) + 0.2 * ((r.volume_ratio - 1) / 1)) + self.rng.normal(0, 0.06)
            score = float(np.clip(score, 0, 1))
            
        if score <= self.cfg["min_ai_score"]:
            return None
            
        if rs > self.cfg["min_rs_rank"] and r.supertrend_dir > 0 and r.close > r.vwap and r.close > r.ema_21:
            return {"symbol": symbol, "side": "BUY", "ts": r.Index, "i": i, "atr": r.atr_14, "score": score}
        if rs < 0.95 and r.supertrend_dir < 0 and r.close < r.vwap and r.close < r.ema_21:
            return {"symbol": symbol, "side": "SELL", "ts": r.Index, "i": i, "atr": r.atr_14, "score": score}
        return None

    def _signal_gamma_scalper(self, symbol: str, opt: pd.DataFrame, spot: pd.DataFrame, i: int) -> dict[str, Any] | None:
        """Evaluate Gamma Scalper rules."""
        r = self.engine.rows[symbol][i]
        s = self.engine.row_by_ts["SENSEX_5M"].get(r.Index)
        if s is None:
            return None
        strength = (r.close - r.open) / max(r.open, 1e-9) * 100
        vals = [strength, s.ema_9, s.vwap, s.ADX_14, r.vwap, r.rsi_14, r.volume_ratio]
        if not np.isfinite(vals).all() or strength < self.cfg["min_candle_strength_pct"] or r.rsi_14 <= 60 or r.volume_ratio <= self.cfg["min_vol_ratio"] or s.ADX_14 < self.cfg["min_adx"] or r.close <= r.vwap:
            return None
        is_ce = "CE" in symbol
        if is_ce and not (s.close > s.ema_9 and s.close > s.vwap):
            return None
        if not is_ce and not (s.close < s.ema_9 and s.close < s.vwap):
            return None
        score = sigmoid(0.4 * ((strength - 10) / 10) + 0.3 * ((s.ADX_14 - 20) / 30) + 0.3 * ((r.rsi_14 - 50) / 50)) + self.rng.normal(0, 0.10)
        score = float(np.clip(score, 0, 1))
        return {"symbol": symbol, "side": "BUY", "ts": r.Index, "i": i, "trigger_low": float(r.low), "score": score} if score >= self.cfg["min_ai_score"] else None

    def _signal_mean_reversion(self, symbol: str, df: pd.DataFrame, i: int) -> dict[str, Any] | None:
        """Evaluate 200-SMA mean-reversion rules."""
        r, p = self.engine.rows[symbol][i], self.engine.rows[symbol][i - 1]
        if not np.isfinite([r.sma200, r.rsi_14, r.ADX_14, r.ema_21, r.vwap]).all() or r.ADX_14 >= self.cfg["max_adx"] or not (r.low <= r.sma200 <= r.high):
            return None
        prior = df.iloc[max(0, i - 50):i]
        dist = (prior["close"].sub(prior["sma200"]).abs() / prior["sma200"].replace(0, np.nan) * 100).max()
        body = max(abs(r.close - r.open), 1e-9)
        wr_up = (r.high - max(r.close, r.open)) / body
        wr_dn = (min(r.close, r.open) - r.low) / body
        pattern = candle_pattern(df, i)
        if not np.isfinite(dist) or dist < self.cfg["min_distance_pct"]:
            return None
        side, wick = None, None
        if p.close > p.sma200 and r.rsi_14 < self.cfg["rsi_oversold"] and pattern in ("hammer", "bullish_engulfing") and wr_dn >= self.cfg["min_wick_ratio"]:
            side, wick = "BUY", wr_dn
        if p.close < p.sma200 and r.rsi_14 > self.cfg["rsi_overbought"] and pattern in ("shooting_star", "bearish_engulfing") and wr_up >= self.cfg["min_wick_ratio"]:
            side, wick = "SELL", wr_up
        if side is None:
            return None
        score = sigmoid(0.4 * ((dist - 3) / 3) + 0.3 * ((wick - 2) / 2) + 0.3 * ((abs(r.rsi_14 - 50)) / 50)) + self.rng.normal(0, 0.07)
        score = float(np.clip(score, 0, 1))
        if score < self.cfg["min_ai_score"]:
            return None
        buf = self.cfg["sl_buffer_inr"]
        sl = min(r.sma200 - buf, r.low - buf) if side == "BUY" else max(r.sma200 + buf, r.high + buf)
        if side == "BUY" and not (r.ema_21 > r.close and r.vwap > r.close):
            return None
        if side == "SELL" and not (r.ema_21 < r.close and r.vwap < r.close):
            return None
        return {"symbol": symbol, "side": side, "ts": r.Index, "i": i, "sl": float(sl), "target1": float(r.ema_21), "target2": float(r.vwap), "score": score}

    def _blocked(self, ts: pd.Timestamp) -> bool:
        """Return whether daily loss circuit breaker is active (including unrealised)."""
        day = ts.normalize()
        realised = self.daily_pnl.get(day, 0.0)
        unrealised = sum(self._pnl(p.side, p.entry, self._row(p.symbol, ts).close if self._row(p.symbol, ts) else p.entry, p.qty_open) for p in self.positions)
        return (realised + unrealised) <= -float(self.cfg["daily_loss_limit_inr"])

    def _open_count(self) -> int:
        """Return open plus pending strategy exposure."""
        return len(self.positions) + len(self.pending)

    def _gamma_side_open(self, leg: str) -> bool:
        """Return whether a CE or PE leg is already pending/open."""
        return any(leg in p.symbol for p in self.positions) or any(leg in p["symbol"] for p in self.pending)

    def _bar_i(self, symbol: str, ts: pd.Timestamp) -> int:
        """Return integer bar index for timestamp."""
        return int(self.engine.index_pos[symbol][ts])

    def _has_next_same_day(self, df: pd.DataFrame, i: int) -> bool:
        """Return whether a next same-day bar exists for fill."""
        return i + 1 < len(df) and df.index[i + 1].date() == df.index[i].date()


class BacktestEngine:
    """Load data, run all strategies, and produce reports."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the backtest engine."""
        self.config = config
        self.output_dir = Path("backtest_output")
        self.output_dir.mkdir(exist_ok=True)
        self.rng = np.random.default_rng(config["random_seed"])
        random.seed(config["random_seed"])
        self.data: dict[str, pd.DataFrame] = {}
        self.rows: dict[str, list[Any]] = {}
        self.row_by_ts: dict[str, dict[pd.Timestamp, Any]] = {}
        self.index_pos: dict[str, dict[pd.Timestamp, int]] = {}
        self.eval_start_by_id: dict[int, pd.Timestamp] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load production models for inference."""
        self.equity_model = XGBoostModel("models/xgboost/model.json")
        self.equity_model_loaded = self.equity_model.load()
        
        self.rsmb_model = RSMBAIFilter(Path("models/rsmb_xgb.pkl"))
        self.rsmb_model_loaded = self.rsmb_model.is_loaded
        
        if not self.equity_model_loaded:
            print("WARNING: Equity XGBoost model could not be loaded. Backtest will use heuristic scores for Ensemble AI.")
        if not self.rsmb_model_loaded:
            print("WARNING: RSMB XGBoost model could not be loaded. Backtest will use heuristic scores for RSMB.")

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Fetch yfinance daily data or generate synthetic OHLCV."""
        daily = self._download_daily_batch() if self.config.get("use_yfinance", True) else {}
        for symbol in EQUITY_SYMBOLS:
            df = self._daily_to_intraday(symbol, daily.get(symbol), "15min") if symbol in daily else self._synthetic_intraday(symbol, "15min")
            self.data[symbol] = add_indicators(df)
        sensex_15 = self._daily_to_intraday("SENSEX", daily.get("SENSEX"), "15min") if "SENSEX" in daily else self._synthetic_intraday("SENSEX", "15min")
        sensex_5 = self._daily_to_intraday("SENSEX", daily.get("SENSEX"), "5min") if "SENSEX" in daily else self._synthetic_intraday("SENSEX", "5min")
        self.data["SENSEX_15M"] = add_indicators(sensex_15)
        self.data["SENSEX_5M"] = add_indicators(sensex_5)
        self.data["SENSEX_CE_5M"] = add_indicators(self._simulate_option("SENSEX_CE_5M", self.data["SENSEX_5M"], 1))
        self.data["SENSEX_PE_5M"] = add_indicators(self._simulate_option("SENSEX_PE_5M", self.data["SENSEX_5M"], -1))
        self._build_caches()
        self._print_data_quality()
        return self.data

    def _build_caches(self) -> None:
        """Build fast row and index lookup caches."""
        class Row:
            def __init__(self, index, data):
                self.Index = index
                for k, v in data.items():
                    setattr(self, k, v)

        for symbol, df in self.data.items():
            # Use to_dict('index') to avoid itertuples name mangling for dots (e.g. BBL_20_2.0)
            data_dict = df.to_dict('index')
            rows = [Row(ts, data) for ts, data in data_dict.items()]
            
            self.rows[symbol] = rows
            self.row_by_ts[symbol] = {row.Index: row for row in rows}
            self.index_pos[symbol] = {row.Index: i for i, row in enumerate(rows)}
            warmup = self.config["warmup_bars"]
            self.eval_start_by_id[id(df)] = df.index[warmup] if len(df) > warmup else df.index[0]

    def _download_daily_batch(self) -> dict[str, pd.DataFrame]:
        """Download yfinance daily data in one batch."""
        try:
            import yfinance as yf
            cache_dir = self.output_dir / "yf_cache"
            cache_dir.mkdir(exist_ok=True)
            if hasattr(yf, "set_tz_cache_location"):
                yf.set_tz_cache_location(str(cache_dir))
            if os.getenv("BACKTEST_DISABLE_YFINANCE", "").lower() in {"1", "true", "yes"}:
                print("Skipping yfinance because BACKTEST_DISABLE_YFINANCE is set.")
                return {}
            mapping = {**{s: f"{s}.NS" for s in EQUITY_SYMBOLS}, "SENSEX": "^BSESN", "NIFTY": "^NSEI"}
            print("Trying yfinance daily download...")
            raw = yf.download(list(mapping.values()), period="2y", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True, timeout=10)
            out = {}
            for symbol, ticker in mapping.items():
                try:
                    df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
                    df = df.rename(columns={"adj_close": "adj_close"})
                    df = df[["open", "high", "low", "close", "volume"]].dropna().tail(280)
                    if len(df) >= 80:
                        out[symbol] = df
                    else:
                        print(f"WARNING: yfinance returned insufficient rows for {symbol}; using synthetic data.")
                except Exception as exc:
                    print(f"WARNING: could not parse yfinance data for {symbol}: {exc}")
            return out
        except Exception as exc:
            print(f"WARNING: yfinance download failed; using synthetic data. Reason: {exc}")
            return {}

    def _daily_to_intraday(self, symbol: str, daily: pd.DataFrame | None, freq: str) -> pd.DataFrame:
        """Expand daily OHLCV into constrained intraday bars."""
        if daily is None or daily.empty:
            return self._synthetic_intraday(symbol, freq)
        times = trading_times(freq)
        prof = volume_profile(len(times))
        rows = []
        for d, day in daily.tail(140).iterrows():
            d = pd.Timestamp(d).date()
            # Fix CRIT-1: Generate path without knowing day high/low
            base = np.linspace(float(day.open), float(day.close), len(times) + 1)[1:]
            # Only clamp the final value to day.close, allow intermediate bars to wander
            noise = self.rng.normal(0, max(float(day.high - day.low), 1e-9) * 0.08, len(times))
            noise_cumsum = noise.cumsum()
            noise_final = noise_cumsum[-1]
            # Drift correction to hit day.close
            closes = base + noise_cumsum - noise_final * np.linspace(0, 1, len(times))
            opens = np.r_[float(day.open), closes[:-1]]
            vols = np.maximum(1, (float(day.volume) * prof * self.rng.lognormal(0, 0.25, len(times))).astype(int))
            for t, o, c, v in zip(times, opens, closes, vols):
                spread = max(abs(c - o), float(day.high - day.low) / len(times) * self.rng.uniform(0.5, 1.5))
                h = min(float(day.high), max(o, c) + spread * self.rng.uniform(0.1, 0.6))
                l = max(float(day.low), min(o, c) - spread * self.rng.uniform(0.1, 0.6))
                rows.append((pd.Timestamp.combine(d, t), o, h, l, c, v))
        return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"]).set_index("datetime").sort_index()

    def _synthetic_intraday(self, symbol: str, freq: str) -> pd.DataFrame:
        """Generate six months of realistic synthetic intraday OHLCV."""
        times = trading_times(freq)
        prof = volume_profile(len(times))
        days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=140, freq="B")
        price = float(START_PRICES[symbol])
        sigma = 0.008 * math.sqrt((5 if freq == "5min" else 15) / 15)
        mu, kappa, anchor = 0.0003 * ((5 if freq == "5min" else 15) / 15), 0.02, price
        rows = []
        for d in days:
            day_base_vol = BASE_VOLUME[symbol] * self.rng.lognormal(0, 0.25)
            for j, t in enumerate(times):
                o = price
                ret = mu + kappa * (math.log(anchor) - math.log(max(price, 1e-9))) / len(times) + sigma * self.rng.normal()
                c = max(1.0, price * math.exp(ret))
                wiggle = abs(c - o) + max(o, c) * sigma * self.rng.uniform(0.2, 0.8)
                h = max(o, c) + wiggle * self.rng.uniform(0.1, 0.5)
                l = max(0.5, min(o, c) - wiggle * self.rng.uniform(0.1, 0.5))
                v = int(max(1, day_base_vol * prof[j] * self.rng.lognormal(0, 0.5)))
                rows.append((pd.Timestamp.combine(d.date(), t), o, h, l, c, v))
                price = c
        return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"]).set_index("datetime")

    def _simulate_option(self, symbol: str, spot: pd.DataFrame, direction: int) -> pd.DataFrame:
        """Simulate ATM option premiums from Sensex spot movement."""
        ret = spot["close"].pct_change().fillna(0)
        premium = max(100.0, spot["close"].iloc[0] * 0.006)
        rows = []
        for ts, sret in ret.items():
            srow = spot.loc[ts]
            atr_pct = max(float(srow.atr_14) / max(float(srow.close), 1e-9), 0.0005)
            o = premium
            noise = self.rng.normal(0, atr_pct * 3.0)
            move = direction * float(sret) * 8.0 + abs(float(sret)) * 1.5 + noise
            c = float(np.clip(premium * (1 + move), 10.0, float(srow.close) * 0.03))
            wiggle = max(abs(c - o), premium * atr_pct * 6)
            h = min(float(srow.close) * 0.04, max(o, c) + wiggle * self.rng.uniform(0.1, 0.8))
            l = max(5.0, min(o, c) - wiggle * self.rng.uniform(0.1, 0.8))
            vol = int(max(100, float(srow.volume) * 0.15 * self.rng.lognormal(0, 0.6)))
            rows.append((ts, o, h, l, c, vol))
            premium = c
        return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"]).set_index("datetime")

    def run_all(self) -> BacktestResults:
        """Run all configured strategies."""
        results = {}
        specs = {"ensemble_ai": EQUITY_SYMBOLS, "rsmb": EQUITY_SYMBOLS, "gamma_scalper": ["SENSEX_CE_5M", "SENSEX_PE_5M"], "mean_reversion": EQUITY_SYMBOLS}
        for name, symbols in specs.items():
            results[name] = self.run_strategy(name, symbols)
        for name, res in results.items():
            res.sensitivity = self.sensitivity(name, specs[name])
            res.flags = self.review_flags(res)
        return BacktestResults(results)

    def run_strategy(self, strategy_name: str, symbols: list[str], overrides: dict[str, Any] | None = None) -> StrategyResult:
        """Run a single strategy."""
        return EventDrivenSimulator(self, strategy_name, symbols, overrides).run()

    def rs_rank(self, symbol: str, ts: pd.Timestamp) -> float:
        """Estimate 20-day relative strength versus NIFTY (fallback RELIANCE)."""
        ref_sym = "NIFTY" if "NIFTY" in self.data else "RELIANCE"
        df, ref = self.data[symbol], self.data[ref_sym]
        if ts not in df.index or ts not in ref.index:
            return 1.0
        i, ri = df.index.get_loc(ts), ref.index.get_loc(ts)
        lookback = 25 * 20
        if i < lookback or ri < lookback:
            return 1.0
        stock_ret = df.close.iloc[i] / df.close.iloc[i - lookback] - 1
        ref_ret = ref.close.iloc[ri] / ref.close.iloc[ri - lookback] - 1
        return float((1 + stock_ret) / max(1 + ref_ret, 1e-9))

    def calculate_metrics(self, name: str, trades: list[dict[str, Any]], signals: int, loss_days: set[pd.Timestamp]) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
        """Calculate complete strategy metrics."""
        cap = float(self.config[name]["capital"])
        df = pd.DataFrame(trades)
        if df.empty:
            days = self._all_eval_days()
            daily = pd.Series(0.0, index=days)
            return self._zero_metrics(signals, len(loss_days)), pd.DataFrame(columns=["Month", "Trades", "Wins", "Losses", "Win Rate", "Net P&L", "Drawdown"]), daily
        df["dt"] = pd.to_datetime(df["date"] + " " + df["time"])
        df["day"] = pd.to_datetime(df["date"])
        daily = df.groupby("day")["net_pnl"].sum().reindex(self._all_eval_days(), fill_value=0.0)
        equity = daily.cumsum()
        dd = equity - equity.cummax()
        max_dd = float(dd.min())
        max_dd_pct = abs(max_dd) / cap * 100
        total = float(df.net_pnl.sum())
        days = max(1, len(daily))
        total_pct = total / cap * 100
        ann_ret = ((1 + total / cap) ** (252 / days) - 1) * 100 if total > -cap else -100.0
        daily_ret = daily / cap
        vol = float(daily_ret.std(ddof=0) * math.sqrt(252) * 100)
        sharpe = (ann_ret / 100 - self.config["risk_free_rate_annual"]) / (vol / 100) if vol > 0 else 0.0
        calmar = ann_ret / max(max_dd_pct, 1e-9)
        wins, losses = df[df.net_pnl > 0], df[df.net_pnl <= 0]
        gross_wins, gross_losses = wins.net_pnl.sum(), abs(losses.net_pnl.sum())
        wr = len(wins) / len(df) * 100
        monthly = self._monthly(df)
        metrics = {
            "total_net_pnl": total, "total_net_pnl_pct": total_pct, "annualised_return_pct": ann_ret,
            "max_drawdown_inr": max_dd, "max_drawdown_pct": max_dd_pct, "max_drawdown_duration_days": self._dd_duration(dd),
            "annualised_volatility_pct": vol, "sharpe": sharpe, "calmar": calmar,
            "total_signals": signals, "total_trades": len(df), "signal_acceptance_rate_pct": len(df) / max(signals, 1) * 100,
            "wins": len(wins), "losses": len(losses), "win_rate_pct": wr, "avg_win_inr": float(wins.net_pnl.mean() or 0),
            "avg_loss_inr": float(losses.net_pnl.mean() or 0), "profit_factor": float(gross_wins / gross_losses) if gross_losses > 0 else float("inf"),
            "expectancy_inr": float(df.net_pnl.mean()), "avg_r_multiple": float(df.r_multiple.mean()), "avg_holding_minutes": float(df.holding_minutes.mean()),
            "max_consecutive_wins": self._streak(df.net_pnl > 0), "max_consecutive_losses": self._streak(df.net_pnl <= 0),
            "largest_win_inr": float(df.net_pnl.max()), "largest_loss_inr": float(df.net_pnl.min()),
            "daily_loss_limit_days": len(loss_days), "win_rate_by_hour": self._win_rate(df, df.dt.dt.hour),
            "win_rate_by_day": self._win_rate(df, df.dt.dt.day_name()), "avg_pnl_per_symbol": df.groupby("symbol")["net_pnl"].mean().to_dict(),
        }
        return metrics, monthly, daily

    def _monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build monthly metrics table."""
        rows = []
        for month, g in df.groupby(df["dt"].dt.to_period("M")):
            daily = g.groupby("day")["net_pnl"].sum().cumsum()
            dd = daily - daily.cummax()
            wins = int((g.net_pnl > 0).sum())
            losses = int((g.net_pnl <= 0).sum())
            rows.append({"Month": str(month), "Trades": len(g), "Wins": wins, "Losses": losses, "Win Rate": wins / max(len(g), 1) * 100, "Net P&L": g.net_pnl.sum(), "Drawdown": dd.min() if len(dd) else 0})
        return pd.DataFrame(rows)

    def sensitivity(self, name: str, symbols: list[str]) -> pd.DataFrame:
        """Run one-parameter sensitivity sweep."""
        sweeps = {
            "ensemble_ai": ("min_confidence", [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.72]),
            "rsmb": ("min_rs_rank", [1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15]),
            "gamma_scalper": ("min_candle_strength_pct", [8, 9, 10, 11, 12, 13, 14]),
            "mean_reversion": ("min_distance_pct", [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        }
        key, values = sweeps[name]
        rows = []
        print(f"Running sensitivity sweep for {name}...")
        for value in values:
            res = self.run_strategy(name, symbols, {key: value})
            m = res.metrics
            rows.append({"Parameter Value": value, "Current": value == self.config[name][key], "Trades": m["total_trades"], "Win Rate": m["win_rate_pct"], "Profit Factor": m["profit_factor"], "Sharpe": m["sharpe"], "Unreliable": m["total_trades"] < 20})
        return pd.DataFrame(rows)

    def review_flags(self, res: StrategyResult) -> dict[str, str]:
        """Create quant review flags."""
        m, cap = res.metrics, float(self.config[res.name]["capital"])
        monthly = res.monthly
        sens = res.sensitivity if res.sensitivity is not None else pd.DataFrame()
        def flag(value: float, limit: float, higher: bool = True) -> str:
            if higher:
                return "PASS" if value >= limit else "WARN" if value >= limit * 0.9 else "FAIL"
            return "PASS" if value <= limit else "WARN" if value <= limit * 1.1 else "FAIL"
        total_profit = float(monthly["Net P&L"].sum()) if not monthly.empty else 0.0
        concentration = float(monthly["Net P&L"].max() / total_profit * 100) if total_profit > 0 and not monthly.empty else 100.0
        wr_std = float(monthly["Win Rate"].std(ddof=0)) if len(monthly) > 1 else 0.0
        flat = False
        if not sens.empty:
            peak = sens["Sharpe"].replace([np.inf, -np.inf], np.nan).max()
            flat = bool((sens["Sharpe"] >= peak - 0.1).sum() >= 3) if np.isfinite(peak) else False
        flags = {
            "Win rate > 45%": flag(m["win_rate_pct"], 45),
            "Profit factor > 1.2": flag(m["profit_factor"], 1.2),
            "Sharpe ratio > 0.8": flag(m["sharpe"], 0.8),
            "Max drawdown < 20% capital": flag(m["max_drawdown_pct"], 20, False),
            "Trades > 30": flag(m["total_trades"], 30),
            "No month > 40% profit": flag(concentration, 40, False),
            "Monthly win rate std < 15pp": flag(wr_std, 15, False),
            "Sensitivity has flat region": "PASS" if flat else "FAIL",
        }
        fails = sum(1 for v in flags.values() if v == "FAIL")
        flags["Overall verdict"] = "VIABLE" if fails == 0 else "NEEDS TUNING" if fails <= 2 else "NOT READY"
        return flags

    def report(self, results: BacktestResults) -> None:
        """Print console report and write all output files."""
        BacktestReporter(self).report(results)

    def _print_data_quality(self) -> None:
        """Print data coverage summary."""
        total_bars = sum(len(df) for df in self.data.values())
        symbols = len(self.data)
        starts = [df.index.min() for df in self.data.values() if not df.empty]
        ends = [df.index.max() for df in self.data.values() if not df.empty]
        eval_days = len(self._all_eval_days())
        print(f"Loaded {total_bars:,} bars for {symbols} symbols covering {min(starts).date()} to {max(ends).date()}")
        print(f"Warm-up uses first {self.config['warmup_bars']} bars per symbol. Backtest evaluates approximately {eval_days} trading days.")

    def _all_eval_days(self) -> pd.DatetimeIndex:
        """Return common evaluation day index (excluding warmup)."""
        df = self.data[EQUITY_SYMBOLS[0]] if self.data else pd.DataFrame(index=pd.date_range(end=pd.Timestamp.today(), periods=60, freq="B"))
        warmup = self.config["warmup_bars"]
        eval_start = df.index[warmup] if len(df) > warmup else df.index[0]
        dates = pd.DatetimeIndex(sorted(pd.unique(df[df.index >= eval_start].index.normalize())))
        return dates

    def _zero_metrics(self, signals: int, loss_days: int) -> dict[str, Any]:
        """Return zero metrics for no-trade strategies."""
        return {"total_net_pnl": 0.0, "total_net_pnl_pct": 0.0, "annualised_return_pct": 0.0, "max_drawdown_inr": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_duration_days": 0, "annualised_volatility_pct": 0.0, "sharpe": 0.0, "calmar": 0.0, "total_signals": signals, "total_trades": 0, "signal_acceptance_rate_pct": 0.0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "avg_win_inr": 0.0, "avg_loss_inr": 0.0, "profit_factor": 0.0, "expectancy_inr": 0.0, "avg_r_multiple": 0.0, "avg_holding_minutes": 0.0, "max_consecutive_wins": 0, "max_consecutive_losses": 0, "largest_win_inr": 0.0, "largest_loss_inr": 0.0, "daily_loss_limit_days": loss_days, "win_rate_by_hour": {}, "win_rate_by_day": {}, "avg_pnl_per_symbol": {}}

    def _win_rate(self, df: pd.DataFrame, groups: pd.Series) -> dict[Any, float]:
        """Calculate grouped win rates."""
        return {k: float((g.net_pnl > 0).mean() * 100) for k, g in df.groupby(groups)}

    def _streak(self, wins: pd.Series) -> int:
        """Calculate maximum consecutive true values."""
        best = cur = 0
        for v in wins:
            cur = cur + 1 if bool(v) else 0
            best = max(best, cur)
        return best

    def _dd_duration(self, dd: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        best = cur = 0
        for v in dd:
            cur = cur + 1 if v < 0 else 0
            best = max(best, cur)
        return best

def main() -> None:
    """Run the standalone backtest."""
    engine = BacktestEngine(CONFIG)
    engine.load_data()
    results = engine.run_all()
    engine.report(results)


if __name__ == "__main__":
    main()
