import pandas as pd
from typing import Dict, Any
from models.ensemble import AIEnsemble
from features.price_features import PriceFeatures
from features.volume_features import VolumeFeatures
from features.time_features import TimeFeatures
from loguru import logger
from tracking.signal_logger import SignalLogger

class EquitySignalEngine:
    """
    Core engine for Equity/F&O signals.
    Combines AI predictions with rule-based validation.
    """
    def __init__(self, config: dict, risk_engine=None):
        self.config = config
        self.ensemble = AIEnsemble(config)
        self.ensemble.load_models()
        self.signal_logger = SignalLogger()
        self.risk_engine = risk_engine  # Optional: used for position sizing

        equity_cfg = config.get("equity_signal", {})
        paper_cfg = config.get("paper_trading", {})
        self.paper_mode = bool(config.get("paper_mode", True))
        self.paper_relaxed = self.paper_mode and bool(paper_cfg.get("relaxed_signals", False))
        self.paper_min_conf = self._float_config(paper_cfg, "equity_min_confidence", 0.15)
        self.paper_technical_min_score = self._float_config(paper_cfg, "equity_technical_min_score", 3.0)
        self.paper_technical_enabled = self.paper_mode and bool(paper_cfg.get("technical_fallback_enabled", True))
        self.paper_cooldown_minutes = self._float_config(paper_cfg, "equity_signal_cooldown_minutes", 0)
        self.min_buy_conf = self._float_config(equity_cfg, "min_buy_confidence", 0.38)
        self.min_sell_conf = self._float_config(equity_cfg, "min_sell_confidence", 0.38)

        try:
            import json, os
            if os.path.exists("config/adaptive_params.json"):
                with open("config/adaptive_params.json", "r") as f:
                    adapt = json.load(f)
                    self.min_buy_conf = self._bounded_float(
                        adapt.get("equity_min_buy_confidence", self.min_buy_conf),
                        default=self.min_buy_conf,
                        low=0.20,
                        high=0.80,
                        name="adaptive equity_min_buy_confidence",
                    )
                    self.min_sell_conf = self._bounded_float(
                        adapt.get("equity_min_sell_confidence", self.min_sell_conf),
                        default=self.min_sell_conf,
                        low=0.20,
                        high=0.80,
                        name="adaptive equity_min_sell_confidence",
                    )
        except Exception as e:
            logger.warning(f"Failed to load adaptive params: {e}")

        self.min_rel_vol = self._float_config(equity_cfg, "min_relative_volume", 1.0)
        self.min_adx = self._float_config(equity_cfg, "min_adx", 12)
        self.min_ai_confidence_floor = self._float_config(equity_cfg, "min_ai_confidence_floor", 0.25)
        self.performance_guard_enabled = bool(equity_cfg.get("performance_guard_enabled", True))
        self.performance_guard_min_trades = int(equity_cfg.get("performance_guard_min_trades", 5))
        self.performance_guard_min_win_rate = self._float_config(equity_cfg, "performance_guard_min_win_rate", 0.35)
        self.performance_guard_min_profit_factor = self._float_config(equity_cfg, "performance_guard_min_profit_factor", 0.80)
        self.trade_journal_path = str(equity_cfg.get("trade_journal_path", "data/trade_journal.csv"))
        self._performance_cache = {"mtime": None, "by_symbol": {}}
        self.time_features = TimeFeatures()
        self._last_signal_bar = {}  # symbol -> timestamp
        self._last_paper_signal_time = {}  # symbol -> timestamp
        logger.info(
            "EquitySignalEngine thresholds loaded | "
            f"buy={self.min_buy_conf:.2f}, sell={self.min_sell_conf:.2f}, "
            f"rel_vol={self.min_rel_vol:.2f}, adx={self.min_adx:.1f}"
        )

    @staticmethod
    def _float_config(section: dict, key: str, default: float) -> float:
        try:
            return float(section.get(key, default))
        except (TypeError, ValueError):
            logger.warning(f"Invalid equity_signal.{key}; using default {default}")
            return default

    @staticmethod
    def _bounded_float(value, default: float, low: float, high: float, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid {name}; using default {default}")
            return default
        bounded = min(max(parsed, low), high)
        if bounded != parsed:
            logger.warning(f"{name}={parsed} outside [{low}, {high}]; clamped to {bounded}")
        return bounded

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            if pd.isna(parsed):
                return default
            return parsed
        except (TypeError, ValueError):
            return default

    def _paper_cooldown_reason(self, symbol: str, last_ts) -> str | None:
        cooldown = float(getattr(self, "paper_cooldown_minutes", 0) or 0)
        if cooldown <= 0 or last_ts is None:
            return None
        previous = self._last_paper_signal_time.get(symbol)
        if previous is None:
            return None
        try:
            elapsed = (pd.Timestamp(last_ts) - pd.Timestamp(previous)).total_seconds() / 60.0
        except Exception:
            return None
        if elapsed < cooldown:
            return f"cooldown active ({elapsed:.0f}/{cooldown:.0f} minutes)"
        return None

    def _mark_paper_signal(self, symbol: str, last_ts) -> None:
        if last_ts is not None:
            self._last_paper_signal_time[symbol] = last_ts

    def _paper_relaxed_quality_gate(self, df: pd.DataFrame, side: str) -> tuple[bool, str]:
        if df is None or df.empty:
            return False, "missing timing data"
        if len(df) < 20:
            return False, "indicator warm-up incomplete"

        last_row = df.iloc[-1]
        try:
            flags = self.time_features.add_session_flags(df.tail(1))
            if bool(flags["is_noise_window"].iloc[0]):
                return False, "noise-window blocked"
        except Exception:
            pass

        close = self._safe_float(last_row.get("close"), 0.0)
        ema_9 = self._safe_float(last_row.get("ema_9"), close)
        ema_21 = self._safe_float(last_row.get("ema_21"), close)
        vwap = self._safe_float(last_row.get("vwap"), close)
        adx = self._safe_float(last_row.get("ADX_14"), float("nan"))
        rel_vol = self._safe_float(last_row.get("rel_vol"), 0.0)

        score = 0
        reasons = []
        if side == "BUY" and ema_9 > ema_21:
            score += 1
            reasons.append("EMA")
        elif side == "SELL" and ema_9 < ema_21:
            score += 1
            reasons.append("EMA")

        if side == "BUY" and close > vwap:
            score += 1
            reasons.append("VWAP")
        elif side == "SELL" and close < vwap:
            score += 1
            reasons.append("VWAP")

        if pd.notna(adx) and adx >= self.min_adx:
            score += 1
            reasons.append(f"ADX={adx:.1f}")
        if rel_vol >= self.min_rel_vol:
            score += 1
            reasons.append(f"VOL={rel_vol:.1f}")

        if score < 4:
            return False, f"paper quality gate {score}/4 [{', '.join(reasons) or 'no alignment'}]"
        return True, f"paper quality gate {score}/4 [{', '.join(reasons)}]"

    async def process_symbol(self, symbol: str, df_15m: pd.DataFrame, df_5m: pd.DataFrame = None):
        """
        Dual-Timeframe Engine:
        - Brain: 15m candles (indicators, AI inference).
        - Feet: 5m candles (entry timing and VWAP validation).
        """
        if df_15m is None or df_15m.empty:
            return None

        # 0. Deduplication check (Hoist to prevent double logging)
        last_ts = df_15m.index[-1]
        if last_ts is not None and self._last_signal_bar.get(symbol) == last_ts:
            return None 

        # 1. Brain: 15m Indicator Calculation
        df_15m = PriceFeatures.add_indicators(df_15m)
        df_15m = VolumeFeatures.add_volume_analysis(df_15m)
        
        if len(df_15m) < 50:
            if getattr(self, "paper_technical_enabled", False):
                if df_5m is not None and not df_5m.empty:
                    timing_df = VolumeFeatures.add_volume_analysis(PriceFeatures.add_indicators(df_5m))
                else:
                    timing_df = df_15m
                signal = self._emit_paper_technical_signal(
                    symbol=symbol,
                    last_ts=last_ts,
                    timing_df=timing_df,
                    reason="insufficient AI warm-up bars",
                )
                if signal:
                    return signal
            return None

        # 2. Prepare Features for AI (Align with Training)
        feature_cols = [
            'dist_ema_9', 'dist_ema_21', 'dist_ema_50', 'rsi_14', 'atr_pct',
            'dist_vwap', 'ADX_14', 'DMP_14', 'DMN_14', 'bb_pct'
        ]
        
        missing_cols = [col for col in feature_cols if col not in df_15m.columns]
        if missing_cols:
            logger.warning(f"NO TRADE {symbol}: missing feature columns {missing_cols}")
            return None

        current_df = df_15m.tail(1)[feature_cols]
        if current_df.isna().any(axis=None):
            return None
        current_features = current_df
        
        sequence_features = df_15m.tail(30)[feature_cols].values 
        returns = df_15m['close'].pct_change().tail(50).values
        
        # Get AI Score from Ensemble (Trained on 15m)
        ai_score = self.ensemble.get_combined_score(current_features, sequence_features, returns)
        
        # 3. Feet: Validate Signal using 5m data (Timing Layer)
        side = "BUY" if ai_score > 0 else "SELL"
        
        # If no 5m data provided, fallback to 15m for validation
        if df_5m is not None and not df_5m.empty:
            timing_df = VolumeFeatures.add_volume_analysis(PriceFeatures.add_indicators(df_5m))
        else:
            timing_df = df_15m   # indicators already computed in Brain layer
        
        validation = self.validate_setup(symbol, timing_df, abs(ai_score), side)
        
        # 4. LOG TO CSV (For Dashboard)
        status = "TRADE" if validation['valid'] else "NO_TRADE"
        if validation['valid']:
            guard_ok, guard_reason = self._performance_guard_allows(symbol, "Ensemble_AI")
            if not guard_ok:
                validation = {"valid": False, "reason": guard_reason}

        if validation['valid']:
            self._last_signal_bar[symbol] = last_ts
            reason = f"AI: {abs(ai_score):.2f} | 5m-ADX: {validation['metrics']['adx']:.1f} | 15m-Brain Valid"
        else:
            reason = validation['reason']
        
        entry = float(timing_df.iloc[-1]['close'])
        sl = validation.get('sl', entry * 0.99)
        target = validation.get('target', entry * 1.02)
        
        self.signal_logger.log_signal(
            symbol=symbol,
            side=side,
            strategy="Ensemble_AI",
            entry=entry,
            sl=sl,
            target=target,
            score=abs(ai_score),
            status=status,
            reason=reason
        )

        if validation['valid']:
            logger.info(f"🚀 ENSEMBLE SIGNAL: {side} {symbol} | Brain(15m): {abs(ai_score):.2f} | Feet(5m) Valid")

            qty = 1 
            if self.risk_engine is not None:
                sized_qty = self.risk_engine.get_equity_position_size(entry, sl)
                if sized_qty > 0:
                    qty = sized_qty

            return {
                "symbol": symbol,
                "side": side,
                "strategy": "Ensemble_AI",
                "qty": qty,
                "entry": entry,
                "sl": sl,
                "target": target,
                "confidence": abs(ai_score),
            }
            
        if getattr(self, "paper_relaxed", False) and abs(ai_score) >= getattr(self, "paper_min_conf", 0.15):
            cooldown_reason = self._paper_cooldown_reason(symbol, last_ts)
            gate_ok, gate_reason = self._paper_relaxed_quality_gate(timing_df, side)
            if cooldown_reason:
                logger.info(f"PAPER_RELAXED skipped {symbol}: {cooldown_reason}")
            elif not gate_ok:
                logger.info(f"PAPER_RELAXED skipped {symbol}: {gate_reason}")
            else:
                guard_ok, guard_reason = self._performance_guard_allows(symbol, "Ensemble_AI_PAPER_RELAXED")
                if not guard_ok:
                    logger.info(f"PAPER_RELAXED skipped {symbol}: {guard_reason}")
                    return None
                self._last_signal_bar[symbol] = last_ts
                self._mark_paper_signal(symbol, last_ts)
                relaxed_signal = self._build_paper_relaxed_signal(symbol, timing_df, abs(ai_score), side)
                self.signal_logger.log_signal(
                    symbol=symbol,
                    side=side,
                    strategy=relaxed_signal["strategy"],
                    entry=relaxed_signal["entry"],
                    sl=relaxed_signal["sl"],
                    target=relaxed_signal["target"],
                    score=abs(ai_score),
                    status="TRADE",
                    reason=f"PAPER_RELAXED: {validation['reason']} | {gate_reason}",
                )
                logger.warning(
                    f"PAPER_RELAXED equity signal: {side} {symbol} | "
                    f"AI={abs(ai_score):.2f} | {gate_reason} | bypassed={validation['reason']}"
                )
                return relaxed_signal

        if getattr(self, "paper_technical_enabled", False):
            technical_signal = self._emit_paper_technical_signal(
                symbol=symbol,
                last_ts=last_ts,
                timing_df=timing_df,
                reason=f"AI blocked: {validation['reason']}",
                ai_score=abs(ai_score),
            )
            if technical_signal:
                return technical_signal

        return None

    def _build_paper_relaxed_signal(self, symbol: str, df: pd.DataFrame, confidence: float, side: str) -> dict:
        """Build a paper-only signal when strict timing filters block all forward-test trades."""
        last_row = df.iloc[-1]
        entry = float(last_row["close"])
        atr = float(last_row.get("atr_14", entry * 0.01) or entry * 0.01)
        risk = max(atr * self.config.get("risk", {}).get("atr_sl_multiplier", 1.5), entry * 0.002)
        rr = float(self.config.get("risk", {}).get("rr_ratio", 1.5))
        sl = entry - risk if side == "BUY" else entry + risk
        target = entry + (risk * rr) if side == "BUY" else entry - (risk * rr)

        qty = 1
        if self.risk_engine is not None:
            sized_qty = self.risk_engine.get_equity_position_size(entry, sl)
            if sized_qty > 0:
                qty = sized_qty

        return {
            "symbol": symbol,
            "side": side,
            "strategy": "Ensemble_AI_PAPER_RELAXED",
            "qty": qty,
            "entry": entry,
            "sl": sl,
            "target": target,
            "confidence": confidence,
        }

    def _build_paper_technical_signal(self, symbol: str, df: pd.DataFrame) -> dict | None:
        """Paper-only fallback for forward testing when the AI ensemble is neutral."""
        if symbol.endswith(("CE", "PE")):
            return None

        if df.empty or len(df) < 20:
            return None

        last_row = df.iloc[-1]
        close = float(last_row.get("close", 0) or 0)
        if close <= 0:
            return None

        close_series = pd.to_numeric(df["close"], errors="coerce").ffill()
        high_series = pd.to_numeric(df.get("high", close_series), errors="coerce").ffill()
        low_series = pd.to_numeric(df.get("low", close_series), errors="coerce").ffill()
        volume_series = pd.to_numeric(df.get("volume", pd.Series(1, index=df.index)), errors="coerce").fillna(0)

        ema_9 = float(last_row.get("ema_9") or close_series.ewm(span=9, min_periods=1).mean().iloc[-1])
        ema_21 = float(last_row.get("ema_21") or close_series.ewm(span=21, min_periods=1).mean().iloc[-1])

        if last_row.get("vwap") is not None:
            vwap = float(last_row.get("vwap") or close)
        else:
            typical_price = (high_series + low_series + close_series) / 3
            volume_sum = volume_series.cumsum().replace(0, float("nan"))
            vwap_series = (typical_price * volume_series).cumsum() / volume_sum
            vwap = float(vwap_series.ffill().fillna(close_series.expanding().mean()).iloc[-1])

        if last_row.get("rsi_14") is not None:
            rsi = float(last_row.get("rsi_14") or 50)
        else:
            delta = close_series.diff().fillna(0)
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi = float((100 - (100 / (1 + rs))).ffill().fillna(50).iloc[-1])

        adx = self._safe_float(last_row.get("ADX_14"), float("nan"))
        if last_row.get("rel_vol") is not None:
            rel_vol = self._safe_float(last_row.get("rel_vol"), 0.0)
        else:
            avg_volume = volume_series.rolling(20, min_periods=1).mean().replace(0, float("nan"))
            rel_vol = float((volume_series / avg_volume).ffill().fillna(1.0).iloc[-1] or 1.0)

        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        if ema_9 > ema_21:
            buy_score += 1.0
            reasons.append("EMA_BUY")
        elif ema_9 < ema_21:
            sell_score += 1.0
            reasons.append("EMA_SELL")

        if close > vwap:
            buy_score += 1.0
            reasons.append("VWAP_BUY")
        elif close < vwap:
            sell_score += 1.0
            reasons.append("VWAP_SELL")

        if 52 <= rsi <= 72:
            buy_score += 1.0
            reasons.append(f"RSI_BUY={rsi:.1f}")
        elif 28 <= rsi <= 48:
            sell_score += 1.0
            reasons.append(f"RSI_SELL={rsi:.1f}")

        momentum = float(close_series.pct_change(3).iloc[-1] or 0)
        if momentum > 0.0005:
            buy_score += 1.0
            reasons.append(f"MOM_BUY={momentum:.3%}")
        elif momentum < -0.0005:
            sell_score += 1.0
            reasons.append(f"MOM_SELL={momentum:.3%}")

        if pd.notna(adx) and adx >= self.min_adx:
            buy_score += 0.5
            sell_score += 0.5
            reasons.append(f"ADX={adx:.1f}")

        if rel_vol >= self.min_rel_vol:
            buy_score += 0.5
            sell_score += 0.5
            reasons.append(f"VOL={rel_vol:.1f}")

        side = "BUY" if buy_score >= sell_score else "SELL"
        score = max(buy_score, sell_score)
        if score < getattr(self, "paper_technical_min_score", 3.0):
            return None

        return self._build_paper_relaxed_signal(
            symbol=symbol,
            df=df,
            confidence=score,
            side=side,
        ) | {
            "strategy": "Ensemble_AI_PAPER_TECHNICAL",
            "reason": f"PAPER_TECHNICAL: score={score:.1f} [{' '.join(reasons)}]",
        }

    def _emit_paper_technical_signal(
        self,
        symbol: str,
        last_ts,
        timing_df: pd.DataFrame,
        reason: str,
        ai_score: float = 0.0,
    ) -> dict | None:
        cooldown_reason = self._paper_cooldown_reason(symbol, last_ts)
        if cooldown_reason:
            logger.info(f"PAPER_TECHNICAL skipped {symbol}: {cooldown_reason}")
            return None

        technical_signal = self._build_paper_technical_signal(symbol, timing_df)
        if not technical_signal:
            return None

        guard_ok, guard_reason = self._performance_guard_allows(symbol, technical_signal["strategy"])
        if not guard_ok:
            logger.info(f"PAPER_TECHNICAL skipped {symbol}: {guard_reason}")
            return None

        self._last_signal_bar[symbol] = last_ts
        self._mark_paper_signal(symbol, last_ts)
        log_reason = f"{technical_signal['reason']} | {reason}"
        self.signal_logger.log_signal(
            symbol=symbol,
            side=technical_signal["side"],
            strategy=technical_signal["strategy"],
            entry=technical_signal["entry"],
            sl=technical_signal["sl"],
            target=technical_signal["target"],
            score=technical_signal["confidence"],
            status="TRADE",
            reason=log_reason,
        )
        logger.warning(
            f"PAPER_TECHNICAL equity signal: {technical_signal['side']} {symbol} | "
            f"score={technical_signal['confidence']:.2f} | AI={ai_score:.2f} | {reason}"
        )
        technical_signal["reason"] = log_reason
        return technical_signal

    def _performance_guard_allows(self, symbol: str, strategy: str) -> tuple[bool, str]:
        if not getattr(self, "performance_guard_enabled", True):
            return True, "performance guard disabled"

        stats = self._performance_stats_by_symbol().get(symbol)
        if not stats or stats["trades"] < self.performance_guard_min_trades:
            return True, "insufficient performance sample"

        weak_win_rate = stats["win_rate"] < self.performance_guard_min_win_rate
        weak_profit_factor = stats["profit_factor"] < self.performance_guard_min_profit_factor
        losing_net = stats["net_pnl"] < 0
        if losing_net and (weak_win_rate or weak_profit_factor):
            return False, (
                f"Performance guard blocked {strategy} for {symbol}: "
                f"{stats['trades']} trades, win_rate={stats['win_rate']:.1%}, "
                f"PF={stats['profit_factor']:.2f}, net={stats['net_pnl']:.2f}"
            )
        return True, "performance acceptable"

    def _performance_stats_by_symbol(self) -> dict:
        import os
        path = self.trade_journal_path
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self._performance_cache = {"mtime": None, "by_symbol": {}}
            return {}

        if self._performance_cache.get("mtime") == mtime:
            return self._performance_cache.get("by_symbol", {})

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"Performance guard could not read {path}: {exc}")
            self._performance_cache = {"mtime": mtime, "by_symbol": {}}
            return {}

        if df.empty or "symbol" not in df.columns:
            self._performance_cache = {"mtime": mtime, "by_symbol": {}}
            return {}

        pnl_col = "pnl_after_costs" if "pnl_after_costs" in df.columns else "pnl_inr"
        if pnl_col not in df.columns:
            self._performance_cache = {"mtime": mtime, "by_symbol": {}}
            return {}

        df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
        by_symbol = {}
        for sym, group in df.groupby("symbol"):
            pnl = group[pnl_col]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            gross_profit = float(wins.sum())
            gross_loss = abs(float(losses.sum()))
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float("inf") if gross_profit > 0 else 0.0
            by_symbol[str(sym)] = {
                "trades": int(len(group)),
                "win_rate": float(len(wins) / len(group)) if len(group) else 0.0,
                "profit_factor": float(profit_factor),
                "net_pnl": float(pnl.sum()),
            }

        self._performance_cache = {"mtime": mtime, "by_symbol": by_symbol}
        return by_symbol

    def validate_setup(self, symbol: str, df: pd.DataFrame, ai_score: float, side: str) -> Dict[str, Any]:
        """
        Validates the entry timing using a scoring system.
        """
        if df.empty or len(df) < 20:
            return {"valid": False, "reason": "Insufficient timing data"}

        last_row = df.iloc[-1]

        target_conf = self.min_buy_conf if side == "BUY" else self.min_sell_conf

        # Scoring system (0-100+)
        score = ai_score * 100
        min_ai_base = self.min_ai_confidence_floor
        if ai_score < min_ai_base:
            return {"valid": False, "reason": f"AI confidence too low ({ai_score:.2f} < {min_ai_base})"}

        df_flags = self.time_features.add_session_flags(df.tail(1))
        reason_parts = [f"AI={ai_score:.2f}"]

        # 1. Session Penalties
        if df_flags['is_noise_window'].iloc[0]:
            score -= 20
            reason_parts.append("-20(noise)")
        elif df_flags['is_chop_zone'].iloc[0]:
            score -= 10
            reason_parts.append("-10(chop)")

        close = self._safe_float(last_row.get('close'), 0.0)
        ema_9 = self._safe_float(last_row.get('ema_9'), close)
        ema_21 = self._safe_float(last_row.get('ema_21'), close)
        vwap = self._safe_float(last_row.get('vwap'), close)
        rsi = self._safe_float(last_row.get('rsi_14'), 50.0)

        # 2. Timing: EMA Alignment
        ema_aligned = False
        if side == "BUY":
            ema_aligned = ema_9 > ema_21
        else:
            ema_aligned = ema_9 < ema_21
            
        if ema_aligned:
            score += 15
            reason_parts.append("+15(EMA)")
        else:
            score -= 10
            reason_parts.append("-10(EMA)")

        # 3. Timing: VWAP Position
        vwap_aligned = False
        if side == "BUY" and close > vwap:
            vwap_aligned = True
        elif side == "SELL" and close < vwap:
            vwap_aligned = True

        if vwap_aligned:
            score += 15
            reason_parts.append("+15(VWAP)")
        else:
            score -= 10
            reason_parts.append("-10(VWAP)")

        # 4. Trend Strength
        import math
        adx = self._safe_float(last_row.get('ADX_14'), float('nan'))
        if not math.isnan(adx) and adx >= self.min_adx:
            score += 10
            reason_parts.append(f"+10(ADX={adx:.1f})")
        else:
            score -= 5
            reason_parts.append("-5(ADX)")

        # 5. Volume
        rel_vol = self._safe_float(last_row.get("rel_vol"), 0.0)
        if rel_vol >= self.min_rel_vol:
            score += 10
            reason_parts.append(f"+10(Vol={rel_vol:.1f})")
        else:
            score -= 10
            reason_parts.append(f"-10(Vol={rel_vol:.1f})")

        threshold_score = target_conf * 100
        is_valid = score >= threshold_score
        reason = f"Score: {score:.1f}/{threshold_score:.1f} [{' '.join(reason_parts)}]"

        if not is_valid:
            return {"valid": False, "reason": reason}

        entry = close
        atr = self._safe_float(last_row.get('atr_14'), entry * 0.01)
        risk = max(atr * self.config.get("risk", {}).get("atr_sl_multiplier", 1.5), entry * 0.002)
        rr = self.config.get("risk", {}).get("rr_ratio", 1.5)
        sl = entry - risk if side == "BUY" else entry + risk
        target = entry + (risk * rr) if side == "BUY" else entry - (risk * rr)

        return {
            "valid": True,
            "side": side,
            "entry": entry,
            "sl": sl,
            "target": target,
            "confidence": ai_score,
            "metrics": {
                "adx": adx,
                "rsi": rsi,
                "rel_vol": rel_vol,
            }
        }

    async def run(self):
        """Main loop for signal generation (placeholder for main.py integration)."""
        logger.info("Equity Signal Engine active.")
        # This would be called by the scheduler on candle close
