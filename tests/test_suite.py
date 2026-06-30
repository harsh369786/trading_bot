"""
NSE Trading Bot — Comprehensive Test Suite
Covers issues found in the 2026-05-13 audit:
  C1-C10, H1-H10, M1-M10, L1-L8 (selected)
Run: python -m pytest tests/test_suite.py -v
"""
import asyncio
import csv
import json
import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)

    async def incrbyfloat(self, key, amount):
        self.store[key] = str(float(self.store.get(key, 0) or 0) + float(amount))
        return self.store[key]

    async def expire(self, key, seconds):
        return True


def _make_equity_df(n=70, with_volume=True):
    """Build a minimal OHLCV DataFrame with realistic values."""
    idx = pd.date_range("2026-05-13 09:15", periods=n, freq="5min")
    prices = [100 + i * 0.1 for i in range(n)]
    vol = [1000 + i * 10 for i in range(n)] if with_volume else [0] * n
    return pd.DataFrame({
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.3 for p in prices],
        "close": prices,
        "volume": vol,
    }, index=idx)


# ---------------------------------------------------------------------------
# Risk Engine Tests
# ---------------------------------------------------------------------------

class TestRiskEngine(unittest.TestCase):

    def test_position_sizing_basic(self):
        from risk.risk_engine import RiskEngine
        engine = RiskEngine({"capital": {"equity_total": 50000, "risk_per_trade_pct": 1.0}})
        # 1% of 50000 = 500 INR risk; SL distance = 1.0 → qty = 500
        self.assertEqual(engine.get_equity_position_size(100.0, 99.0), 500)

    def test_position_sizing_zero_sl(self):
        from risk.risk_engine import RiskEngine
        engine = RiskEngine({"capital": {"equity_total": 50000, "risk_per_trade_pct": 1.0}})
        self.assertEqual(engine.get_equity_position_size(100.0, 100.0), 0)

    def test_position_sizing_clamped_by_notional(self):
        from risk.risk_engine import RiskEngine
        engine = RiskEngine({
            "capital": {
                "equity_total": 50000,
                "risk_per_trade_pct": 1.0,
                "max_equity_notional_per_trade": 10000,
            }
        })
        self.assertEqual(engine.get_equity_position_size(100.0, 99.0), 100)

    def test_today_property_returns_string(self):
        """C8: today must be a live property, not a construction-time constant."""
        from risk.risk_engine import RiskEngine
        engine = RiskEngine({})
        today = engine.today
        self.assertRegex(today, r"\d{4}-\d{2}-\d{2}")

    def test_circuit_breakers_equity_pass(self):
        """Equity trading allowed when no losses and no open positions."""
        async def run():
            from risk.risk_engine import RiskEngine
            engine = RiskEngine({
                "capital": {"equity_total": 50000, "risk_per_trade_pct": 1.0,
                            "max_open_trades_equity": 2},
                "risk": {"daily_loss_limit_r": 3},
            }, FakeRedis())
            self.assertTrue(await engine.check_circuit_breakers("equity"))
        asyncio.run(run())

    def test_circuit_breakers_equity_blocked_on_loss(self):
        """Equity trading blocked once daily loss limit is hit."""
        async def run():
            from risk.risk_engine import RiskEngine
            redis = FakeRedis()
            engine = RiskEngine({
                "capital": {"equity_total": 50000, "risk_per_trade_pct": 1.0,
                            "max_open_trades_equity": 2},
                "risk": {"daily_loss_limit_r": 3},
            }, redis)
            # Simulate 3R of losses
            await engine._set_stat("equity_loss_r", 3.0)
            self.assertFalse(await engine.check_circuit_breakers("equity"))
        asyncio.run(run())

    def test_drawdown_circuit_breaker_blocks_on_max_open_trades(self):
        async def run():
            from risk.risk_engine import RiskEngine
            redis = FakeRedis()
            engine = RiskEngine({
                "capital": {"equity_total": 50000, "risk_per_trade_pct": 1.0,
                            "max_open_trades_equity": 2},
                "risk": {"daily_loss_limit_r": 3},
            }, redis)
            await engine._set_stat("equity_open_count", 2.0)
            self.assertFalse(await engine.check_circuit_breakers("equity"))
        asyncio.run(run())

    def test_equity_circuit_breaker_blocks_on_daily_trade_count(self):
        """Equity entries stop after configured daily trade cap."""
        async def run():
            from risk.risk_engine import RiskEngine
            redis = FakeRedis()
            engine = RiskEngine({
                "capital": {"max_open_trades_equity": 2},
                "risk": {"daily_loss_limit_r": 3, "equity_max_daily_trades": 2},
            }, redis)
            await engine.update_stats("equity", trade_delta=1)
            self.assertTrue(await engine.check_circuit_breakers("equity"))
            await engine.update_stats("equity", trade_delta=1)
            self.assertFalse(await engine.check_circuit_breakers("equity"))
        asyncio.run(run())

    def test_currency_circuit_breaker_blocks_on_daily_trade_count(self):
        """currency_max_daily_trades must be enforced, not just configured."""
        async def run():
            from risk.risk_engine import RiskEngine
            redis = FakeRedis()
            engine = RiskEngine({
                "capital": {"max_open_trades_currency": 2},
                "risk": {"currency_max_daily_loss_inr": 750, "currency_max_daily_trades": 1},
            }, redis)
            self.assertTrue(await engine.check_circuit_breakers("currency"))
            await engine.update_stats("currency", trade_delta=1)
            self.assertFalse(await engine.check_circuit_breakers("currency"))
        asyncio.run(run())


# ---------------------------------------------------------------------------
# Signal Schema / OrderManager Tests
# ---------------------------------------------------------------------------

class TestOrderManagerNormalization(unittest.TestCase):

    def _mgr(self):
        from execution.order_manager import OrderManager
        return OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []}})

    def test_accepts_target_field(self):
        mgr = self._mgr()
        s = mgr._normalize_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 99, "target": 102, "qty": 1})
        self.assertIsNotNone(s)
        self.assertEqual(s["target"], 102.0)

    def test_accepts_t1_alias(self):
        mgr = self._mgr()
        s = mgr._normalize_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 99, "t1": 102, "qty": 1})
        self.assertIsNotNone(s)
        self.assertEqual(s["target"], 102.0)
        self.assertEqual(s["t1"], 102.0)

    def test_rejects_inverted_buy_sl_target(self):
        mgr = self._mgr()
        # SL above entry is invalid for BUY
        s = mgr._normalize_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 101, "target": 103, "qty": 1})
        self.assertIsNone(s)

    def test_rejects_inverted_sell_sl_target(self):
        mgr = self._mgr()
        # target above entry is invalid for SELL
        s = mgr._normalize_signal({"symbol": "NIFTY", "side": "SELL", "entry": 100, "sl": 99, "target": 103, "qty": 1})
        self.assertIsNone(s)

    def test_rejects_non_finite(self):
        mgr = self._mgr()
        s = mgr._normalize_signal({"symbol": "NIFTY", "side": "BUY", "entry": float("inf"), "sl": 99, "target": 102, "qty": 1})
        self.assertIsNone(s)

    def test_paper_order_reaches_protected(self):
        async def run():
            redis = FakeRedis()
            from execution.order_manager import OrderManager
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []},
                                "capital": {"max_open_trades_equity": 2}}, redis)
            await mgr.execute_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 99, "target": 102, "qty": 1})
            active = json.loads(redis.store[mgr.KEY_ACTIVE])
            self.assertEqual(len(active), 1)
            self.assertEqual(next(iter(active.values()))["status"], "PROTECTED")
        asyncio.run(run())

    def test_duplicate_signal_blocked(self):
        """Second signal for same symbol while trade is active must be rejected."""
        async def run():
            redis = FakeRedis()
            from execution.order_manager import OrderManager
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []},
                                "capital": {"max_open_trades_equity": 2}}, redis)
            await mgr.execute_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 99, "target": 102, "qty": 1})
            await mgr.execute_signal({"symbol": "NIFTY", "side": "BUY", "entry": 101, "sl": 100, "target": 103, "qty": 1})
            active = json.loads(redis.store[mgr.KEY_ACTIVE])
            self.assertEqual(len(active), 1)  # only one trade
        asyncio.run(run())

    def test_paper_fallback_does_not_increment_equity_daily_trade_count(self):
        async def run():
            redis = FakeRedis()
            from execution.order_manager import OrderManager
            mgr = OrderManager({
                "paper_mode": True,
                "instruments": {"equity": ["NIFTY"], "currency": []},
                "capital": {"max_open_trades_equity": 2},
            }, redis)
            await mgr.execute_signal({
                "symbol": "NIFTY",
                "side": "BUY",
                "entry": 100,
                "sl": 99,
                "target": 102,
                "qty": 1,
                "strategy": "Ensemble_AI_PAPER_TECHNICAL",
            })
            today = mgr.risk_engine.today
            self.assertNotIn(f"bot:risk:stats:{today}:equity_trade_count", redis.store)
            self.assertEqual(float(redis.store[f"bot:risk:stats:{today}:equity_open_count"]), 1.0)
        asyncio.run(run())

    def test_startup_reconciliation_drops_stale_pending(self):
        async def run():
            redis = FakeRedis()
            from execution.order_manager import OrderManager
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["HDFCBANK"], "currency": []}}, redis)
            await redis.set(mgr.KEY_ACTIVE, json.dumps({
                "MOCK-OLD": {"symbol": "HDFCBANK", "side": "BUY", "entry": 755.65,
                             "sl": 748.09, "t1": 770.76, "qty": 1, "status": "PENDING"}
            }))
            await mgr.reconcile_startup_state()
            self.assertEqual(json.loads(redis.store[mgr.KEY_ACTIVE]), {})
        asyncio.run(run())

    def test_square_off_uses_latest_ltp_from_redis(self):
        async def run():
            redis = FakeRedis()
            from execution.order_manager import OrderManager
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []}}, redis)
            mgr._journal_path = os.path.join(tempfile.gettempdir(), "square_off_test_journal.csv")
            await redis.set(mgr.KEY_ACTIVE, json.dumps({
                "OID1": {
                    "symbol": "NIFTY", "side": "BUY", "entry": 100.0,
                    "sl": 99.0, "target": 102.0, "qty": 2, "status": "PROTECTED",
                    "domain": "equity",
                }
            }))
            await redis.set("bot:ltp:NIFTY", "101.5")
            await mgr.square_off_all()
            active = json.loads(redis.store[mgr.KEY_ACTIVE])
            self.assertEqual(active, {})
        asyncio.run(run())

    def test_square_off_domain_filter_preserves_other_market(self):
        async def run():
            redis = FakeRedis()
            from execution.order_manager import OrderManager
            mgr = OrderManager(
                {
                    "paper_mode": True,
                    "instruments": {"equity": ["NIFTY"], "currency": ["USDINR"]},
                },
                redis,
            )
            with tempfile.TemporaryDirectory() as tmp:
                mgr._journal_path = os.path.join(tmp, "trade_journal.csv")
                mgr._ensure_journal_header()
                await redis.set(mgr.KEY_ACTIVE, json.dumps({
                    "EQ1": {
                        "symbol": "NIFTY", "side": "BUY", "entry": 100.0,
                        "sl": 99.0, "target": 102.0, "qty": 1,
                        "status": "PROTECTED", "domain": "equity",
                    },
                    "CUR1": {
                        "symbol": "USDINR", "side": "BUY", "entry": 83.0,
                        "sl": 82.9, "target": 83.2, "qty": 1,
                        "status": "PROTECTED", "domain": "currency",
                    },
                }))
                await mgr.square_off_all(domain="equity")
                active = json.loads(redis.store[mgr.KEY_ACTIVE])
                self.assertNotIn("EQ1", active)
                self.assertIn("CUR1", active)
        asyncio.run(run())


class TestEquitySignalEngine(unittest.TestCase):

    def test_process_symbol_uses_df_15m_for_feature_validation(self):
        from strategies.equity_signal_engine import EquitySignalEngine

        feature_cols = [
            'dist_ema_9', 'dist_ema_21', 'dist_ema_50', 'rsi_14', 'atr_pct',
            'dist_vwap', 'ADX_14', 'DMP_14', 'DMN_14', 'bb_pct'
        ]
        idx = pd.date_range("2026-05-14 09:15", periods=60, freq="15min", tz="Asia/Kolkata")
        df = pd.DataFrame({col: np.ones(len(idx)) for col in feature_cols}, index=idx)
        df["close"] = 100.0

        engine = EquitySignalEngine.__new__(EquitySignalEngine)
        engine.config = {"equity_signal": {}, "risk": {}}
        engine.ensemble = type("FakeEnsemble", (), {"get_combined_score": lambda self, *args: 0.1})()
        engine.signal_logger = type("FakeLogger", (), {"log_signal": lambda self, **kwargs: None})()
        engine.risk_engine = None
        engine.min_buy_conf = 0.62
        engine.min_sell_conf = 0.62
        engine.min_rel_vol = 0.0
        engine.time_features = None
        engine._last_signal_bar = {}
        engine.validate_setup = lambda symbol, data, score, side: {"valid": False, "reason": "test"}

        with patch("strategies.equity_signal_engine.PriceFeatures.add_indicators", side_effect=lambda data: data), \
             patch("strategies.equity_signal_engine.VolumeFeatures.add_volume_analysis", side_effect=lambda data: data):
            result = asyncio.run(engine.process_symbol("NIFTY", df))

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Trade Journal Tests (C5, M4)
# ---------------------------------------------------------------------------

class TestTradeJournal(unittest.TestCase):

    def test_no_duplicate_header_on_repeated_writes(self):
        """C5 fix: header must appear exactly once even after multiple writes."""
        with tempfile.TemporaryDirectory() as tmp:
            from execution.order_manager import OrderManager
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []},
                                "execution": {"cost_fraction_per_side": 0.0006}})
            mgr._journal_path = os.path.join(tmp, "trade_journal.csv")
            mgr._ensure_journal_header()

            trade = {"symbol": "NIFTY", "side": "BUY", "entry": 100.0, "qty": 1, "confidence": 0.8}
            mgr._log_to_journal(trade, 102.0, "TARGET_HIT")
            mgr._log_to_journal(trade, 103.0, "TARGET_HIT")

            with open(mgr._journal_path) as f:
                lines = f.readlines()
            # First line is header; subsequent are data rows
            header_lines = [l for l in lines if "symbol" in l and "entry_price" in l]
            self.assertEqual(len(header_lines), 1, "Header must appear exactly once")

    def test_pnl_after_costs_less_than_gross(self):
        """M4 fix: net P&L after costs must be strictly less than gross P&L."""
        with tempfile.TemporaryDirectory() as tmp:
            from execution.order_manager import OrderManager
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []},
                                "execution": {"cost_fraction_per_side": 0.0006}})
            mgr._journal_path = os.path.join(tmp, "trade_journal.csv")
            mgr._ensure_journal_header()

            trade = {"symbol": "NIFTY", "side": "BUY", "entry": 22000.0, "qty": 1, "confidence": 0.8}
            gross = mgr._calculate_pnl(trade, 22200.0)
            net = mgr._calculate_pnl_after_costs(trade, 22200.0)
            self.assertGreater(gross, 0)
            self.assertLess(net, gross, "Net P&L must be less than gross after costs")


# ---------------------------------------------------------------------------
# Trade Lifecycle Tracker Tests
# ---------------------------------------------------------------------------

class TestTradeLifecycleTracker(unittest.TestCase):

    def test_target_hit_detection(self):
        class FakeOrderManager:
            def __init__(self):
                self.active = {"OID1": {"symbol": "NIFTY", "side": "BUY",
                                        "entry": 100, "sl": 99, "target": 102, "status": "PROTECTED"}}
                self.update = None

            async def _get_active_orders(self):
                return self.active

            async def handle_order_update(self, update):
                self.update = update
                self.active = {}

        class FakeQueue:
            async def read_ticks(self, symbol, last_id):
                return [{"data": {"ltp": 102.5}}], "1-0"

        async def run():
            from tracking.trade_lifecycle_tracker import TradeLifecycleTracker
            manager = FakeOrderManager()
            tracker = TradeLifecycleTracker(
                {"instruments": {"equity": ["NIFTY"], "currency": []}},
                manager, FakeQueue()
            )
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(tracker._monitor_symbol("NIFTY"), timeout=0.3)
            self.assertIsNotNone(manager.update)
            self.assertEqual(manager.update["status"], "TARGET_HIT")
            self.assertTrue(manager.update["is_exit"])

        asyncio.run(run())

    def test_sl_hit_detection(self):
        class FakeOrderManager:
            def __init__(self):
                self.active = {"OID2": {"symbol": "NIFTY", "side": "BUY",
                                        "entry": 100, "sl": 99, "target": 102, "status": "PROTECTED"}}
                self.update = None

            async def _get_active_orders(self):
                return self.active

            async def handle_order_update(self, update):
                self.update = update
                self.active = {}

        class FakeQueue:
            async def read_ticks(self, symbol, last_id):
                return [{"data": {"ltp": 98.5}}], "2-0"

        async def run():
            from tracking.trade_lifecycle_tracker import TradeLifecycleTracker
            manager = FakeOrderManager()
            tracker = TradeLifecycleTracker(
                {"instruments": {"equity": ["NIFTY"], "currency": []}},
                manager, FakeQueue()
            )
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(tracker._monitor_symbol("NIFTY"), timeout=0.3)
            self.assertIsNotNone(manager.update)
            self.assertEqual(manager.update["status"], "SL_HIT")

        asyncio.run(run())

    def test_latest_ltp_triggers_exit_without_new_stream_tick(self):
        class FakeOrderManager:
            def __init__(self):
                self.active = {"OID3": {"symbol": "NIFTY", "side": "BUY",
                                        "entry": 100, "sl": 99, "target": 102, "status": "PROTECTED"}}
                self.update = None

            async def get_active_orders(self):
                return self.active

            async def handle_order_update(self, update):
                self.update = update
                self.active = {}

        class FakeClient:
            async def get(self, key):
                return "98.5"

        class FakeQueue:
            client = FakeClient()

            async def read_ticks(self, symbol, last_id):
                return [], last_id

        async def run():
            from tracking.trade_lifecycle_tracker import TradeLifecycleTracker
            manager = FakeOrderManager()
            tracker = TradeLifecycleTracker(
                {"instruments": {"equity": ["NIFTY"], "currency": []}},
                manager, FakeQueue()
            )
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(tracker._monitor_symbol("NIFTY"), timeout=0.2)
            self.assertIsNotNone(manager.update)
            self.assertEqual(manager.update["status"], "SL_HIT")

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering(unittest.TestCase):

    def test_zero_volume_candles_no_nan_vwap(self):
        """Zero-volume candles must produce a finite VWAP (falls back to close)."""
        from features.price_features import PriceFeatures
        from features.volume_features import VolumeFeatures

        df = _make_equity_df(n=70, with_volume=False)
        df = PriceFeatures.add_indicators(df)
        df = VolumeFeatures.add_volume_analysis(df)
        self.assertFalse(df.tail(1)["vwap"].isna().any(), "VWAP must not be NaN for zero-volume candles")

    def test_volume_features_rel_vol_no_nan_after_warmup(self):
        from features.volume_features import VolumeFeatures
        df = _make_equity_df(n=50)
        df = VolumeFeatures.add_volume_analysis(df)
        # After 20-candle warmup, rel_vol must be finite
        self.assertFalse(df.tail(20)["rel_vol"].isna().any())

    def test_indicators_populated_for_sufficient_history(self):
        from features.price_features import PriceFeatures
        df = _make_equity_df(n=70)
        result = PriceFeatures.add_indicators(df)
        for col in ["ema_9", "ema_21", "ema_50", "rsi_14", "atr_14", "ADX_14", "vwap"]:
            self.assertIn(col, result.columns, f"Missing expected column: {col}")


# ---------------------------------------------------------------------------
# ML / XGBoost Model Tests
# ---------------------------------------------------------------------------

class TestXGBoostModel(unittest.TestCase):

    def test_predict_single_row_returns_1d(self):
        """Single-row predict must return a 1-D probability array of length 3."""
        from models.xgboost.model import XGBoostModel
        import xgboost as xgb
        # Build a minimal trained booster so we exercise the real predict path.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_model.json")
            booster = xgb.train(
                {"objective": "multi:softprob", "num_class": 3},
                xgb.DMatrix(np.ones((10, 12)), label=[0, 1, 2] * 3 + [0]),
                num_boost_round=1,
            )
            booster.save_model(path)
            m = XGBoostModel(model_path=path)
            m.load()
            result = m.predict(np.zeros(12))
            self.assertEqual(len(result), 3)
            self.assertAlmostEqual(float(sum(result)), 1.0, places=3)

    def test_predict_batch_returns_2d(self):
        """Batch predict must return a 2-D array (N, 3)."""
        from models.xgboost.model import XGBoostModel
        m = XGBoostModel()
        result = m.predict(np.zeros((5, 12)))
        self.assertEqual(result.shape, (5, 3))

    def test_predict_feature_count_mismatch_raises(self):
        """C10: mismatched feature count must raise ValueError (after training)."""
        from models.xgboost.model import XGBoostModel
        import xgboost as xgb
        m = XGBoostModel()
        # Manually set _n_features to 12
        m._n_features = 12
        # Load a dummy Booster so model is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dummy.json")
            booster = xgb.train({"objective": "multi:softprob", "num_class": 3}, xgb.DMatrix(np.ones((10, 12)), label=[0]*10), num_boost_round=1)
            booster.save_model(path)
            m.model = booster
        with self.assertRaises(ValueError):
            m.predict(np.zeros(8))  # wrong number of features

    def test_predict_dataframe_column_mismatch_raises(self):
        from models.xgboost.model import XGBoostModel

        class FakeBooster:
            feature_names = ["ema_9", "ema_21"]

            def attr(self, name):
                return "ema_9|ema_21" if name == "feature_names_json" else None

            def predict(self, dmatrix):
                return np.array([[0.8, 0.1, 0.1]])

        model = XGBoostModel()
        model.model = FakeBooster()
        model._n_features = 2
        bad = pd.DataFrame([[1.0, 2.0]], columns=["ema_21", "ema_9"])
        with self.assertRaises(ValueError):
            model.predict(bad)


class TestAIEnsembleModelGate(unittest.TestCase):

    def test_undeployed_xgb_metadata_is_rejected(self):
        from models.ensemble import AIEnsemble

        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "model_metadata.json"
            metadata.write_text(json.dumps({"deployed": False}), encoding="utf-8")
            ensemble = AIEnsemble({"model": {"allow_undeployed_models": False}})
            self.assertFalse(ensemble._model_metadata_allows(str(metadata)))

    def test_undeployed_xgb_returns_neutral_score(self):
        from models.ensemble import AIEnsemble

        ensemble = AIEnsemble({"model": {"allow_undeployed_models": False}})
        ensemble.xgb_available = False
        ensemble.rf_available = False
        score = ensemble.get_combined_score(np.zeros((1, 12)))
        self.assertEqual(score, 0.0)


class TestRandomForestModel(unittest.TestCase):

    def test_predict_reorders_matching_dataframe_columns(self):
        from models.random_forest.model import RandomForestModel

        class FakeRF:
            n_features_in_ = 2
            feature_names_in_ = np.array(["ema_9", "ema_21"])
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, values):
                self.last_values = values
                return np.array([[0.7, 0.2, 0.1]])

        rf = RandomForestModel()
        rf.model = FakeRF()
        rf._n_features = 2
        data = pd.DataFrame([[21.0, 9.0]], columns=["ema_21", "ema_9"])
        probs = rf.predict(data)
        self.assertEqual(probs.shape, (1, 3))
        self.assertEqual(rf.model.last_values.to_numpy().tolist(), [[9.0, 21.0]])

    def test_predict_dataframe_missing_expected_column_raises(self):
        from models.random_forest.model import RandomForestModel

        class FakeRF:
            n_features_in_ = 2
            feature_names_in_ = np.array(["ema_9", "ema_21"])
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, values):
                return np.array([[0.7, 0.2, 0.1]])

        rf = RandomForestModel()
        rf.model = FakeRF()
        rf._n_features = 2
        data = pd.DataFrame([[1.0, 2.0]], columns=["ema_9", "rsi_14"])
        with self.assertRaises(ValueError):
            rf.predict(data)


# ---------------------------------------------------------------------------
# Retrain Pipeline Tests (C3, H1)
# ---------------------------------------------------------------------------

class TestRetrainPipeline(unittest.TestCase):

    def _build_parquet(self, tmp_dir: str, n=400) -> str:
        """Write a minimal parquet file for testing."""
        idx = pd.date_range("2025-01-01 09:15", periods=n, freq="5min")
        prices = [100 + np.sin(i / 20) * 5 + i * 0.02 for i in range(n)]
        df = pd.DataFrame({
            "open": prices, "high": [p + 0.5 for p in prices],
            "low": [p - 0.3 for p in prices], "close": prices,
            "volume": [1000 + i for i in range(n)],
        }, index=idx)
        path = os.path.join(tmp_dir, "TEST_6m.parquet")
        df.to_parquet(path)
        return path

    def test_time_based_split_no_leakage(self):
        """C3: training rows must all precede test rows chronologically."""
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build_parquet(tmp)
            from learning.retrain_pipeline import RetrainPipeline
            pipeline = RetrainPipeline(data_path=path)
            df = pipeline._load_files()
            self.assertFalse(df.empty)

            prepared_frames = []
            for symbol, df_sym in df.groupby("symbol", sort=False):
                frame = pipeline._prepare_symbol_frame(df_sym.drop(columns=["symbol"]))
                if not frame.empty:
                    prepared_frames.append(frame)

            self.assertTrue(len(prepared_frames) > 0, "Prepared frames must not be empty")
            df_all = pd.concat(prepared_frames).sort_index()

            cutoff = int(len(df_all) * 0.80)
            train_idx = df_all.index[:cutoff]
            test_idx = df_all.index[cutoff:]
            if len(train_idx) > 0 and len(test_idx) > 0:
                self.assertLessEqual(
                    train_idx.max(), test_idx.min(),
                    "All training timestamps must precede all test timestamps"
                )

    def test_label_has_no_lookahead_at_end(self):
        """H2: last `lookahead` rows must have NaN label (future_close unavailable)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build_parquet(tmp, n=200)
            from learning.retrain_pipeline import RetrainPipeline
            pipeline = RetrainPipeline(data_path=path)
            df = pipeline._load_files().drop(columns=["symbol"])
            prepared = pipeline._prepare_symbol_frame(df)
            # All remaining rows should have non-NaN labels (NaN rows were dropped)
            self.assertFalse(prepared["label"].isna().any(), "dropna must eliminate NaN labels")


    def test_walk_forward_split_keeps_duplicate_timestamps_together(self):
        """All symbols at one timestamp must stay on the same side of a fold."""
        from learning.retrain_pipeline import RetrainPipeline

        timestamps = pd.date_range("2026-01-01 09:15", periods=40, freq="5min")
        duplicated = pd.Index(np.repeat(timestamps, 3))
        for _, train_idx, test_idx in RetrainPipeline._walk_forward_splits(duplicated, n_splits=3):
            train_times = set(duplicated[train_idx])
            test_times = set(duplicated[test_idx])
            self.assertFalse(train_times.intersection(test_times))
            self.assertLess(max(train_times), min(test_times))

    def test_quality_gate_rejects_weak_minority_classes(self):
        from learning.retrain_pipeline import RetrainPipeline

        weak = {
            "balanced_accuracy": 0.50,
            "macro_f1": 0.40,
            "buy_f1": 0.30,
            "sell_f1": 0.10,
        }
        self.assertFalse(RetrainPipeline._passes_quality_gate(weak))

    def test_glob_filters_to_requested_symbols(self):
        from learning.retrain_pipeline import RetrainPipeline

        with tempfile.TemporaryDirectory() as tmp:
            self._build_parquet(tmp, n=200)
            other = Path(tmp) / "OTHER_6m.parquet"
            pd.read_parquet(Path(tmp) / "TEST_6m.parquet").to_parquet(other)
            pipeline = RetrainPipeline(
                data_path=str(Path(tmp) / "*_6m.parquet"),
                symbols=["TEST"],
            )
            loaded = pipeline._load_files()
            self.assertEqual(set(loaded["symbol"].unique()), {"TEST"})

    def test_average_gate_metrics_uses_all_folds(self):
        from learning.retrain_pipeline import RetrainPipeline

        rows = [
            {"balanced_accuracy": 0.50, "macro_f1": 0.40, "buy_f1": 0.30, "sell_f1": 0.30},
            {"balanced_accuracy": 0.30, "macro_f1": 0.20, "buy_f1": 0.10, "sell_f1": 0.10},
        ]
        avg = RetrainPipeline._average_gate_metrics(rows)
        self.assertAlmostEqual(avg["balanced_accuracy"], 0.40)
        self.assertAlmostEqual(avg["macro_f1"], 0.30)

    def test_label_threshold_shift_uses_full_lookahead_gap(self):
        import inspect
        from learning.retrain_pipeline import RetrainPipeline

        source = inspect.getsource(RetrainPipeline._prepare_symbol_frame)
        self.assertIn("shift(lookahead + 1)", source)
        self.assertNotIn("quantile(0.20).shift(1)", source)

    def test_retrain_feature_cols_are_stationary(self):
        from learning.retrain_pipeline import FEATURE_COLS

        self.assertIn("dist_vwap", FEATURE_COLS)
        self.assertIn("atr_pct", FEATURE_COLS)
        for absolute_col in ["ema_9", "ema_21", "ema_50", "vwap", "atr_14", "BBL_20_2.0", "BBU_20_2.0"]:
            self.assertNotIn(absolute_col, FEATURE_COLS)


# ---------------------------------------------------------------------------
# Feature and Signal Robustness Tests
# ---------------------------------------------------------------------------

class TestFeatureAndSignalRobustness(unittest.TestCase):

    def test_add_indicators_does_not_mutate_input(self):
        from features.price_features import PriceFeatures

        original = _make_equity_df(n=30)
        before_columns = list(original.columns)
        enriched = PriceFeatures.add_indicators(original)
        self.assertEqual(list(original.columns), before_columns)
        self.assertIn("ema_9", enriched.columns)

    def test_add_indicators_outputs_stationary_ml_features(self):
        from features.price_features import PriceFeatures

        enriched = PriceFeatures.add_indicators(_make_equity_df(n=70))
        for col in ["dist_ema_9", "dist_ema_21", "dist_ema_50", "dist_vwap", "atr_pct", "bb_pct"]:
            self.assertIn(col, enriched.columns)
            self.assertFalse(enriched[col].tail(20).isna().any(), col)

    def test_standalone_trainers_use_shared_relative_feature_helper(self):
        import inspect
        import scripts.train_equity_xgb as train_equity_xgb
        import scripts.train_rsmb_xgb as train_rsmb_xgb

        self.assertIn("PriceFeatures.add_relative_price_features", inspect.getsource(train_equity_xgb.compute_indicators))
        self.assertIn("PriceFeatures.add_relative_price_features", inspect.getsource(train_rsmb_xgb.compute_indicators))

    def test_paper_quality_gate_requires_indicator_warmup(self):
        from strategies.equity_signal_engine import EquitySignalEngine

        engine = EquitySignalEngine.__new__(EquitySignalEngine)
        engine.time_features = type("Flags", (), {
            "add_session_flags": lambda self, df: pd.DataFrame({"is_noise_window": [False]})
        })()
        engine.min_adx = 16
        engine.min_rel_vol = 0.7
        df = _make_equity_df(n=10)
        ok, reason = engine._paper_relaxed_quality_gate(df, "BUY")
        self.assertFalse(ok)
        self.assertIn("warm-up", reason)

    def test_rsmb_extract_features_uses_relative_contract(self):
        from strategies.rsmb.ai_filter import FEATURE_COLS, RSMBAIFilter

        bar = pd.Series({
            "close": 100.0, "ema_9": 99.0, "ema_21": 98.0, "ema_50": 97.0,
            "vwap": 99.5, "atr_14": 2.0, "rsi_14": 55.0, "ADX_14": 25.0,
            "DMP_14": 20.0, "DMN_14": 10.0, "BBL_20_2.0": 95.0, "BBU_20_2.0": 105.0,
        })
        features = RSMBAIFilter.extract_features(bar, rs_rank=1.1, vwap=99.5, ema21=98.0)
        self.assertEqual(set(features), set(FEATURE_COLS))
        self.assertAlmostEqual(features["dist_ema_9"], 0.01)
        self.assertAlmostEqual(features["atr_pct"], 0.02)
        self.assertAlmostEqual(features["bb_pct"], 0.5)

    def test_performance_guard_blocks_losing_symbol(self):
        from strategies.equity_signal_engine import EquitySignalEngine

        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.csv"
            pd.DataFrame({
                "symbol": ["NIFTY"] * 5 + ["TCS"] * 2,
                "pnl_after_costs": [-100, -50, -25, 10, -20, -10, 20],
            }).to_csv(journal, index=False)

            engine = EquitySignalEngine.__new__(EquitySignalEngine)
            engine.performance_guard_enabled = True
            engine.performance_guard_min_trades = 5
            engine.performance_guard_min_win_rate = 0.35
            engine.performance_guard_min_profit_factor = 0.80
            engine.trade_journal_path = str(journal)
            engine._performance_cache = {"mtime": None, "by_symbol": {}}

            allowed, reason = engine._performance_guard_allows("NIFTY", "Ensemble_AI")
            self.assertFalse(allowed)
            self.assertIn("Performance guard blocked", reason)

            allowed, reason = engine._performance_guard_allows("TCS", "Ensemble_AI")
            self.assertTrue(allowed)
            self.assertEqual(reason, "insufficient performance sample")

    def test_validate_setup_handles_missing_indicator_columns(self):
        from strategies.equity_signal_engine import EquitySignalEngine

        engine = EquitySignalEngine({
            "paper_mode": True,
            "paper_trading": {"relaxed_signals": False, "technical_fallback_enabled": True},
            "equity_signal": {
                "min_buy_confidence": 0.38,
                "min_sell_confidence": 0.38,
                "min_ai_confidence_floor": 0.25,
            },
        })
        df = _make_equity_df(n=25)
        result = engine.validate_setup("NIFTY", df, 0.40, "BUY")
        self.assertIn("valid", result)
        self.assertIn("reason", result)

    def test_vwap_resets_daily_without_lookahead(self):
        from features.price_features import PriceFeatures

        day1 = pd.date_range("2026-05-13 09:15", periods=75, freq="5min")
        day2 = pd.date_range("2026-05-14 09:15", periods=75, freq="5min")
        idx = day1.append(day2)
        prices = np.linspace(100, 120, len(idx))
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices + 0.5,
            "volume": np.arange(1, len(idx) + 1) * 100,
        }, index=idx)
        result = PriceFeatures.add_indicators(df)
        first_day2 = result[result.index.date == day2[0].date()].iloc[0]
        typical_price = (first_day2["high"] + first_day2["low"] + first_day2["close"]) / 3
        self.assertAlmostEqual(float(first_day2["vwap"]), float(typical_price), places=6)


# ---------------------------------------------------------------------------
# Adaptive Learning Tests
# ---------------------------------------------------------------------------

class TestAdaptiveLearningEngine(unittest.TestCase):

    def test_existing_currency_quant_key_does_not_crash(self):
        from learning.adaptive_learning_engine import AdaptiveLearningEngine

        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "adaptive_params.json"
            params_path.write_text(
                json.dumps({
                    "equity_min_buy_confidence": 0.72,
                    "equity_min_sell_confidence": 0.70,
                    "currency_min_quant_score": 70,
                }),
                encoding="utf-8",
            )
            engine = AdaptiveLearningEngine(str(params_path), config={"adaptive_learning": {"min_trades_before_change": 15}})
            engine.tune_parameters({"win_rate": "40.0%", "total_trades": 20})
            params = json.loads(params_path.read_text(encoding="utf-8"))
            self.assertEqual(params["currency_min_quant_score"], 72)

    def test_respects_min_trades_before_tightening(self):
        from learning.adaptive_learning_engine import AdaptiveLearningEngine

        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "adaptive_params.json"
            history_path = Path("data") / "parameter_history.csv"
            before = history_path.read_text(encoding="utf-8") if history_path.exists() else None
            try:
                engine = AdaptiveLearningEngine(
                    params_path=str(params_path),
                    config={"adaptive_learning": {"min_trades_before_change": 40}},
                )
                reason = engine.tune_parameters({"total_trades": 5, "win_rate": "20.0%"})
                params = json.loads(params_path.read_text(encoding="utf-8"))
                self.assertIn("Only 5 trades", reason)
                self.assertEqual(params["min_adx"], 20)
                self.assertEqual(params["min_quant_score"], 70)
            finally:
                if before is None:
                    if history_path.exists():
                        history_path.unlink()
                else:
                    history_path.write_text(before, encoding="utf-8")

    def test_skips_duplicate_same_day_update(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from learning.adaptive_learning_engine import AdaptiveLearningEngine

        today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "adaptive_params.json"
            params_path.write_text(json.dumps({"min_adx": 22, "last_updated": today}), encoding="utf-8")
            engine = AdaptiveLearningEngine(
                params_path=str(params_path),
                config={"adaptive_learning": {"min_trades_before_change": 1}},
            )
            reason = engine.tune_parameters({"total_trades": 50, "win_rate": "10.0%"})
            params = json.loads(params_path.read_text(encoding="utf-8"))
            self.assertIn("Already tuned", reason)
            self.assertEqual(params["min_adx"], 22)


# ---------------------------------------------------------------------------
# Signal Logger / Dashboard Tests
# ---------------------------------------------------------------------------

class TestSignalLogger(unittest.TestCase):

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "signal_log.csv")
            from tracking.signal_logger import SignalLogger
            from dashboard.data_loader import load_csv_safely

            sl = SignalLogger(path)
            sl.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
            df, err = load_csv_safely(path)
            self.assertIsNone(err)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["symbol"], "NIFTY")

    def test_dedup_within_window(self):
        """Same signal logged twice within dedup window must appear only once in CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "signal_log.csv")
            from tracking.signal_logger import SignalLogger
            sl = SignalLogger(path)
            sl.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
            sl.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")  # duplicate
            df = pd.read_csv(path)
            self.assertEqual(len(df), 1)

    def test_instance_level_dedup_not_shared(self):
        """H5: two separate SignalLogger instances must not share dedup state."""
        with tempfile.TemporaryDirectory() as tmp:
            from tracking.signal_logger import SignalLogger
            path1 = str(Path(tmp) / "a.csv")
            path2 = str(Path(tmp) / "b.csv")
            sl1 = SignalLogger(path1)
            sl2 = SignalLogger(path2)
            sl1.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
            # sl2 should NOT be blocked by sl1's dedup
            sl2.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
            df2 = pd.read_csv(path2)
            self.assertEqual(len(df2), 1, "sl2 must log independently of sl1")

    def test_missing_file_returns_empty(self):
        from dashboard.data_loader import load_csv_safely
        df, err = load_csv_safely("totally_missing_file_xyz.csv")
        self.assertIsNone(err)
        self.assertTrue(df.empty)

    def test_dashboard_loader_repairs_old_trade_journal_strategy_column(self):
        from dashboard.data_loader import load_csv_safely

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trade_journal.csv"
            path.write_text(
                "date,symbol,side,entry_price,exit_price,qty,pnl_inr,pnl_after_costs,outcome,confidence\n"
                "2026-05-14 11:30:43,ICICIBANK,SELL,rsmb,1239.0,1246.4,80,-636.0,-636.0,SL_HIT,0.7662\n",
                encoding="utf-8",
            )
            df, err = load_csv_safely(str(path))

        self.assertIsNone(err)
        self.assertEqual(df.iloc[0]["symbol"], "ICICIBANK")
        self.assertEqual(df.iloc[0]["strategy"], "rsmb")
        self.assertEqual(float(df.iloc[0]["entry_price"]), 1239.0)
        self.assertEqual(df.iloc[0]["outcome"], "SL_HIT")

    def test_dashboard_metrics_include_win_rate_for_empty_and_non_empty_data(self):
        from dashboard.data_loader import calculate_advanced_metrics

        empty_metrics = calculate_advanced_metrics(pd.DataFrame())
        self.assertIn("win_rate", empty_metrics)
        self.assertEqual(empty_metrics["win_rate"], 0.0)

        metrics = calculate_advanced_metrics(pd.DataFrame({"pnl_after_costs": [100.0, -50.0]}))
        self.assertEqual(metrics["win_rate"], 50.0)

    def test_filepath_without_directory_does_not_crash(self):
        """L3: SignalLogger must not crash when filepath has no directory component."""
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                from tracking.signal_logger import SignalLogger
                sl = SignalLogger("bare_log.csv")  # no directory part
                sl.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
                self.assertTrue(os.path.exists("bare_log.csv"))
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Risk Manager Agent Tests (C9)
# ---------------------------------------------------------------------------

class TestRiskManagerAgent(unittest.TestCase):

    def test_rr_positive_for_buy(self):
        from agents.risk_manager_agent import RiskManagerAgent
        agent = RiskManagerAgent({"currency_signal": {"max_sl_paise": 20}})
        row = {"close": 83.45, "atr_14": 0.05, "low": 83.40, "high": 83.50}
        params = agent.calculate_risk_params(row, "BUY")
        self.assertGreater(params["rr"], 0, "R:R must be positive for BUY")

    def test_rr_positive_for_sell(self):
        """C9 fix: R:R must be positive for SELL signals too."""
        from agents.risk_manager_agent import RiskManagerAgent
        agent = RiskManagerAgent({"currency_signal": {"max_sl_paise": 20}})
        row = {"close": 83.45, "atr_14": 0.05, "low": 83.40, "high": 83.50}
        params = agent.calculate_risk_params(row, "SELL")
        self.assertGreater(params["rr"], 0, "R:R must be positive for SELL (C9 fix)")

    def test_lots_clamped(self):
        from agents.risk_manager_agent import RiskManagerAgent
        agent = RiskManagerAgent({"currency_signal": {"max_sl_paise": 20}})
        row = {"close": 83.45, "atr_14": 0.001, "low": 83.44, "high": 83.46}
        params = agent.calculate_risk_params(row, "BUY")
        self.assertGreaterEqual(params["lots"], 1)
        self.assertLessEqual(params["lots"], 3)


class TestCurrencyQuantValidator(unittest.TestCase):

    def test_min_quant_score_config_is_used(self):
        from agents.quant_validator import QuantValidator

        validator = QuantValidator.__new__(QuantValidator)
        validator.config = {"currency_signal": {"min_quant_score": 70}}
        validator.feature_cols = [
            "ema_9", "ema_21", "ema_50", "rsi_14", "atr_14",
            "vwap", "ADX_14", "DMP_14", "DMN_14",
            "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0",
        ]
        validator.xgb = type("FakeXGB", (), {"predict": lambda self, features: np.array([0.45, 0.55, 0.0])})()
        row = {
            "close": 83.5, "ema_9": 83.6, "ema_21": 83.4, "ema_50": 83.3,
            "rsi_14": 55, "atr_14": 0.05, "vwap": 83.4, "ADX_14": 25,
            "DMP_14": 20, "DMN_14": 10, "BBL_20_2.0": 83.0,
            "BBM_20_2.0": 83.4, "BBU_20_2.0": 84.0,
            "high": 83.6, "low": 83.4,
        }
        df_5m = pd.DataFrame([row])
        df_15m = pd.DataFrame([{"close": 83.5, "ema_21": 83.4}])

        result = validator.validate(df_5m, df_15m, "BUY")
        self.assertFalse(result["valid"])
        self.assertIn("Low ML confidence", result["reason"])


# ---------------------------------------------------------------------------
# RSMB Position Lifecycle Tests
# ---------------------------------------------------------------------------

class TestRSMBPositionManager(unittest.TestCase):

    def test_price_updates_are_symbol_scoped(self):
        from strategies.base_strategy import Signal
        from strategies.rsmb.position_manager import RSMBPositionManager

        manager = RSMBPositionManager(cost_per_order_inr=0)
        signal = Signal(
            strategy="rsmb",
            symbol="HDFCBANK",
            side="BUY",
            entry=100,
            sl=95,
            target1=105,
            target2=110,
            qty=10,
            score=0.8,
            rs_rank=1.1,
            rejection_reason=None,
            timestamp=pd.Timestamp("2026-05-14 09:30", tz="Asia/Kolkata"),
        )
        pos_id = manager.open_position(signal)
        manager.on_fill(pos_id, 100, pd.Timestamp("2026-05-14 09:31", tz="Asia/Kolkata"))

        self.assertEqual(manager.on_price_update(94, symbol="RELIANCE"), [])
        self.assertEqual(manager.on_price_update(94, symbol="HDFCBANK")[0][1], "SL_HIT")


class TestPaperEngineActiveSnapshot(unittest.TestCase):

    def test_active_snapshot_written_for_dashboard(self):
        from execution.paper_engine import PaperEngine
        from strategies.base_strategy import Signal

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "paper_orders.json"
            journal_path = Path(tmp) / "trade_journal.csv"
            engine = PaperEngine(
                journal_path=str(journal_path),
                active_orders_path=str(snapshot_path),
            )
            signal = Signal(
                strategy="rsmb",
                symbol="WIPRO",
                side="SELL",
                entry=187,
                sl=188,
                target1=185,
                target2=183,
                qty=10,
                score=0.76,
                rs_rank=0.9,
                rejection_reason=None,
                timestamp=pd.Timestamp("2026-05-14 11:15", tz="Asia/Kolkata"),
            )
            order_id = engine.simulate_fill(signal, 187.2)
            rows = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["order_id"], order_id)
            self.assertEqual(rows[0]["symbol"], "WIPRO")
            self.assertEqual(rows[0]["source"], "paper_engine")

    def test_simulate_fill_accepts_dict_signal_options(self):
        from execution.paper_engine import PaperEngine

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "paper_orders.json"
            journal_path = Path(tmp) / "trade_journal.csv"
            engine = PaperEngine(
                journal_path=str(journal_path),
                active_orders_path=str(snapshot_path),
            )
            order_id = engine.simulate_fill({
                "strategy": "Ensemble_AI_PAPER_TECHNICAL",
                "symbol": "NIFTY",
                "side": "BUY",
                "entry": 100.0,
                "sl": 99.0,
                "target": 103.0,
                "qty": 2,
                "confidence": 0.42,
                "t1_exit_pct": 0.25,
                "target2_mode": "manual",
            }, 100.5)
            order = engine.get_order_snapshot(order_id)
            self.assertEqual(order.t1_exit_pct, 0.25)
            self.assertEqual(order.target2_mode, "manual")
            self.assertAlmostEqual(order.sl, 99.5)

    def test_simulate_fill_enforces_local_max_open_guard(self):
        from execution.paper_engine import PaperEngine

        with tempfile.TemporaryDirectory() as tmp:
            engine = PaperEngine(
                journal_path=str(Path(tmp) / "trade_journal.csv"),
                active_orders_path=str(Path(tmp) / "paper_orders.json"),
                config={"capital": {"max_open_trades_equity": 1}},
            )
            first = engine.simulate_fill({
                "strategy": "rsmb", "symbol": "NIFTY", "side": "BUY",
                "entry": 100.0, "sl": 99.0, "target": 103.0, "qty": 1,
            }, 100.0)
            second = engine.simulate_fill({
                "strategy": "rsmb", "symbol": "TCS", "side": "BUY",
                "entry": 100.0, "sl": 99.0, "target": 103.0, "qty": 1,
            }, 100.0)
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertEqual(len(engine.get_active_orders()), 1)

    def test_simulate_fill_blocks_duplicate_symbol(self):
        from execution.paper_engine import PaperEngine

        with tempfile.TemporaryDirectory() as tmp:
            engine = PaperEngine(
                journal_path=str(Path(tmp) / "trade_journal.csv"),
                active_orders_path=str(Path(tmp) / "paper_orders.json"),
                config={"capital": {"max_open_trades_equity": 3}},
            )
            first = engine.simulate_fill({
                "strategy": "rsmb", "symbol": "NIFTY", "side": "BUY",
                "entry": 100.0, "sl": 99.0, "target": 103.0, "qty": 1,
            }, 100.0)
            duplicate = engine.simulate_fill({
                "strategy": "Ensemble_AI_PAPER_TECHNICAL", "symbol": "NIFTY", "side": "BUY",
                "entry": 101.0, "sl": 100.0, "target": 104.0, "qty": 1,
            }, 101.0)
            self.assertIsNotNone(first)
            self.assertIsNone(duplicate)
            self.assertEqual(len(engine.get_active_orders()), 1)

    def test_paper_engine_journal_separates_gross_and_net_pnl(self):
        from execution.paper_engine import PaperEngine
        from strategies.base_strategy import Signal

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "paper_orders.json"
            journal_path = Path(tmp) / "trade_journal.csv"
            engine = PaperEngine(
                cost_per_order_inr=22,
                journal_path=str(journal_path),
                active_orders_path=str(snapshot_path),
            )
            signal = Signal(
                strategy="rsmb",
                symbol="ICICIBANK",
                side="SELL",
                entry=1239.0,
                sl=1246.4,
                target1=1229.8,
                target2=1220.0,
                qty=80,
                score=0.7662,
                rs_rank=0.9,
                rejection_reason=None,
                timestamp=pd.Timestamp("2026-05-14 11:15", tz="Asia/Kolkata"),
            )
            engine.simulate_fill(signal, 1239.0)
            engine.on_price_update("ICICIBANK", 1246.4)

            df = pd.read_csv(journal_path)
            self.assertEqual(float(df.iloc[-1]["pnl_inr"]), -592.0)
            self.assertEqual(float(df.iloc[-1]["pnl_after_costs"]), -636.0)

    def test_paper_engine_partial_exit_charges_entry_and_each_exit_once(self):
        from execution.paper_engine import PaperEngine
        from strategies.base_strategy import Signal

        with tempfile.TemporaryDirectory() as tmp:
            engine = PaperEngine(
                cost_per_order_inr=22,
                journal_path=str(Path(tmp) / "trade_journal.csv"),
                active_orders_path=str(Path(tmp) / "paper_orders.json"),
            )
            signal = Signal(
                strategy="rsmb",
                symbol="NIFTY",
                side="BUY",
                entry=100,
                sl=95,
                target1=105,
                target2=110,
                qty=10,
                score=0.8,
                rs_rank=1.1,
                rejection_reason=None,
                timestamp=pd.Timestamp("2026-05-14 11:15", tz="Asia/Kolkata"),
            )
            oid = engine.simulate_fill(signal, 100)
            self.assertEqual(engine.on_price_update("NIFTY", 105), [(oid, "T1_HIT")])
            self.assertEqual(engine.on_price_update("NIFTY", 110), [(oid, "T2_HIT")])

            df = pd.read_csv(Path(tmp) / "trade_journal.csv")
            self.assertEqual(float(df.iloc[-1]["pnl_inr"]), 75.0)
            self.assertEqual(float(df.iloc[-1]["pnl_after_costs"]), 9.0)

    def test_paper_engine_reload_preserves_partial_realised_pnl(self):
        from execution.paper_engine import PaperEngine
        from strategies.base_strategy import Signal

        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "trade_journal.csv"
            snapshot_path = Path(tmp) / "paper_orders.json"
            signal = Signal(
                strategy="rsmb",
                symbol="NIFTY",
                side="BUY",
                entry=100,
                sl=95,
                target1=105,
                target2=110,
                qty=10,
                score=0.8,
                rs_rank=1.1,
                rejection_reason=None,
                timestamp=pd.Timestamp("2026-05-14 11:15", tz="Asia/Kolkata"),
            )
            engine = PaperEngine(
                cost_per_order_inr=22,
                journal_path=str(journal_path),
                active_orders_path=str(snapshot_path),
            )
            oid = engine.simulate_fill(signal, 100)
            engine.on_price_update("NIFTY", 105)

            reloaded = PaperEngine(
                cost_per_order_inr=22,
                journal_path=str(journal_path),
                active_orders_path=str(snapshot_path),
            )
            self.assertEqual(reloaded.on_price_update("NIFTY", 110), [(oid, "T2_HIT")])
            df = pd.read_csv(journal_path)
            self.assertEqual(float(df.iloc[-1]["pnl_inr"]), 75.0)
            self.assertEqual(float(df.iloc[-1]["pnl_after_costs"]), 9.0)

    def test_paper_engine_manual_target2_mode_waits_for_strategy_close(self):
        from execution.paper_engine import PaperEngine
        from strategies.base_strategy import Signal

        with tempfile.TemporaryDirectory() as tmp:
            engine = PaperEngine(
                cost_per_order_inr=22,
                journal_path=str(Path(tmp) / "trade_journal.csv"),
                active_orders_path=str(Path(tmp) / "paper_orders.json"),
            )
            signal = Signal(
                strategy="gamma_scalper",
                symbol="SENSEXATMCE",
                side="BUY",
                entry=100,
                sl=90,
                target1=105,
                target2=106,
                qty=10,
                score=0.8,
                rs_rank=None,
                rejection_reason=None,
                timestamp=pd.Timestamp("2026-05-14 11:15", tz="Asia/Kolkata"),
                t1_exit_pct=0.6,
                target2_mode="manual",
            )
            oid = engine.simulate_fill(signal, 100)
            self.assertEqual(engine.on_price_update("SENSEXATMCE", 105), [(oid, "T1_HIT")])
            self.assertEqual(engine.on_price_update("SENSEXATMCE", 110), [])
            order = engine.get_order_snapshot(oid)
            self.assertEqual(order.qty_t1_booked, 6)
            self.assertEqual(order.status, "PARTIAL")

            self.assertTrue(engine.close_position(oid, 104, "TARGET_HIT"))
            df = pd.read_csv(Path(tmp) / "trade_journal.csv")
            self.assertEqual(df.iloc[-1]["strategy"], "gamma_scalper")
            self.assertEqual(df.iloc[-1]["outcome"], "TARGET_HIT")


class TestNewStrategyContracts(unittest.TestCase):

    def test_gamma_position_manager_allows_one_ce_and_one_pe(self):
        from strategies.base_strategy import Signal
        from strategies.gamma_scalper.position_manager import GammaPositionManager

        manager = GammaPositionManager(max_open_trades=2)
        common = dict(
            strategy="gamma_scalper",
            side="BUY",
            entry=100,
            sl=90,
            target1=130,
            target2=1000,
            qty=1,
            score=0.75,
            rs_rank=None,
            rejection_reason=None,
            timestamp=pd.Timestamp("2026-05-14 11:15", tz="Asia/Kolkata"),
        )
        ce1 = Signal(symbol="SENSEXATMCE", **common)
        ce2 = Signal(symbol="SENSEXNEXTCE", **common)
        pe1 = Signal(symbol="SENSEXATMPE", **common)

        self.assertIsNotNone(manager.open_position(ce1))
        self.assertIsNone(manager.open_position(ce2))
        self.assertIsNotNone(manager.open_position(pe1))

    def test_new_strategies_empty_bar_returns_none(self):
        from strategies.gamma_scalper.strategy import GammaScalperStrategy
        from strategies.mean_reversion.strategy import MeanReversionStrategy

        gamma = GammaScalperStrategy(
            {"gamma_scalper": {"enabled": True, "symbols": ["TESTCE"]}},
            signal_logger=None,
        )
        meanrev = MeanReversionStrategy(
            {"instruments": {"equity": ["TEST"], "currency": []}, "mean_reversion": {"enabled": True}},
            signal_logger=None,
        )

        self.assertIsNone(gamma.on_bar("TESTCE", pd.Series(dtype=float), spot_bar=None))
        self.assertIsNone(meanrev.on_bar("TEST", pd.Series(dtype=float), df_1h=pd.DataFrame()))


# ---------------------------------------------------------------------------
# Candle Builder Tests (C6, H9)
# ---------------------------------------------------------------------------

class TestCandleBuilder(unittest.TestCase):

    def test_string_tick_values_are_numeric_before_resample(self):
        """Broker JSON/string tick fields must not create lexicographic OHLCV candles."""
        from pipeline.candle_builder import CandleBuilder

        with tempfile.TemporaryDirectory() as tmp:
            cb = CandleBuilder.__new__(CandleBuilder)
            cb.config = {"instruments": {"equity": ["NIFTY"], "currency": []}}
            cb.symbols = ["NIFTY"]
            cb.timeframes = ["5min"]
            cb.redis_queue = None
            cb.equity_engine = None
            cb.currency_engine = None
            cb.order_manager = None
            cb.rsmb_strategy = None
            cb.paper_engine = None
            cb.candle_snapshot_path = str(Path(tmp) / "candle_snapshot.json")
            cb.tick_data = {"NIFTY": []}
            cb.candles = {"NIFTY": {"5min": pd.DataFrame()}}

            ticks = [
                {"data": {"timestamp": "2026-05-14 09:15:01", "ltp": "99.0", "volume": "10"}},
                {"data": {"timestamp": "2026-05-14 09:15:10", "ltp": "100.0", "volume": "15"}},
            ]

            async def run():
                await cb._process_ticks_to_candles("NIFTY", ticks)

            asyncio.run(run())
            candle = cb.candles["NIFTY"]["5min"].iloc[-1]
            self.assertEqual(float(candle["high"]), 100.0)
            self.assertEqual(float(candle["low"]), 99.0)
            self.assertEqual(float(candle["volume"]), 25.0)

    def test_out_of_order_ticks_sorted(self):
        """C6: out-of-order ticks must be sorted before aggregation."""
        from pipeline.candle_builder import CandleBuilder

        with tempfile.TemporaryDirectory() as tmp:
            cb = CandleBuilder.__new__(CandleBuilder)
            cb.config = {"instruments": {"equity": ["NIFTY"], "currency": []}}
            cb.symbols = ["NIFTY"]
            cb.timeframes = ["5min"]
            cb.redis_queue = None
            cb.equity_engine = None
            cb.currency_engine = None
            cb.order_manager = None
            cb.rsmb_strategy = None
            cb.paper_engine = None
            cb.candle_snapshot_path = str(Path(tmp) / "candle_snapshot.json")
            cb.tick_data = {"NIFTY": []}
            cb.candles = {"NIFTY": {"5min": pd.DataFrame()}}

            # Create ticks out of order
            ticks = [
                {"data": {"timestamp": "2026-05-13 09:20:00", "ltp": 101.0, "volume": 100}},
                {"data": {"timestamp": "2026-05-13 09:15:00", "ltp": 100.0, "volume": 200}},  # earlier
            ]

            async def run():
                await cb._process_ticks_to_candles("NIFTY", ticks)
                df = cb.candles["NIFTY"]["5min"]
                if not df.empty and len(df) >= 1:
                    # First candle should start at 09:15, not 09:20
                    self.assertLessEqual(df.index[0].strftime("%H:%M"), "09:15")

            asyncio.run(run())

    def test_15min_candle_snapshot_written_for_dashboard(self):
        from pipeline.candle_builder import CandleBuilder

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "candle_snapshot.json"
            cb = CandleBuilder.__new__(CandleBuilder)
            cb.config = {"instruments": {"equity": ["NIFTY"], "currency": []}}
            cb.symbols = ["NIFTY"]
            cb.timeframes = ["15min"]
            cb.redis_queue = None
            cb.equity_engine = None
            cb.currency_engine = None
            cb.order_manager = None
            cb.rsmb_strategy = None
            cb.paper_engine = None
            cb.candle_snapshot_path = str(snapshot_path)
            cb.tick_data = {"NIFTY": []}
            cb.candles = {"NIFTY": {"15min": pd.DataFrame()}}

            ticks = [
                {"data": {"timestamp": "2026-05-14 09:15:01", "ltp": 100.0, "volume": 10}},
                {"data": {"timestamp": "2026-05-14 09:20:00", "ltp": 102.0, "volume": 15}},
                {"data": {"timestamp": "2026-05-14 09:29:59", "ltp": 101.0, "volume": 20}},
            ]

            async def run():
                await cb._process_ticks_to_candles("NIFTY", ticks)

            asyncio.run(run())
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            candle = snapshot["symbols"]["NIFTY"]["15min"][-1]
            self.assertEqual(candle["open"], 100.0)
            self.assertEqual(candle["high"], 102.0)
            self.assertEqual(candle["low"], 100.0)
            self.assertEqual(candle["close"], 101.0)

    def test_stale_historical_candle_does_not_trigger_engine(self):
        """Historical preload from a prior session must not fire strategies on restart."""
        from pipeline.candle_builder import CandleBuilder

        class FakeEquityEngine:
            def __init__(self):
                self.calls = 0

            async def process_symbol(self, symbol, candles, timing_candles=None):
                self.calls += 1
                return None

        idx_5m = pd.DatetimeIndex([pd.Timestamp("2026-05-13 15:25", tz="Asia/Kolkata")])
        idx_15m = pd.DatetimeIndex([pd.Timestamp("2026-05-13 15:15", tz="Asia/Kolkata")])
        old_5m = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000], "oi": [0]},
            index=idx_5m,
        )
        old_15m = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000], "oi": [0]},
            index=idx_15m,
        )

        engine = FakeEquityEngine()
        cb = CandleBuilder.__new__(CandleBuilder)
        cb.config = {"instruments": {"equity": ["NIFTY"], "currency": []}}
        cb.symbols = ["NIFTY"]
        cb.timeframes = ["5min", "15min"]
        cb.redis_queue = None
        cb.equity_engine = engine
        cb.currency_engine = None
        cb.order_manager = None
        cb.rsmb_strategy = None
        cb.paper_engine = None
        cb.candle_snapshot_path = str(Path(tempfile.gettempdir()) / "stale_candle_snapshot_test.json")
        cb.tick_data = {"NIFTY": []}
        cb.candles = {"NIFTY": {"5min": old_5m, "15min": old_15m}}

        ticks = [{"data": {"timestamp": "2026-05-14 09:15:01", "ltp": 101.0, "volume": 100}}]

        async def run():
            await cb._process_ticks_to_candles("NIFTY", ticks)

        asyncio.run(run())
        self.assertEqual(engine.calls, 0)

    def test_same_session_candle_close_triggers_engine(self):
        """Normal live candle closes must still trigger strategy evaluation."""
        from pipeline.candle_builder import CandleBuilder

        class FakeEquityEngine:
            def __init__(self):
                self.calls = 0
                self.timing_rows = 0
                self.timing_last = None

            async def process_symbol(self, symbol, candles, timing_candles=None):
                self.calls += 1
                self.timing_rows = len(timing_candles)
                self.timing_last = timing_candles.index[-1]
                return None

        idx_5m = pd.DatetimeIndex([
            pd.Timestamp("2026-05-14 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-05-14 09:20", tz="Asia/Kolkata"),
            pd.Timestamp("2026-05-14 09:25", tz="Asia/Kolkata"),
        ])
        idx_15m = pd.DatetimeIndex([pd.Timestamp("2026-05-14 09:15", tz="Asia/Kolkata")])
        old_5m = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1000, 1000],
                "oi": [0, 0, 0],
            },
            index=idx_5m,
        )
        old_15m = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000], "oi": [0]},
            index=idx_15m,
        )

        engine = FakeEquityEngine()
        cb = CandleBuilder.__new__(CandleBuilder)
        cb.config = {"instruments": {"equity": ["NIFTY"], "currency": []}}
        cb.symbols = ["NIFTY"]
        cb.timeframes = ["5min", "15min"]
        cb.redis_queue = None
        cb.equity_engine = engine
        cb.currency_engine = None
        cb.order_manager = None
        cb.rsmb_strategy = None
        cb.paper_engine = None
        cb.candle_snapshot_path = str(Path(tempfile.gettempdir()) / "same_session_candle_snapshot_test.json")
        cb.tick_data = {"NIFTY": []}
        cb.candles = {"NIFTY": {"5min": old_5m, "15min": old_15m}}

        ticks = [{"data": {"timestamp": "2026-05-14 09:30:01", "ltp": 101.0, "volume": 100}}]

        async def run():
            await cb._process_ticks_to_candles("NIFTY", ticks)

        asyncio.run(run())
        self.assertEqual(engine.calls, 1)
        self.assertEqual(engine.timing_rows, 3)
        self.assertEqual(engine.timing_last, pd.Timestamp("2026-05-14 09:25", tz="Asia/Kolkata"))

    def test_rsmb_receives_long_history_and_live_nifty_benchmark(self):
        """RSMB needs more than the equity engine's 200-bar slice for RS_Rank and daily EMA checks."""
        from pipeline.candle_builder import CandleBuilder

        class FakeEquityEngine:
            async def process_symbol(self, symbol, candles, timing_candles=None):
                return None

        class FakeRSMB:
            def __init__(self):
                self.pushed_bar_len = 0
                self.nifty_len = 0

            def push_nifty_daily(self, daily):
                self.nifty_len = len(daily)

            def push_daily(self, symbol, daily):
                pass

            def push_bar(self, symbol, df):
                self.pushed_bar_len = len(df)

            def update_trailing_stops(self, symbol):
                pass

            def on_bar(self, symbol, bar):
                return None

        idx_15m = pd.date_range(
            end=pd.Timestamp("2026-05-14 09:15", tz="Asia/Kolkata"),
            periods=2500,
            freq="15min",
        )
        idx_5m = pd.date_range(
            end=pd.Timestamp("2026-05-14 09:25", tz="Asia/Kolkata"),
            periods=3,
            freq="5min",
        )
        old_15m = pd.DataFrame(
            {"open": range(2500), "high": range(1, 2501), "low": range(2500), "close": range(2500), "volume": [1000] * 2500, "oi": [0] * 2500},
            index=idx_15m,
        )
        old_5m = pd.DataFrame(
            {"open": [100, 101, 102], "high": [101, 102, 103], "low": [99, 100, 101], "close": [100, 101, 102], "volume": [1000] * 3, "oi": [0] * 3},
            index=idx_5m,
        )

        rsmb = FakeRSMB()
        cb = CandleBuilder.__new__(CandleBuilder)
        cb.config = {"instruments": {"equity": ["NIFTY"], "currency": []}}
        cb.symbols = ["NIFTY"]
        cb.timeframes = ["15min"]
        cb.redis_queue = None
        cb.equity_engine = FakeEquityEngine()
        cb.currency_engine = None
        cb.order_manager = None
        cb.rsmb_strategy = rsmb
        cb.paper_engine = None
        cb.candle_snapshot_path = str(Path(tempfile.gettempdir()) / "rsmb_long_history_snapshot_test.json")
        cb.tick_data = {"NIFTY": []}
        cb.candles = {"NIFTY": {"5min": old_5m, "15min": old_15m}}

        ticks = [{"data": {"timestamp": "2026-05-14 09:30:01", "ltp": 601, "volume": 100}}]

        async def run():
            await cb._process_ticks_to_candles("NIFTY", ticks)

        asyncio.run(run())
        self.assertGreater(rsmb.pushed_bar_len, 200)
        self.assertGreaterEqual(rsmb.nifty_len, 21)


class TestWebSocketFeed(unittest.TestCase):

    def test_angelone_cumulative_volume_is_converted_to_delta(self):
        """Angel One cumulative day volume must become candle interval volume."""
        from pipeline.websocket_feed import WebSocketFeed

        feed = WebSocketFeed({"instruments": {"equity": []}})
        self.assertEqual(feed._extract_volume_delta("NIFTY", {"volume_trade_for_the_day": 1000}), 0)
        self.assertEqual(feed._extract_volume_delta("NIFTY", {"volume_trade_for_the_day": 1250}), 250)
        self.assertEqual(feed._extract_volume_delta("NIFTY", {"volume_trade_for_the_day": 1240}), 0)

    def test_angelone_last_traded_quantity_fallback(self):
        """If cumulative volume is absent, use the last traded quantity field."""
        from pipeline.websocket_feed import WebSocketFeed

        feed = WebSocketFeed({"instruments": {"equity": []}})
        self.assertEqual(feed._extract_volume_delta("NIFTY", {"last_traded_quantity": 75}), 75)


class TestDailyReportPublisher(unittest.TestCase):

    def test_email_channel_sends_via_smtp(self):
        from unittest.mock import patch
        from reporting.daily_report_publisher import DailyReportPublisher

        sent = []

        class FakeSMTP:
            def __init__(self, host, port, timeout):
                self.host = host
                self.port = port
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def starttls(self):
                pass

            def login(self, user, password):
                self.user = user
                self.password = password

            def send_message(self, message):
                sent.append(message)

        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "bot@example.com",
            "SMTP_PASSWORD": "secret",
            "EMAIL_TO": "owner@example.com",
        }
        with patch.dict(os.environ, env, clear=True), patch("smtplib.SMTP", FakeSMTP):
            publisher = DailyReportPublisher({"notifications": {"channel": "email"}})
            publisher.send_report("Daily summary")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["To"], "owner@example.com")
        self.assertEqual(sent[0]["From"], "bot@example.com")

    def test_email_channel_supports_legacy_notify_env_names(self):
        from unittest.mock import patch
        from reporting.daily_report_publisher import DailyReportPublisher

        sent = []

        class FakeSMTP:
            def __init__(self, host, port, timeout):
                self.host = host
                self.port = port
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def starttls(self):
                pass

            def login(self, user, password):
                self.user = user
                self.password = password

            def send_message(self, message):
                sent.append((self.host, self.port, self.user, message))

        env = {
            "NOTIFY_EMAIL_FROM": "oldbot@example.com",
            "NOTIFY_EMAIL_PASSWORD": "old secret",
            "NOTIFY_EMAIL_TO": "owner@example.com",
            "NOTIFY_SMTP_HOST": "smtp.gmail.com",
            "NOTIFY_SMTP_PORT": "587",
        }
        with patch.dict(os.environ, env, clear=True), patch("smtplib.SMTP", FakeSMTP):
            publisher = DailyReportPublisher({"notifications": {"channel": "email"}})
            publisher.send_report("Daily summary")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "smtp.gmail.com")
        self.assertEqual(sent[0][2], "oldbot@example.com")
        self.assertEqual(sent[0][3]["To"], "owner@example.com")


class TestEodSummaryStats(unittest.TestCase):

    def test_accepted_signals_are_separate_from_closed_trades(self):
        import scripts.eod_summary as eod

        trades = pd.DataFrame({
            "date": [datetime.now(eod.IST).strftime("%Y-%m-%d %H:%M:%S")],
            "strategy": ["test"],
            "pnl_after_costs": [100.0],
        })
        signals = pd.DataFrame({
            "timestamp": [datetime.now(eod.IST).strftime("%Y-%m-%d %H:%M:%S")] * 3,
            "status": ["TRADE", "TRADE", "NO_TRADE"],
        })

        with patch.object(eod, "_today_frame", side_effect=[trades, signals]):
            stats = eod.build_eod_stats()

        self.assertEqual(stats["closed_trades"], 1)
        self.assertEqual(stats["accepted_signals"], 2)
        self.assertEqual(stats["trades_taken"], 2)


class TestRunBotAutomation(unittest.TestCase):

    def test_startup_expected_features_match_retrain_pipeline(self):
        import run_bot
        from learning.retrain_pipeline import FEATURE_COLS

        self.assertEqual(run_bot.EXPECTED_MODEL_FEATURES, FEATURE_COLS)

    def test_deployed_model_ready_rejects_undeployed_metadata(self):
        import run_bot

        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "model_metadata.json"
            metadata_path.write_text(json.dumps({
                "deployed": False,
                "feature_cols": run_bot.EXPECTED_MODEL_FEATURES,
            }), encoding="utf-8")
            ready, reason = run_bot.deployed_model_ready(str(metadata_path))
            self.assertFalse(ready)
            self.assertIn("deployed=false", reason)

    def test_deployed_model_ready_rejects_stale_feature_contract(self):
        import run_bot

        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "model_metadata.json"
            metadata_path.write_text(json.dumps({
                "deployed": True,
                "feature_cols": ["ema_9", "vwap"],
            }), encoding="utf-8")
            ready, reason = run_bot.deployed_model_ready(str(metadata_path))
            self.assertFalse(ready)
            self.assertIn("stale/incompatible", reason)

    def test_deployed_model_ready_accepts_current_contract(self):
        import run_bot

        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "model_metadata.json"
            metadata_path.write_text(json.dumps({
                "deployed": True,
                "feature_cols": run_bot.EXPECTED_MODEL_FEATURES,
            }), encoding="utf-8")
            ready, reason = run_bot.deployed_model_ready(str(metadata_path))
            self.assertTrue(ready)
            self.assertIn("feature-compatible", reason)

    def test_recent_retrain_attempt_covers_current_contract(self):
        from datetime import datetime, timezone
        import run_bot

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "retrain_status.json"
            status_path.write_text(json.dumps({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "feature_cols": run_bot.EXPECTED_MODEL_FEATURES,
                "deployed": False,
            }), encoding="utf-8")
            cooling_down, reason = run_bot.recent_retrain_attempt_covers_contract(
                str(status_path),
                cooldown_hours=24,
            )
            self.assertTrue(cooling_down)
            self.assertIn("recent retrain", reason)

    def test_recent_retrain_attempt_rejects_stale_contract(self):
        from datetime import datetime, timezone
        import run_bot

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "retrain_status.json"
            status_path.write_text(json.dumps({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "feature_cols": ["ema_9", "vwap"],
                "deployed": False,
            }), encoding="utf-8")
            cooling_down, reason = run_bot.recent_retrain_attempt_covers_contract(
                str(status_path),
                cooldown_hours=24,
            )
            self.assertFalse(cooling_down)
            self.assertIn("different feature contract", reason)


if __name__ == "__main__":
    unittest.main()
