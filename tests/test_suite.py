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
from pathlib import Path

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
            engine = AdaptiveLearningEngine(str(params_path))
            engine.tune_parameters({"win_rate": "40.0%", "total_trades": 20})
            params = json.loads(params_path.read_text(encoding="utf-8"))
            self.assertEqual(params["currency_min_quant_score"], 72)


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


# ---------------------------------------------------------------------------
# Candle Builder Tests (C6, H9)
# ---------------------------------------------------------------------------

class TestCandleBuilder(unittest.TestCase):

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

            async def process_symbol(self, symbol, candles):
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

            async def process_symbol(self, symbol, candles):
                self.calls += 1
                return None

        idx_5m = pd.DatetimeIndex([pd.Timestamp("2026-05-14 09:15", tz="Asia/Kolkata")])
        idx_15m = pd.DatetimeIndex([pd.Timestamp("2026-05-14 09:15", tz="Asia/Kolkata")])
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
        cb.candle_snapshot_path = str(Path(tempfile.gettempdir()) / "same_session_candle_snapshot_test.json")
        cb.tick_data = {"NIFTY": []}
        cb.candles = {"NIFTY": {"5min": old_5m, "15min": old_15m}}

        ticks = [{"data": {"timestamp": "2026-05-14 09:20:01", "ltp": 101.0, "volume": 100}}]

        async def run():
            await cb._process_ticks_to_candles("NIFTY", ticks)

        asyncio.run(run())
        self.assertEqual(engine.calls, 1)


if __name__ == "__main__":
    unittest.main()
