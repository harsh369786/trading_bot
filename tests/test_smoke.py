import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from dashboard.data_loader import load_csv_safely
from execution.order_manager import OrderManager
from risk.risk_engine import RiskEngine
from tracking.signal_logger import SignalLogger
from tracking.trade_lifecycle_tracker import TradeLifecycleTracker
from features.price_features import PriceFeatures
from features.volume_features import VolumeFeatures


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


class SmokeTests(unittest.TestCase):
    def test_risk_engine_position_sizing(self):
        engine = RiskEngine({"capital": {"equity_total": 50000, "risk_per_trade_pct": 1.0}})
        self.assertEqual(engine.get_equity_position_size(100.0, 99.0), 500)
        self.assertEqual(engine.get_equity_position_size(100.0, 100.0), 0)

    def test_signal_schema_validation_accepts_target_aliases(self):
        manager = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []}})
        signal = manager._normalize_signal({
            "symbol": "NIFTY",
            "side": "BUY",
            "entry": 100,
            "sl": 99,
            "t1": 102,
            "qty": 1,
        })
        self.assertIsNotNone(signal)
        self.assertEqual(signal["target"], 102.0)
        self.assertEqual(signal["t1"], 102.0)

    def test_order_manager_paper_order_placement(self):
        async def run():
            redis = FakeRedis()
            manager = OrderManager({
                "paper_mode": True,
                "instruments": {"equity": ["NIFTY"], "currency": []},
                "capital": {"max_open_trades_equity": 2},
            }, redis)
            await manager.execute_signal({
                "symbol": "NIFTY",
                "side": "BUY",
                "entry": 100,
                "sl": 99,
                "target": 102,
                "qty": 1,
            })
            active = json.loads(redis.store[manager.KEY_ACTIVE])
            self.assertEqual(len(active), 1)
            self.assertEqual(next(iter(active.values()))["status"], "PROTECTED")

        asyncio.run(run())

    def test_startup_reconciliation_drops_stale_paper_pending_orders(self):
        async def run():
            redis = FakeRedis()
            manager = OrderManager({"paper_mode": True, "instruments": {"equity": ["HDFCBANK"], "currency": []}}, redis)
            await redis.set(manager.KEY_ACTIVE, json.dumps({
                "MOCK-OLD": {
                    "symbol": "HDFCBANK",
                    "side": "BUY",
                    "entry": 755.65,
                    "sl": 748.09,
                    "t1": 770.76,
                    "qty": 1,
                    "status": "PENDING",
                }
            }))
            await manager.reconcile_startup_state()
            self.assertEqual(json.loads(redis.store[manager.KEY_ACTIVE]), {})

        asyncio.run(run())

    def test_trade_lifecycle_tracker_detects_target(self):
        class FakeOrderManager:
            def __init__(self):
                self.active = {
                    "OID1": {
                        "symbol": "NIFTY",
                        "side": "BUY",
                        "entry": 100,
                        "sl": 99,
                        "target": 102,
                        "status": "PROTECTED",
                    }
                }
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
            manager = FakeOrderManager()
            tracker = TradeLifecycleTracker({"instruments": {"equity": ["NIFTY"], "currency": []}}, manager, FakeQueue())
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(tracker._monitor_symbol("NIFTY"), timeout=0.2)
            self.assertEqual(manager.update["status"], "TARGET_HIT")
            self.assertTrue(manager.update["is_exit"])

        asyncio.run(run())

    def test_csv_logger_write_and_dashboard_loader_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signal_log.csv"
            logger = SignalLogger(str(path))
            logger.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
            df, err = load_csv_safely(str(path))
            self.assertIsNone(err)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["symbol"], "NIFTY")

    def test_dashboard_loader_missing_file(self):
        df, err = load_csv_safely("missing-file-for-smoke-test.csv")
        self.assertIsNone(err)
        self.assertTrue(df.empty)

    def test_zero_volume_candles_do_not_create_nan_features(self):
        import pandas as pd

        idx = pd.date_range("2026-05-13 09:15", periods=60, freq="5min")
        prices = [100 + i * 0.1 for i in range(60)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [0] * 60,
        }, index=idx)
        df = PriceFeatures.add_indicators(df)
        df = VolumeFeatures.add_volume_analysis(df)
        self.assertFalse(df.tail(1)[["vwap", "rel_vol", "vol_spike_ratio"]].isna().any(axis=None))


if __name__ == "__main__":
    unittest.main()
