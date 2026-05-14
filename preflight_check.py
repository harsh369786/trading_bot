"""
Preflight validation for the NSE paper-trading bot.

Run this before starting the bot:

    python preflight_check.py

Optional:

    python preflight_check.py --skip-redis
    python preflight_check.py --run-tests

Exit code:
    0 = no blocking errors
    1 = at least one blocking error
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
SIGNAL_PATH = ROOT / "data" / "signal_log.csv"
TRADE_PATH = ROOT / "data" / "trade_journal.csv"
PAPER_ORDERS_PATH = ROOT / "data" / "paper_orders.json"
CANDLE_SNAPSHOT_PATH = ROOT / "data" / "candle_snapshot.json"


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Preflight:
    def __init__(self) -> None:
        self.results: list[Check] = []
        self.config: dict[str, Any] = {}

    def ok(self, name: str, detail: str = "") -> None:
        self.results.append(Check(name, "OK", detail))

    def warn(self, name: str, detail: str) -> None:
        self.results.append(Check(name, "WARN", detail))

    def fail(self, name: str, detail: str) -> None:
        self.results.append(Check(name, "FAIL", detail))

    def run(self, skip_redis: bool, run_tests: bool) -> int:
        os.chdir(ROOT)
        self.check_structure()
        self.check_python_syntax()
        self.check_requirements()
        self.check_critical_imports()
        self.check_config()
        self.check_live_safety()
        self.check_logs_and_dashboard_data()
        self.check_trade_journal_math()
        self.check_active_order_state()
        self.check_historical_warmup()
        self.check_core_component_contracts()
        if not skip_redis:
            asyncio.run(self.check_redis())
        if run_tests:
            self.run_unit_tests()
        return self.report()

    def check_structure(self) -> None:
        required = [
            "main.py",
            "requirements.txt",
            "config/config.yaml",
            "pipeline/websocket_feed.py",
            "pipeline/redis_queue.py",
            "pipeline/candle_builder.py",
            "execution/order_manager.py",
            "execution/paper_engine.py",
            "risk/risk_engine.py",
            "dashboard/app.py",
            "dashboard/data_loader.py",
        ]
        missing = [p for p in required if not (ROOT / p).exists()]
        if missing:
            self.fail("Project structure", f"Missing required files: {missing}")
        else:
            self.ok("Project structure", "Required runtime files exist.")

    def check_python_syntax(self) -> None:
        skip_dirs = {".git", "__pycache__", ".pytest_cache"}
        runtime_roots = {
            "agents", "api", "backtesting", "config", "dashboard", "execution",
            "features", "learning", "models", "pipeline", "reporting", "risk",
            "scheduler", "strategies", "tests", "tracking", "utils",
        }
        root_runtime_files = {"main.py", "run_bot.py", "preflight_check.py", "train_rsmb.py"}
        failed = []
        for path in ROOT.rglob("*.py"):
            if any(part in skip_dirs for part in path.parts):
                continue
            rel = path.relative_to(ROOT)
            if len(rel.parts) == 1 and rel.name not in root_runtime_files:
                continue
            if len(rel.parts) > 1 and rel.parts[0] not in runtime_roots:
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
            except Exception as exc:
                failed.append(f"{rel}: {exc}")
        if failed:
            self.fail("Python syntax", "\n".join(failed[:20]))
        else:
            self.ok("Python syntax", "Runtime Python files parse without writing .pyc files.")

    def check_requirements(self) -> None:
        req_path = ROOT / "requirements.txt"
        if not req_path.exists():
            self.fail("Dependencies", "requirements.txt missing.")
            return

        import_names = {
            "python-dotenv": "dotenv",
            "pyyaml": "yaml",
            "scikit-learn": "sklearn",
            "smartapi-python": "SmartApi",
            "python-telegram-bot": "telegram",
        }
        missing = []
        for raw in req_path.read_text(encoding="utf-8").splitlines():
            pkg = raw.strip()
            if not pkg or pkg.startswith("#"):
                continue
            base = pkg.split("==")[0].split(">=")[0].split("<")[0].strip()
            module = import_names.get(base, base.replace("-", "_"))
            if importlib.util.find_spec(module) is None:
                missing.append(f"{base} (import {module})")
        if missing:
            self.fail("Dependencies", "Missing packages: " + ", ".join(missing))
        else:
            self.ok("Dependencies", "All requirements import successfully.")

    def check_critical_imports(self) -> None:
        modules = [
            "main",
            "pipeline.websocket_feed",
            "pipeline.redis_queue",
            "pipeline.candle_builder",
            "strategies.equity_signal_engine",
            "agents.pipeline",
            "risk.risk_engine",
            "execution.order_manager",
            "execution.paper_engine",
            "tracking.trade_lifecycle_tracker",
            "tracking.signal_logger",
            "dashboard.data_loader",
            "strategies.rsmb.strategy",
        ]
        failed = []
        for module in modules:
            try:
                importlib.import_module(module)
            except Exception as exc:
                failed.append(f"{module}: {type(exc).__name__}: {exc}")
        if failed:
            self.fail("Critical imports", "\n".join(failed))
        else:
            self.ok("Critical imports", "Main pipeline modules import cleanly.")

    def check_config(self) -> None:
        path = ROOT / "config" / "config.yaml"
        try:
            self.config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self.fail("Config parse", f"Could not read config/config.yaml: {exc}")
            return

        instruments = self.config.get("instruments", {})
        equity = instruments.get("equity", [])
        currency = instruments.get("currency", [])
        if not equity:
            self.fail("Config instruments", "No equity instruments configured.")
        elif not currency:
            self.warn("Config instruments", "No currency instruments configured.")
        else:
            self.ok("Config instruments", f"{len(equity)} equity, {len(currency)} currency symbols.")

        if any(sym == "TATAMOTORS" for sym in equity):
            self.fail("Instrument aliases", "Use TMPV, not obsolete TATAMOTORS, in config instruments.")
        else:
            self.ok("Instrument aliases", "No obsolete TATAMOTORS config symbol found.")

        risk = self.config.get("risk", {})
        capital = self.config.get("capital", {})
        if float(risk.get("daily_loss_limit_r", 0) or 0) <= 0:
            self.fail("Risk config", "risk.daily_loss_limit_r must be positive.")
        elif float(capital.get("risk_per_trade_pct", 0) or 0) <= 0:
            self.fail("Risk config", "capital.risk_per_trade_pct must be positive.")
        else:
            self.ok("Risk config", "Daily loss and per-trade risk controls are configured.")

    def check_live_safety(self) -> None:
        mode = os.environ.get("TRADING_MODE", "paper").strip().lower()
        paper_mode = bool(self.config.get("paper_mode", True))
        if mode == "live" and not paper_mode:
            self.fail("Trading mode safety", "LIVE mode is active. Preflight refuses to approve live trading.")
        elif not paper_mode:
            self.fail("Trading mode safety", "config.paper_mode is false without approved live mode.")
        else:
            self.ok("Trading mode safety", f"Paper mode enforced. TRADING_MODE={mode!r}, paper_mode={paper_mode}.")

    def check_logs_and_dashboard_data(self) -> None:
        try:
            from dashboard.data_loader import load_csv_safely
        except Exception as exc:
            self.fail("Dashboard loader", f"Cannot import dashboard loader: {exc}")
            return

        for name, path in [("Signal log", SIGNAL_PATH), ("Trade journal", TRADE_PATH)]:
            df, err = load_csv_safely(str(path))
            if err:
                self.fail(name, err)
            elif not path.exists():
                self.warn(name, f"{path.relative_to(ROOT)} is missing; dashboard will be empty until runtime writes it.")
            else:
                self.ok(name, f"Readable with {len(df)} rows.")

        if TRADE_PATH.exists():
            df, _ = load_csv_safely(str(TRADE_PATH))
            required = {"date", "symbol", "side", "strategy", "entry_price", "exit_price", "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence"}
            missing = required - set(df.columns)
            if missing:
                self.fail("Trade journal schema", f"Missing columns after loader normalization: {sorted(missing)}")
            else:
                self.ok("Trade journal schema", "Closed-trade columns are dashboard-compatible.")

        if not CANDLE_SNAPSHOT_PATH.exists():
            self.warn("Candle snapshot", "No data/candle_snapshot.json yet; 15-minute candles appear after bot processes ticks.")
        else:
            try:
                data = json.loads(CANDLE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
                symbols = data.get("symbols", {}) if isinstance(data, dict) else {}
                with_15m = [sym for sym, by_tf in symbols.items() if by_tf.get("15min")]
                if with_15m:
                    self.ok("Candle snapshot", f"15-minute candles available for {len(with_15m)} symbols.")
                else:
                    self.warn("Candle snapshot", "Snapshot exists but has no 15-minute candle rows yet.")
            except Exception as exc:
                self.fail("Candle snapshot", f"Invalid candle snapshot JSON: {exc}")

    def check_trade_journal_math(self) -> None:
        if not TRADE_PATH.exists():
            self.warn("Trade journal math", "No trade_journal.csv yet.")
            return
        from dashboard.data_loader import load_csv_safely

        df, err = load_csv_safely(str(TRADE_PATH))
        if err or df.empty:
            self.warn("Trade journal math", err or "No closed trades to validate.")
            return

        cost_per_order = float(self.config.get("execution", {}).get("cost_per_order_inr", 22.0))
        bad = []
        for i, row in df.iterrows():
            strategy = str(row.get("strategy", "")).lower()
            if strategy != "rsmb":
                continue
            try:
                entry = float(row["entry_price"])
                exit_price = float(row["exit_price"])
                qty = int(float(row["qty"]))
                pnl = float(row["pnl_inr"])
                net = float(row["pnl_after_costs"])
            except Exception:
                bad.append(f"row {i}: non-numeric trade fields")
                continue

            side = str(row.get("side", "")).upper()
            expected_gross = (exit_price - entry) * qty if side == "BUY" else (entry - exit_price) * qty
            expected_net = expected_gross - (cost_per_order * 2)
            if abs(pnl - expected_gross) > 0.05 or abs(net - expected_net) > 0.05:
                bad.append(
                    f"row {i} {row.get('symbol')} {side}: expected gross={expected_gross:.2f}, "
                    f"net={expected_net:.2f}; got gross={pnl:.2f}, net={net:.2f}"
                )

        if bad:
            self.fail("Trade journal math", "\n".join(bad[:20]))
        else:
            self.ok("Trade journal math", "RSMB closed-trade gross/net P&L is consistent.")

    def check_active_order_state(self) -> None:
        rows = []
        if PAPER_ORDERS_PATH.exists():
            try:
                loaded = json.loads(PAPER_ORDERS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    rows.extend(loaded)
                else:
                    self.fail("Paper active orders", "data/paper_orders.json must contain a JSON list.")
                    return
            except Exception as exc:
                self.fail("Paper active orders", f"Invalid JSON: {exc}")
                return

        required = {"order_id", "symbol", "side", "entry", "sl", "target", "qty", "status"}
        bad = []
        seen_symbols = set()
        for row in rows:
            if not isinstance(row, dict):
                bad.append("Non-object row in paper_orders.json")
                continue
            missing = required - set(row)
            if missing:
                bad.append(f"{row.get('order_id', '<unknown>')}: missing {sorted(missing)}")
            symbol = row.get("symbol")
            if symbol in seen_symbols:
                bad.append(f"{symbol}: duplicate active paper order")
            seen_symbols.add(symbol)
            try:
                entry = float(row["entry"])
                sl = float(row["sl"])
                target = float(row["target"])
                qty = int(float(row["qty"]))
                side = str(row["side"]).upper()
                if qty <= 0 or not all(math.isfinite(v) for v in [entry, sl, target]):
                    bad.append(f"{symbol}: invalid numeric order fields")
                if side == "BUY" and not (sl < entry < target):
                    bad.append(f"{symbol}: invalid BUY SL/target relation")
                if side == "SELL" and not (target < entry < sl):
                    bad.append(f"{symbol}: invalid SELL SL/target relation")
            except Exception as exc:
                bad.append(f"{symbol}: invalid active order fields: {exc}")

        if bad:
            self.fail("Paper active orders", "\n".join(bad[:20]))
        else:
            self.ok("Paper active orders", f"{len(rows)} active paper order(s) validated.")

    def check_historical_warmup(self) -> None:
        instruments = self.config.get("instruments", {})
        symbols = instruments.get("equity", []) + instruments.get("currency", [])
        if not symbols:
            return
        missing = []
        empty = []
        for symbol in symbols:
            path = ROOT / "data" / "historical" / f"{symbol}_6m.parquet"
            if not path.exists():
                missing.append(symbol)
                continue
            try:
                if pd.read_parquet(path, columns=["close"]).empty:
                    empty.append(symbol)
            except Exception as exc:
                empty.append(f"{symbol} ({exc})")
        if empty:
            self.fail("Historical warm-up", f"Unreadable/empty parquet files: {empty}")
        elif missing:
            self.warn("Historical warm-up", f"Missing warm-up files: {missing}")
        else:
            self.ok("Historical warm-up", "All configured symbols have readable parquet warm-up files.")

    def check_core_component_contracts(self) -> None:
        try:
            from execution.order_manager import OrderManager
            from execution.paper_engine import PaperEngine
            from strategies.base_strategy import Signal
            from tracking.signal_logger import SignalLogger
            from dashboard.data_loader import load_csv_safely
        except Exception as exc:
            self.fail("Core contracts", f"Could not import core classes: {exc}")
            return

        try:
            mgr = OrderManager({"paper_mode": True, "instruments": {"equity": ["NIFTY"], "currency": []}})
            normalized = mgr._normalize_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 99, "target": 102, "qty": 1})
            if not normalized or normalized.get("target") != 102:
                self.fail("OrderManager contract", "Valid target signal was not normalized.")
            elif mgr._normalize_signal({"symbol": "NIFTY", "side": "BUY", "entry": 100, "sl": 101, "target": 102, "qty": 1}) is not None:
                self.fail("OrderManager contract", "Invalid BUY SL relation was accepted.")
            else:
                self.ok("OrderManager contract", "Signal normalization and validation behave correctly.")
        except Exception as exc:
            self.fail("OrderManager contract", str(exc))

        try:
            with tempfile.TemporaryDirectory() as tmp:
                journal = Path(tmp) / "trade_journal.csv"
                active = Path(tmp) / "paper_orders.json"
                engine = PaperEngine(cost_per_order_inr=22, journal_path=str(journal), active_orders_path=str(active))
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
                df = pd.read_csv(journal)
                gross = float(df.iloc[-1]["pnl_inr"])
                net = float(df.iloc[-1]["pnl_after_costs"])
                if gross != -592.0 or net != -636.0:
                    self.fail("PaperEngine P&L contract", f"Expected gross=-592/net=-636, got gross={gross}/net={net}")
                else:
                    self.ok("PaperEngine P&L contract", "Gross and after-cost P&L are separated correctly.")
        except Exception as exc:
            self.fail("PaperEngine P&L contract", str(exc))

        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "signal_log.csv"
                logger = SignalLogger(str(path))
                logger.log_signal("NIFTY", "BUY", 100, 99, 102, 0.9, "TRADE")
                df, err = load_csv_safely(str(path))
                if err or df.empty or df.iloc[0].get("symbol") != "NIFTY":
                    self.fail("Signal logger contract", err or "Logged signal did not round-trip.")
                else:
                    self.ok("Signal logger contract", "Signal CSV writes and dashboard loader reads.")
        except Exception as exc:
            self.fail("Signal logger contract", str(exc))

    async def check_redis(self) -> None:
        try:
            import redis.asyncio as redis
            url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
            client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
            await client.ping()
            await client.aclose()
            self.ok("Redis connectivity", f"Connected to {url}.")
        except Exception as exc:
            self.fail("Redis connectivity", f"Redis is required before main.py starts: {exc}")

    def run_unit_tests(self) -> None:
        cmd = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode == 0:
            self.ok("Unit tests", "unittest discover passed.")
        else:
            output = (proc.stdout + "\n" + proc.stderr).strip()
            self.fail("Unit tests", output[-4000:])

    def report(self) -> int:
        fail_count = sum(1 for r in self.results if r.status == "FAIL")
        warn_count = sum(1 for r in self.results if r.status == "WARN")

        print("\nNSE Bot Preflight Report")
        print("=" * 72)
        for result in self.results:
            print(f"[{result.status:4}] {result.name}")
            if result.detail:
                for line in str(result.detail).splitlines():
                    print(f"       {line}")
        print("=" * 72)
        print(f"Summary: {fail_count} failure(s), {warn_count} warning(s), {len(self.results)} checks.")
        if fail_count:
            print("Decision: DO NOT START run_bot.py/main.py. Fix the failures above first.")
            return 1
        print("Decision: Preflight passed. Paper-mode bot startup is allowed.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the trading bot before startup.")
    parser.add_argument("--skip-redis", action="store_true", help="Do not require Redis connectivity.")
    parser.add_argument("--run-tests", action="store_true", help="Run the full unittest suite too.")
    args = parser.parse_args()
    return Preflight().run(skip_redis=args.skip_redis, run_tests=args.run_tests)


if __name__ == "__main__":
    raise SystemExit(main())
