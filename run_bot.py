"""
run_bot.py - safe bot launcher.

Runs preflight checks, kills any running main.py instance, then starts fresh.

Usage:
    python run_bot.py

Emergency bypass only:
    python run_bot.py --skip-preflight

Optional:
    python run_bot.py --skip-history-sync
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pandas as pd
import yaml

EXPECTED_MODEL_FEATURES = [
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "rsi_14", "atr_pct",
    "dist_vwap", "ADX_14", "DMP_14", "DMN_14", "bb_pct",
]


def load_config() -> dict:
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[run_bot] Could not read config/config.yaml: {exc}")
        return {}


def configured_equity_symbols(config: dict) -> list[str]:
    return list((config.get("instruments", {}) or {}).get("equity", []) or [])


def deployed_model_ready(metadata_path: str = "models/xgboost/model_metadata.json") -> tuple[bool, str]:
    path = Path(metadata_path)
    if not path.exists():
        return False, f"metadata missing: {metadata_path}"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"metadata unreadable: {exc}"

    if not metadata.get("deployed", False):
        return False, "model metadata says deployed=false"

    feature_cols = list(metadata.get("feature_cols") or [])
    if feature_cols != EXPECTED_MODEL_FEATURES:
        return False, "model feature contract is stale/incompatible"

    return True, "model metadata is deployed and feature-compatible"


def recent_retrain_attempt_covers_contract(
    status_path: str = "data/retrain_status.json",
    cooldown_hours: float = 24.0,
) -> tuple[bool, str]:
    path = Path(status_path)
    if cooldown_hours <= 0:
        return False, "cooldown disabled"
    if not path.exists():
        return False, "no retrain status found"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"retrain status unreadable: {exc}"

    if status.get("status") != "completed":
        return False, "last retrain did not complete"
    if list(status.get("feature_cols") or []) != EXPECTED_MODEL_FEATURES:
        return False, "last retrain used a different feature contract"

    completed_raw = status.get("completed_at")
    if not completed_raw:
        return False, "last retrain has no completed_at"
    try:
        completed_at = datetime.fromisoformat(str(completed_raw).replace("Z", "+00:00"))
    except ValueError:
        return False, "last retrain completed_at is invalid"
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)

    age_hours = (datetime.now(timezone.utc) - completed_at.astimezone(timezone.utc)).total_seconds() / 3600.0
    if age_hours < cooldown_hours:
        return True, f"recent retrain attempted {age_hours:.1f}h ago; waiting {cooldown_hours:.1f}h cooldown"
    return False, f"last retrain is older than cooldown ({age_hours:.1f}h)"


def auto_retrain_if_needed() -> bool:
    config = load_config()
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    if model_cfg.get("auto_retrain_on_startup", True) is False:
        print("[run_bot] Startup auto-retrain disabled by config.")
        return True

    ready, reason = deployed_model_ready()
    if ready:
        print(f"[run_bot] AI model ready: {reason}")
        return True

    if not historical_warmup_ready():
        print(f"[run_bot] AI model not ready ({reason}), but historical data is not ready either.")
        print("[run_bot] Continuing in safe mode; undeployed/incompatible AI models remain disabled.")
        return True

    cooldown_hours = float(model_cfg.get("startup_retrain_cooldown_hours", 24))
    cooling_down, cooldown_reason = recent_retrain_attempt_covers_contract(cooldown_hours=cooldown_hours)
    if cooling_down:
        print(f"[run_bot] AI model not ready ({reason}); {cooldown_reason}.")
        print("[run_bot] Continuing in safe mode; strict AI remains disabled until a quality-gated retrain deploys.")
        return True

    print(f"[run_bot] AI model not ready: {reason}")
    print("[run_bot] Running automatic ensemble retraining before startup ...")
    cmd = [sys.executable, "-B", "scripts/retrain_ensemble.py"]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("[run_bot] Automatic retraining failed. Continuing in safe mode with strict AI disabled.")
        return True

    ready, reason = deployed_model_ready()
    if ready:
        print(f"[run_bot] Automatic retraining deployed a compatible model: {reason}")
    else:
        print(f"[run_bot] Automatic retraining completed but model is still not deployed: {reason}")
        print("[run_bot] Continuing in safe mode; strict AI remains disabled until quality gates pass.")
    return True


def kill_existing_bot():
    """Terminate any process running main.py from this directory."""
    killed = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if (
                any("python" in c.lower() for c in cmdline)
                and any("main.py" in c for c in cmdline)
                and proc.pid != os.getpid()
            ):
                proc.terminate()
                killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed:
        print(f"[run_bot] Terminated existing bot PIDs: {killed}")
        time.sleep(1.5)
        for pid in killed:
            try:
                psutil.Process(pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        print("[run_bot] No existing bot process found.")


def run_preflight() -> bool:
    """Block startup if the pipeline/data preflight finds failures."""
    print("[run_bot] Running preflight checks ...\n")
    sys.stdout.flush()
    result = subprocess.run([sys.executable, "-B", "preflight_check.py"], check=False)
    if result.returncode != 0:
        print("\n[run_bot] Preflight failed. Bot startup blocked.")
        print("[run_bot] Fix the reported failures, then run: python run_bot.py")
        print("[run_bot] Emergency bypass only: python run_bot.py --skip-preflight")
        return False
    print("\n[run_bot] Preflight passed.")
    return True


def historical_warmup_ready() -> bool:
    """Return True when configured symbols already have readable real candle files."""
    config = load_config()
    if not config:
        return False

    instruments = config.get("instruments", {})
    symbols = list(instruments.get("equity", []) or []) + list(instruments.get("currency", []) or [])
    missing = []
    for symbol in symbols:
        path = os.path.join("data", "historical", f"{symbol}_6m.parquet")
        if not os.path.exists(path):
            missing.append(symbol)
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                missing.append(symbol)
        except Exception:
            missing.append(symbol)

    if missing:
        print(f"[run_bot] Historical warm-up missing/unreadable for: {missing[:8]}")
        return False
    return True


def sync_real_intraday_history() -> bool:
    """Fetch latest Angel One 5-minute candles before paper trading starts."""
    print("[run_bot] Syncing real Angel One intraday history ...\n")
    sys.stdout.flush()
    cmd = [sys.executable, "-B", "data/fetch_angelone_historical.py", "--days", "7"]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("\n[run_bot] Historical sync failed.")
        if historical_warmup_ready():
            print("[run_bot] Existing real warm-up candles are readable; continuing startup.")
            print("[run_bot] Live Angel One ticks will update candles after startup.")
            return True
        print("[run_bot] Bot startup blocked because warm-up files are missing/unreadable.")
        print("[run_bot] Fix Angel One credentials/network/data errors, then run: python run_bot.py")
        print("[run_bot] Emergency bypass only: python run_bot.py --skip-history-sync")
        return False
    print("\n[run_bot] Real intraday history synced.")
    return True


if __name__ == "__main__":
    skip_preflight = "--skip-preflight" in sys.argv
    skip_history_sync = "--skip-history-sync" in sys.argv
    kill_existing_bot()
    if skip_history_sync:
        print("[run_bot] WARNING: Angel One history sync skipped by user flag.")
    elif not sync_real_intraday_history():
        raise SystemExit(1)

    if not auto_retrain_if_needed():
        raise SystemExit(1)

    if skip_preflight:
        print("[run_bot] WARNING: preflight skipped by user flag.")
    elif not run_preflight():
        raise SystemExit(1)

    print("[run_bot] Starting main.py ...\n")
    sys.stdout.flush()
    result = subprocess.run([sys.executable, "-B", "main.py"], check=False)
    sys.exit(result.returncode)
