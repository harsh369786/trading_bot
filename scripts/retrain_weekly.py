import json
import msvcrt
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.retrain_pipeline import RetrainPipeline

IST = ZoneInfo("Asia/Kolkata")
LOCK_PATH = ROOT / "data" / "retrain_weekly.lock"
STATUS_PATH = ROOT / "data" / "retrain_weekly_status.json"
BACKFILL_MARKER = ROOT / "data" / "history_backfill_status.json"
LOG_PATH = ROOT / "logs" / "retrain_weekly.log"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp_path, path)


def acquire_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def load_training_config() -> dict:
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("training", {}) or {}


def configured_equity_symbols() -> list[str]:
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return list(config.get("instruments", {}).get("equity", []) or [])


def history_coverage(symbols: list[str]) -> dict:
    coverage = {}
    for symbol in symbols:
        path = ROOT / "data" / "historical" / f"{symbol}_6m.parquet"
        if not path.exists():
            coverage[symbol] = {"rows": 0, "span_days": 0}
            continue
        try:
            df = pd.read_parquet(path, columns=["close"])
            if df.empty:
                coverage[symbol] = {"rows": 0, "span_days": 0}
                continue
            index = pd.to_datetime(df.index, errors="coerce").dropna()
            span_days = max(0, int((index.max() - index.min()).total_seconds() // 86400))
            coverage[symbol] = {
                "rows": len(df),
                "span_days": span_days,
                "first": index.min(),
                "last": index.max(),
            }
        except Exception as exc:
            coverage[symbol] = {"rows": 0, "span_days": 0, "error": str(exc)}
    return coverage


def run_data_sync(full_backfill: bool, history_days: int, incremental_days: int, symbols: list[str]) -> int:
    cmd = [sys.executable, "-B", "data/fetch_angelone_historical.py", "--days"]
    cmd.append(str(history_days if full_backfill else incremental_days))
    if full_backfill:
        cmd.append("--full")
    cmd.extend(["--symbols", *symbols])
    logger.info(f"Running data sync: {' '.join(cmd)}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n{'=' * 70}\n{datetime.now(IST).isoformat()} {' '.join(cmd)}\n")
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def main() -> int:
    lock_handle = acquire_lock()
    if lock_handle is None:
        logger.warning("Weekly retraining is already running; duplicate invocation skipped.")
        return 0

    started_at = datetime.now(IST)
    status = {
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "status": "running",
        "data_sync": None,
        "training": None,
    }
    write_json_atomic(STATUS_PATH, status)

    try:
        training_cfg = load_training_config()
        history_days = int(training_cfg.get("history_days", 730))
        incremental_days = int(training_cfg.get("incremental_sync_days", 14))
        minimum_history_days = int(training_cfg.get("minimum_history_days", 365))
        symbols = configured_equity_symbols()

        before = history_coverage(symbols)
        needs_backfill = any(item.get("span_days", 0) < minimum_history_days for item in before.values())
        backfill_complete = False
        if BACKFILL_MARKER.exists():
            try:
                backfill_complete = bool(json.loads(BACKFILL_MARKER.read_text(encoding="utf-8")).get("complete"))
            except Exception:
                backfill_complete = False
        full_backfill = needs_backfill and not backfill_complete

        sync_code = run_data_sync(full_backfill, history_days, incremental_days, symbols)
        after = history_coverage(symbols)
        status["data_sync"] = {
            "return_code": sync_code,
            "full_backfill": full_backfill,
            "coverage": after,
        }
        if full_backfill:
            write_json_atomic(
                BACKFILL_MARKER,
                {
                    "attempted_at": datetime.now(IST).isoformat(),
                    "return_code": sync_code,
                    "complete": all(
                        item.get("span_days", 0) >= minimum_history_days
                        for item in after.values()
                    ),
                    "minimum_history_days": minimum_history_days,
                    "coverage": after,
                },
            )
        if sync_code != 0:
            raise RuntimeError(f"Angel One historical sync failed with exit code {sync_code}")

        pipeline = RetrainPipeline(
            data_path="data/historical/*_6m.parquet",
            status_path="data/retrain_status.json",
            symbols=symbols,
        )
        training_result = pipeline.run()
        status["training"] = training_result
        if training_result["status"] != "completed":
            raise RuntimeError(training_result.get("error", "Ensemble retraining failed"))

        status["status"] = "completed"
        if training_result.get("deployed"):
            logger.success("Weekly retraining completed and a validated ensemble was deployed.")
        else:
            logger.warning("Weekly retraining completed; candidate rejected and production models preserved.")
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        logger.exception(f"Weekly retraining failed: {exc}")
        return 1
    finally:
        status["completed_at"] = datetime.now(IST).isoformat()
        write_json_atomic(STATUS_PATH, status)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
