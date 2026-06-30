"""
scripts/eod_summary.py
----------------------
Generate the one authoritative End-of-Day email summary.

This script is run by bot_daemon.py and/or Windows Task Scheduler after 17:06 IST.
It reads data/trade_journal.csv directly so it includes both OrderManager trades
and shared PaperEngine trades.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from notifications.email_notifier import EmailNotifier

IST = ZoneInfo("Asia/Kolkata")
JOURNAL_PATH = ROOT / "data" / "trade_journal.csv"
SIGNAL_PATH = ROOT / "data" / "signal_log.csv"
STATE_PATH = ROOT / "data" / "eod_summary_sent.json"


def _today_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(path, on_bad_lines="skip")
    if df.empty:
        return pd.DataFrame()

    date_col = "date" if "date" in df.columns else "timestamp" if "timestamp" in df.columns else None
    if date_col is None:
        return pd.DataFrame()

    dates = pd.to_datetime(df[date_col], errors="coerce")
    today = datetime.now(IST).date()
    return df.loc[dates.dt.date == today].copy()


def _max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    peak = equity.expanding(min_periods=1).max()
    return float((peak - equity).max())


def _stats_for_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
        }

    pnl_col = "pnl_after_costs" if "pnl_after_costs" in df.columns else "pnl_inr"
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    return {
        "net_pnl": float(pnl.sum()),
        "win_rate": float((len(wins) / len(pnl)) * 100) if len(pnl) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "max_drawdown": _max_drawdown(pnl),
        "total_trades": int(len(df)),
    }


def build_eod_stats() -> dict:
    trades = _today_frame(JOURNAL_PATH)
    stats = _stats_for_frame(trades)

    by_strategy = {}
    if not trades.empty and "strategy" in trades.columns:
        for strategy, group in trades.groupby("strategy", dropna=False):
            name = str(strategy or "Unknown")
            by_strategy[name] = _stats_for_frame(group)
    stats["by_strategy"] = by_strategy

    signals = _today_frame(SIGNAL_PATH)
    stats["signals_evaluated"] = int(len(signals))
    accepted_signals = int((signals.get("status", pd.Series(dtype=str)) == "TRADE").sum()) if not signals.empty else 0
    stats["accepted_signals"] = accepted_signals
    stats["closed_trades"] = int(stats.get("total_trades", 0))
    # Backwards-compatible alias for older templates. This is accepted signals, not broker executions.
    stats["trades_taken"] = accepted_signals
    return stats


def _sent_today() -> bool:
    if not STATE_PATH.exists():
        return False
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("sent_date") == datetime.now(IST).date().isoformat()


def _mark_sent() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sent_date": datetime.now(IST).date().isoformat(),
        "sent_at": datetime.now(IST).isoformat(),
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send the NSE bot EOD summary email.")
    parser.add_argument("--dry-run", action="store_true", help="Print computed stats without sending email.")
    parser.add_argument("--force", action="store_true", help="Send even if today's EOD marker already exists.")
    args = parser.parse_args()

    logger.info("Generating EOD Summary from trade_journal.csv...")
    stats = build_eod_stats()
    if args.dry_run:
        print(json.dumps(stats, indent=2, default=str))
        return 0

    if _sent_today() and not args.force:
        logger.info("EOD Summary already sent today. Skipping duplicate email.")
        return 0

    notifier = EmailNotifier()
    if not notifier.enabled:
        logger.warning("Email notifications are disabled in .env. Skipping EOD email.")
        return 1

    if notifier.send_eod_summary(stats):
        _mark_sent()
        logger.success("EOD Summary email sent successfully.")
        return 0

    logger.error("Failed to send EOD Summary email.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
