"""
bot_daemon.py
-------------
Watchdog daemon for the NSE Trading Bot.

Features:
  - Auto-starts the bot when market opens (09:10 IST, Mon-Fri)
  - Auto-restarts the bot within seconds of a crash (with exponential backoff)
  - Stops the bot cleanly at market close (17:05 IST)
  - Guards against starting during 15:20-17:05 (equity already closed) â€” 
    waits until next morning instead of letting it run pointlessly
  - Writes a heartbeat file every 60 s so external monitors can detect a dead daemon
  - Logs every crash with exit code + timestamp to logs/daemon.log
  - Sends a Telegram alert on repeated crashes (if configured in .env)
  - Sends EOD summary once per day after 17:06
"""

import os
import subprocess
import sys
import time
import logging
if os.name == "nt":
    import msvcrt
else:
    import fcntl
from datetime import datetime, time as dtime, date
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BOT_DIR = Path(__file__).parent.resolve()
os.chdir(BOT_DIR)
PYTHON_EXE = Path(sys.executable)
if PYTHON_EXE.name.lower() == "pythonw.exe":
    PYTHON_EXE = PYTHON_EXE.with_name("python.exe")

LOG_DIR = BOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
BOT_OUTPUT_LOG = LOG_DIR / "bot_launcher.log"
DAEMON_LOCK_FILE = LOG_DIR / "bot_daemon.lock"

HEARTBEAT_FILE = BOT_DIR / "data" / "daemon_heartbeat.txt"
HEARTBEAT_FILE.parent.mkdir(exist_ok=True)

IST = ZoneInfo("Asia/Kolkata")

# Market window (IST)
MARKET_START   = dtime(9, 10)
MARKET_END     = dtime(17, 5)

# Don't bother starting if equity is already closed and only currency remains â€”
# if launched after this threshold the bot will immediately EOD-square and idle.
LATE_START_WARN = dtime(15, 20)

# How long (seconds) to wait between restart attempts â€” exponential back-off
BACKOFF_BASE    = 10     # first retry after 10 s
BACKOFF_MAX     = 300    # cap at 5 minutes
BACKOFF_RESET   = 600    # if bot ran for >10 min before crashing â†’ reset backoff

# Max consecutive crashes before skipping until next day
MAX_CONSECUTIVE_CRASHES = 6

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAEMON] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "daemon.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("daemon")

# ---------------------------------------------------------------------------
# Optional Telegram alert (reads from environment â€” no crash if unconfigured)
# ---------------------------------------------------------------------------
def _send_telegram(msg: str):
    try:
        from dotenv import load_dotenv
        load_dotenv(BOT_DIR / ".env")
        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id or "mock" in token:
            return
        import urllib.request, urllib.parse, json as _json
        payload = urllib.parse.urlencode({"chat_id": chat_id, "text": f"ðŸ¤– TradingBot Daemon\n{msg}"}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Telegram failure must never crash the daemon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_ist() -> datetime:
    return datetime.now(IST)


def is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5   # Mon=0 â€¦ Fri=4


def is_market_hours(dt: datetime) -> bool:
    return is_weekday(dt) and MARKET_START <= dt.time() < MARKET_END


def write_heartbeat(status: str = "running"):
    try:
        HEARTBEAT_FILE.write_text(
            f"{now_ist().isoformat()}  status={status}\n", encoding="utf-8"
        )
    except Exception:
        pass


def kill_existing_bot():
    """Kill any stale main.py / run_bot.py processes from a previous session."""
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if (
                    any("python" in c.lower() for c in cmdline)
                    and any(c in ("main.py", "run_bot.py") for c in cmdline)
                    and proc.pid != os.getpid()
                ):
                    log.info(f"Killing stale bot process PID {proc.pid}")
                    proc.terminate()
                    time.sleep(1)
                    try:
                        proc.kill()
                    except Exception:
                        pass
            except Exception:
                pass
    except ImportError:
        pass  # psutil not installed â€” skip cleanup


# ---------------------------------------------------------------------------
# Daemon process guard
# ---------------------------------------------------------------------------
def acquire_daemon_lock():
    """Keep scheduled tasks from running multiple watchdogs at the same time."""
    lock_handle = open(DAEMON_LOCK_FILE, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.warning("Another bot_daemon.py instance is already running. Exiting duplicate daemon.")
        lock_handle.close()
        return None

    return lock_handle


def is_external_termination(exit_code: int | None) -> bool:
    """Clean shutdowns and Windows terminate/SIGTERM are not crashes."""
    return exit_code in {0, 15, -15}


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------
def run_daemon():
    daemon_lock = acquire_daemon_lock()
    if daemon_lock is None:
        return

    log.info("=" * 60)
    log.info("NSE Trading Bot Watchdog Daemon â€” STARTED")
    log.info(f"Bot directory : {BOT_DIR}")
    log.info(f"Market window : {MARKET_START} â€“ {MARKET_END} IST (weekdays)")
    log.info("=" * 60)

    bot_proc: subprocess.Popen | None = None
    bot_log_handle = None
    bot_start_time: float | None = None
    last_eod_date: date | None = None

    consecutive_crashes = 0
    skip_until_tomorrow: date | None = None
    backoff_seconds = BACKOFF_BASE

    kill_existing_bot()

    while True:
        try:
            write_heartbeat("running")
            dt = now_ist()

            # ----------------------------------------------------------------
            # INSIDE market hours
            # ----------------------------------------------------------------
            if is_market_hours(dt):

                # Reset the daily crash-skip flag when a new day begins
                if skip_until_tomorrow is not None and dt.date() > skip_until_tomorrow:
                    log.info("New trading day â€” resetting crash counters.")
                    skip_until_tomorrow = None
                    consecutive_crashes = 0
                    backoff_seconds = BACKOFF_BASE

                # Circuit-break: too many crashes in a row â†’ wait until tomorrow
                if skip_until_tomorrow is not None and dt.date() <= skip_until_tomorrow:
                    log.warning(f"Crash limit hit. Skipping rest of trading day ({dt.date()}).")
                    time.sleep(60)
                    continue

                # Late start guard: if past 15:20 equity is already done â†’ don't bother
                if dt.time() >= LATE_START_WARN and (bot_proc is None or bot_proc.poll() is not None):
                    log.warning(
                        f"Current time {dt.strftime('%H:%M')} IST is past equity cutoff (15:20). "
                        "Skipping bot start until next session."
                    )
                    time.sleep(60)
                    continue

                # Bot is not running (never started, or just crashed)
                if bot_proc is None or bot_proc.poll() is not None:
                    exit_code = bot_proc.poll() if bot_proc else None

                    if exit_code is not None:
                        if bot_log_handle is not None:
                            try:
                                bot_log_handle.close()
                            except Exception:
                                pass
                            bot_log_handle = None
                        # It was running before â€” it crashed
                        external_termination = is_external_termination(exit_code)
                        if external_termination:
                            log.warning(
                                f"Bot was externally terminated with code {exit_code}; "
                                "restarting without incrementing crash counter."
                            )
                        else:
                            consecutive_crashes += 1
                        if not external_termination:
                            log.error(
                                f"Bot exited with code {exit_code} "
                                f"(crash #{consecutive_crashes}). "
                                f"Restarting in {backoff_seconds}s..."
                            )

                        if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                            log.critical(
                                f"Bot crashed {consecutive_crashes} times. "
                                "Giving up for today. Check logs/daemon.log."
                            )
                            _send_telegram(
                                f"âŒ Bot crashed {consecutive_crashes} times in a row on {dt.date()}.\n"
                                "Auto-restart SUSPENDED. Check daemon.log immediately."
                            )
                            skip_until_tomorrow = dt.date()
                            bot_proc = None
                            time.sleep(60)
                            continue

                        if not external_termination:
                            _send_telegram(
                                f"Bot crashed (exit {exit_code}) at {dt.strftime('%H:%M')} IST.\n"
                                f"Restart #{consecutive_crashes} in {backoff_seconds}s"
                            )
                        if external_termination:
                            time.sleep(BACKOFF_BASE)
                        else:
                            time.sleep(backoff_seconds)
                            # Exponential back-off, capped at BACKOFF_MAX
                            backoff_seconds = min(backoff_seconds * 2, BACKOFF_MAX)

                    else:
                        # First start of the day
                        log.info(f"Market is open ({dt.strftime('%H:%M')} IST). Starting botâ€¦")

                    log.info(
                        "Launching: python run_bot.py --skip-history-sync  "
                        f"(skip-preflight={consecutive_crashes > 0})"
                    )
                    cmd = [str(PYTHON_EXE), "run_bot.py", "--skip-history-sync"]
                    if consecutive_crashes > 0:
                        # After a crash, skip the slow preflight so recovery is fast
                        cmd.append("--skip-preflight")

                    bot_log_handle = open(BOT_OUTPUT_LOG, "a", encoding="utf-8", buffering=1)
                    bot_log_handle.write(
                        f"\n{'=' * 70}\n"
                        f"{dt.isoformat()} launching: {' '.join(cmd)}\n"
                    )
                    bot_proc = subprocess.Popen(
                        cmd,
                        cwd=str(BOT_DIR),
                        stdout=bot_log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    bot_start_time = time.monotonic()
                    log.info(f"Bot PID {bot_proc.pid} launched.")
                    _send_telegram(f"âœ… Bot (re)started at {dt.strftime('%H:%M')} IST (PID {bot_proc.pid}).")

                else:
                    # Bot is alive â€” check if it ran long enough to reset backoff
                    if consecutive_crashes > 0:
                        uptime = time.monotonic() - (bot_start_time or time.monotonic())
                        if uptime >= BACKOFF_RESET:
                            log.info(f"Bot stable for >{BACKOFF_RESET}s â€” resetting crash counters.")
                            consecutive_crashes = 0
                            backoff_seconds = BACKOFF_BASE

            # ----------------------------------------------------------------
            # OUTSIDE market hours
            # ----------------------------------------------------------------
            else:
                # Stop bot if still running
                if bot_proc is not None and bot_proc.poll() is None:
                    log.info(f"Market closed ({dt.strftime('%H:%M')} IST). Stopping botâ€¦")
                    bot_proc.terminate()
                    try:
                        bot_proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        log.warning("Bot did not stop in 15 s â€” killing.")
                        bot_proc.kill()
                    bot_proc = None
                    if bot_log_handle is not None:
                        try:
                            bot_log_handle.close()
                        except Exception:
                            pass
                        bot_log_handle = None
                    log.info("Bot stopped cleanly.")
                    _send_telegram(f"ðŸ”´ Bot stopped at market close ({dt.strftime('%H:%M')} IST).")

                # Reset counters at night so tomorrow starts fresh
                if dt.time() < MARKET_START:
                    if consecutive_crashes > 0:
                        log.info("Pre-market: resetting crash counters for new day.")
                        consecutive_crashes = 0
                        backoff_seconds = BACKOFF_BASE
                        skip_until_tomorrow = None
                    write_heartbeat("pre-market")

                # EOD summary (once per weekday, after 17:06)
                if is_weekday(dt) and dt.time() >= dtime(17, 6):
                    if last_eod_date != dt.date():
                        log.info("Triggering EOD summaryâ€¦")
                        eod_script = BOT_DIR / "scripts" / "eod_summary.py"
                        if eod_script.exists():
                            subprocess.run([str(PYTHON_EXE), str(eod_script)], cwd=str(BOT_DIR))
                        last_eod_date = dt.date()

            time.sleep(30)

        except KeyboardInterrupt:
            log.info("Daemon interrupted by user (Ctrl+C). Shutting downâ€¦")
            if bot_proc is not None and bot_proc.poll() is None:
                log.info("Stopping bot processâ€¦")
                bot_proc.terminate()
                bot_proc.wait()
            write_heartbeat("stopped")
            break

        except Exception as exc:
            log.exception(f"Unexpected daemon error: {exc}")
            time.sleep(60)


if __name__ == "__main__":
    run_daemon()
