"""
run_bot.py - safe bot launcher.

Runs preflight checks, kills any running main.py instance, then starts fresh.

Usage:
    python run_bot.py

Emergency bypass only:
    python run_bot.py --skip-preflight
"""
import os
import subprocess
import sys
import time

import psutil


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


if __name__ == "__main__":
    skip_preflight = "--skip-preflight" in sys.argv
    kill_existing_bot()
    if skip_preflight:
        print("[run_bot] WARNING: preflight skipped by user flag.")
    elif not run_preflight():
        raise SystemExit(1)

    print("[run_bot] Starting main.py ...\n")
    sys.stdout.flush()
    subprocess.run([sys.executable, "-B", "main.py"], check=False)
