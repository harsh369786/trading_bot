"""
run_bot.py — Safe bot launcher.
Kills any running main.py instance before starting a fresh one.
Usage: python run_bot.py
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
            # Match python processes running main.py (but not this script)
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
        time.sleep(1.5)          # Grace period for clean shutdown
        # Force-kill stragglers
        for pid in killed:
            try:
                psutil.Process(pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        print("[run_bot] No existing bot process found.")


if __name__ == "__main__":
    kill_existing_bot()
    print("[run_bot] Starting main.py ...\n")
    # Flush so logs appear immediately
    sys.stdout.flush()
    subprocess.run([sys.executable, "main.py"], check=False)
