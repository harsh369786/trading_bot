import csv
import os
import time
from datetime import datetime
from loguru import logger

class SignalLogger:
    """
    Logs all signal generation attempts (Success & Rejections) to a CSV.
    This provides the data for the Dashboard's Signal Feed and Rejection Analytics.
    """

    def __init__(self, filepath: str = "data/signal_log.csv"):
        self.filepath = filepath
        # H5 fix: per-instance dedup map (was class-level, shared across instances)
        self._last_logged: dict = {}
        self.headers = [
            "timestamp", "symbol", "side", "strategy", "entry", "sl", "target",
            "quant_score", "status", "rejection_reason"
        ]
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filepath):
            # L3 fix: os.path.dirname may return '' if filepath has no directory part
            dir_part = os.path.dirname(self.filepath)
            if dir_part:
                os.makedirs(dir_part, exist_ok=True)
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_signal(self, symbol: str, side: str, strategy: str, entry: float, sl: float, target: float, 
                   score: float, status: str, reason: str = ""):
        """
        status: 'TRADE' or 'NO_TRADE'
        """
        try:
            dedupe_window = 240 if status == "TRADE" else 60
            dedupe_key = (symbol, side, status, reason)
            now = time.time()
            last_logged = self._last_logged.get(dedupe_key, 0)
            if now - last_logged < dedupe_window:
                logger.debug(f"Skipping duplicate signal log: {symbol} | {status} | {reason}")
                return
            self._last_logged[dedupe_key] = now

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "side": side,
                "strategy": strategy,
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "target": round(target, 2),
                "quant_score": round(score, 2),
                "status": status,
                "rejection_reason": reason
            }
            
            for attempt in range(3):
                try:
                    with open(self.filepath, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.headers)
                        writer.writerow(row)
                    break
                except PermissionError as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Signal log locked, retrying write: {e}")
                    time.sleep(0.2)
            
            logger.info(f"📝 Signal Logged: {symbol} | {status} | {reason}")
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")
