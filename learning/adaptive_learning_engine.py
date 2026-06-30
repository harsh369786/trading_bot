import json
import os
from loguru import logger
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

class AdaptiveLearningEngine:
    """
    Module 10: Nightly threshold tuning based on recent performance.
    """
    def __init__(self, params_path: str = "config/adaptive_params.json", config: dict | None = None):
        self.params_path = params_path
        learning_cfg = (config or {}).get("adaptive_learning", {})
        self.min_trades_before_change = int(learning_cfg.get("min_trades_before_change", 40))
        self.default_params = {
            "min_quant_score": 70,
            "min_adx": 20,
            "max_sl_paise": 28,
        }

    def tune_parameters(self, stats: dict):
        """
        Adjust thresholds once per session day after enough closed trades.
        """
        if not stats:
            logger.info("Adaptive Learning: no trade stats available; skipping.")
            return "No trade stats available"

        params = self._load_params()
        today = datetime.now(IST).date().isoformat()
        if params.get("last_updated") == today:
            reason = f"Already tuned for {today}; skipping duplicate update."
            logger.info(f"Adaptive Learning: {reason}")
            return reason

        reason = "Performance stable"
        win_rate = float(str(stats.get("win_rate", "0%")).replace("%", ""))
        total_trades = int(stats.get("total_trades", 0) or 0)

        if total_trades < self.min_trades_before_change:
            reason = (
                f"Only {total_trades} trades today; need "
                f"{self.min_trades_before_change} before changing parameters."
            )
        elif win_rate < 50:
            quant_key = "currency_min_quant_score" if "currency_min_quant_score" in params else "min_quant_score"
            params[quant_key] = min(float(params.get(quant_key, 70)) + 2, 85)
            params["min_adx"] = min(float(params.get("min_adx", 20)) + 2, 30)
            reason = f"Win rate low ({win_rate}%). Tightened thresholds."

        params["last_updated"] = today
        os.makedirs(os.path.dirname(self.params_path) or ".", exist_ok=True)
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=4)

        self._log_history(params, reason)
        logger.info(f"Adaptive Learning: {reason}")
        return reason

    def _load_params(self) -> dict:
        params = self.default_params.copy()
        if not os.path.exists(self.params_path):
            return params
        try:
            with open(self.params_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                params.update(loaded)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not load adaptive params from {self.params_path}: {exc}")
        return params

    def _log_history(self, params: dict, reason: str):
        history_path = "data/parameter_history.csv"
        file_exists = os.path.exists(history_path)
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
        with open(history_path, "a") as f:
            if not file_exists:
                f.write("timestamp,params,reason\n")
            f.write(f"{datetime.now(IST).isoformat()},{json.dumps(params)},{reason}\n")
