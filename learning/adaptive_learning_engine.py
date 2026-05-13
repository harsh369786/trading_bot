import json
import os
from loguru import logger
from datetime import datetime

class AdaptiveLearningEngine:
    """
    Module 10: Nightly threshold tuning based on recent performance.
    """
    def __init__(self, params_path: str = "config/adaptive_params.json"):
        self.params_path = params_path
        self.default_params = {
            "min_quant_score": 70,
            "min_adx": 20,
            "max_sl_paise": 20
        }

    def tune_parameters(self, stats: dict):
        """
        Adjust thresholds if performance drops.
        Safety Bounds: Module 10 logic.
        """
        if not stats: return
        
        # Load current params
        params = self.default_params
        if os.path.exists(self.params_path):
            with open(self.params_path, 'r') as f:
                params = json.load(f)
        
        reason = "Performance stable"
        win_rate = float(stats.get("win_rate", "0%").replace("%", ""))
        
        # Logic: If win rate < 50%, tighten quality threshold
        if win_rate < 50 and stats.get("total_trades", 0) >= 15:
            params["min_quant_score"] = min(params["min_quant_score"] + 2, 85)
            params["min_adx"] = min(params["min_adx"] + 2, 30)
            reason = f"Win rate low ({win_rate}%). Tightened thresholds."
            
        # Save updated params
        with open(self.params_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        # Log to parameter_history.csv
        self._log_history(params, reason)
        logger.info(f"Adaptive Learning: {reason}")

    def _log_history(self, params: dict, reason: str):
        history_path = "data/parameter_history.csv"
        file_exists = os.path.exists(history_path)
        with open(history_path, 'a') as f:
            if not file_exists:
                f.write("timestamp,params,reason\n")
            f.write(f"{datetime.now()},{json.dumps(params)},{reason}\n")
