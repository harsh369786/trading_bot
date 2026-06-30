import pandas as pd
from loguru import logger


class OpportunityScanner:
    """
    Agent 2: OpportunityScanner
    Scans for N-of-6 conditions to identify a valid currency setup.
    All thresholds are read from config so config.yaml is the single source of truth.
    """

    def __init__(self, config: dict):
        self.config = config
        sig_cfg = config.get("currency_signal", {})
        self.min_conditions = max(1, min(int(sig_cfg.get("conditions_required", 3)), 6))
        self.min_adx = sig_cfg.get("min_adx", 12)
        self.min_vol_ratio = sig_cfg.get("min_volume_ratio", 0.5)

    def scan(self, row: pd.Series) -> tuple[str, str]:
        """Returns ('BUY'/'SELL'/'NONE', 'StrategyName')."""
        adx = row.get("ADX_14", 0)
        mode = "TrendFollowing" if adx > self.min_adx else "MeanReversion"
        
        buy_score = self._calculate_score(row, "BUY")
        sell_score = self._calculate_score(row, "SELL")
        
        if buy_score >= self.min_conditions:
            return "BUY", mode
        if sell_score >= self.min_conditions:
            return "SELL", mode
        return "NONE", "None"

    def _calculate_score(self, row: pd.Series, side: str) -> int:
        score = 0.0
        try:
            if row.get("ADX_14", 0) > self.min_adx:
                if side == "BUY":
                    if row["ema_9"] > row["ema_21"] > row["ema_50"]:         score += 1.0
                    if 45 <= row["rsi_14"] <= 70:                             score += 1.25
                    if row.get("SUPERTd_10_3.0", 0) == 1:                    score += 0.75
                    if row["close"] > row["vwap"]:                            score += 1.25
                    if row["ADX_14"] > self.min_adx:                          score += 1.0
                    if row.get("rel_vol", 1.0) > self.min_vol_ratio:          score += 0.75
                else:  # SELL
                    if row["ema_9"] < row["ema_21"] < row["ema_50"]:         score += 1.0
                    if 30 <= row["rsi_14"] <= 55:                             score += 1.25
                    if row.get("SUPERTd_10_3.0", 0) == -1:                   score += 0.75
                    if row["close"] < row["vwap"]:                            score += 1.25
                    if row["ADX_14"] > self.min_adx:                          score += 1.0
                    if row.get("rel_vol", 1.0) > self.min_vol_ratio:          score += 0.75
            else:
                # Mean Reversion Logic (Ranging Market)
                # Weighted heavier to meet the conditions_required=4 threshold
                if side == "BUY":
                    if row.get("rsi_14", 50) < 35:                               score += 2
                    if row.get("close", 0) <= row.get("BBL_20_2.0", 0):          score += 2
                else:  # SELL
                    if row.get("rsi_14", 50) > 65:                               score += 2
                    if row.get("close", 0) >= row.get("BBU_20_2.0", 0):          score += 2
        except KeyError as e:
            logger.warning(f"Scanner: Missing feature {e}")
        return int(round(score))

    def debug_score(self, row: pd.Series) -> dict:
        """Return per-condition breakdown for logging/dashboard debug."""
        if row.get("ADX_14", 0) > self.min_adx:
            labels = ["trend", "rsi_range", "supertrend", "vwap_side", "adx", "rel_vol"]
            buy_vals = [
                row.get("ema_9", 0) > row.get("ema_21", 0) > row.get("ema_50", 0),
                45 <= row.get("rsi_14", 0) <= 70,
                row.get("SUPERTd_10_3.0", 0) == 1,
                row.get("close", 0) > row.get("vwap", 0),
                row.get("ADX_14", 0) > self.min_adx,
                row.get("rel_vol", 0) > self.min_vol_ratio,
            ]
            sell_vals = [
                row.get("ema_9", 0) < row.get("ema_21", 0) < row.get("ema_50", 0),
                30 <= row.get("rsi_14", 0) <= 55,
                row.get("SUPERTd_10_3.0", 0) == -1,
                row.get("close", 0) < row.get("vwap", 0),
                row.get("ADX_14", 0) > self.min_adx,
                row.get("rel_vol", 0) > self.min_vol_ratio,
            ]
            return {
                "BUY": {l: v for l, v in zip(labels, buy_vals)},
                "SELL": {l: v for l, v in zip(labels, sell_vals)},
                "buy_score": sum(buy_vals),
                "sell_score": sum(sell_vals),
                "min_needed": self.min_conditions,
                "mode": "TREND"
            }
        else:
            labels = ["mr_rsi", "mr_bollinger"]
            buy_vals = [
                row.get("rsi_14", 50) < 35,
                row.get("close", 0) <= row.get("BBL_20_2.0", 0)
            ]
            sell_vals = [
                row.get("rsi_14", 50) > 65,
                row.get("close", 0) >= row.get("BBU_20_2.0", 0)
            ]
            return {
                "BUY": {l: v for l, v in zip(labels, buy_vals)},
                "SELL": {l: v for l, v in zip(labels, sell_vals)},
                "buy_score": sum(buy_vals) * 2,
                "sell_score": sum(sell_vals) * 2,
                "min_needed": self.min_conditions,
                "mode": "MEAN_REVERSION"
            }
