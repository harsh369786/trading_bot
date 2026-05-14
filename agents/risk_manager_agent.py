import math
from loguru import logger

class RiskManagerAgent:
    """
    Agent 4: RiskManagerAgent
    Calculates dynamic Stop Loss, Targets, and Lot Sizing.
    """
    def __init__(self, config: dict):
        self.config = config
        self.max_sl_paise = config.get("currency_signal", {}).get("max_sl_paise", 20) / 100.0
        self.max_lots = int(config.get("currency_signal", {}).get("max_lots", 3))

    def calculate_risk_params(self, row: dict, side: str) -> dict:
        """
        Returns {sl, t1, t2, lots, rr}
        """
        entry = row['close']
        atr = row.get('atr_14', 0.05)
        
        # 1. Dynamic Stop Loss
        if side == "BUY":
            sl = min(row.get('low', entry), row.get('SUPERT_10_3.0', entry)) - (0.5 * atr)
        else:
            sl = max(row.get('high', entry), row.get('SUPERT_10_3.0', entry)) + (0.5 * atr)
            
        # Hard Cap SL at 20 paise
        sl_paise = abs(entry - sl)
        if sl_paise > self.max_sl_paise:
            sl = entry - self.max_sl_paise if side == "BUY" else entry + self.max_sl_paise
            sl_paise = self.max_sl_paise

        # 2. Targets
        t1 = entry + (1.5 * sl_paise) if side == "BUY" else entry - (1.5 * sl_paise)
        t2 = entry + (2.5 * sl_paise) if side == "BUY" else entry - (2.5 * sl_paise)
        
        # 3. Dynamic Lot Sizing
        currency_cap = self.config.get('capital', {}).get('currency_total', 25000)
        risk_pct = self.config.get('capital', {}).get('risk_per_trade_pct', 1.0) / 100.0
        risk_per_trade_inr = currency_cap * risk_pct
        
        # 1 lot of USDINR is 1000 units. 1 paise move = 10 INR PnL.
        if sl_paise > 0:
            lots = math.floor(risk_per_trade_inr / (sl_paise * 100 * 10))
        else:
            lots = 1
            
        lots = max(1, min(lots, self.max_lots))
        
        return {
            "entry": entry,
            "sl": sl,
            "t1": t1,
            "t2": t2,
            "lots": lots,
            # C9 fix: use abs() so R:R is always positive for both BUY and SELL
            "rr": round(abs(t1 - entry) / abs(entry - sl), 2) if entry != sl else 0,
        }
