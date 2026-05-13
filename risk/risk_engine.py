import math
from datetime import datetime
from loguru import logger

class RiskEngine:
    """
    Shared Risk Management Engine for Equity and Currency.
    Handles position sizing and hard circuit breakers.
    State is persisted in Redis to survive restarts.
    """
    def __init__(self, config: dict, redis_client=None):
        self.config = config
        self.redis = redis_client
        capital_cfg = config.get("capital", {})
        risk_cfg = config.get("risk", {})
        self.capital_equity = float(capital_cfg.get("equity_total", 50000))
        self.capital_currency = float(capital_cfg.get("currency_total", 25000))
        self.risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100
        self.max_open_trades_equity = int(capital_cfg.get("max_open_trades_equity", 2))
        self.max_open_trades_currency = int(capital_cfg.get("max_open_trades_currency", 2))
        self.daily_loss_limit_r = float(risk_cfg.get("daily_loss_limit_r", 3))
        self.currency_max_daily_loss_inr = float(risk_cfg.get("currency_max_daily_loss_inr", 1500))
        self.currency_max_daily_trades = int(risk_cfg.get("currency_max_daily_trades", 5))
        
        # State keys
        self.KEY_PREFIX = "bot:risk:stats:"

    @property
    def today(self) -> str:
        """Always return the current IST date so keys roll over at midnight."""
        return datetime.now().strftime("%Y-%m-%d")

    async def _get_stat(self, key: str, default: float = 0.0) -> float:
        if not self.redis: return default
        val = await self.redis.get(f"{self.KEY_PREFIX}{self.today}:{key}")
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            logger.warning(f"Invalid risk stat for {key}: {val!r}. Using default {default}.")
            return default

    async def _set_stat(self, key: str, val: float):
        if not self.redis: return
        await self.redis.set(f"{self.KEY_PREFIX}{self.today}:{key}", val, ex=86400) # Expire in 24h

    def get_equity_position_size(self, entry: float, sl: float) -> int:
        """Calculate quantity based on fixed fractional risk."""
        risk_amount = self.capital_equity * self.risk_pct
        sl_points = abs(entry - sl)
        if sl_points == 0: return 0
        
        quantity = math.floor(risk_amount / sl_points)
        return quantity

    def get_currency_lots(self, sl_paise: float) -> int:
        """Calculate lots for currency based on fixed INR risk (Module 5)."""
        risk_amount_inr = 500 
        if sl_paise == 0: return 0
        
        lots = math.floor(risk_amount_inr / (sl_paise * 10))
        return max(1, min(lots, 3))

    async def check_circuit_breakers(self, domain: str) -> bool:
        """Verify if trading is allowed based on daily limits (Persisted in Redis)."""
        if domain == "equity":
            loss_r = await self._get_stat("equity_loss_r")
            open_count = await self._get_stat("equity_open_count")
            
            if loss_r >= self.daily_loss_limit_r:
                logger.warning(f"Equity daily loss limit ({self.daily_loss_limit_r}R) hit. Current: {loss_r}R")
                return False
            if open_count >= self.max_open_trades_equity:
                logger.warning(f"Equity max open trades reached: {open_count}")
                return False
        
        if domain == "currency":
            loss_inr = await self._get_stat("currency_loss_inr")
            open_count = await self._get_stat("currency_open_count")
            
            if loss_inr >= self.currency_max_daily_loss_inr:
                logger.warning(f"Currency daily loss limit (Rs {self.currency_max_daily_loss_inr}) hit. Current: Rs {loss_inr}")
                return False
            if open_count >= self.max_open_trades_currency:
                logger.warning(f"Currency max open trades reached: {open_count}")
                return False
                
        return True

    async def update_stats(self, domain: str, pnl_r: float = 0, pnl_inr: float = 0, trade_delta: int = 0):
        """Update stats in Redis after trade execution/closure."""
        if domain == "equity":
            curr_r = await self._get_stat("equity_loss_r")
            curr_open = await self._get_stat("equity_open_count")
            # We only track losses in R (negative pnl_r)
            if pnl_r < 0:
                await self._set_stat("equity_loss_r", curr_r + abs(pnl_r))
            await self._set_stat("equity_open_count", max(0, curr_open + trade_delta))
            
        if domain == "currency":
            curr_inr = await self._get_stat("currency_loss_inr")
            curr_open = await self._get_stat("currency_open_count")
            if pnl_inr < 0:
                await self._set_stat("currency_loss_inr", curr_inr + abs(pnl_inr))
            await self._set_stat("currency_open_count", max(0, curr_open + trade_delta))
