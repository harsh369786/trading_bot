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
        self.capital_gamma = float(capital_cfg.get("gamma_total", 30000))
        self.capital_meanrev = float(capital_cfg.get("meanrev_total", 40000))
        self.risk_pct = float(capital_cfg.get("risk_per_trade_pct", 1.0)) / 100
        self.max_open_trades_equity = int(capital_cfg.get("max_open_trades_equity", 2))
        self.max_open_trades_currency = int(capital_cfg.get("max_open_trades_currency", 2))
        self.max_open_trades_gamma = int(risk_cfg.get("gamma_max_open_trades", capital_cfg.get("gamma_max_open_trades", 2)))
        self.max_open_trades_meanrev = int(risk_cfg.get("meanrev_max_open_trades", capital_cfg.get("meanrev_max_open_trades", 3)))
        self.daily_loss_limit_r = float(risk_cfg.get("daily_loss_limit_r", 3))
        self.equity_max_daily_trades = int(risk_cfg.get("equity_max_daily_trades", 6))
        self.currency_max_daily_loss_inr = float(risk_cfg.get("currency_max_daily_loss_inr", 1500))
        self.gamma_max_daily_loss_inr = float(risk_cfg.get("gamma_max_daily_loss_inr", 3000))
        self.meanrev_max_daily_loss_inr = float(risk_cfg.get("meanrev_max_daily_loss_inr", 2000))
        self.currency_max_daily_trades = int(risk_cfg.get("currency_max_daily_trades", 5))
        self.max_equity_notional_per_trade = float(
            capital_cfg.get("max_equity_notional_per_trade", self.capital_equity)
        )
        
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

    async def _incr_stat(self, key: str, delta: float) -> float:
        """Atomically increment a daily stat when Redis supports it."""
        if not self.redis:
            return 0.0
        redis_key = f"{self.KEY_PREFIX}{self.today}:{key}"
        try:
            new_value = await self.redis.incrbyfloat(redis_key, float(delta))
            try:
                await self.redis.expire(redis_key, 86400)
            except Exception:
                pass
            return float(new_value)
        except AttributeError:
            current = await self._get_stat(key)
            new_value = current + float(delta)
            await self._set_stat(key, new_value)
            return new_value

    async def _get_stats_batch(self, *keys: str) -> list[float]:
        """Fetch multiple stats in a single Redis round-trip (pipelined)."""
        if not self.redis:
            return [0.0] * len(keys)
        if not hasattr(self.redis, "pipeline"):
            return [await self._get_stat(k) for k in keys]
        
        pipe = self.redis.pipeline()
        for k in keys:
            pipe.get(f"{self.KEY_PREFIX}{self.today}:{k}")
        
        results = await pipe.execute()
        final = []
        for v in results:
            try:
                final.append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                final.append(0.0)
        return final

    def get_equity_position_size(self, entry: float, sl: float) -> int:
        """Calculate quantity based on fixed fractional risk."""
        risk_amount = self.capital_equity * self.risk_pct
        sl_points = abs(entry - sl)
        if sl_points == 0: return 0
        
        quantity = math.floor(risk_amount / sl_points)
        if self.max_equity_notional_per_trade > 0 and entry > 0:
            quantity = min(quantity, math.floor(self.max_equity_notional_per_trade / entry))
        return quantity

    def get_currency_lots(self, sl_paise: float) -> int:
        """Calculate lots for currency based on configured risk (Module 5 fix)."""
        risk_amount_inr = self.capital_currency * self.risk_pct
        if sl_paise == 0: return 0
        
        # sl_paise * 10 = INR impact per lot (tick size 0.0025, lot size 1000)
        lots = math.floor(risk_amount_inr / (sl_paise * 10))
        return max(1, min(lots, 3))

    def get_gamma_position_size(self, entry: float, sl: float) -> int:
        risk_amount = self.capital_gamma * self.risk_pct
        sl_points = abs(entry - sl)
        if sl_points == 0:
            return 0
        qty = math.floor(risk_amount / sl_points)
        return max(0, min(qty, math.floor(self.capital_gamma / entry) if entry > 0 else qty))

    def get_meanrev_position_size(self, entry: float, sl: float) -> int:
        risk_amount = self.capital_meanrev * self.risk_pct
        sl_points = abs(entry - sl)
        if sl_points == 0:
            return 0
        qty = math.floor(risk_amount / sl_points)
        return max(0, min(qty, math.floor(self.capital_meanrev / entry) if entry > 0 else qty))

    async def check_circuit_breakers(self, domain: str) -> bool:
        """Verify if trading is allowed based on daily limits (Persisted in Redis)."""
        if domain == "equity":
            loss_r, open_count, trade_count = await self._get_stats_batch(
                "equity_loss_r", "equity_open_count", "equity_trade_count"
            )
            
            if loss_r >= self.daily_loss_limit_r:
                logger.warning(f"Equity daily loss limit ({self.daily_loss_limit_r}R) hit. Current: {loss_r}R")
                return False
            if open_count >= self.max_open_trades_equity:
                logger.warning(f"Equity max open trades reached: {open_count}")
                return False
            if self.equity_max_daily_trades > 0 and trade_count >= self.equity_max_daily_trades:
                logger.warning(
                    f"Equity daily trade limit ({self.equity_max_daily_trades}) hit. Current: {trade_count}"
                )
                return False
        
        if domain == "currency":
            loss_inr, open_count, trade_count = await self._get_stats_batch(
                "currency_loss_inr", "currency_open_count", "currency_trade_count"
            )
            
            if loss_inr >= self.currency_max_daily_loss_inr:
                logger.warning(f"Currency daily loss limit (Rs {self.currency_max_daily_loss_inr}) hit. Current: Rs {loss_inr}")
                return False
            if open_count >= self.max_open_trades_currency:
                logger.warning(f"Currency max open trades reached: {open_count}")
                return False
            if self.currency_max_daily_trades > 0 and trade_count >= self.currency_max_daily_trades:
                logger.warning(
                    f"Currency daily trade limit ({self.currency_max_daily_trades}) hit. Current: {trade_count}"
                )
                return False

        if domain == "gamma":
            loss_inr, open_count = await self._get_stats_batch("gamma_loss_inr", "gamma_open_count")
            if loss_inr >= self.gamma_max_daily_loss_inr:
                logger.warning(f"Gamma daily loss limit (Rs {self.gamma_max_daily_loss_inr}) hit. Current: Rs {loss_inr}")
                return False
            if open_count >= self.max_open_trades_gamma:
                logger.warning(f"Gamma max open trades reached: {open_count}")
                return False

        if domain == "mean_reversion":
            loss_inr, open_count = await self._get_stats_batch("mean_reversion_loss_inr", "mean_reversion_open_count")
            if loss_inr >= self.meanrev_max_daily_loss_inr:
                logger.warning(f"Mean reversion daily loss limit (Rs {self.meanrev_max_daily_loss_inr}) hit. Current: Rs {loss_inr}")
                return False
            if open_count >= self.max_open_trades_meanrev:
                logger.warning(f"Mean reversion max open trades reached: {open_count}")
                return False
                
        return True

    async def update_stats(
        self,
        domain: str,
        pnl_r: float = 0,
        pnl_inr: float = 0,
        trade_delta: int = 0,
        count_daily_trade: bool = True,
    ):
        """Update stats in Redis after trade execution/closure."""
        if domain == "equity":
            # We only track losses in R (negative pnl_r)
            if pnl_r < 0:
                await self._incr_stat("equity_loss_r", abs(pnl_r))
            if count_daily_trade and trade_delta > 0:
                await self._incr_stat("equity_trade_count", trade_delta)
            new_open = await self._incr_stat("equity_open_count", trade_delta)
            if new_open < 0:
                await self._set_stat("equity_open_count", 0)
            
        if domain == "currency":
            if pnl_inr < 0:
                await self._incr_stat("currency_loss_inr", abs(pnl_inr))
            if count_daily_trade and trade_delta > 0:
                await self._incr_stat("currency_trade_count", trade_delta)
            new_open = await self._incr_stat("currency_open_count", trade_delta)
            if new_open < 0:
                await self._set_stat("currency_open_count", 0)

        if domain == "gamma":
            if pnl_inr < 0:
                await self._incr_stat("gamma_loss_inr", abs(pnl_inr))
            new_open = await self._incr_stat("gamma_open_count", trade_delta)
            if new_open < 0:
                await self._set_stat("gamma_open_count", 0)

        if domain == "mean_reversion":
            if pnl_inr < 0:
                await self._incr_stat("mean_reversion_loss_inr", abs(pnl_inr))
            new_open = await self._incr_stat("mean_reversion_open_count", trade_delta)
            if new_open < 0:
                await self._set_stat("mean_reversion_open_count", 0)

    async def reconcile_open_counts(
        self,
        equity_open_count: int = 0,
        currency_open_count: int = 0,
        gamma_open_count: int = 0,
        mean_reversion_open_count: int = 0,
    ):
        """Reset open-count circuit breaker stats from recovered runtime state."""
        await self._set_stat("equity_open_count", max(0, int(equity_open_count)))
        await self._set_stat("currency_open_count", max(0, int(currency_open_count)))
        await self._set_stat("gamma_open_count", max(0, int(gamma_open_count)))
        await self._set_stat("mean_reversion_open_count", max(0, int(mean_reversion_open_count)))
