import pandas as pd
from loguru import logger

from .market_watcher import MarketWatcher
from .opportunity_scanner import OpportunityScanner
from .quant_validator import QuantValidator
from .risk_manager_agent import RiskManagerAgent
from .signal_publisher import SignalPublisher
from tracking.signal_logger import SignalLogger


class CurrencyAgentPipeline:
    """
    Sequential multi-agent pipeline for currency trading.
    Coordinates watcher -> scanner -> validator -> risk manager -> publisher.
    """
    def __init__(self, config: dict):
        self.config = config
        self.watcher = MarketWatcher(config)
        self.scanner = OpportunityScanner(config)
        self.validator = QuantValidator(config)
        self.risk_manager = RiskManagerAgent(config)
        self.publisher = SignalPublisher(config)
        self.signal_logger = SignalLogger()
        self.max_signals = config.get("capital", {}).get("max_open_trades_currency", 2)
        self._last_signal_bar = {}  # symbol -> timestamp

    async def process_symbol(self, symbol: str, df_5m: pd.DataFrame, df_15m: pd.DataFrame):
        """Run one currency-pair evaluation from closed 5m and 15m candles."""
        # C7 fix: active_signals was never decremented. Use max_signals as an
        # advisory limit only; actual gating is done by RiskEngine circuit breakers.
        if df_5m is None or df_15m is None or df_5m.empty or df_15m.empty:
            logger.debug(f"Swarm: NO TRADE {symbol}: missing candle data")
            return None
        if len(df_5m) < 50 or len(df_15m) < 50:
            logger.debug(f"Swarm: NO TRADE {symbol}: insufficient candle history")
            return None

        df_5m, df_15m = self.watcher.prepare_data(df_5m, df_15m)
        required_cols = {"close", "ema_9", "ema_21", "ema_50", "rsi_14", "vwap", "ADX_14", "atr_14"}
        missing = required_cols - set(df_5m.columns)
        available_required = list(required_cols - missing)
        if missing or df_5m.tail(1)[available_required].isna().any(axis=None):
            logger.debug(f"Swarm: NO TRADE {symbol}: missing/NaN 5m features {sorted(missing)}")
            return None
        if "ema_21" not in df_15m.columns or df_15m.tail(1)[["close", "ema_21"]].isna().any(axis=None):
            logger.debug(f"Swarm: NO TRADE {symbol}: missing/NaN 15m features")
            return None

        last_row = df_5m.iloc[-1]
        last_ts = df_5m.index[-1] if not df_5m.empty and isinstance(df_5m.index, pd.DatetimeIndex) else None
        if last_ts and self._last_signal_bar.get(symbol) == last_ts:
            return None  # Already fired a signal for this candle
            
        side, strategy = self.scanner.scan(last_row)
        if side == "NONE":
            debug = self.scanner.debug_score(last_row)
            logger.info(
                f"📊 {symbol} scanner: BUY={debug['buy_score']}/{debug['min_needed']} "
                f"SELL={debug['sell_score']}/{debug['min_needed']} | "
                f"BUY conds={debug['BUY']} | SELL conds={debug['SELL']}"
            )
            self.signal_logger.log_signal(
                symbol=symbol,
                side="NONE",
                strategy="None",
                entry=last_row.get("close", 0),
                sl=0,
                target=0,
                score=0,
                status="NO_TRADE",
                reason=f"Scanner: BUY={debug['buy_score']} SELL={debug['sell_score']} of {debug['min_needed']} needed",
            )
            return None

        logger.info(f"Swarm: {strategy} setup detected for {symbol} ({side})")

        validation = self.validator.validate(df_5m, df_15m, side)
        if not validation["valid"]:
            logger.info(f"Swarm: signal rejected by QuantValidator: {validation['reason']}")
            self.signal_logger.log_signal(symbol, side, strategy, last_row["close"], 0, 0, 0, "NO_TRADE", f"Quant: {validation['reason']}")
            return None

        risk_params = self.risk_manager.calculate_risk_params(last_row.to_dict(), side)
        if risk_params["rr"] < 1.5:
            logger.info(f"Swarm: signal rejected by RiskManager: low R:R ({risk_params['rr']})")
            self.signal_logger.log_signal(
                symbol,
                side,
                strategy,
                last_row["close"],
                risk_params.get("sl", 0),
                risk_params.get("t1", 0),
                validation["quant_score"],
                "NO_TRADE",
                "Low R:R",
            )
            return None

        signal_data = {
            "symbol": symbol,
            "side": side,
            "strategy": strategy,
            "quant_score": validation["quant_score"],
            **risk_params,
        }

        self.signal_logger.log_signal(
            symbol=symbol,
            side=side,
            strategy=strategy,
            entry=risk_params["entry"],
            sl=risk_params["sl"],
            target=risk_params["t1"],
            score=validation["quant_score"],
            status="TRADE",
        )

        formatted_card = self.publisher.format_signal(signal_data)
        self.publisher.publish(formatted_card)
        if last_ts:
            self._last_signal_bar[symbol] = last_ts
        # Note: RiskEngine tracks open count in Redis; no local counter needed (C7 fix).
        return signal_data

    async def run(self):
        logger.info("Currency Agent Pipeline active.")
