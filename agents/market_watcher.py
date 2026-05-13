import pandas as pd
from loguru import logger
from features.price_features import PriceFeatures
from features.currency_features import CurrencyFeatures

class MarketWatcher:
    """
    Agent 1: MarketWatcher
    Fetches currency data and computes indicators for the 5-min and 15-min candles.
    """
    def __init__(self, config: dict):
        self.config = config

    def prepare_data(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add all necessary indicators to the dataframes."""
        logger.debug("MarketWatcher: Adding indicators to currency data.")
        
        # Add basic price indicators (EMA, RSI, ADX, etc.)
        df_5m = PriceFeatures.add_indicators(df_5m)
        df_15m = PriceFeatures.add_indicators(df_15m)
        
        # Add currency-specific filters
        df_5m = CurrencyFeatures.add_currency_filters(df_5m)
        
        return df_5m, df_15m
