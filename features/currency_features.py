import pandas as pd
import numpy as np

class CurrencyFeatures:
    """
    Module 2: Specific features for Currency Derivatives (USDINR, etc.)
    Includes Pivot Points, BB, and specific filters.
    """
    
    @staticmethod
    def add_currency_filters(df: pd.DataFrame) -> pd.DataFrame:
        """Add filters specifically used by the Currency Agent Swarm."""
        if len(df) < 20:
            return df
            
        from ta.volatility import BollingerBands, AverageTrueRange
        
        # Bollinger Bands (20, 2.0)
        bb_ind = BollingerBands(close=df['close'], window=20, window_dev=2.0)
        df['BBL_20_2.0'] = bb_ind.bollinger_lband()
        df['BBM_20_2.0'] = bb_ind.bollinger_mavg()
        df['BBU_20_2.0'] = bb_ind.bollinger_hband()
        
        # Volatility normalization for SL/Targets
        df['atr_5'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=5).average_true_range()
        
        return df

    @staticmethod
    def calculate_pivot_proximity(df: pd.DataFrame) -> pd.Series:
        """
        Check proximity to major pivot levels (in paise).
        Reject signals if price is within 3 paise of a major level.
        """
        if 'pivot_p' not in df.columns:
            return pd.Series([False] * len(df))
            
        levels = ['pivot_p', 'pivot_r1', 'pivot_s1', 'pivot_r2', 'pivot_s2']
        close_col = df['close']
        min_dist = df[levels].apply(
            lambda row: np.min(np.abs(row.values - close_col.loc[row.name])), axis=1
        )

        # Proximity threshold: 3 paise (0.03)
        return min_dist < 0.03
