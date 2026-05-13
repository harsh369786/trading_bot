import pandas as pd
import numpy as np
from typing import Union

class PriceFeatures:
    """
    Computes technical indicators related to price action.
    Uses pandas_ta for high-performance vectorized calculations.
    """
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA, VWAP, RSI, MACD, ATR, Supertrend, and ADX."""
        if len(df) < 50:
            return df
            
        import ta
        from ta.trend import EMAIndicator, ADXIndicator
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange, BollingerBands
        
        # EMA
        df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        
        # Momentum
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # Trend Strength
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['ADX_14'] = adx_ind.adx()
        df['DMP_14'] = adx_ind.adx_pos()
        df['DMN_14'] = adx_ind.adx_neg()
        
        # Volatility
        df['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # Bollinger Bands
        bb_ind = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BBL_20_2.0'] = bb_ind.bollinger_lband()
        df['BBM_20_2.0'] = bb_ind.bollinger_mavg()
        df['BBU_20_2.0'] = bb_ind.bollinger_hband()
        
        # VWAP (Reset daily)
        df['date'] = df.index.date
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        df['tpv'] = df['tp'] * df['volume']
        cum_tpv = df.groupby('date')['tpv'].cumsum()
        cum_volume = df.groupby('date')['volume'].cumsum()
        df['vwap'] = (cum_tpv / cum_volume.replace(0, np.nan)).fillna(df['close'])
        df.drop(['date', 'tp', 'tpv'], axis=1, inplace=True)

        # True Supertrend Implementation (Vectorized/Numpy loop)
        m = 3.0
        n = 10
        hl2 = (df['high'] + df['low']) / 2
        df['atr_st'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=n).average_true_range()
        
        bub = hl2 + (m * df['atr_st'])
        blb = hl2 - (m * df['atr_st'])
        
        close_arr = df['close'].values
        bub_arr = bub.values
        blb_arr = blb.values
        
        final_ub = np.zeros(len(df))
        final_lb = np.zeros(len(df))
        st_dir = np.zeros(len(df))
        
        for i in range(len(df)):
            if i == 0 or np.isnan(bub_arr[i]) or np.isnan(blb_arr[i]):
                final_ub[i] = bub_arr[i] if not np.isnan(bub_arr[i]) else close_arr[i]
                final_lb[i] = blb_arr[i] if not np.isnan(blb_arr[i]) else close_arr[i]
                st_dir[i] = 1
                continue
                
            # Upper Band
            if bub_arr[i] < final_ub[i-1] or close_arr[i-1] > final_ub[i-1]:
                final_ub[i] = bub_arr[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            # Lower Band
            if blb_arr[i] > final_lb[i-1] or close_arr[i-1] < final_lb[i-1]:
                final_lb[i] = blb_arr[i]
            else:
                final_lb[i] = final_lb[i-1]
                
            # Directional Trend
            if st_dir[i-1] == 1 and close_arr[i] < final_lb[i]:
                st_dir[i] = -1
            elif st_dir[i-1] == -1 and close_arr[i] > final_ub[i]:
                st_dir[i] = 1
            else:
                st_dir[i] = st_dir[i-1]
                
        df['SUPERTd_10_3.0'] = st_dir
        
        return df

    @staticmethod
    def add_pivots(df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Add Daily Pivot Points to the intraday dataframe."""
        if daily_df.empty:
            return df
            
        prev_day = daily_df.iloc[-1]
        p = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
        r1 = (2 * p) - prev_day['low']
        s1 = (2 * p) - prev_day['high']
        r2 = p + (prev_day['high'] - prev_day['low'])
        s2 = p - (prev_day['high'] - prev_day['low'])
        
        df['pivot_p'] = p
        df['pivot_r1'] = r1
        df['pivot_s1'] = s1
        df['pivot_r2'] = r2
        df['pivot_s2'] = s2
        
        return df
