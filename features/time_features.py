from datetime import datetime, time
import pandas as pd
import pytz

class TimeFeatures:
    """
    Adds market session flags and time-based bias features.
    """
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def add_session_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize signals based on time-of-day market character."""
        df = df.copy()
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        times = df.index.time
        
        # 09:15–09:30 IST → Noise window
        df.loc[:, 'is_noise_window'] = [time(9, 15) <= t < time(9, 30) for t in times]
        
        # 11:30–13:30 IST → Chop zone (Low liquidity / Mean reversion)
        df.loc[:, 'is_chop_zone'] = [time(11, 30) <= t < time(13, 30) for t in times]
        
        # 14:30–15:15 IST → Trend window (Closing momentum)
        df.loc[:, 'is_trend_window'] = [time(14, 30) <= t < time(15, 15) for t in times]
        
        return df
