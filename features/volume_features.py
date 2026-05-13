import pandas as pd
import numpy as np

class VolumeFeatures:
    """
    Computes volume-based features for breakout and momentum confirmation.
    """
    
    @staticmethod
    def add_volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Volume, Spikes, and Delta metrics."""
        if len(df) < 20:
            return df
            
        # Relative Volume vs 20-period average
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
        df['rel_vol'] = (df['volume'] / df['vol_sma_20'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Volume Spike (ratio of current volume to previous volume)
        df['vol_spike_ratio'] = (df['volume'] / df['volume'].shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Approximate Delta Volume (Simplification: using candle direction)
        # Production versions would use tick-level bid/ask data for true delta
        df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
        df['delta_vol'] = df['volume'] * df['candle_dir']
        
        return df
