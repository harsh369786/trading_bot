import pandas as pd
import numpy as np

class OptionsFeatures:
    """
    Module 2: Options chain analysis for Equity/F&O signals.
    Processes PCR, OI Buildup, and IV Rank.
    """
    
    @staticmethod
    def analyze_oi_buildup(options_df: pd.DataFrame) -> str:
        """
        Determines the OI buildup regime:
        Long Buildup, Short Buildup, Long Unwinding, Short Covering.
        """
        if options_df.empty: return "NEUTRAL"
        
        price_change = options_df['price_change'].iloc[-1]
        oi_change = options_df['oi_change'].iloc[-1]
        
        if price_change > 0 and oi_change > 0: return "LONG_BUILDUP"
        if price_change < 0 and oi_change > 0: return "SHORT_BUILDUP"
        if price_change < 0 and oi_change < 0: return "LONG_UNWINDING"
        if price_change > 0 and oi_change < 0: return "SHORT_COVERING"
        
        return "NEUTRAL"

    @staticmethod
    def calculate_pcr(ce_oi: float, pe_oi: float) -> float:
        """Calculate Put-Call Ratio."""
        if ce_oi == 0: return 0.0
        return pe_oi / ce_oi

    @staticmethod
    def calculate_iv_rank(current_iv: float, iv_history: pd.Series) -> float:
        """Calculate IV Rank (0-100)."""
        if iv_history.empty: return 50.0
        min_iv = iv_history.min()
        max_iv = iv_history.max()
        if max_iv == min_iv: return 50.0
        return (current_iv - min_iv) / (max_iv - min_iv) * 100
