import pandas as pd
import numpy as np
from loguru import logger
import os
from datetime import datetime
from zoneinfo import ZoneInfo

class AccuracyAnalyzer:
    """
    Module 9: Post-trade and End-of-Day performance analysis.
    Calculates granular metrics across symbols, sessions, and market regimes.
    """
    def __init__(self, journal_path: str = "data/trade_journal.csv"):
        self.journal_path = journal_path

    def _load_data(self, today_only: bool = False) -> pd.DataFrame:
        if not os.path.exists(self.journal_path):
            return pd.DataFrame()
        df = pd.read_csv(self.journal_path)
        if today_only and not df.empty and "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
            df = df.loc[dates.dt.date == today].copy()
        return df

    def get_stats(self, today_only: bool = False) -> dict:
        """Core high-level metrics for the dashboard and daily reports."""
        df = self._load_data(today_only=today_only)
        if df.empty: return {}
        pnl_col = "pnl_after_costs" if "pnl_after_costs" in df.columns else "pnl_inr"
        
        wins = df[df[pnl_col] > 0]
        total = len(df)
        win_rate = (len(wins) / total) * 100 if total > 0 else 0
        
        # Calculate Profit Factor
        gross_profit = df[df[pnl_col] > 0][pnl_col].sum()
        gross_loss = abs(df[df[pnl_col] < 0][pnl_col].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_trades": total,
            "win_rate": f"{win_rate:.1f}%",
            "avg_pnl": df[pnl_col].mean(),
            "profit_factor": round(profit_factor, 2),
            "net_pnl": df[pnl_col].sum(),
            "max_drawdown": self.calculate_max_drawdown(df[pnl_col])
        }

    def get_detailed_breakdown(self) -> dict:
        """End-of-Day breakdown by symbol, session, and character."""
        df = self._load_data()
        if df.empty: return {}

        breakdown = {
            "by_symbol": df.groupby('symbol')['pnl_inr'].agg(['count', 'sum', 'mean']).to_dict('index'),
            "by_session": df.groupby('market_session')['pnl_inr'].agg(['count', 'sum', 'mean']).to_dict('index') if 'market_session' in df.columns else {},
            "by_setup": df.groupby('setup_type')['pnl_inr'].agg(['count', 'sum', 'mean']).to_dict('index') if 'setup_type' in df.columns else {},
            "by_regime": df.groupby('market_character')['pnl_inr'].agg(['count', 'sum', 'mean']).to_dict('index') if 'market_character' in df.columns else {}
        }
        
        # Add rolling metrics
        if len(df) >= 10:
            df['win'] = np.where(df['pnl_inr'] > 0, 1, 0)
            breakdown['rolling_win_rate_10'] = df['win'].rolling(10).mean().iloc[-1]
            
        return breakdown

    def calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """Calculates the maximum peak-to-trough decline in the equity curve."""
        equity_curve = pnl_series.cumsum()
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = peak - equity_curve
        return float(drawdown.max())

    def check_sl_efficiency(self) -> dict:
        """
        Module 9: SL Efficiency analysis.
        Checks if SL was hit then price reversed (SL too tight).
        """
        df = self._load_data()
        if df.empty or 'mae_paise' not in df.columns: return {}
        
        # SL too tight if MAE was close to SL but PnL was negative
        # Simplified: check how many trades had MAE within 2 paise of SL
        tight_sl = df[df['mae_paise'] >= (df['sl_paise'] - 2)]
        return {
            "tight_sl_count": len(tight_sl),
            "efficiency_ratio": len(tight_sl) / len(df) if len(df) > 0 else 0
        }

    async def run(self):
        """Scheduler integration for EOD analysis."""
        logger.info("Accuracy Analyzer triggered for EOD breakdown.")
        stats = self.get_stats()
        logger.info(f"Day Summary: Win Rate {stats.get('win_rate')} | Net P&L {stats.get('net_pnl')}")
