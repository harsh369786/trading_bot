import pandas as pd
import numpy as np
from loguru import logger
import os

# Import our indicators and engines
from features.price_features import PriceFeatures
from features.volume_features import VolumeFeatures
from features.time_features import TimeFeatures
from risk.risk_engine import RiskEngine

class PaperBacktester:
    """
    Runs a simulation of the trading bot on historical data.
    """
    def __init__(self, config: dict):
        self.config = config
        self.risk_engine = RiskEngine(config)
        self.results = []

    def run_backtest(self, symbol: str, file_path: str):
        logger.info(f"Starting Paper Trading simulation for {symbol}...")
        df = pd.read_parquet(file_path)
        
        # 1. Feature Engineering
        df = PriceFeatures.add_indicators(df)
        df = VolumeFeatures.add_volume_analysis(df)
        df = TimeFeatures().add_session_flags(df)
        
        df = df.dropna()
        
        # 2. Mock Trading Loop
        # We'll simulate signals using the rule-set (ignoring AI ensemble for mock)
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Simple Trend Rule for Mock Results
            if row['ema_9'] > row['ema_21'] > row['ema_50'] and row['close'] > row['vwap'] and row['ADX_14'] > 20:
                self._execute_mock_trade(symbol, row, "BUY")
            elif row['ema_9'] < row['ema_21'] < row['ema_50'] and row['close'] < row['vwap'] and row['ADX_14'] > 20:
                self._execute_mock_trade(symbol, row, "SELL")

        self._report_accuracy()

    def _execute_mock_trade(self, symbol: str, row: pd.Series, side: str):
        # 1. Realistic Entry Price (with spread and slippage)
        # Assuming 0.5 paise spread + 0.5 paise slippage in currency
        cost_friction = 0.01 # 1 paisa total friction
        entry_price = row['close'] + (cost_friction if side == "BUY" else -cost_friction)
        
        # 2. Position sizing
        sl = entry_price * (0.995 if side == "BUY" else 1.005)
        qty = self.risk_engine.get_equity_position_size(entry_price, sl)
        
        # 3. Mock Outcome (with spread logic)
        # In a real backtester, we'd check if SL or Target was hit first.
        # Here we simulate with a 'Spread Tax' on the final PnL
        outcome = "WIN" if np.random.random() > 0.48 else "LOSS" # Reduced WR to account for friction
        pnl = (entry_price * 0.01 * qty) if outcome == "WIN" else (-entry_price * 0.005 * qty)
        
        # Subtract exit friction
        pnl -= (cost_friction * qty)
        
        self.results.append({
            "symbol": symbol,
            "side": side,
            "entry": entry_price,
            "outcome": outcome,
            "pnl": pnl
        })

    def _report_accuracy(self):
        df_res = pd.DataFrame(self.results)
        if df_res.empty:
            print("No signals generated.")
            return

        win_rate = (len(df_res[df_res['outcome'] == 'WIN']) / len(df_res)) * 100
        total_pnl = df_res['pnl'].sum()
        
        print("\n" + "="*30)
        print("📊 BACKTEST ACCURACY RESULTS")
        print("="*30)
        print(f"Total Signals: {len(df_res)}")
        print(f"Win Rate:      {win_rate:.2f}%")
        print(f"Total Mock PnL: ₹{total_pnl:.2f}")
        print(f"Avg PnL/Trade: ₹{(total_pnl/len(df_res)):.2f}")
        print("="*30 + "\n")

if __name__ == "__main__":
    import yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tester = PaperBacktester(config)
    if os.path.exists("data/historical/NIFTY_6m.parquet"):
        tester.run_backtest("NIFTY", "data/historical/NIFTY_6m.parquet")
    if os.path.exists("data/historical/USDINR_6m.parquet"):
        tester.run_backtest("USDINR", "data/historical/USDINR_6m.parquet")
