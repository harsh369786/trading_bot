import os
import time
import numpy as np
import pandas as pd


def load_csv_safely(file_path: str, retries: int = 3, delay_seconds: float = 0.2):
    """Read a bot CSV log without crashing on missing files or transient locks."""
    if not os.path.exists(file_path):
        return pd.DataFrame(), None

    last_err = None
    for _ in range(retries):
        try:
            df = pd.read_csv(file_path, engine="c", on_bad_lines="skip")
            for col in ["date", "timestamp", "entry_time", "exit_time"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df, None
        except (PermissionError, OSError, pd.errors.ParserError) as e:
            last_err = str(e)
            time.sleep(delay_seconds)

    return pd.DataFrame(), f"CSV Read Error: {last_err}"

def calculate_advanced_metrics(df: pd.DataFrame, pnl_col: str = "pnl_after_costs") -> dict:
    """Calculates quantitative performance metrics for the dashboard."""
    if df.empty or pnl_col not in df.columns:
        return {"profit_factor": 0.0, "max_drawdown": 0.0, "expectancy": 0.0}

    # Gross Win/Loss Math
    gross_profit = df[df[pnl_col] > 0][pnl_col].sum()
    gross_loss = abs(df[df[pnl_col] < 0][pnl_col].sum())
    
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

    # Max Drawdown Calculation
    df_sorted = df.sort_values("date") if "date" in df.columns else df
    cum_pnl = df_sorted[pnl_col].cumsum()
    peak = cum_pnl.cummax()
    drawdown = peak - cum_pnl
    max_drawdown = round(drawdown.max(), 2)

    # Expectancy (Average PnL per trade)
    expectancy = round(df[pnl_col].mean(), 2)

    return {
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy
    }
