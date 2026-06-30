import csv
import os
import time
import numpy as np
import pandas as pd


TRADE_JOURNAL_FIELDS = [
    "date", "symbol", "side", "strategy", "entry_price", "exit_price",
    "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence",
]


def _load_trade_journal_csv(file_path: str) -> pd.DataFrame:
    """
    Read trade_journal.csv with compatibility for older files that missed the
    strategy header while rows already included a strategy value.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return pd.DataFrame(columns=TRADE_JOURNAL_FIELDS)

    header = rows[0]
    data_rows = rows[1:]
    normalized = []

    old_header = [
        "date", "symbol", "side", "entry_price", "exit_price",
        "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence",
    ]

    for row in data_rows:
        if not row:
            continue
        if header == TRADE_JOURNAL_FIELDS and len(row) >= len(TRADE_JOURNAL_FIELDS):
            normalized.append(row[:len(TRADE_JOURNAL_FIELDS)])
        elif header == old_header and len(row) >= len(TRADE_JOURNAL_FIELDS):
            normalized.append(row[:len(TRADE_JOURNAL_FIELDS)])
        elif header == old_header and len(row) == len(old_header):
            normalized.append(row[:3] + ["Unknown"] + row[3:])

    return pd.DataFrame(normalized, columns=TRADE_JOURNAL_FIELDS)


def load_csv_safely(file_path: str, retries: int = 3, delay_seconds: float = 0.2):
    """Read a bot CSV log without crashing on missing files or transient locks."""
    if not os.path.exists(file_path):
        return pd.DataFrame(), None

    last_err = None
    for _ in range(retries):
        try:
            if os.path.basename(file_path).lower() == "trade_journal.csv":
                df = _load_trade_journal_csv(file_path)
            else:
                try:
                    # Try UTF-8 (standard)
                    df = pd.read_csv(file_path, engine="c", on_bad_lines="skip", encoding='utf-8')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    # Fallback to UTF-16 (Windows/PowerShell default for some tools)
                    df = pd.read_csv(file_path, engine="c", on_bad_lines="skip", encoding='utf-16')
                
            for col in ["date", "timestamp", "entry_time", "exit_time"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df, None
        except (PermissionError, OSError, pd.errors.ParserError, UnicodeDecodeError) as e:
            last_err = str(e)
            time.sleep(delay_seconds)

    return pd.DataFrame(), f"CSV Read Error: {last_err}"

def calculate_advanced_metrics(df: pd.DataFrame, pnl_col: str = "pnl_after_costs") -> dict:
    """Calculates quantitative performance metrics for the dashboard."""
    if df.empty or pnl_col not in df.columns:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
        }

    # Gross Win/Loss Math
    gross_profit = df[df[pnl_col] > 0][pnl_col].sum()
    gross_loss = abs(df[df[pnl_col] < 0][pnl_col].sum())
    
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')
    win_rate = round((df[pnl_col] > 0).sum() / len(df) * 100, 2) if len(df) else 0.0

    # Max Drawdown Calculation
    df_sorted = (df.sort_values("date") if "date" in df.columns else df.sort_index()).copy()
    cum_pnl = df_sorted[pnl_col].cumsum()
    peak = cum_pnl.cummax()
    drawdown = peak - cum_pnl
    max_drawdown = round(drawdown.max(), 2)

    # Expectancy (Average PnL per trade)
    expectancy = round(df[pnl_col].mean(), 2)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy
    }
