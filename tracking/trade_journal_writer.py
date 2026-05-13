import csv
import os
from datetime import datetime
from loguru import logger

class TradeJournalWriter:
    """
    Module 8: Trade Journal Writer
    Logs every signal and trade outcome with full environmental context.
    """
    
    def __init__(self, filepath: str = "data/trade_journal.csv"):
        self.filepath = filepath
        # Full schema as per Module 8 specifications
        self.headers = [
            "signal_id", "date", "symbol", "direction", "entry_price", 
            "exit_price", "sl", "sl_paise", "target_1", "target_2", "rr_ratio",
            "quant_score", "setup_type", "market_session", "market_character",
            "adx_at_entry", "volume_ratio", "pnl_inr", "outcome", 
            "mfe_paise", "mae_paise", "duration_min", "notes"
        ]
        self._init_file()

    def _init_file(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Initialized Trade Journal at {self.filepath}")

    def log_trade(self, trade_data: dict):
        """
        Append a closed trade to the CSV journal.
        Ensures all keys in trade_data match the headers.
        """
        try:
            # Fill missing keys with None to avoid DictWriter errors
            row = {header: trade_data.get(header, None) for header in self.headers}
            
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(row)
            logger.info(f"📓 Journal Entry: {trade_data['symbol']} | PnL: ₹{trade_data['pnl_inr']}")
        except Exception as e:
            logger.error(f"Failed to write to journal: {e}")
