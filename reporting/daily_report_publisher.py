import os
from datetime import datetime

import requests
from loguru import logger


class DailyReportPublisher:
    """
    Module 14: daily and weekly summary reports via Telegram.
    """
    def __init__(self, config: dict):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.config = config

    def send_report(self, message: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials missing. Printing report to console.")
            print(message)
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            res = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }, timeout=10)
            if res.status_code == 200:
                logger.info("Daily report sent via Telegram.")
            else:
                logger.error(f"Telegram failed: {res.text}")
        except Exception as e:
            logger.error(f"Report publisher error: {e}")

    def format_daily_summary(self, stats: dict):
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"""
*DAILY TRADING REPORT - {date_str}*

SUMMARY
Total Trades: {stats.get('total_trades', 0)}
Win Rate: {stats.get('win_rate', '0%')}
Profit Factor: {stats.get('profit_factor', 0):.2f}
Net P&L: INR {stats.get('net_pnl', 0):.2f}

PARAMETER CHANGES
{stats.get('notes', 'No adaptive changes today.')}
        """
