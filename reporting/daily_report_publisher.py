import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

import requests
from loguru import logger


class DailyReportPublisher:
    """
    Module 14: daily and weekly summary reports via email or Telegram.
    """
    def __init__(self, config: dict):
        notifications = config.get("notifications", {}) if isinstance(config, dict) else {}
        self.channel = str(notifications.get("channel") or os.environ.get("REPORT_CHANNEL") or "email").lower()
        self.smtp_host = os.environ.get("SMTP_HOST") or os.environ.get("NOTIFY_SMTP_HOST") or "smtp.gmail.com"
        self.smtp_port = int(os.environ.get("SMTP_PORT") or os.environ.get("NOTIFY_SMTP_PORT") or "587")
        self.smtp_user = os.environ.get("SMTP_USER") or os.environ.get("NOTIFY_EMAIL_FROM")
        self.smtp_password = (
            os.environ.get("SMTP_PASSWORD")
            or os.environ.get("NOTIFY_EMAIL_PASSWORD", "").replace(" ", "")
        )
        self.smtp_use_tls = os.environ.get("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}
        self.email_from = os.environ.get("EMAIL_FROM") or os.environ.get("NOTIFY_EMAIL_FROM") or self.smtp_user
        self.email_to = (
            os.environ.get("EMAIL_TO")
            or os.environ.get("REPORT_EMAIL_TO")
            or os.environ.get("MAIL_TO")
            or os.environ.get("NOTIFY_EMAIL_TO")
            or self.email_from
        )
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.config = config

    def send_report(self, message: str):
        if self.channel == "email":
            if self._send_email_report(message):
                return
            logger.error(
                "Email report was not sent. Check SMTP_* / EMAIL_TO or legacy "
                "NOTIFY_EMAIL_* settings in .env."
            )
            return

        if self.channel == "telegram":
            self._send_telegram_report(message)
            return

        logger.warning(f"Unknown notifications.channel={self.channel!r}. Printing report to console.")
        print(message)

    def _send_email_report(self, message: str) -> bool:
        missing = [
            name for name, value in {
                "SMTP_HOST": self.smtp_host,
                "SMTP_USER": self.smtp_user,
                "SMTP_PASSWORD": self.smtp_password,
                "EMAIL_FROM": self.email_from,
                "EMAIL_TO": self.email_to,
            }.items()
            if not value
        ]
        if missing:
            logger.warning(f"Email credentials missing: {', '.join(missing)}")
            return False

        email = EmailMessage()
        email["Subject"] = f"Trading Bot EOD Summary - {datetime.now().strftime('%Y-%m-%d')}"
        email["From"] = self.email_from
        email["To"] = self.email_to
        email.set_content(message.replace("*", ""))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as smtp:
                if self.smtp_use_tls:
                    smtp.starttls()
                smtp.login(self.smtp_user, self.smtp_password)
                smtp.send_message(email)
            logger.info(f"Daily report sent via email to {self.email_to}.")
            return True
        except Exception as e:
            logger.error(f"Email report publisher error: {e}")
            return False

    def _send_telegram_report(self, message: str):
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
