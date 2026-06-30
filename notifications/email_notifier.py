"""
notifications/email_notifier.py
---------------------------------
SMTP email notification service for the NSE trading bot.
Credentials are read from .env environment variables.

Required .env keys:
    NOTIFY_EMAIL_FROM      e.g. yourbot@gmail.com
    NOTIFY_EMAIL_PASSWORD  Gmail App Password (16-char, no spaces)
    NOTIFY_EMAIL_TO        recipient address (defaults to FROM)
    NOTIFY_SMTP_HOST       (optional) default: smtp.gmail.com
    NOTIFY_SMTP_PORT       (optional) default: 587
"""
from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card(label: str, value: str, bg: str = "white") -> str:
    return (
        f'<tr style="background:{bg};">'
        f'<td style="padding:9px 12px;color:#555;font-size:14px;">{label}</td>'
        f'<td style="padding:9px 12px;font-weight:600;font-size:14px;">{value}</td>'
        f"</tr>"
    )


# ---------------------------------------------------------------------------
# EmailNotifier
# ---------------------------------------------------------------------------

class EmailNotifier:
    """Thread-safe SMTP email sender."""

    def __init__(self) -> None:
        load_dotenv()
        self.from_addr = os.getenv("NOTIFY_EMAIL_FROM", "")
        self.password  = os.getenv("NOTIFY_EMAIL_PASSWORD", "").replace(" ", "")
        self.to_addr   = os.getenv("NOTIFY_EMAIL_TO", self.from_addr)
        self.smtp_host = os.getenv("NOTIFY_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("NOTIFY_SMTP_PORT", "587"))
        self.enabled   = bool(self.from_addr and self.password)
        if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("BOT_ALLOW_TEST_EMAILS") != "1":
            self.enabled = False
            logger.warning("EmailNotifier: disabled during pytest. Set BOT_ALLOW_TEST_EMAILS=1 to override.")

        if not self.enabled:
            logger.warning(
                "EmailNotifier: NOTIFY_EMAIL_FROM / NOTIFY_EMAIL_PASSWORD not set "
                "in .env — notifications disabled."
            )

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, subject: str, html_body: str) -> bool:
        if not self.enabled:
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self.from_addr
            msg["To"]      = self.to_addr
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as srv:
                srv.ehlo()
                srv.starttls()
                srv.login(self.from_addr, self.password)
                srv.sendmail(self.from_addr, self.to_addr, msg.as_string())

            logger.info(f"EmailNotifier: sent '{subject}' → {self.to_addr}")
            return True
        except Exception as exc:
            logger.error(f"EmailNotifier: failed to send '{subject}': {exc}")
            return False

    # ------------------------------------------------------------------
    # Trade fill notification
    # ------------------------------------------------------------------

    def send_trade_fill(
        self,
        symbol: str,
        side: str,
        strategy: str,
        entry: float,
        sl: float,
        target: float,
        score: float,
    ) -> bool:
        color   = "#27ae60" if side == "BUY" else "#e74c3c"
        emoji   = "🟢" if side == "BUY" else "🔴"
        rr      = abs(target - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        subject = f"{emoji} NSE Bot: {side} {symbol} Triggered — ₹{entry:.2f}"

        html = f"""
<html><body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,sans-serif;">
<div style="max-width:520px;margin:30px auto;background:white;border-radius:12px;
            box-shadow:0 4px 16px rgba(0,0,0,.10);overflow:hidden;">
  <div style="background:{color};padding:22px 24px;">
    <div style="font-size:22px;font-weight:700;color:white;">{emoji} {side} — {symbol}</div>
    <div style="color:rgba(255,255,255,.85);font-size:13px;margin-top:4px;">
      {strategy} &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y  %H:%M IST')}
    </div>
  </div>
  <table style="width:100%;border-collapse:collapse;">
    {_card('Entry Price', f'₹{entry:.2f}', '#fafafa')}
    {_card('Stop Loss',   f'₹{sl:.2f} &nbsp;<span style="color:#e74c3c;font-size:12px;">▼ {abs(entry-sl):.2f}</span>')}
    {_card('Target',      f'₹{target:.2f} &nbsp;<span style="color:#27ae60;font-size:12px;">▲ {abs(target-entry):.2f}</span>', '#fafafa')}
    {_card('R:R Ratio',   f'{rr:.2f}x')}
    {_card('AI Score',    f'{score:.1%}', '#fafafa')}
  </table>
  <div style="padding:12px 24px;color:#aaa;font-size:11px;border-top:1px solid #f0f0f0;">
    NSE Paper Trading Bot · Paper mode active · No real orders placed
  </div>
</div></body></html>"""
        return self.send(subject, html)

    # ------------------------------------------------------------------
    # Trade close notification
    # ------------------------------------------------------------------

    def send_trade_close(
        self,
        symbol: str,
        side: str,
        strategy: str,
        entry: float,
        exit_price: float,
        pnl: float,
        outcome: str,
        qty: int,
    ) -> bool:
        pnl_color = "#27ae60" if pnl >= 0 else "#e74c3c"
        emoji     = "✅" if pnl >= 0 else "❌"
        subject   = f"{emoji} NSE Bot: {symbol} Closed — ₹{pnl:+.0f} ({outcome})"

        html = f"""
<html><body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,sans-serif;">
<div style="max-width:520px;margin:30px auto;background:white;border-radius:12px;
            box-shadow:0 4px 16px rgba(0,0,0,.10);overflow:hidden;">
  <div style="background:{pnl_color};padding:22px 24px;">
    <div style="font-size:22px;font-weight:700;color:white;">{emoji} {symbol} Closed</div>
    <div style="color:rgba(255,255,255,.85);font-size:13px;margin-top:4px;">
      {outcome} &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y  %H:%M IST')}
    </div>
  </div>
  <div style="padding:16px 24px;background:{pnl_color}10;text-align:center;">
    <div style="font-size:36px;font-weight:700;color:{pnl_color};">₹{pnl:+.2f}</div>
    <div style="font-size:12px;color:#888;margin-top:2px;">Net P&amp;L after costs</div>
  </div>
  <table style="width:100%;border-collapse:collapse;">
    {_card('Strategy',    strategy, '#fafafa')}
    {_card('Side',        side)}
    {_card('Entry',       f'₹{entry:.2f}', '#fafafa')}
    {_card('Exit',        f'₹{exit_price:.2f}')}
    {_card('Qty',         str(qty), '#fafafa')}
    {_card('Move',        f'₹{abs(exit_price - entry):.2f} ({abs(exit_price - entry)/entry*100:.2f}%)')}
  </table>
  <div style="padding:12px 24px;color:#aaa;font-size:11px;border-top:1px solid #f0f0f0;">
    NSE Paper Trading Bot · Paper mode active · No real orders placed
  </div>
</div></body></html>"""
        return self.send(subject, html)

    # ------------------------------------------------------------------
    # EOD summary
    # ------------------------------------------------------------------

    def send_eod_summary(self, stats: dict) -> bool:
        date_str  = datetime.now().strftime("%d %b %Y")
        net_pnl   = stats.get("net_pnl", 0.0)
        pnl_color = "#27ae60" if net_pnl >= 0 else "#e74c3c"
        emoji     = "📈" if net_pnl >= 0 else "📉"
        subject   = f"{emoji} NSE Bot EOD — {date_str} — ₹{net_pnl:+.0f}"

        pf_val = stats.get("profit_factor", 0)
        pf_str = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"

        # Per-strategy breakdown rows
        strat_rows = ""
        for strat, sd in stats.get("by_strategy", {}).items():
            sp   = sd.get("net_pnl", 0)
            sc   = "#27ae60" if sp >= 0 else "#e74c3c"
            strat_rows += (
                f'<tr>'
                f'<td style="padding:8px 12px;">{strat}</td>'
                f'<td style="padding:8px 12px;text-align:center;">{sd.get("total_trades",0)}</td>'
                f'<td style="padding:8px 12px;text-align:center;">{sd.get("win_rate",0):.1f}%</td>'
                f'<td style="padding:8px 12px;text-align:right;color:{sc};font-weight:700;">₹{sp:+.2f}</td>'
                f'</tr>'
            )

        html = f"""
<html><body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,sans-serif;">
<div style="max-width:600px;margin:30px auto;background:white;border-radius:12px;
            box-shadow:0 4px 16px rgba(0,0,0,.10);overflow:hidden;">
  <div style="background:#2c3e50;padding:22px 24px;">
    <div style="font-size:22px;font-weight:700;color:white;">📊 EOD Summary — {date_str}</div>
    <div style="color:rgba(255,255,255,.7);font-size:13px;margin-top:4px;">
      Generated at {datetime.now().strftime('%H:%M IST')}
    </div>
  </div>

  <div style="padding:20px 24px;background:{pnl_color}12;text-align:center;border-bottom:2px solid {pnl_color}30;">
    <div style="font-size:13px;color:#666;text-transform:uppercase;letter-spacing:1px;">Net P&amp;L Today</div>
    <div style="font-size:44px;font-weight:700;color:{pnl_color};margin:6px 0;">₹{net_pnl:+.2f}</div>
  </div>

  <table style="width:100%;border-collapse:collapse;margin-bottom:4px;">
    {_card('Total Trades',       str(stats.get('total_trades', 0)),      '#fafafa')}
    {_card('Win Rate',           f'{stats.get("win_rate", 0):.1f}%')}
    {_card('Profit Factor',      pf_str,                                  '#fafafa')}
    {_card('Max Drawdown',       f'₹{stats.get("max_drawdown", 0):.2f}')}
    {_card('Signals Evaluated',  str(stats.get('signals_evaluated', 0)), '#fafafa')}
    {_card('Accepted Signals',   f'{stats.get("accepted_signals", stats.get("trades_taken", 0))} / {stats.get("signals_evaluated", 1)}')}
    {_card('Closed Trades',      str(stats.get('closed_trades', stats.get('total_trades', 0))), '#fafafa')}
  </table>

  <div style="padding:16px 24px 8px;font-weight:700;color:#2c3e50;font-size:15px;">By Strategy</div>
  <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
    <tr style="background:#2c3e50;color:white;font-size:13px;">
      <th style="padding:9px 12px;text-align:left;">Strategy</th>
      <th style="padding:9px 12px;text-align:center;">Trades</th>
      <th style="padding:9px 12px;text-align:center;">Win %</th>
      <th style="padding:9px 12px;text-align:right;">P&amp;L</th>
    </tr>
    {strat_rows if strat_rows else '<tr><td colspan="4" style="padding:12px;color:#aaa;text-align:center;">No closed trades today</td></tr>'}
  </table>

  <div style="padding:12px 24px;color:#aaa;font-size:11px;border-top:1px solid #f0f0f0;">
    NSE Paper Trading Bot · Paper mode active · No real orders placed
  </div>
</div></body></html>"""
        return self.send(subject, html)
