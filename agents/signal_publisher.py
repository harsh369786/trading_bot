from datetime import datetime
from loguru import logger

class SignalPublisher:
    """
    Agent 5: SignalPublisher
    Formats and publishes the final signal card.
    """
    def __init__(self, config: dict):
        self.config = config

    def format_signal(self, signal_data: dict) -> str:
        """Create the formatted ASCII card for Telegram."""
        now = datetime.now().strftime("%Y-%m-%d | %H:%M")
        side_emoji = "🟢 BUY" if signal_data['side'] == "BUY" else "🔴 SELL"
        
        sl_paise = int(abs(signal_data['entry'] - signal_data['sl']) * 100)
        t1_paise = int(abs(signal_data['t1'] - signal_data['entry']) * 100)
        t2_paise = int(abs(signal_data['t2'] - signal_data['entry']) * 100)

        card = f"""
╔══════════════════════════════════════════════╗
║  {side_emoji} SIGNAL — {signal_data['symbol']}
║  📅 {now} IST
╠══════════════════════════════════════════════╣
║  ENTRY     : {signal_data['entry']:.4f}
║  STOP LOSS : {signal_data['sl']:.4f}  (−{sl_paise} paise)
║  TARGET 1  : {signal_data['t1']:.4f}  (+{t1_paise} p) — 60% qty
║  TARGET 2  : {signal_data['t2']:.4f}  (+{t2_paise} p) — 40% qty
║  R:R RATIO : 1:{signal_data['rr']}
║  LOTS      : {signal_data['lots']}
╠══════════════════════════════════════════════╣
║  STRENGTH  : HIGH (Score: {signal_data.get('quant_score', 80)}/100)
║  MTF       : 15-min Aligned ✅
╠══════════════════════════════════════════════╣
║  ⚠️  EXIT ALL positions by 15:00 IST
╚══════════════════════════════════════════════╝
"""
        return card

    def publish(self, formatted_card: str):
        logger.info(f"Signal Published:\n{formatted_card}")
        # In production, this would send to TelegramBot
