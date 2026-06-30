import os
import sys

# Ensure we're in the right directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from notifications.email_notifier import EmailNotifier
import subprocess

def test_emails():
    notifier = EmailNotifier()
    if not notifier.enabled:
        print("[ERROR] NOT ENABLED: Please check your .env file. NOTIFY_EMAIL_FROM or NOTIFY_EMAIL_PASSWORD is missing.")
        return

    print("[OK] Credentials found in .env. Attempting to send test emails...")

    # 1. Test Trade Fill
    print("Sending Test Trade Fill email...")
    success1 = notifier.send_trade_fill(
        symbol="RELIANCE",
        side="BUY",
        strategy="Ensemble_AI",
        entry=2950.50,
        sl=2920.00,
        target=3011.50,
        score=0.89
    )
    if success1:
        print("  -> Trade Fill Email Sent Successfully!")
    else:
        print("  -> Failed to send Trade Fill Email.")

    # 2. Test Trade Close
    print("Sending Test Trade Close email...")
    success2 = notifier.send_trade_close(
        symbol="RELIANCE",
        side="BUY",
        strategy="Ensemble_AI",
        entry=2950.50,
        exit_price=3011.50,
        pnl=610.00,
        outcome="TARGET_HIT",
        qty=10
    )
    if success2:
        print("  -> Trade Close Email Sent Successfully!")
    else:
        print("  -> Failed to send Trade Close Email.")

    # 3. Test EOD Summary
    print("Triggering actual EOD Summary script...")
    result = subprocess.run([sys.executable, "scripts/eod_summary.py"], capture_output=True, text=True)
    if "success" in result.stdout.lower() or result.returncode == 0:
         print("  -> EOD Summary Email Sent Successfully!")
    else:
         print(f"  -> EOD Summary Failed. Output: {result.stdout}")

    if success1 and success2:
        print("\n[SUCCESS] ALL TESTS PASSED! Check your email inbox.")
    else:
        print("\n[WARNING] SOME TESTS FAILED. Please double-check your App Password in the .env file.")

if __name__ == "__main__":
    test_emails()
