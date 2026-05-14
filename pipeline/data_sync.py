import os
import yaml
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

def login_angelone():
    from SmartApi import SmartConnect
    import pyotp
    
    api_key = os.environ.get("BROKER_API_KEY")
    client_id = os.environ.get("ANGEL_CLIENT_ID")
    password = os.environ.get("ANGEL_PASSWORD")
    totp_secret = os.environ.get("ANGEL_TOTP_SECRET")
    
    if not all([api_key, client_id, password, totp_secret]):
        return None
        
    try:
        smart_api = SmartConnect(api_key=api_key)
        session = smart_api.generateSession(client_id, password, pyotp.TOTP(totp_secret).now())
        if session.get("status"):
            return smart_api
    except Exception as e:
        logger.error(f"Sync login failed: {e}")
    return None

def sync_morning_data():
    """Fetch missing candles for today's morning session from Angel One."""
    load_dotenv()
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    equity = config.get("instruments", {}).get("equity", [])
    currency = config.get("instruments", {}).get("currency", [])
    symbols = equity + currency
    
    smart_api = login_angelone()
    if not smart_api:
        logger.warning("DataSync: Skipping live sync (Broker credentials missing/invalid).")
        return

    from utils.broker_utils import AngelOneMaster
    
    end_time = datetime.now()
    # Market opens at 09:15
    start_time = end_time.replace(hour=9, minute=15, second=0, microsecond=0)
    
    if end_time < start_time:
        logger.info("DataSync: Market hasn't opened yet today. Skipping.")
        return

    os.makedirs("data/historical", exist_ok=True)
    
    for symbol in symbols:
        token, exch, tradingsymbol = AngelOneMaster.get_token(symbol)
        if not token:
            continue
            
        path = f"data/historical/{symbol}_6m.parquet"
        last_ts = None
        existing_df = pd.DataFrame()
        
        if os.path.exists(path):
            existing_df = pd.read_parquet(path)
            if not existing_df.empty:
                last_ts = existing_df.index.max()
                if last_ts.tzinfo:
                    last_ts = last_ts.tz_localize(None)
        
        # If last_ts is already from today and > 15:15 (or close to now), skip
        if last_ts and last_ts >= end_time - timedelta(minutes=10):
            continue
            
        fetch_start = last_ts + timedelta(minutes=5) if last_ts else start_time
        if fetch_start > end_time:
            continue

        params = {
            "exchange": exch,
            "symboltoken": str(token),
            "interval": "FIVE_MINUTE",
            "fromdate": fetch_start.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M"),
        }
        
        try:
            res = smart_api.getCandleData(params)
            if res and res.get("status") and res.get("data"):
                new_df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
                new_df = new_df.set_index("timestamp")
                for col in ["open", "high", "low", "close", "volume"]:
                    new_df[col] = pd.to_numeric(new_df[col])
                new_df["oi"] = 0
                
                if not existing_df.empty:
                    df = pd.concat([existing_df, new_df])
                    df = df[~df.index.duplicated(keep='last')].sort_index()
                else:
                    df = new_df
                
                df.to_parquet(path)
                logger.info(f"DataSync: {symbol} synchronized up to {df.index.max()}")
        except Exception as e:
            logger.error(f"DataSync: {symbol} fetch failed: {e}")

if __name__ == "__main__":
    sync_morning_data()
