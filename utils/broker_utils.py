import requests
import pandas as pd
import os
import re
from loguru import logger

class AngelOneMaster:
    """
    Utility to fetch and parse Angel One Master Contract.
    Ensures tokens for USDINR, NIFTY, etc. are always current.
    """
    URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    CACHE_FILE = "data/master_contract.json"
    _is_downloading = False
    EXCHANGE_ALIASES = {
        "CDE_FO": "CDS",
        "CDS": "CDS",
        "NSE": "NSE",
        "NFO": "NFO",
    }

    @classmethod
    def update_contract(cls):
        """Download latest scrip master from Angel One."""
        if cls._is_downloading:
            return False
            
        try:
            cls._is_downloading = True
            logger.info("📡 Downloading Angel One Master Contract (30MB+)...")
            os.makedirs("data", exist_ok=True)
            response = requests.get(cls.URL, timeout=60)
            if response.status_code == 200:
                with open(cls.CACHE_FILE, "wb") as f:
                    f.write(response.content)
                logger.success("✅ Master Contract updated.")
                return True
        except Exception as e:
            logger.error(f"Failed to update master contract: {e}")
        finally:
            cls._is_downloading = False
        return False

    @classmethod
    def get_token(cls, symbol: str, exchange: str = "NSE"):
        """
        Find token for a given symbol.
        Example symbols: NIFTY, USDINR, RELIANCE
        """
        import time
        if not os.path.exists(cls.CACHE_FILE):
            cls.update_contract()
            
        # Wait if another thread is downloading
        retries = 0
        while cls._is_downloading and retries < 60:
            time.sleep(1)
            retries += 1
            
        try:
            if not os.path.exists(cls.CACHE_FILE):
                return None, None, None
            
            df = pd.read_json(cls.CACHE_FILE)
            exchange = cls.EXCHANGE_ALIASES.get(str(exchange).upper(), str(exchange).upper())
            symbol = str(symbol).upper()
            df['exch_seg'] = df['exch_seg'].astype(str).str.upper()
            df['name'] = df['name'].astype(str).str.upper()
            df['symbol'] = df['symbol'].astype(str).str.upper()
            df['instrumenttype'] = df['instrumenttype'].astype(str).str.upper()
            
            # Handle Futures (NIFTY, BANKNIFTY, USDINR, etc.)
            if symbol in ["USDINR", "EURINR", "GBPINR", "JPYINR", "NIFTY", "BANKNIFTY"]:
                exch_seg = 'CDS' if "INR" in symbol else 'NFO'
                instrument_type = 'FUTCUR' if "INR" in symbol else 'FUTIDX'
                match = df[(df['exch_seg'] == exch_seg) & 
                           (df['name'] == symbol) &
                           (df['instrumenttype'] == instrument_type) &
                           (df['symbol'].str.contains('FUT', na=False))]
                if not match.empty:
                    expiry = pd.to_datetime(match['expiry'], format='%d%b%Y', errors='coerce')
                    today = pd.Timestamp.today().normalize()
                    match = match.assign(_expiry=expiry)
                    future_expiries = match[match['_expiry'] >= today]
                    match = future_expiries if not future_expiries.empty else match
                    if "INR" in symbol:
                        monthly_pattern = re.compile(rf"^{re.escape(symbol)}\d{{2}}[A-Z]{{3}}FUT$")
                        monthly = match[match['symbol'].str.match(monthly_pattern, na=False)]
                        if not monthly.empty:
                            match = monthly
                    match = match.sort_values(by='_expiry')
                    return match.iloc[0]['token'], exch_seg, match.iloc[0]['symbol']
            
            # Standard Equity Lookup
            match = df[(df['exch_seg'] == exchange) & (df['name'] == symbol)]
            if not match.empty:
                return match.iloc[0]['token'], exchange, match.iloc[0]['symbol']
                
            # Fallback for symbols like "NIFTY 50"
            match = df[(df['exch_seg'] == exchange) & (df['symbol'] == symbol)]
            if not match.empty:
                return match.iloc[0]['token'], exchange, match.iloc[0]['symbol']
                
        except Exception as e:
            logger.error(f"Token lookup failed for {symbol}: {e}")
        
        return None, None, None
