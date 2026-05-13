import asyncio
import sqlite3
import collections
from loguru import logger
from typing import Dict, Any, Deque
from .redis_queue import RedisQueue
import time

class LivePriceStreamer:
    """
    Consumes ticks from Redis, maintains a RAM buffer of last 500 ticks per symbol,
    and persists them to SQLite periodically.
    """
    def __init__(self, config: dict, redis_queue=None):
        self.config = config
        self.symbols = config.get("instruments", {}).get("equity", []) + config.get("instruments", {}).get("currency", [])
        self.redis_queue = redis_queue
        # In-memory buffer of last 500 ticks per symbol
        self.tick_buffers: Dict[str, Deque[Dict[str, Any]]] = {
            sym: collections.deque(maxlen=500) for sym in self.symbols
        }
        self.sqlite_db_path = "data/ticks.db"
        self._init_db()
        self.last_persist_time = time.time()
        self.pending_ticks = []

    def _init_db(self):
        """Initialize SQLite database for ticks."""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                ltp REAL,
                volume INTEGER,
                bid REAL,
                ask REAL,
                oi INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def _persist_ticks(self):
        """Persist pending ticks to SQLite."""
        if not self.pending_ticks:
            return
            
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            
            # Prepare data for bulk insert
            data = [
                (t.get("symbol"), t.get("timestamp"), t.get("ltp"), t.get("volume"), 
                 t.get("bid"), t.get("ask"), t.get("oi"))
                for t in self.pending_ticks
            ]
            
            cursor.executemany('''
                INSERT INTO ticks (symbol, timestamp, ltp, volume, bid, ask, oi)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
            conn.close()
            
            # Clear pending after successful insert
            self.pending_ticks.clear()
            self.last_persist_time = time.time()
        except Exception as e:
            logger.error(f"Failed to persist ticks to SQLite: {e}")

    async def _process_symbol(self, symbol: str, last_id: str):
        """Process ticks for a single symbol."""
        while True:
            ticks, new_last_id = await self.redis_queue.read_ticks(symbol, last_id)
            for tick_wrapper in ticks:
                tick_data = tick_wrapper["data"]
                # Add to memory buffer
                self.tick_buffers[symbol].append(tick_data)
                # Add to pending for DB persistence
                self.pending_ticks.append(tick_data)
                
            last_id = new_last_id
            
            # Persist to SQLite every 60 seconds
            if time.time() - self.last_persist_time >= 60:
                self._persist_ticks()
                
            await asyncio.sleep(0.01) # small sleep to prevent CPU hogging

    async def run(self):
        """Main entry point for live price streamer."""
        logger.info("Initializing Live Price Streamer...")
        
        # Start a processing task for each symbol
        tasks = []
        for symbol in self.symbols:
            tasks.append(asyncio.create_task(self._process_symbol(symbol, "$")))
            
        await asyncio.gather(*tasks)
