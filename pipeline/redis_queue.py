import redis.asyncio as redis
import json
import os
from loguru import logger
from typing import Dict, Any

class RedisQueue:
    """
    Handles connection and communication with Redis for the trading bot.
    Uses Redis Streams for tick data.
    """
    def __init__(self, url: str = None):
        if url is None:
            url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.url = url
        self.client = None

    async def connect(self):
        """Establish async connection to Redis with a stable pool."""
        try:
            self.client = await redis.from_url(
                self.url, 
                decode_responses=True,
                health_check_interval=0 # Disable potentially problematic health checks on Windows
            )
            await self.client.ping()
            logger.info(f"Connected to Redis at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self):
        """Close the Redis connection."""
        if self.client:
            await self.client.aclose()
            logger.info("Redis connection closed")

    async def publish_tick(self, symbol: str, tick_data: Dict[str, Any]):
        """
        Publish a tick to a Redis Stream named `market:ticks:{symbol}`.
        """
        if not self.client:
            await self.connect()
        stream_name = f"market:ticks:{symbol}"
        try:
            # We convert dict values to string for Redis stream
            # Alternatively, store JSON string
            await self.client.xadd(stream_name, {"data": json.dumps(tick_data)}, maxlen=50000)
        except Exception as e:
            logger.error(f"Error publishing tick for {symbol}: {e}")

    async def read_ticks(self, symbol: str, last_id: str = "$") -> list:
        """
        Read new ticks for a symbol from the Redis Stream.
        last_id: "$" means only new messages, "0" means from beginning.
        """
        if not self.client:
            await self.connect()
        stream_name = f"market:ticks:{symbol}"
        try:
            # Returns list of [ (stream_name, [(msg_id, msg_dict), ...]) ]
            messages = await self.client.xread({stream_name: last_id}, count=1000, block=100)
            if not messages:
                return [], last_id
            
            stream_msgs = messages[0][1]
            new_last_id = stream_msgs[-1][0]
            
            parsed_ticks = []
            for msg_id, msg_dict in stream_msgs:
                parsed_ticks.append({
                    "msg_id": msg_id,
                    "data": json.loads(msg_dict.get("data", "{}"))
                })
                
            return parsed_ticks, new_last_id
        except Exception as e:
            logger.error(f"Error reading ticks for {symbol}: {e}")
            return [], last_id
