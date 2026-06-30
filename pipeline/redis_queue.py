import asyncio
import redis.asyncio as redis
import json
import os
import time
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
        self.max_connections = max(1, int(os.environ.get("REDIS_MAX_CONNECTIONS", "80")))
        requested_concurrent_ops = max(1, int(os.environ.get("REDIS_MAX_CONCURRENT_OPS", "20")))
        self.max_concurrent_ops = min(requested_concurrent_ops, self.max_connections)
        self.client = None
        self._connect_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._operation_semaphore = asyncio.Semaphore(self.max_concurrent_ops)
        self._client_generation = 0
        self._last_warning_ts = {}

    async def connect(self):
        """Establish async connection to Redis with a stable pool."""
        async with self._connect_lock:
            if self.client is not None:
                try:
                    await self.client.ping()
                    return
                except Exception:
                    await self._close_client_unlocked()

            await self._connect_unlocked()

    async def _connect_unlocked(self):
        """Open a Redis client. Caller must hold _connect_lock."""
        try:
            self.client = redis.from_url(
                self.url, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=self.max_connections,
            )
            await self.client.ping()
            self._client_generation += 1
            logger.info(f"Connected to Redis at {self.url}")
        except Exception as e:
            self.client = None
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self):
        """Close the Redis connection."""
        async with self._connect_lock:
            await self._close_client_unlocked()

    async def _close_client_unlocked(self):
        """Close current Redis client. Caller must hold _connect_lock."""
        if self.client:
            try:
                await self.client.aclose()
            except Exception:
                pass
            self.client = None
            logger.info("Redis connection closed")

    async def _reconnect(self, seen_generation: int | None = None):
        """Reset and reopen Redis once, with concurrent reconnects coalesced."""
        async with self._reconnect_lock:
            if (
                seen_generation is not None
                and self.client is not None
                and self._client_generation != seen_generation
            ):
                return
            async with self._connect_lock:
                await self._close_client_unlocked()
                await self._connect_unlocked()

    def _warn_throttled(self, key: str, message: str, interval: float = 10.0):
        """Log repeated transient Redis warnings at a bounded rate."""
        now = time.monotonic()
        last = self._last_warning_ts.get(key, 0.0)
        if now - last >= interval:
            self._last_warning_ts[key] = now
            logger.warning(message)

    @staticmethod
    def _is_pool_exhausted(exc: Exception) -> bool:
        """redis-py raises this when concurrency exceeds max_connections."""
        return "Too many connections" in str(exc)

    async def publish_tick(self, symbol: str, tick_data: Dict[str, Any]):
        """
        Publish a tick to a Redis Stream named `market:ticks:{symbol}`.
        """
        if not self.client:
            await self.connect()
        stream_name = f"market:ticks:{symbol}"
        payload = {"data": json.dumps(tick_data)}
        seen_generation = self._client_generation
        try:
            async with self._operation_semaphore:
                await self.client.xadd(stream_name, payload, maxlen=50000, approximate=True)
        except Exception as e:
            if self._is_pool_exhausted(e):
                self._warn_throttled(
                    "publish_pool",
                    f"Redis publish delayed; connection pool saturated. First symbol={symbol}. Error: {e}",
                )
                try:
                    await asyncio.sleep(0.05)
                    async with self._operation_semaphore:
                        await self.client.xadd(stream_name, payload, maxlen=50000, approximate=True)
                except Exception as retry_exc:
                    self._warn_throttled(
                        "publish_pool_retry",
                        f"Redis publish still failing after pool backoff. First symbol={symbol}. Error: {retry_exc}",
                        interval=10.0,
                    )
                return

            self._warn_throttled(
                "publish",
                f"Redis publish failed; reconnecting once. First symbol={symbol}. Error: {e}",
            )
            try:
                await self._reconnect(seen_generation)
                async with self._operation_semaphore:
                    await self.client.xadd(stream_name, payload, maxlen=50000, approximate=True)
            except Exception as retry_exc:
                self._warn_throttled(
                    "publish_retry",
                    f"Redis publish still failing after reconnect. First symbol={symbol}. Error: {retry_exc}",
                    interval=10.0,
                )

    async def read_ticks(self, symbol: str, last_id: str = "$") -> list:
        """
        Read new ticks for a symbol from the Redis Stream.
        last_id: "$" means only new messages, "0" means from beginning.
        """
        if not self.client:
            await self.connect()
        stream_name = f"market:ticks:{symbol}"
        seen_generation = self._client_generation
        try:
            # Returns list of [ (stream_name, [(msg_id, msg_dict), ...]) ]
            async with self._operation_semaphore:
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
            if self._is_pool_exhausted(e):
                self._warn_throttled(
                    "read_pool",
                    f"Redis read skipped; connection pool saturated. First symbol={symbol}. Error: {e}",
                )
                return [], last_id

            self._warn_throttled(
                "read",
                f"Redis read failed; reconnecting once. First symbol={symbol}. Error: {e}",
            )
            try:
                await self._reconnect(seen_generation)
            except Exception as retry_exc:
                self._warn_throttled(
                    "read_retry",
                    f"Redis reconnect failed while reading ticks. First symbol={symbol}. Error: {retry_exc}",
                    interval=10.0,
                )
            return [], last_id
