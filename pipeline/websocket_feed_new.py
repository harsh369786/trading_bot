"""
Compatibility module for older imports.

The active implementation lives in pipeline.websocket_feed.
"""

from .websocket_feed import WebSocketFeed

__all__ = ["WebSocketFeed"]
