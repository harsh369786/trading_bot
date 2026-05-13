from abc import ABC, abstractmethod
from datetime import datetime
import random
from loguru import logger

class BaseBroker(ABC):
    @abstractmethod
    def place_order(self, symbol: str, qty: int, direction: str, order_type: str, price: float = None) -> dict:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_positions(self) -> list:
        pass

    @abstractmethod
    def get_balance(self) -> float:
        pass

class AngelOneBroker(BaseBroker):
    """
    Concrete implementation of Angel One (SmartAPI).
    """
    def __init__(self, api_key: str = None, secret: str = None):
        from SmartApi import SmartConnect
        import os
        self.api_key = api_key or os.environ.get("BROKER_API_KEY")
        self.secret = secret or os.environ.get("BROKER_SECRET")
        if not self.api_key:
            raise ValueError("BROKER_API_KEY is required for AngelOneBroker.")
        self.smart_api = SmartConnect(api_key=self.api_key)
        self.session_data = None

    def login(self, client_id: str, password: str, totp_secret: str):
        """Perform login and handle TOTP."""
        import pyotp
        if not all([client_id, password, totp_secret]):
            raise ValueError("Angel One login requires client_id, password, and totp_secret.")
        totp = pyotp.TOTP(totp_secret).now()
        self.session_data = self.smart_api.generateSession(client_id, password, totp)
        if self.session_data.get("status"):
            logger.info("Angel One: Login successful.")
        else:
            logger.error(f"Angel One: Login failed: {self.session_data.get('message')}")

    def place_order(self, symbol: str, qty: int, direction: str, order_type: str, price: float = None) -> dict:
        logger.error("Angel One live place_order is not implemented. Refusing to fake a live order.")
        return {"status": "FAILED", "reason": "AngelOneBroker.place_order is not implemented"}

    def cancel_order(self, order_id: str) -> bool:
        logger.info(f"Angel One: Cancelling order {order_id}")
        return True

    def get_positions(self) -> list:
        return []

    def get_balance(self) -> float:
        # Placeholder for SmartAPI rms/funds call
        return 100000.0

class MockBroker(BaseBroker):
    """
    High-fidelity Mock Broker for Paper Trading.
    Simulates fills, slippage, and order IDs.
    """
    def __init__(self):
        self.orders = {}
        self.positions = []

    def place_order(self, symbol: str, qty: int, direction: str, order_type: str, price: float = None) -> dict:
        order_id = f"MOCK-{random.randint(10000, 99999)}"
        
        # Simulate Slippage (0.01% - 0.05%)
        slippage = 1 + (random.uniform(0.0001, 0.0005) * (1 if direction == "BUY" else -1))
        fill_price = (price or 0) * slippage if order_type == "LIMIT" else price # Simplified
        
        logger.info(f"MockBroker: {direction} {qty} {symbol} @ {fill_price:.2f} (ID: {order_id})")
        
        res = {
            "status": "SUCCESS",
            "order_id": order_id,
            "fill_price": fill_price,
            "timestamp": datetime.now()
        }
        self.orders[order_id] = res
        return res

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    def get_positions(self) -> list:
        return self.positions

    def get_balance(self) -> float:
        return 50000.0 # Standard Mock Balance
