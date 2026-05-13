"""
strategies/rsmb/vix_monitor.py
---------------------------------
VIX spike detection for the RSMB strategy.

Spec:
- Store VIX values in a deque of length 12 (5-min poll × 12 = 60-min window)
- At signal time: vix_now = latest; vix_60m_ago = deque[0]
- Spike = (vix_now - vix_60m_ago) / vix_60m_ago × 100
- If spike > 5.0%: veto = True
- Thread-safe: all mutations under threading.Lock
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# VIXMonitor
# ---------------------------------------------------------------------------

class VIXMonitor:
    """
    Maintains a 60-minute rolling window of India VIX values.

    Usage
    -----
    monitor = VIXMonitor(spike_threshold_pct=5.0, window_size=12)
    monitor.update(vix_value)          # called every 5 min by scheduler
    veto, reason = monitor.is_veto()   # called at signal time
    """

    def __init__(
        self,
        spike_threshold_pct: float = 5.0,
        window_size: int = 12,
    ) -> None:
        self._threshold = spike_threshold_pct
        self._window_size = window_size
        self._deque: deque[Tuple[datetime, float]] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update(self, vix_value: float) -> None:
        """
        Record a new VIX observation.

        Parameters
        ----------
        vix_value : Current India VIX level (positive float).
        """
        if vix_value <= 0:
            logger.warning(f"VIXMonitor.update: received invalid VIX value {vix_value}; ignored")
            return

        now = datetime.now()
        with self._lock:
            self._deque.append((now, vix_value))
        logger.debug(f"VIXMonitor: VIX={vix_value:.2f} recorded at {now.strftime('%H:%M:%S')}")

    # ------------------------------------------------------------------
    # Veto check
    # ------------------------------------------------------------------

    def is_veto(self) -> Tuple[bool, str]:
        """
        Determine if a VIX spike veto is currently active.

        Returns
        -------
        (veto: bool, reason: str)
        veto=True means all RSMB trades must be blocked.
        """
        with self._lock:
            if len(self._deque) < 2:
                # Not enough data to compute spike — be conservative
                logger.debug("VIXMonitor: insufficient data for spike check; no veto")
                return False, "insufficient_data"

            vix_now = self._deque[-1][1]
            vix_old = self._deque[0][1]

        if vix_old == 0:
            return False, "zero_base_vix"

        spike_pct = (vix_now - vix_old) / vix_old * 100.0

        if spike_pct > self._threshold:
            reason = (
                f"VIX spike {spike_pct:.1f}% in last "
                f"{len(self._deque) * 5}min "
                f"(now={vix_now:.2f}, was={vix_old:.2f})"
            )
            logger.warning(f"VIXMonitor: VETO ACTIVE — {reason}")
            return True, reason

        return False, f"vix_ok:{spike_pct:.1f}%"

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_vix(self) -> Optional[float]:
        """Latest VIX value, or None if no data yet."""
        with self._lock:
            if not self._deque:
                return None
            return self._deque[-1][1]

    @property
    def spike_pct_60m(self) -> float:
        """
        Current spike percentage over the 60-minute window.
        Returns 0.0 if insufficient data.
        """
        with self._lock:
            if len(self._deque) < 2:
                return 0.0
            vix_now = self._deque[-1][1]
            vix_old = self._deque[0][1]

        if vix_old == 0:
            return 0.0
        return (vix_now - vix_old) / vix_old * 100.0

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def readings_count(self) -> int:
        with self._lock:
            return len(self._deque)

    def reset(self) -> None:
        """Clear all stored VIX values (called at session start)."""
        with self._lock:
            self._deque.clear()
        logger.info("VIXMonitor: reset — all VIX readings cleared")
