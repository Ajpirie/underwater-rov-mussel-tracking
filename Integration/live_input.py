"""Thread-safe, timestamped observations from an external tracking source.

No frame metadata, transport or camera API is assumed. The external adapter
calls publish_observation() once for each (ID, x, y, z) measurement.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
import math
import time


@dataclass(frozen=True)
class Observation:
    mussel_id: int
    position: tuple[float, float, float]
    time_s: float


class LiveObservationBuffer:
    """Bounded FIFO; timestamps are assigned at receipt using a monotonic clock."""

    def __init__(self, max_observations: int = 2000):
        if max_observations < 1:
            raise ValueError('Queue capacity must be positive')
        self._lock = Lock()
        self._queue = deque()
        self._capacity = max_observations
        self._origin_ns = time.monotonic_ns()
        self.dropped_observations = 0

    def clear(self):
        """Start a fresh Run session and reset the receipt-time origin."""
        with self._lock:
            self._queue.clear()
            self._origin_ns = time.monotonic_ns()
            self.dropped_observations = 0

    def elapsed_time(self) -> float:
        """Seconds since clear(), on the same clock as queued observations."""
        with self._lock:
            return max(0.0, (time.monotonic_ns() - self._origin_ns) * 1e-9)

    def submit_detection(self, mussel_id: int, x: float, y: float, z: float):
        """Accept one observation immediately; no finish_frame() is needed."""
        if isinstance(mussel_id, bool) or not isinstance(mussel_id, int) or mussel_id < 0:
            raise ValueError('Mussel ID must be a non-negative integer')
        pos = (float(x), float(y), float(z))
        if not all(math.isfinite(v) for v in pos):
            raise ValueError('Coordinates must be finite metres')
        with self._lock:
            timestamp = (time.monotonic_ns() - self._origin_ns) * 1e-9
            if len(self._queue) >= self._capacity:
                self.dropped_observations += 1
                self._queue.popleft()
            self._queue.append(Observation(mussel_id, pos, max(0.0, timestamp)))

    def read(self, max_observations: int = 256) -> list[Observation]:
        """Drain observations in arrival order, on the consumer thread."""
        if max_observations < 1:
            raise ValueError('Read limit must be positive')
        with self._lock:
            return [self._queue.popleft() for _ in range(min(max_observations, len(self._queue)))]


live_source = LiveObservationBuffer()


def publish_observation(mussel_id: int, x: float, y: float, z: float):
    """Connect push callbacks or pull/polling results here."""
    live_source.submit_detection(mussel_id, x, y, z)
