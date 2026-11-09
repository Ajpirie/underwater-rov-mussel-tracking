"""Standalone recorded-data lull detector used by verification scripts."""

from collections import deque
import numpy as np
from Data_Classes import LullStatus


class LullDetector:
    """
    Flags a calm ("lull") window in otherwise wave-driven oscillatory
    motion of a tracked point (e.g. a mussel-cluster centroid), so a
    docking/collection manoeuvre can be timed to coincide with reduced
    motion rather than the ROV continuously chasing a swaying target.

    Deliberately literal, so it's easy to reason about and to explain in
    a report:

        calm  <=>  windowed speed < speed_threshold, held for >= min_calm_seconds

    Speed is a short rolling window's path length divided by its
    duration (not a first-to-last-point displacement -- a point that
    swings out and back over the window would otherwise look falsely
    calm). The debounce (min_calm_seconds) avoids flagging/unflagging
    every frame right at the threshold crossing -- the same
    consecutive-frame pattern already used for dropout/stale-track
    detection elsewhere in this pipeline.

    speed_threshold has no universal default: how fast counts as "calm"
    depends on the sea state and how the tracked point is defined (a
    single mussel vs. a group centroid). Use lull_verification.py to plot
    a real track's speed against candidate thresholds and pick one that
    visually separates genuine lulls from active swings, rather than
    guessing a number.

    An earlier version of this class self-calibrated a threshold from a
    slow-moving average of the point's own recent speed. That added a
    second time constant and a dimensionless ratio that were hard to
    reason about for only a modest robustness gain, so it's been dropped
    in favour of this simpler, directly-in-physical-units version.

    Feed this continuously from tracking start (not only once locked or
    aligned).
    """

    def __init__(
        self,
        dt: float,
        speed_threshold: float,
        window_seconds: float = 2.0,
        min_calm_seconds: float = 1.0,
    ):
        """
        Args:
            dt (float): Time between updates (matches data_time_step).
            speed_threshold (float): Speed (m/s) below which the current
                window counts as "calm". Pick this from real data via
                lull_verification.py -- there's no safe universal default.
            window_seconds (float): Length of the rolling window used for
                the speed estimate. Should be shorter than the expected
                duration of a lull (or the dip gets averaged away) but
                long enough (a few frames) to smooth measurement noise.
            min_calm_seconds (float): How long speed must stay below
                speed_threshold, continuously, before is_calm reports True.
        """
        self.dt = dt
        self.speed_threshold = speed_threshold
        self.window_len = max(2, round(window_seconds / dt))
        self.min_calm_frames = max(1, round(min_calm_seconds / dt))

        self._pos_buffer = deque(maxlen=self.window_len)
        self._calm_streak = 0

    def update(self, position: tuple) -> LullStatus:
        """
        Feed one new position sample (e.g. this frame's centroid) and get
        back the current lull status.

        Args:
            position (tuple): (x, y, z) position for this timestep.

        Returns:
            LullStatus: current calm classification, the speed it was
            computed from, and how long it's been calm.
        """
        self._pos_buffer.append(np.asarray(position, dtype=float))

        if len(self._pos_buffer) < 2:
            speed = 0.0
        else:
            window = np.array(self._pos_buffer)
            step_lengths = np.linalg.norm(np.diff(window, axis=0), axis=1)
            duration = (len(self._pos_buffer) - 1) * self.dt
            speed = float(np.sum(step_lengths) / duration) if duration > 0 else 0.0

        if speed < self.speed_threshold:
            self._calm_streak += 1
        else:
            self._calm_streak = 0

        return LullStatus(
            is_calm=(self._calm_streak >= self.min_calm_frames),
            speed=speed,
            calm_streak=self._calm_streak,
        )
