"""Live per-ID UKFs for tracking streams without camera frame boundaries.

Every observation updates only its own filter. A snapshot is an event-level
view of the most recently received positions, not a simultaneous camera frame.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math

import numpy as np

from Integration.live_input import Observation
from Integration.tracking_session import Score, Snapshot, TrackingSession, _Track


class ObservationSession(TrackingSession):
    """Live tracking session with asynchronous, individually timed observations."""

    def __init__(self, dt: float, horizon_s: float, stale_after_s: float = None,
                 max_id: int = 999, lull_threshold_mps: float = 0.015,
                 calm_window_s: float = 2.0, calm_min_s: float = 1.0):
        super().__init__(dt, horizon_s, stale_after_s, max_id,
                         lull_threshold_mps, calm_window_s, calm_min_s)
        self._due_by_id = defaultdict(deque)
        self._previous_observation = {}
        self._last_event_time_s = None
        self.observation_count = 0
        self._centroid_ids = frozenset()
        self._calm_start_s = None

    def process(self, frame):
        """Frame-based processing is reserved for the recorded Test session."""
        raise TypeError('Use observe(Observation) for live input; no frame boundaries exist')

    def observe(self, observation: Observation) -> Snapshot:
        """Process one incoming observation without stepping any other track."""
        t = float(observation.time_s)
        mid = observation.mussel_id
        pos = tuple(observation.position)
        if not math.isfinite(t) or t < 0:
            raise ValueError('Observation time must be a non-negative finite number')
        if self._last_event_time_s is not None and t < self._last_event_time_s:
            raise ValueError('Observation timestamps must be non-decreasing')
        if isinstance(mid, bool) or not isinstance(mid, int) or mid < 0:
            raise ValueError('Mussel ID must be a non-negative integer')
        if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
            raise ValueError('Invalid observation coordinates')
        self._last_event_time_s = t
        observation_index = self.observation_count
        self.observation_count += 1
        index = len(self.snapshots)
        if mid > self.max_id or pos == (0.0, 0.0, 0.0):
            return self._snapshot(t)

        old_track = self.tracks.get(mid)
        new_track = (old_track is None or
                     t - old_track.last_seen_time_s > self.stale_after_s)
        if new_track:
            predictor = self.predictor.predictionInstance(
                (*pos, mid), timestep=self.dt, num_timesteps=self.steps)
            generation = 0 if old_track is None else old_track.generation + 1
            track = _Track(predictor, index, t, generation)
            if old_track is not None:
                track.history = old_track.history
            self.tracks[mid] = track
            self._due_by_id[mid].clear()
            self._previous_observation.pop(mid, None)
        else:
            track = old_track
            previous_t, previous_pos, previous_generation = self._previous_observation[mid]
            elapsed = t - previous_t
            if elapsed <= 0:
                raise ValueError(f'ID {mid} received two observations at the same timestamp')
            due_queue = self._due_by_id[mid]
            while due_queue and due_queue[0][0] <= t:
                due, forecast, hold, origin = due_queue.popleft()
                if due < previous_t:
                    continue  # No measurement spanning the requested target time.
                fraction = (due - previous_t) / elapsed
                actual = tuple(float(a + fraction * (b - a))
                               for a, b in zip(previous_pos, pos))
                error = tuple(float(a - b) for a, b in zip(forecast, actual))
                baseline = tuple(float(a - b) for a, b in zip(hold, actual))
                score = Score(index, t, mid, error, baseline, origin, due)
                self.scores.append(score)
                self._sum_pred_squares += score.error_m ** 2
                self._sum_hold_squares += score.hold_m ** 2

            track.predictor.x_history.append(pos[0])
            track.predictor.y_history.append(pos[1])
            track.predictor.z_history.append(pos[2])
            track.last_seen_frame = index
            track.last_seen_time_s = t

        track.observed += 1
        p = track.predictor
        # The first observation only initialises the UKF; its elapsed value is unused.
        p.predictPositionUKF(elapsed_s=self.dt if new_track else elapsed)
        rejected = bool(p.rejected_meas)
        if rejected:
            track.rejected_count += 1
        filtered = tuple(float(x) for x in p.ukf.x[:3])
        velocity = tuple(float(x) for x in p.ukf.x[3:6])
        forecast = tuple(float(v[-1]) for v in
                         (p.x_future_history, p.y_future_history, p.z_future_history))
        row = {
            'frame': index, 'observation_index': observation_index,
            'time_s': t, 'raw': pos, 'filtered': filtered,
            'forecast': forecast, 'velocity': velocity, 'fresh': True,
            'age': 0, 'age_s': 0.0,
            'frame_interval_s': self.dt if new_track else elapsed,
            'rejected': rejected, 'rejected_count': track.rejected_count,
        }
        track.history.append(row)
        self._previous_observation[mid] = (t, pos, track.generation)
        if track.observed >= 2:
            self._due_by_id[mid].append((t + self.horizon_s, forecast, pos, index))
        return self._snapshot(t, updated_id=mid)

    def expire_at(self, time_s: float):
        """Update the display only when tracks have become inactive in silence."""
        if not self.snapshots or time_s <= self.snapshots[-1].time_s:
            return None
        previous_active = {mid for mid, track in self.tracks.items()
                           if self.snapshots[-1].time_s - track.last_seen_time_s <= self.stale_after_s}
        current_active = {mid for mid, track in self.tracks.items()
                          if time_s - track.last_seen_time_s <= self.stale_after_s}
        if previous_active != current_active:
            return self._snapshot(time_s)
        return None

    def _snapshot(self, t: float, updated_id=None) -> Snapshot:
        """Build a view from last-known positions; no UKFs are advanced here."""
        index = len(self.snapshots)
        views = {}
        for mid, track in self.tracks.items():
            age_s = t - track.last_seen_time_s
            if age_s > self.stale_after_s or not track.history:
                continue
            view = dict(track.history[-1])
            view.update(frame=index, age=index - track.last_seen_frame,
                        age_s=age_s, fresh=(mid == updated_id),
                        raw=track.history[-1]['raw'] if mid == updated_id else None)
            views[mid] = view

        members = frozenset(views)
        centroid = None
        centroid_future = None
        speed = None
        calm = False
        if views:
            centroid = tuple(float(x) for x in np.mean([v['filtered'] for v in views.values()], axis=0))
            centroid_future = tuple(float(x) for x in np.mean([v['forecast'] for v in views.values()], axis=0))
        if updated_id is None or members != self._centroid_ids:
            self.centroid_window.clear()
            self._calm_start_s = None
            self.calm_elapsed_s = 0.0
            self.calm_streak = 0
        if updated_id is not None and centroid is not None:
            self.centroid_window.append((t, np.asarray(centroid)))
            while len(self.centroid_window) > 2 and t - self.centroid_window[1][0] >= self.calm_window_s:
                self.centroid_window.popleft()
            if len(self.centroid_window) >= 2:
                samples = list(self.centroid_window)
                duration = samples[-1][0] - samples[0][0]
                if duration > 0:
                    distance = sum(float(np.linalg.norm(b[1] - a[1]))
                                   for a, b in zip(samples, samples[1:]))
                    speed = distance / duration
                    if duration >= self.calm_window_s and speed < self.lull_threshold_mps:
                        if self._calm_start_s is None:
                            self._calm_start_s = t
                        self.calm_streak += 1
                        self.calm_elapsed_s = t - self._calm_start_s
                        calm = self.calm_elapsed_s >= self.calm_min_s
                    else:
                        self._calm_start_s = None
                        self.calm_streak = 0
                        self.calm_elapsed_s = 0.0
        self._centroid_ids = members
        n = len(self.scores)
        rmse = math.sqrt(self._sum_pred_squares / n) if n else None
        hold_rmse = math.sqrt(self._sum_hold_squares / n) if n else None
        rejected_total = sum(track.rejected_count for track in self.tracks.values())
        snap = Snapshot(index, t, views, centroid, centroid_future,
                        int(updated_id in views), len(views), len(self.tracks),
                        speed, calm, self.calm_streak, n, rmse, hold_rmse,
                        bool(n), rejected_total)
        self.snapshots.append(snap)
        return snap
