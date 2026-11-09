"""Tracking data, UKF integration and online forecast scoring.

All spatial values are metres and timestamps are seconds.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import re
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TRACK_COLS = {"xpos", "ypos", "zpos", "ID"}


@dataclass(frozen=True)
class Frame:
    index: int
    time_s: float
    detections: dict[int, tuple[float, float, float]]


@dataclass(frozen=True)
class Score:
    frame: int
    time_s: float
    mussel_id: int
    forecast_error_m: tuple[float, float, float]
    hold_error_m: tuple[float, float, float]
    origin_frame: int
    target_time_s: float = 0.0

    @property
    def error_m(self):
        return float(np.linalg.norm(self.forecast_error_m))

    @property
    def hold_m(self):
        return float(np.linalg.norm(self.hold_error_m))


@dataclass(frozen=True)
class Snapshot:
    index: int
    time_s: float
    tracks: dict[int, dict]
    centroid: Optional[tuple[float, float, float]]
    centroid_forecast: Optional[tuple[float, float, float]]
    fresh: int
    active: int
    total: int
    speed_mps: Optional[float]
    is_calm: bool
    calm_frames: int
    n_scores: int
    rmse_m: Optional[float]
    hold_rmse_m: Optional[float]
    evaluated: bool
    rejected_total: int = 0


def discover_datasets(root: Path = ROOT) -> list[Path]:
    """Find CSVs with the project's tracking columns; exclude unrelated wave files."""
    results = []
    for path in (root / "Data").rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8-sig") as f:
                columns = set(next(csv.reader(f)))
            if TRACK_COLS <= columns:
                results.append(path)
        except (OSError, StopIteration, UnicodeError):
            continue
    return sorted(results, key=lambda p: str(p).lower())


def recommended_dt(path: Path) -> float:
    """RD recordings were collected at 2 fps; the 260721 tracker at 15 fps."""
    return 1.0 / 15.0 if path.parent.name == "15fps" else 0.5


def _f(value, name: str) -> float:
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        return number
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number; received {value!r}") from exc


def _frame_numbers_15fps(raw_path: Path) -> list[int]:
    """Restore frame indices from the original 15 fps multi-object tracker dump.

    Adapted from Test_Verification/Benchmark_UKF.py; the CSV
    deliberately does not contain a frame index.
    """
    text = raw_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    num = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
    pat = re.compile(rf"^\s*(\[{{1,2}})\s*{num}\s+{num}\s+{num}\s+{num}\s*(\]{{1,2}})\s*$")
    frames: list[int] = []
    frame, rows_in_frame = 0, 0
    for line in text.splitlines():
        match = pat.match(line)
        if match is None:
            continue
        if match.group(1) == "[[" and rows_in_frame:
            frame += 1
            rows_in_frame = 0
        frames.append(frame)
        rows_in_frame += 1
        if match.group(2) == "]]":
            frame += 1
            rows_in_frame = 0
    return frames


def load_csv_frames(path: Path, dt: float, grouping: str = "auto") -> list[Frame]:
    """Load rows and reconstruct frames without treating all-zero ID 0 as an object.

    Modes: auto, rows (one row/frame), repeated (new frame when an ID repeats).
    For Data/Real/15fps/260721.csv, auto uses adjacent 260721.txt to
    recover the exact video frame index. Other RD files are single-row frames.
    """
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("Frame period must be greater than 0 seconds")
    path = Path(path).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not TRACK_COLS <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(TRACK_COLS))}")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                mid_float = _f(row["ID"], "ID")
                if not mid_float.is_integer() or mid_float < 0:
                    raise ValueError("ID must be a non-negative integer")
                pos = tuple(_f(row[key], key) for key in ("xpos", "ypos", "zpos"))
                rows.append((int(mid_float), pos))
            except ValueError as exc:
                raise ValueError(f"{path.name}, CSV line {line}: {exc}") from exc
    if not rows:
        raise ValueError(f"No detections in {path.name}")
    if grouping not in {"auto", "rows", "repeated"}:
        raise ValueError("Unknown frame grouping mode")

    is_15fps = path.parent.name == "15fps" and path.stem == "260721"
    if grouping == "auto" and is_15fps:
        txt = path.with_suffix(".txt")
        if not txt.is_file():
            raise ValueError("The 15 fps file requires its companion 260721.txt for correct frame timing")
        numbers = _frame_numbers_15fps(txt)
        if len(numbers) != len(rows):
            raise ValueError(f"Tracker metadata has {len(numbers)} rows, CSV has {len(rows)}")
    elif grouping == "repeated" or (grouping == "auto" and "Synthetic" in path.parts):
        numbers, current, seen = [], 0, set()
        for mid, _ in rows:
            if mid in seen:
                current += 1
                seen.clear()
            numbers.append(current)
            seen.add(mid)
    else:
        numbers = list(range(len(rows)))

    detections = defaultdict(dict)
    for (mid, pos), frame in zip(rows, numbers):
        if mid == 0 and pos == (0.0, 0.0, 0.0):
            # RD_004 has dummy ID-0 dropout frames, not an actual track.
            continue
        if mid in detections[frame]:
            raise ValueError(f"Duplicate ID {mid} in frame {frame}. Choose 'Rows' grouping.")
        detections[frame][mid] = pos
    return [Frame(i, i * dt, detections[i]) for i in range(max(numbers) + 1)]


@dataclass
class _Track:
    predictor: object
    last_seen_frame: int
    last_seen_time_s: float
    generation: int = 0
    observed: int = 0
    history: list[dict] = field(default_factory=list)
    rejected_count: int = 0


class TrackingSession:
    """Per-frame UKF, online accuracy scoring, centroid and lull indication.

    Forecasts are scored only when their future ground-truth frame arrives.
    Missing measurements propagate as (0,0,0) while the track is active;
    the filter is reset on re-acquisition after the stale window.
    """
    def __init__(self, dt: float, horizon_s: float, stale_after_s: float = None,
                 max_id: int = 999,
                 lull_threshold_mps: float = 0.015,
                 calm_window_s: float = 2.0, calm_min_s: float = 1.0):
        if not (math.isfinite(dt) and 0 < dt <= 10):
            raise ValueError("Nominal frame period must be between 0 and 10 s")
        if not (math.isfinite(horizon_s) and 0 < horizon_s <= 120):
            raise ValueError("Prediction horizon must be between 0 and 120 s")
        if stale_after_s is None:
            stale_after_s = 6 * dt  # Previously six nominal frames.
        if not (math.isfinite(stale_after_s) and stale_after_s > 0) or max_id < 0:
            raise ValueError("ID timeout must be positive; maximum ID non-negative")
        if not (math.isfinite(lull_threshold_mps) and lull_threshold_mps > 0):
            raise ValueError("Lull speed threshold must be positive")
        if not (math.isfinite(calm_window_s) and calm_window_s > 0 and
                math.isfinite(calm_min_s) and calm_min_s > 0):
            raise ValueError("Lull durations must be positive")
        from Prediction import prediction_method
        self.predictor = prediction_method
        self.model = "prediction_method"
        self.dt = dt  # Nominal frame period, used for the forecast grid only.
        self.steps = max(1, int(round(horizon_s / dt)))
        if self.steps > 10000:
            raise ValueError("Horizon is too long for this frame period")
        self.horizon_s = self.steps * dt
        self.stale_after_s = float(stale_after_s)
        self.max_id = max_id
        self.lull_threshold_mps = lull_threshold_mps
        self.calm_window_s = calm_window_s
        self.calm_min_s = calm_min_s
        self.centroid_window = deque()
        self.calm_elapsed_s = 0.0
        self.calm_streak = 0  # Counts below-threshold frames for display.
        self.tracks: dict[int, _Track] = {}
        self.snapshots: list[Snapshot] = []
        self.scores: list[Score] = []
        self._pending = deque()  # Ordered by forecast target time.
        self._previous_frame = None
        self._sum_pred_squares = 0.0
        self._sum_hold_squares = 0.0

    def process(self, frame: Frame) -> Snapshot:
        if frame.index != len(self.snapshots):
            raise ValueError(f"Expected frame {len(self.snapshots)}; received {frame.index}")
        if not math.isfinite(frame.time_s) or frame.time_s < 0:
            raise ValueError("Frame time must be a non-negative finite number")
        previous = self._previous_frame
        if previous is not None and frame.time_s <= previous.time_s:
            raise ValueError("Frame time must increase on each completed frame")
        idx, detections = frame.index, frame.detections
        elapsed = self.dt if previous is None else frame.time_s - previous.time_s
        if math.isclose(elapsed, self.dt, rel_tol=1e-12, abs_tol=1e-12):
            elapsed = self.dt  # Avoid timestamp rounding noise in fixed-rate replay.
        tolerance = 1e-10 * max(1.0, abs(frame.time_s))

        # Evaluate forecasts when the target time is reached, not a fixed frame count.
        while self._pending and self._pending[0][0] <= frame.time_s + tolerance:
            due, mid, forecast, hold, origin, generation = self._pending.popleft()
            target = detections.get(mid)
            track = self.tracks.get(mid)
            if (target is None or target == (0.0, 0.0, 0.0) or
                    track is None or track.generation != generation or
                    frame.time_s - track.last_seen_time_s > self.stale_after_s + tolerance):
                continue
            if abs(due - frame.time_s) <= tolerance:
                actual = target
            elif previous is not None and previous.time_s <= due:
                before = previous.detections.get(mid)
                if before is None or before == (0.0, 0.0, 0.0):
                    continue
                weight = (due - previous.time_s) / (frame.time_s - previous.time_s)
                actual = tuple(float(a + weight * (b - a)) for a, b in zip(before, target))
            else:
                continue  # Target time fell in an unobserved gap.
            pred_error = tuple(float(p - v) for p, v in zip(forecast, actual))
            hold_error = tuple(float(p - v) for p, v in zip(hold, actual))
            result = Score(idx, frame.time_s, mid, pred_error, hold_error, origin, due)
            self.scores.append(result)
            self._sum_pred_squares += result.error_m ** 2
            self._sum_hold_squares += result.hold_m ** 2

        views = {}
        for mid, pos in detections.items():
            if mid > self.max_id:
                continue
            if len(pos) != 3 or not all(math.isfinite(p) for p in pos):
                raise ValueError(f"Invalid position for ID {mid}")
            if pos == (0.0, 0.0, 0.0):
                continue
            track = self.tracks.get(mid)
            is_new = track is None or frame.time_s - track.last_seen_time_s > self.stale_after_s + tolerance
            if is_new:
                instance = self.predictor.predictionInstance(
                    (*pos, mid), timestep=self.dt, num_timesteps=self.steps,
                )
                generation = 0 if track is None else track.generation + 1
                replacement = _Track(instance, idx, frame.time_s, generation)
                if track is not None:
                    replacement.history = track.history
                self.tracks[mid] = replacement
                track = replacement
            else:
                track.last_seen_frame = idx
                track.last_seen_time_s = frame.time_s
            track.observed += 1

        for mid, track in self.tracks.items():
            age_s = frame.time_s - track.last_seen_time_s
            if age_s > self.stale_after_s + tolerance:
                continue
            current = detections.get(mid)
            valid = current is not None and current != (0.0, 0.0, 0.0)
            pos = current if valid else (0.0, 0.0, 0.0)
            p = track.predictor
            if p.x_future_history:
                p.x_history.append(pos[0]); p.y_history.append(pos[1]); p.z_history.append(pos[2])
            p.predictPositionUKF(elapsed_s=elapsed)
            rejected = bool(getattr(p, "rejected_meas", []))
            if rejected and valid:
                track.rejected_count += 1
            velocity = tuple(float(x) for x in p.ukf.x[3:6])
            filtered = tuple(float(x) for x in p.ukf.x[:3])
            forecast = (float(p.x_future_history[-1]), float(p.y_future_history[-1]),
                        float(p.z_future_history[-1]))
            entry = {"frame": idx, "time_s": frame.time_s,
                     "raw": tuple(float(x) for x in pos) if valid else None,
                     "filtered": filtered, "forecast": forecast,
                     "fresh": valid, "age": idx - track.last_seen_frame, "age_s": age_s,
                     "frame_interval_s": elapsed,
                     "velocity": velocity, "rejected": rejected,
                     "rejected_count": track.rejected_count}
            track.history.append(entry)
            views[mid] = entry
            if valid and track.observed >= 2:
                self._pending.append((frame.time_s + self.horizon_s, mid, forecast,
                                      tuple(float(v) for v in pos), idx, track.generation))

        fresh = sum(int(e["fresh"]) for e in views.values())
        speed = None
        calm = False
        centroid = None
        centroid_future = None
        if views:
            recent = list(views.values())
            centroid = tuple(float(v) for v in np.mean([r["filtered"] for r in recent], axis=0))
            centroid_future = tuple(float(v) for v in np.mean([r["forecast"] for r in recent], axis=0))
        if centroid is not None and fresh > 0:
            if previous is not None and elapsed > min(self.calm_window_s, self.stale_after_s):
                self.centroid_window.clear()
                self.calm_elapsed_s = 0.0
                self.calm_streak = 0
            self.centroid_window.append((frame.time_s, np.asarray(centroid)))
            while len(self.centroid_window) > 2 and (frame.time_s - self.centroid_window[1][0] >= self.calm_window_s):
                self.centroid_window.popleft()
            if len(self.centroid_window) >= 2:
                samples = list(self.centroid_window)
                distance = sum(float(np.linalg.norm(b[1] - a[1])) for a, b in zip(samples, samples[1:]))
                duration = samples[-1][0] - samples[0][0]
                if duration > 0:
                    speed = distance / duration
                    if speed < self.lull_threshold_mps:
                        self.calm_streak += 1
                        self.calm_elapsed_s += elapsed
                    else:
                        self.calm_streak = 0
                        self.calm_elapsed_s = 0.0
                    calm = (duration >= self.calm_window_s and
                            self.calm_elapsed_s >= self.calm_min_s)
        else:
            self.calm_streak = 0
            self.calm_elapsed_s = 0.0
            self.centroid_window.clear()
        n = len(self.scores)
        rmse = math.sqrt(self._sum_pred_squares / n) if n else None
        hold = math.sqrt(self._sum_hold_squares / n) if n else None
        rejected_total = sum(t.rejected_count for t in self.tracks.values())
        snap = Snapshot(idx, frame.time_s, views, centroid, centroid_future, fresh, len(views),
                        len(self.tracks), speed, calm, self.calm_streak, n, rmse, hold,
                        bool(n), rejected_total)
        self.snapshots.append(snap)
        self._previous_frame = frame
        return snap

    def scores_at(self, frame: int, mussel_id: int | None = None) -> list[Score]:
        return [s for s in self.scores if s.frame <= frame and (mussel_id is None or s.mussel_id == mussel_id)]

    def track_history(self, frame: int, mussel_id: int) -> list[dict]:
        track = self.tracks.get(mussel_id)
        return [row for row in track.history if row["frame"] <= frame] if track else []

    def metrics_at(self, frame: int, mussel_id: int | None = None) -> dict:
        scores = self.scores_at(frame, mussel_id)
        if not scores:
            return {"count": 0, "rmse_m": None, "hold_rmse_m": None,
                    "improvement_pct": None, "rmse_axes_m": None, "mae_m": None}
        a = np.array([s.forecast_error_m for s in scores])
        b = np.array([s.hold_error_m for s in scores])
        rmse = float(np.sqrt(np.mean(np.sum(a * a, axis=1))))
        hold = float(np.sqrt(np.mean(np.sum(b * b, axis=1))))
        return {"count": len(scores), "rmse_m": rmse, "hold_rmse_m": hold,
                "improvement_pct": 100 * (1 - rmse / hold) if hold > 1e-12 else None,
                "rmse_axes_m": tuple(float(x) for x in np.sqrt(np.mean(a * a, axis=0))),
                "mae_m": float(np.mean(np.linalg.norm(a, axis=1)))}
