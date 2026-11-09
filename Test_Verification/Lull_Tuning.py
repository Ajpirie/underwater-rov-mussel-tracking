"""
lull_tuning.py

Generalised version of lull_verification.py. Instead of checking one
hand-picked speed_threshold against one track, this runs over every
track in every dataset, and tunes the detector so it finds the lulls
you describe in the "target lull specs" block below.

Why this exists: a fixed speed_threshold in m/s only suits the dataset
it was picked on. The windowed speed's floor moves with measurement
noise, with how hard the mussel is actually swinging, and with frame
rate, so 25 mm/s that works on RD_002 never triggers on the others.
The detector here (AdaptiveLullDetector) keeps LullDetector's logic but
makes two changes so one set of settings can travel between datasets:

  1. Position is low-pass filtered (causal EMA) before the speed is
     measured, so the path length reflects motion rather than jitter.
  2. The threshold is a *ratio* of the point's own recent typical speed
     (running median), not a fixed number of m/s.

How the tuning works:

  1. A reference labelling is built for each track straight from the
     target specs, offline and with zero lag (centred filters, whole
     track visible): a lull is any stretch at least
     TARGET_MIN_LULL_SECONDS long where the motion is below
     TARGET_SPEED_RATIO x the track's typical motion (and below
     TARGET_MAX_SPEED, if set).
  2. Every combination in TUNING_GRID is run through the causal,
     frame-by-frame detector and scored on how well its calm flags
     overlap that reference (F-beta, see SCORE_BETA).
  3. Reported: the single setting that scores best averaged across all
     tracks (the generalised one to use at runtime), and the best
     setting for each track on its own (how much is lost by not tuning
     per dataset).

One figure per track shows the reference lulls against what the best
general setting detected, with the adaptive threshold drawn on the
speed panel.

Run directly: `python3 Lull_Tuning.py`
"""

import sys
from collections import deque
from itertools import product
from pathlib import Path
import numpy as np

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder))

import pandas as pd
import matplotlib.pyplot as plt

from Test_Verification.lull_detector import LullDetector, LullStatus

# ------------------------------------------------------- target lull specs
# What counts as a lull worth docking in. These define the reference the
# detector is tuned against -- change them and the tuned settings follow.
TARGET_SPEED_RATIO = 0.8  # lull = motion below this fraction of the track's typical (median) motion
TARGET_MIN_LULL_SECONDS = 1.0  # shortest lull worth flagging
TARGET_MAX_SPEED = None  # m/s -- optional absolute cap on top of the ratio (None = ratio only)

# --------------------------------------------------------------- datasets
DATASETS = ["RD_001.csv", "RD_002.csv", "RD_003.csv", "RD_004.csv", "260721.csv"]
DEFAULT_FPS = 15.0
FPS_BY_DATASET = {}  # e.g. {"RD_001.csv": 30.0} -- none of the CSVs carry timestamps, so check these
MIN_TRACK_SECONDS = 3.0  # shorter tracks can't hold a window + a lull, so they're skipped

# ----------------------------------------------------------------- tuning
# Every combination is tried. Times are in seconds so the result carries
# across frame rates.
TUNING_GRID = {
    "threshold_ratio": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "smooth_seconds": [0.0, 0.2, 0.4],
    "window_seconds": [0.5, 1.0, 2.0],
    "min_calm_seconds": [0.25, 0.5, 1.0],
    "baseline_seconds": [10.0, None],  # None = all history so far
}
# Scoring: beta < 1 punishes false "calm" more than a missed lull (docking
# into an active swing is the expensive mistake); 1.0 weighs them equally.
SCORE_BETA = 0.5

# Reference (offline) filters -- these only shape the labelling the
# detector is scored against, not the detector itself.
REF_SMOOTH_SECONDS = 0.4  # centred moving average on position
REF_ENVELOPE_SECONDS = 1.0  # centred moving average on speed (bridges the zero-speed turnarounds of each swing)

SHOW_PLOTS = True
SAVE_PLOTS = False

# ---------------------------------------------------------------- data path
# Same search order as oscillator_investigation.py -- see that file for why.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CANDIDATE_DATA_DIRS = [
    _SCRIPT_DIR,
    _SCRIPT_DIR / "Data",
    _SCRIPT_DIR / "Data" / "Real",
    _SCRIPT_DIR / "Data" / "Real" / "15fps",
    _SCRIPT_DIR.parent / "Data",
    _SCRIPT_DIR.parent / "Data" / "Real",
    _SCRIPT_DIR.parent / "Data" / "Real" / "15fps",
    Path.cwd(),
    Path.cwd() / "Data",
    Path.cwd() / "Data" / "Real",
    Path.cwd() / "Data" / "Real" / "15fps",
]


def _resolve_csv(fname: str) -> Path:
    for d in _CANDIDATE_DATA_DIRS:
        candidate = d / fname
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(d) for d in _CANDIDATE_DATA_DIRS)
    raise FileNotFoundError(
        f"Couldn't find {fname} in any of:\n  {searched}\n"
        f"Either copy the CSV into one of those folders, or add the "
        f"correct folder to _CANDIDATE_DATA_DIRS at the top of this script."
    )


# ----------------------------------------------------------------- detector
class AdaptiveLullDetector(LullDetector):
    """
    LullDetector with a smoothed input and a self-scaling threshold, so
    the same settings work across noise levels, sea states and frame
    rates:

        calm  <=>  windowed speed of the smoothed position
                   < threshold_ratio * (running median of that speed),
                   held for >= min_calm_seconds

    The windowed speed and the debounce are LullDetector's own; this
    class only filters what goes in and sets speed_threshold each frame.
    Fully causal -- it only ever uses samples it has already been given.
    """

    def __init__(
        self,
        dt: float,
        threshold_ratio: float,
        smooth_seconds: float = 0.2,
        window_seconds: float = 1.0,
        min_calm_seconds: float = 0.5,
        baseline_seconds: float = None,
        max_speed: float = None,
    ):
        """
        Args:
            dt (float): Time between updates.
            threshold_ratio (float): Calm when speed is below this
                fraction of the running-median speed.
            smooth_seconds (float): Time constant of the EMA applied to
                position before speed is measured (0 = no smoothing).
            window_seconds (float): As LullDetector.
            min_calm_seconds (float): As LullDetector.
            baseline_seconds (float): How much speed history the running
                median looks back over. None = everything so far.
            max_speed (float): Optional absolute cap (m/s) on the
                threshold, whatever the ratio says.
        """
        super().__init__(dt, 0.0, window_seconds, min_calm_seconds)
        self.threshold_ratio = threshold_ratio
        self.max_speed = max_speed
        self._alpha = dt / (smooth_seconds + dt)
        self._smoothed = None
        baseline_len = None if baseline_seconds is None else max(2, round(baseline_seconds / dt))
        self._speed_history = deque(maxlen=baseline_len)

    def update(self, position: tuple) -> LullStatus:
        """
        As LullDetector.update, but pass None for a dropped frame. The
        last known position is held so the window keeps real time (the
        jump on re-acquisition then carries the distance covered while
        blind), and the calm streak is reset -- never report calm on a
        point that can't be seen.
        """
        if position is None:
            if self._smoothed is not None:
                super().update(self._smoothed)
            self._calm_streak = 0
            return LullStatus(is_calm=False, speed=float("nan"), calm_streak=0)

        position = np.asarray(position, dtype=float)
        if self._smoothed is None:
            self._smoothed = position
        else:
            self._smoothed = self._smoothed + self._alpha * (position - self._smoothed)

        # No threshold until a full window of speed history exists, so the
        # baseline isn't set by the start-up transient.
        if len(self._speed_history) >= self.window_len:
            self.speed_threshold = self.threshold_ratio * float(np.median(self._speed_history))
            if self.max_speed is not None:
                self.speed_threshold = min(self.speed_threshold, self.max_speed)
        else:
            self.speed_threshold = 0.0

        status = super().update(self._smoothed)
        if len(self._pos_buffer) == self.window_len:
            self._speed_history.append(status.speed)
        return status


# --------------------------------------------------------------------- data
def load_tracks(fname):
    """
    Every track in a dataset as (mussel_id, xyz, dropout).

    Rows with ID 0 are dropped frames the tracker couldn't assign. In a
    single-mussel file they're kept in place as dropouts so the time axis
    stays honest; with several mussels there's no telling whose they are.
    """
    df = pd.read_csv(_resolve_csv(fname))
    ids = [i for i in df["ID"].unique() if i != 0]
    tracks = []
    for mid in ids:
        keep = df["ID"] == mid
        if len(ids) == 1:
            keep |= df["ID"] == 0
        xyz = df.loc[keep, ["xpos", "ypos", "zpos"]].to_numpy(dtype=float)
        dropout = np.all(xyz == 0, axis=1)
        tracks.append((int(mid), xyz, dropout))
    return tracks


def interp_dropouts(xyz, dropout):
    xyz = xyz.copy()
    idx = np.arange(len(xyz))
    good = ~dropout
    if good.sum() < 2:
        return xyz
    for axis in range(xyz.shape[1]):
        xyz[dropout, axis] = np.interp(idx[dropout], idx[good], xyz[good, axis])
    return xyz


def calm_spans(is_calm: np.ndarray, t: np.ndarray):
    """Contiguous [start_time, end_time] pairs where is_calm is True."""
    spans = []
    start = None
    for i, calm in enumerate(is_calm):
        if calm and start is None:
            start = t[i]
        elif not calm and start is not None:
            spans.append((start, t[i - 1]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


# ---------------------------------------------------------------- reference
def _centred_mean(a, n):
    """Centred moving average, edges shrunk rather than zero-padded."""
    if n <= 1:
        return a
    kernel = np.ones(n)
    return np.convolve(a, kernel, mode="same") / np.convolve(np.ones(len(a)), kernel, mode="same")


def reference_lulls(xyz, dropout, dt):
    """
    Zero-lag labelling of where the target specs say the lulls are.
    Returns (is_lull, speed, speed_limit).
    """
    n_smooth = max(1, round(REF_SMOOTH_SECONDS / dt))
    n_envelope = max(1, round(REF_ENVELOPE_SECONDS / dt))
    smoothed = np.column_stack([_centred_mean(xyz[:, axis], n_smooth) for axis in range(3)])
    speed = np.linalg.norm(np.gradient(smoothed, dt, axis=0), axis=1)
    speed = _centred_mean(speed, n_envelope)

    speed_limit = TARGET_SPEED_RATIO * float(np.median(speed))
    if TARGET_MAX_SPEED is not None:
        speed_limit = min(speed_limit, TARGET_MAX_SPEED)
    below = speed < speed_limit

    # Keep only runs long enough to be a usable lull, and that were mostly
    # actually seen -- a straight line interpolated across a dropout looks
    # calm but says nothing about the mussel.
    min_frames = max(1, round(TARGET_MIN_LULL_SECONDS / dt))
    is_lull = np.zeros(len(below), dtype=bool)
    start = None
    for i, b in enumerate(np.append(below, False)):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= min_frames and dropout[start:i].mean() < 0.5:
                is_lull[start:i] = True
            start = None
    return is_lull, speed, speed_limit


# ------------------------------------------------------------------- tuning
def run_detector(xyz, dropout, dt, params):
    detector = AdaptiveLullDetector(dt=dt, max_speed=TARGET_MAX_SPEED, **params)
    speed = np.zeros(len(xyz))
    threshold = np.zeros(len(xyz))
    is_calm = np.zeros(len(xyz), dtype=bool)
    for i in range(len(xyz)):
        status = detector.update(None if dropout[i] else xyz[i])
        speed[i] = status.speed
        threshold[i] = detector.speed_threshold
        is_calm[i] = status.is_calm
    return is_calm, speed, threshold


def score(is_calm, is_lull, valid):
    """
    F-beta overlap between detected calm frames and reference lull
    frames (dropped frames excluded). On a track with no reference lull,
    1 minus the fraction of frames wrongly flagged calm.
    """
    is_calm, is_lull = is_calm[valid], is_lull[valid]
    if not is_lull.any():
        return 1.0 - float(is_calm.mean())
    true_calm = np.sum(is_calm & is_lull)
    if true_calm == 0:
        return 0.0
    precision = true_calm / is_calm.sum()
    recall = true_calm / is_lull.sum()
    b2 = SCORE_BETA**2
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def _fmt(params):
    baseline = "all" if params["baseline_seconds"] is None else f"{params['baseline_seconds']:g}s"
    return (
        f"ratio={params['threshold_ratio']:g} smooth={params['smooth_seconds']:g}s "
        f"window={params['window_seconds']:g}s min_calm={params['min_calm_seconds']:g}s baseline={baseline}"
    )


def plot_track(track, params, is_calm, speed, threshold):
    t, xyz = track["t"], track["xyz"]
    fig, (ax_pos, ax_speed) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for axis, label in enumerate("xyz"):
        ax_pos.plot(t, xyz[:, axis] * 1000, label=label, linewidth=1)
    ax_pos.set_ylabel("Position (mm)")
    ax_pos.set_title(f"{track['name']} -- target lulls vs detected (score {score(is_calm, track['is_lull'], track['valid']):.2f})")

    ax_speed.plot(t, speed * 1000, color="tab:gray", linewidth=1, label="detector speed")
    ax_speed.plot(t, threshold * 1000, color="tab:red", linestyle="--", linewidth=1, label="adaptive threshold")
    ax_speed.set_ylabel("Speed (mm/s)")
    ax_speed.set_xlabel(f"Time (s)\n{_fmt(params)}")

    # Target lulls as a strip along the bottom of each panel, detected calm
    # shaded full height, so agreement and lag are both visible.
    for ax in (ax_pos, ax_speed):
        for i, (s, e) in enumerate(calm_spans(track["is_lull"], t)):
            ax.axvspan(s, e, ymin=0.0, ymax=0.06, color="tab:blue", alpha=0.8, label="target lull" if i == 0 else None)
        for i, (s, e) in enumerate(calm_spans(is_calm, t)):
            ax.axvspan(s, e, color="tab:green", alpha=0.2, label="detected calm" if i == 0 else None)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if SAVE_PLOTS:
        out_path = _SCRIPT_DIR / f"lull_tuning_{track['name'].replace('.csv', '').replace(' ', '_').replace('=', '')}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")


def main():
    # ---- load every track and label it from the target specs
    tracks = []
    for fname in DATASETS:
        dt = 1.0 / FPS_BY_DATASET.get(fname, DEFAULT_FPS)
        skipped = 0
        for mid, xyz_raw, dropout in load_tracks(fname):
            if len(xyz_raw) * dt < MIN_TRACK_SECONDS or (~dropout).sum() < 2:
                skipped += 1
                continue
            xyz = interp_dropouts(xyz_raw, dropout)
            is_lull, _, _ = reference_lulls(xyz, dropout, dt)
            tracks.append(
                {
                    "name": f"{fname} ID={mid}",
                    "dt": dt,
                    "t": np.arange(len(xyz)) * dt,
                    "xyz": xyz,
                    "dropout": dropout,
                    "valid": ~dropout,
                    "is_lull": is_lull,
                }
            )
        if skipped:
            print(f"{fname}: skipped {skipped} track(s) shorter than {MIN_TRACK_SECONDS:g}s")
    if not tracks:
        raise SystemExit("No tracks long enough to tune on.")

    # ---- score every grid combination on every track
    keys = list(TUNING_GRID)
    combos = [dict(zip(keys, values)) for values in product(*TUNING_GRID.values())]
    print(f"\nTuning {len(combos)} settings over {len(tracks)} track(s)...")
    scores = np.zeros((len(combos), len(tracks)))
    for c, params in enumerate(combos):
        for k, track in enumerate(tracks):
            is_calm, _, _ = run_detector(track["xyz"], track["dropout"], track["dt"], params)
            scores[c, k] = score(is_calm, track["is_lull"], track["valid"])

    best_general = combos[int(np.argmax(scores.mean(axis=1)))]
    general_scores = scores[int(np.argmax(scores.mean(axis=1)))]

    # ---- report
    print(
        f"\nTarget: motion below {TARGET_SPEED_RATIO:g}x typical for >= {TARGET_MIN_LULL_SECONDS:g}s"
        + (f", and below {TARGET_MAX_SPEED*1000:g} mm/s" if TARGET_MAX_SPEED is not None else "")
    )
    print(f"\nBest general setting (mean score {general_scores.mean():.2f}):\n  {_fmt(best_general)}\n")
    print(f"{'track':<22}{'len':>7}{'target':>8}{'found':>7}{'general':>9}{'own best':>10}  own best setting")
    for k, track in enumerate(tracks):
        is_calm, _, _ = run_detector(track["xyz"], track["dropout"], track["dt"], best_general)
        own = int(np.argmax(scores[:, k]))
        print(
            f"{track['name']:<22}{track['t'][-1]:>6.1f}s"
            f"{len(calm_spans(track['is_lull'], track['t'])):>8}{len(calm_spans(is_calm, track['t'])):>7}"
            f"{general_scores[k]:>9.2f}{scores[own, k]:>10.2f}  {_fmt(combos[own])}"
        )
    print("\n(target/found = number of lulls in the reference / flagged by the general setting)")

    for track in tracks:
        is_calm, speed, threshold = run_detector(track["xyz"], track["dropout"], track["dt"], best_general)
        plot_track(track, best_general, is_calm, speed, threshold)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
