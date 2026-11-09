"""
lull_verification.py

Verification tool for the standalone LullDetector: runs the
detector over one real tracked mussel and plots the regions it flags as
"calm" directly against the measured trajectory, so you can eyeball
whether the flagged regions actually look calm -- and use the same plot
to pick a sensible speed_threshold for your real data instead of
guessing one.

Produces a two-panel figure:
  Top    - the measured x, y, z position over time (the "actual measured
           data"), with the detector's calm regions shaded.
  Bottom - the detector's own speed signal over time, with the
           speed_threshold drawn as a horizontal line and the same shaded
           regions -- shows *why* each region was (or wasn't) flagged,
           not just where.

Edit the constants below (DATASET, MUSSEL_ID, SPEED_THRESHOLD) and rerun
to check a different track or try a different threshold; the shaded
regions will move accordingly.

Run directly: `python3 lull_verification.py`
"""

import sys
from pathlib import Path
import numpy as np

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder))

import pandas as pd
import matplotlib.pyplot as plt

from Test_Verification.lull_detector import LullDetector, LullStatus

# ----------------------------------------------------------------- config
# Defaults below reproduce the RD_002 example in the writeup. Try
# DATASET="RD_004.csv", MUSSEL_ID=3, SPEED_THRESHOLD=0.045 to see a track
# with heavy dropout instead -- worth doing once, because it surfaces a
# real edge case: a single dropped frame gets linearly interpolated into
# a sharp position spike, and that spike inflates the windowed speed for
# up to WINDOW_SECONDS afterwards, which can mask a real calm period
# right after a dropout rather than a genuine burst of motion. Something
# to keep in mind when picking a threshold for a dropout-heavy dataset.
DATASET = "RD_002.csv"
MUSSEL_ID = 1
FPS = 15.0  # this dataset's real sampling rate -- check it matches yours
SPEED_THRESHOLD = 0.025  # m/s -- the number this tool exists to help you pick
WINDOW_SECONDS = 2.0
MIN_CALM_SECONDS = 1.0

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


def load_track(fname, mid):
    df = pd.read_csv(_resolve_csv(fname))
    sub = df[df["ID"] == mid].reset_index(drop=True)
    x = sub["xpos"].to_numpy(dtype=float)
    y = sub["ypos"].to_numpy(dtype=float)
    z = sub["zpos"].to_numpy(dtype=float)
    dropout = (x == 0) & (y == 0) & (z == 0)
    return x, y, z, dropout


def interp_dropouts(a, dropout):
    a = a.copy()
    idx = np.arange(len(a))
    good = ~dropout
    if good.sum() < 2:
        return a
    a[dropout] = np.interp(idx[dropout], idx[good], a[good])
    return a


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


def main():
    dt = 1.0 / FPS
    x_raw, y_raw, z_raw, dropout = load_track(DATASET, MUSSEL_ID)
    x = interp_dropouts(x_raw, dropout)
    y = interp_dropouts(y_raw, dropout)
    z = interp_dropouts(z_raw, dropout)
    t = np.arange(len(x)) * dt

    detector = LullDetector(
        dt=dt,
        speed_threshold=SPEED_THRESHOLD,
        window_seconds=WINDOW_SECONDS,
        min_calm_seconds=MIN_CALM_SECONDS,
    )
    speed = np.zeros(len(x))
    is_calm = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        status = detector.update((x[i], y[i], z[i]))
        speed[i] = status.speed
        is_calm[i] = status.is_calm

    spans = calm_spans(is_calm, t)
    print(
        f"{DATASET} ID={MUSSEL_ID}: {len(spans)} calm region(s) detected "
        f"(threshold={SPEED_THRESHOLD*1000:.1f} mm/s):"
    )
    for s, e in spans:
        print(f"  t = {s:.2f}s to {e:.2f}s  (duration {e - s:.2f}s)")

    fig, (ax_pos, ax_speed) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_pos.plot(t, x * 1000, label="x", linewidth=1)
    ax_pos.plot(t, y * 1000, label="y", linewidth=1)
    ax_pos.plot(t, z * 1000, label="z", linewidth=1)
    ax_pos.set_ylabel("Position (mm)")
    ax_pos.set_title(f"{DATASET} ID={MUSSEL_ID} -- measured position with detected calm regions")
    ax_pos.legend(loc="upper right", fontsize=8)

    ax_speed.plot(t, speed * 1000, color="tab:gray", linewidth=1, label="speed")
    ax_speed.axhline(
        SPEED_THRESHOLD * 1000,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label=f"threshold ({SPEED_THRESHOLD*1000:.0f} mm/s)",
    )
    ax_speed.set_ylabel("Speed (mm/s)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.legend(loc="upper right", fontsize=8)

    for ax in (ax_pos, ax_speed):
        for i, (s, e) in enumerate(spans):
            ax.axvspan(s, e, color="tab:green", alpha=0.2, label="detected calm" if i == 0 else None)
        if spans:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    # out_path = _SCRIPT_DIR / f"lull_verification_{DATASET.replace('.csv', '')}_ID{MUSSEL_ID}.png"
    # fig.savefig(out_path, dpi=150)
    # print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
