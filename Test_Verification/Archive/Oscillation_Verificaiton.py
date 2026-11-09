"""
oscillator_investigation.py

Investigates whether a damped-harmonic-oscillator-augmented state model
out-predicts the existing constant-velocity (CV) model on the real mussel
tracks (RD_001-RD_004). This is the evidence behind the recommendation in
Oscillator_Model.py: a fixed-parameter resonance tuned from a short
calibration window helps a lot on some tracks and hurts a lot on others.
The "cycles_observed" column below shows why: most real single tracks
here give the fit only ~1 oscillation cycle to work with, nowhere near
enough to pin down a frequency with confidence -- a confidently-tuned but
wrong frequency then compounds error across a multi-step forecast, where
CV's "boring" straight-line extrapolation degrades more gracefully.

Runs two variants of the oscillator model for comparison:
  B (plain)  - process noise taken directly from one-step residuals on
               the fit window, no adjustment for how little data that was.
  B (gated)  - same, but process noise is inflated when fewer than
               min_confident_cycles were observed (Oscillator_Model.degrade_to_safe's
               reasoning, applied continuously here rather than as a hard
               swap to zeta=1).

Both models are plain linear Kalman filters (not UKF) -- fair,
reproducible, and don't need UKF's nonlinear machinery since omega/zeta
are fixed constants for this test. Run directly:
    python3 oscillator_investigation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from filterpy.common import Q_discrete_white_noise

FPS = 15.0
DT = 1.0 / FPS

# ---------------------------------------------------------------- data path
# Folders to search for RD_001.csv / RD_002.csv / ... / 260721.csv, in
# order. This script's own folder is checked first, then a few locations
# matching Prediction_Support.py's own convention (data_folder =
# "Data/Real/15fps/", relative to wherever the main pipeline is run from).
# If none of these match your layout, either add your actual folder to
# this list or just copy the CSVs into one of the folders already listed.
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
        f"Either copy the RD_00x.csv / 260721.csv files into one of those "
        f"folders, or add the correct folder to _CANDIDATE_DATA_DIRS at the "
        f"top of this script."
    )


# ---------------------------------------------------------------- data prep
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


def highpass(a, dt, cutoff_hz=0.1, order=2):
    fs = 1.0 / dt
    Wn = cutoff_hz / (fs / 2)
    b, a_coef = signal.butter(order, Wn, btype="highpass")
    padlen = min(len(a) - 1, 3 * max(len(b), len(a_coef)))
    return signal.filtfilt(b, a_coef, a, padlen=padlen)


# ------------------------------------------------------- oscillator tuning
def fit_oscillator_params(sway, dt):
    """omega from the periodogram peak; zeta from its half-power bandwidth."""
    freqs, psd = signal.periodogram(sway, fs=1.0 / dt, window="hann")
    mask = freqs > 0.05
    freqs, psd = freqs[mask], psd[mask]
    peak_idx = np.argmax(psd)
    f0 = freqs[peak_idx]
    half_power = psd[peak_idx] / 2.0

    lo = peak_idx
    while lo > 0 and psd[lo] > half_power:
        lo -= 1
    hi = peak_idx
    while hi < len(psd) - 1 and psd[hi] > half_power:
        hi += 1
    fwhm = freqs[hi] - freqs[lo]
    zeta = np.clip(fwhm / (2 * f0), 0.02, 0.9) if f0 > 0 else 0.2

    return 2 * np.pi * f0, zeta


def oscillator_Phi(omega, zeta, dt):
    """Verified against scipy.linalg.expm to machine precision -- see writeup."""
    a = zeta * omega
    wd = omega * np.sqrt(max(1 - zeta**2, 1e-6))
    e = np.exp(-a * dt)
    c, s = np.cos(wd * dt), np.sin(wd * dt)
    return e * np.array(
        [
            [c + (a / wd) * s, (1 / wd) * s],
            [-(omega**2 / wd) * s, c - (a / wd) * s],
        ]
    )


def fit_process_noise(mean_signal, sway, Phi, dt, omega, min_confident_cycles=None):
    q_mean = np.var(np.diff(mean_signal))

    vel = np.gradient(sway, dt)
    state = np.stack([sway, vel], axis=1)
    pred = (Phi @ state[:-1].T).T
    resid = state[1:] - pred
    q_sway = np.cov(resid.T)
    if q_sway.ndim == 0:
        q_sway = np.array([[float(q_sway), 0], [0, float(q_sway)]])

    if min_confident_cycles is not None:
        cycles_observed = len(sway) * dt * (omega / (2 * np.pi))
        inflate = max(1.0, min_confident_cycles / max(cycles_observed, 0.5))
        q_mean, q_sway = q_mean * inflate, q_sway * inflate

    return q_mean, q_sway


# --------------------------------------------------------------- filters
class KF1D:
    """Minimal linear Kalman filter -- state dim n, scalar measurement of pos."""

    def __init__(self, F, Q, H, R, x0, P0):
        self.F, self.Q, self.H, self.R = F, Q, H, float(R)
        self.x, self.P = x0.astype(float), P0.astype(float)

    def predict(self, F=None):
        F = self.F if F is None else F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = float(z) - self.H @ self.x
        S = float(self.H @ self.P @ self.H.T) + self.R
        K = (self.P @ self.H) / S
        self.x = self.x + K * y
        self.P = self.P - np.outer(K, self.H) @ self.P

    def forecast(self, h):
        x, F = self.x.copy(), self.F
        for _ in range(h):
            x = F @ x
        return float(self.H @ x)


def make_model_A(dt):
    F = np.array([[1, dt], [0, 1]])
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    H = np.array([1.0, 0.0])
    return F, Q, H, 0.1


def make_model_B(dt, omega, zeta, q_mean, q_sway):
    Phi = oscillator_Phi(omega, zeta, dt)
    F = np.zeros((3, 3))
    F[0, 0] = 1.0
    F[1:, 1:] = Phi
    Q = np.zeros((3, 3))
    Q[0, 0] = q_mean
    Q[1:, 1:] = q_sway
    H = np.array([1.0, 1.0, 0.0])
    return F, Q, H, 0.1


# ------------------------------------------------------------- evaluation
def evaluate_track(fname, mid, fit_frac=0.6, horizons=(1, 5, 15, 45), min_confident_cycles=None):
    x_raw, y_raw, z_raw, dropout = load_track(fname, mid)
    n = len(x_raw)
    split = int(n * fit_frac)

    results = {}
    for axis_name, raw in (("x", x_raw), ("y", y_raw), ("z", z_raw)):
        raw_i = interp_dropouts(raw, dropout)
        sway_full = highpass(raw_i, DT)
        mean_full = raw_i - sway_full

        omega, zeta = fit_oscillator_params(sway_full[:split], DT)
        Phi = oscillator_Phi(omega, zeta, DT)
        q_mean, q_sway = fit_process_noise(mean_full[:split], sway_full[:split], Phi, DT, omega, min_confident_cycles)

        FA, QA, HA, RA = make_model_A(DT)
        FB, QB, HB, RB = make_model_B(DT, omega, zeta, q_mean, q_sway)

        kfA = KF1D(FA, QA, HA, RA, x0=np.array([raw_i[0], 0.0]), P0=np.eye(2) * 10.0)
        kfB = KF1D(FB, QB, HB, RB, x0=np.array([raw_i[0], 0.0, 0.0]), P0=np.eye(3) * 10.0)

        errs_A = {h: [] for h in horizons}
        errs_B = {h: [] for h in horizons}

        for i in range(n):
            if dropout[i]:
                kfA.predict()
                kfB.predict()
                continue
            kfA.predict()
            kfA.update(raw_i[i])
            kfB.predict()
            kfB.update(raw_i[i])

            if i >= split:
                for h in horizons:
                    j = i + h
                    if j < n and not dropout[j]:
                        errs_A[h].append(kfA.forecast(h) - raw_i[j])
                        errs_B[h].append(kfB.forecast(h) - raw_i[j])

        cycles_observed = split * DT * (omega / (2 * np.pi))
        results[axis_name] = dict(
            omega_hz=omega / (2 * np.pi),
            zeta=zeta,
            cycles_observed=cycles_observed,
            rmse_A={h: float(np.sqrt(np.mean(np.square(errs_A[h])))) if errs_A[h] else None for h in horizons},
            rmse_B={h: float(np.sqrt(np.mean(np.square(errs_B[h])))) if errs_B[h] else None for h in horizons},
        )
    return results


def print_table(datasets, horizons, min_confident_cycles, title):
    print(f"\n=== {title} ===")
    header = f"{'dataset':<16}{'axis':<5}{'f0(Hz)':<8}{'zeta':<7}{'cycles':<8}"
    header += "".join(f"h={h:<3}(mm) A / B      " for h in horizons)
    print(header)
    for fname, mid in datasets:
        res = evaluate_track(fname, mid, horizons=horizons, min_confident_cycles=min_confident_cycles)
        for axis, r in res.items():
            row = (
                f"{fname + ' ' + str(mid):<16}{axis:<5}{r['omega_hz']:<8.3f}"
                f"{r['zeta']:<7.2f}{r['cycles_observed']:<8.1f}"
            )
            for h in horizons:
                a, b = r["rmse_A"][h], r["rmse_B"][h]
                row += f"{'n/a':<20}" if (a is None or b is None) else f"{a*1000:6.1f} / {b*1000:<6.1f}   "
            print(row)


if __name__ == "__main__":
    datasets = [("RD_001.csv", 1), ("RD_002.csv", 1), ("RD_003.csv", 1), ("RD_003.csv", 15), ("RD_004.csv", 3)]
    horizons = (1, 5, 15, 45)

    print_table(datasets, horizons, min_confident_cycles=None, title="Model B (plain fit)")
    print_table(datasets, horizons, min_confident_cycles=5.0, title="Model B (confidence-gated, min 5 cycles)")
