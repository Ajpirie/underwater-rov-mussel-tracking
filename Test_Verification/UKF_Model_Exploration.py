"""
UKF_Model_Exploration.py

The evidence behind the motion model and Q/R tuning chosen for
prediction_method.py. Compares seven motion models, each tuned
over a small grid, on the same strict per-mussel benchmark as
Benchmark_UKF.py (all six real test cases, horizons of 1, 5, 10
and 20 frames, % RMSE improvement over the zero-order hold):

  Random walk     position only -- a smoothed, outlier-gated zero-order hold
  CV              constant velocity (white-noise acceleration), the original model
  CA              constant acceleration (white-noise jerk)
  Singer          acceleration decays with time constant tau
  DV              damped velocity: velocity decays with time constant tau
  DV + RW         damped velocity plus a random walk on position (the chosen model)
  Oscillator      damped harmonic oscillator about a slowly drifting anchor point

Every model gets the same measurement noise R (measured from the data),
the same outlier gating, and the same initial uncertainty, so only the
motion model and its process noise differ. Each grid is run twice: with a
FIXED Q, and with Q ADAPTED online from the innovations (as in
prediction_method.py). The last section is a leave-one-test-case-
out cross-validation: each test case is scored with the configuration that
does best on the OTHER five, to check the tuning isn't just fitted to the
data it's scored on.

FINDINGS when this script was written (mean / worst case = % improvement
over the ZOH, over all six test cases and all four horizons):
  - Only the damped-velocity family is worth having. DV + RW with adaptive Q
    (tau = 1.5 s, q = 1e-3) scores +15.1 % mean / +0.1 % worst: ahead of the
    ZOH everywhere. This search picks it independently, and it's the
    default in prediction_method.py.
  - With a FIXED Q the best DV + RW only just stays ahead (+11.7 % / +0.1 %),
    by keeping Q small and barely extrapolating (RD_003: +2 % rather than
    +9 %). Q is in physical units and the 15fps recording moves ~20x faster
    (in m/s) than the RD tank tracks, so one fixed Q can't suit both;
    adapting Q lets each recording find its own level.
  - CV, CA and Singer extrapolate a trend across the whole horizon. That
    pays off on RD_003/15's steady drift but overshoots badly whenever the
    mussel sways back, and a tracking glitch becomes a large false velocity
    (RD_004: about -210 % or worse for all three).
  - The random walk (a gated, smoothed hold) is never worse than the ZOH,
    but its only gain is rejecting glitches.
  - The oscillator (critically damped, generic frequency) scores about the
    same mean as DV but is less consistent (worst case -0.6 %). With only
    about one sway cycle per track it can't pin down a frequency: the same
    conclusion Oscillation_Verificaiton.py reached.
  - Cross-validation: with adaptive Q the same configuration wins on every
    5-case training set, and every held-out case stays ahead of the ZOH at
    every horizon. With fixed Q, held-out RD_002 falls behind it (-3.3 % at
    5 frames).

The filters here are plain linear Kalman filters (not UKF), like the ones in
Oscillation_Verificaiton.py. Every model is linear, so a UKF gives the same
answer (see the check printed at the start), and plain matrix maths keeps
the grid search to a few seconds. Run from the repository root:

    python3 Test_Verification/UKF_Model_Exploration.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder))
sys.path.append(str(Path(__file__).resolve().parent))

from Prediction import prediction_method as psi
from Benchmark_UKF import HEADLINE_STEPS, HORIZON_STEPS, load_test_cases, run_filter, strict_errors

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TUNING = psi.default_tuning  # R, gating, initial uncertainty and adaptation settings shared by every model

Q_GRID = [1e-5, 1e-4, 1e-3, 1e-2]  # spans the RD (~1e-5) to 15fps (~1e-2) process-noise regimes
TAU_GRID = [0.75, 1.5, 3.0]  # s

MODEL_GRIDS = {
    "Random walk": [dict(q=q) for q in Q_GRID],
    "CV": [dict(q=q) for q in Q_GRID],
    "CA": [dict(q=q) for q in Q_GRID],
    "Singer": [dict(tau=tau, q=q) for tau in TAU_GRID for q in Q_GRID],
    "DV": [dict(tau=tau, q=q) for tau in TAU_GRID for q in Q_GRID],
    "DV + RW": [dict(tau=tau, q=q, qp=q) for tau in TAU_GRID for q in Q_GRID],
    "Oscillator": [dict(f=f, zeta=zeta, q=q, qa=q) for f in [0.05, 0.1, 0.2] for zeta in [0.5, 1.0] for q in Q_GRID],
}


# ----------------------------------------------------------------------------
# MOTION MODELS
# ----------------------------------------------------------------------------
def van_loan(A, Qc, dt):
    """Exact discretisation of dx = A x dt + (white noise of PSD diag(Qc)). Returns (F, Q)."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = np.diag(Qc)
    M[n:, n:] = A.T
    E = expm(M * dt)
    F = E[n:, n:].T
    Q = F @ E[:n, n:]
    return F, 0.5 * (Q + Q.T)


def axis_model(model, dt, p):
    """Single-axis (F, Q, initial variances) for one model. State 0 is always position."""
    pos_var, vel_var = TUNING.init_pos_std**2, TUNING.init_vel_std**2
    acc_var = 1.0  # (1 m/s^2)^2

    if model == "Random walk":
        A, Qc, P0 = np.zeros((1, 1)), [p["q"]], [pos_var]
    elif model == "CV":
        A, Qc, P0 = np.array([[0, 1], [0, 0.0]]), [0, p["q"]], [pos_var, vel_var]
    elif model == "CA":
        A, Qc, P0 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0.0]]), [0, 0, p["q"]], [pos_var, vel_var, acc_var]
    elif model == "Singer":
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / p["tau"]]])
        Qc, P0 = [0, 0, p["q"]], [pos_var, vel_var, acc_var]
    elif model == "DV":
        A, Qc, P0 = np.array([[0, 1], [0, -1 / p["tau"]]]), [0, p["q"]], [pos_var, vel_var]
    elif model == "DV + RW":
        A, Qc, P0 = np.array([[0, 1], [0, -1 / p["tau"]]]), [p["qp"], p["q"]], [pos_var, vel_var]
    elif model == "Oscillator":
        # state [position, velocity, anchor]: the position is pulled back towards an anchor that drifts slowly
        w = 2 * np.pi * p["f"]
        A = np.array([[0, 1, 0], [-(w**2), -2 * p["zeta"] * w, w**2], [0, 0, 0]])
        Qc, P0 = [0, p["q"], p["qa"]], [pos_var, vel_var, pos_var]
    else:
        raise ValueError(f"Unknown model: {model}")

    F, Q = van_loan(np.asarray(A, dtype=float), Qc, dt)
    return F, Q, np.diag(P0)


# ----------------------------------------------------------------------------
# FILTER
# ----------------------------------------------------------------------------
class LinearKF:
    """
    3-axis linear Kalman filter with the same measurement handling as
    prediction_method.predictionInstance: starts at the first
    measurement, gates glitches (re-initialising on consistent jumps), and
    optionally adapts Q from the normalised innovation squared. The state
    layout is [x, y, z, vx, vy, vz, ...] as in the current UKF.
    """

    def __init__(self, model, params, dt, adaptive):
        F1, Q1, P0 = axis_model(model, dt, params)
        self.n = len(F1)
        self.F = np.kron(F1, np.eye(3))
        self.Q_base = np.kron(Q1, np.eye(3))
        self.P0 = np.kron(P0, np.eye(3))
        if model == "Oscillator":
            self.P0[0:3, 6:9] = self.P0[6:9, 0:3] = np.eye(3) * P0[0, 0]  # anchor starts at the position
        self.H = np.hstack([np.eye(3), np.zeros((3, 3 * (self.n - 1)))])
        self.R = np.diag(np.square(TUNING.meas_std))
        self.model = model
        self.adaptive = adaptive

        self.x = None
        self.q_scale = 1.0
        self.rejected_meas = []
        self.updates_since_init = 0
        self.F_powers = {steps: np.linalg.matrix_power(self.F, steps) for steps in HORIZON_STEPS}

    def _reset(self, z):
        self.x = np.zeros(3 * self.n)
        self.x[0:3] = z
        if self.model == "Oscillator":
            self.x[6:9] = z
        self.P = self.P0.copy()
        self.rejected_meas = []
        self.updates_since_init = 0

    def step(self, z):
        """One frame; z = (x, y, z) measurement or None for a dropout."""
        if self.x is None:
            if z is not None:
                self._reset(z)
            return

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q_base * self.q_scale
        if z is None:
            return

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))

        if nis > TUNING.gate_chi2 and np.linalg.norm(y) > TUNING.gate_min_dist:
            self.rejected_meas.append(z)
            recent = np.array(self.rejected_meas[-TUNING.reacquire_frames :])
            if len(recent) >= TUNING.reacquire_frames:
                if np.max(np.linalg.norm(recent - recent[-1], axis=1)) <= TUNING.reacquire_spread:
                    self._reset(z)
            return

        self.rejected_meas = []
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.updates_since_init += 1

        if self.adaptive and self.updates_since_init > TUNING.adapt_after:
            step = TUNING.adapt_rate * (min(nis / 3, TUNING.nis_clip) - 1)
            self.q_scale = float(np.clip(self.q_scale * np.exp(step), TUNING.q_scale_min, TUNING.q_scale_max))

    def forecast(self, steps):
        return self.H @ (self.F_powers[steps] @ self.x)


# ----------------------------------------------------------------------------
# HARNESS
# ----------------------------------------------------------------------------
def run_config(cases, model, params, adaptive):
    """{steps: {case: % improvement over the ZOH}} for one model configuration."""
    out = {steps: {} for steps in HORIZON_STEPS}

    for name, (dt, tracks) in cases.items():
        errors = {steps: ([], []) for steps in HORIZON_STEPS}

        for track in tracks:
            kf = LinearKF(model, params, dt, adaptive)
            forecasts = {steps: np.full((len(track), 3), np.nan) for steps in HORIZON_STEPS}
            for i, pos in enumerate(track):
                kf.step(None if np.all(pos == 0) else pos)
                if kf.x is not None:
                    for steps in HORIZON_STEPS:
                        forecasts[steps][i] = kf.forecast(steps)

            for steps in HORIZON_STEPS:
                m, z = strict_errors(track, forecasts[steps], steps)
                errors[steps][0].append(m)
                errors[steps][1].append(z)

        for steps in HORIZON_STEPS:
            m, z = np.concatenate(errors[steps][0]), np.concatenate(errors[steps][1])
            out[steps][name] = (1 - np.sqrt(np.mean(m**2)) / np.sqrt(np.mean(z**2))) * 100

    return out


def worst_case(result, cases):
    return min(result[steps][name] for steps in HORIZON_STEPS for name in cases)


def mean_improvement(result, cases):
    return np.mean([result[steps][name] for steps in HORIZON_STEPS for name in cases])


def selection_key(result, cases):
    """Configs that beat the ZOH on every one of `cases` at every horizon rank first, by mean
    improvement; the rest rank by their worst case. Ranking on the worst case alone would hinge
    on RD_002, where every reasonable filter is within a fraction of a percent of the ZOH."""
    worst = worst_case(result, cases)
    return (worst > 0, mean_improvement(result, cases) if worst > 0 else worst)


def check_against_improved_ukf(cases):
    """The linear KF with the chosen model must reproduce prediction_method's UKF."""
    tuning = TUNING
    params = dict(tau=tuning.velocity_tau, q=tuning.accel_noise, qp=tuning.position_noise)
    dt, tracks = cases["RD_003/1"]
    track = tracks[0]

    kf = LinearKF("DV + RW", params, dt, adaptive=True)
    kf_forecasts = []
    for pos in track:
        kf.step(None if np.all(pos == 0) else pos)
        kf_forecasts.append(kf.forecast(HEADLINE_STEPS))

    ukf_forecasts = run_filter(track, dt, HEADLINE_STEPS, tuning)
    diff = np.max(np.linalg.norm(np.array(kf_forecasts) - ukf_forecasts, axis=1))
    print(f"Check: linear KF vs UKF on RD_003/1, max forecast difference = {diff * 1000:.2e} mm\n")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    cases = load_test_cases()
    names = list(cases)

    check_against_improved_ukf(cases)

    rows = []  # (adaptive, model, params, result)
    for adaptive in (False, True):
        for model, grid in MODEL_GRIDS.items():
            for params in grid:
                rows.append((adaptive, model, params, run_config(cases, model, params, adaptive)))

    # --- Best configuration per model: the most consistent one (best worst case), ties to the higher mean ---
    for adaptive in (False, True):
        print("=" * 112)
        print(f"{'ADAPTIVE' if adaptive else 'FIXED'} Q - best configuration of each model (see selection_key)")
        print(
            f"% improvement over ZOH at {HEADLINE_STEPS} frames; mean and worst case over every test case and horizon"
        )
        print("=" * 112)
        print(f"  {'model':12s}{'parameters':34s}" + "".join(f"{n:>10s}" for n in names) + f"{'mean':>8s}{'worst':>9s}")

        for model in MODEL_GRIDS:
            candidates = [r for r in rows if r[0] == adaptive and r[1] == model]
            best = max(candidates, key=lambda r: selection_key(r[3], names))
            params = ", ".join(f"{k}={v:g}" for k, v in best[2].items())
            print(
                f"  {model:12s}{params:34s}"
                + "".join(f"{best[3][HEADLINE_STEPS][n]:+9.1f}%" for n in names)
                + f"{mean_improvement(best[3], names):+7.1f}%{worst_case(best[3], names):+8.1f}%"
            )
        print()

    # --- Leave-one-test-case-out cross-validation ---
    print("=" * 112)
    print("Leave-one-test-case-out: each case scored with the config selection_key picks on the OTHER five")
    print("(% improvement over ZOH on the held-out case, per horizon)")
    print("=" * 112)

    for adaptive in (False, True):
        candidates = [r for r in rows if r[0] == adaptive]
        print(f"  {'ADAPTIVE' if adaptive else 'FIXED'} Q, any model:")
        for held_out in names:
            train = [n for n in names if n != held_out]
            best = max(candidates, key=lambda r: selection_key(r[3], train))
            scores = "  ".join(f"{steps:2d} fr: {best[3][steps][held_out]:+6.1f}%" for steps in HORIZON_STEPS)
            print(f"    {held_out:10s} {scores}   <- {best[1]} ({', '.join(f'{k}={v:g}' for k, v in best[2].items())})")
        print()


if __name__ == "__main__":
    main()
