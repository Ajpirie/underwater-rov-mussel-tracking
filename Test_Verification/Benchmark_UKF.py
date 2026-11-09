"""Benchmark the active UKF against zero-order hold and tuning ablations.

Use the strict per-ID recorded-data protocol and optional report-protocol
comparison to the stationary-position baseline. No retired UKF is imported.
Run from the repository root: python Test_Verification/Benchmark_UKF.py
"""

import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder))
from Prediction import prediction_method as psi

DATA_FOLDER = parent_folder / "Data" / "Real"
RD_DATASETS = ["RD_001", "RD_002", "RD_003", "RD_004"]
DT_RD = 0.5
DT_15FPS = 1.0 / 15.0
HEADLINE_STEPS = 10
HORIZON_STEPS = [1, 5, 10, 20]
SHOW_PLOTS = True

VARIANTS = {
    "UKF": psi.default_tuning,
    "- constant velocity": replace(psi.default_tuning, velocity_tau=None),
    "- no outlier gating": replace(psi.default_tuning, gate_chi2=None),
    "- fixed Q": replace(psi.default_tuning, adapt_rate=None),
    "- high measurement noise": replace(psi.default_tuning, meas_std=(np.sqrt(0.1),) * 3),
}


def frame_index_15fps(txt_path):
    """Restore the recorded video frame number of each 15 fps CSV row."""
    text = Path(txt_path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    num = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
    row_pattern = re.compile(rf"^\s*(\[{{1,2}})\s*{num}\s+{num}\s+{num}\s+{num}\s*(\]{{1,2}})\s*$")
    frames, frame, rows_in_frame = [], 0, 0
    for line in text.splitlines():
        match = row_pattern.match(line)
        if not match:
            continue
        if match.group(1) == "[[" and rows_in_frame:
            frame += 1
            rows_in_frame = 0
        frames.append(frame)
        rows_in_frame += 1
        if match.group(2) == "]]":
            frame += 1
            rows_in_frame = 0
    return np.array(frames)


def load_test_cases():
    """Return recorded cases as (frame period, list of per-ID XYZ tracks)."""
    def positions(df):
        return df[["xpos", "ypos", "zpos"]].to_numpy(dtype=float)

    cases = {}
    for name in ["RD_001", "RD_002"]:
        cases[name] = (DT_RD, [positions(pd.read_csv(DATA_FOLDER / f"{name}.csv"))])
    df = pd.read_csv(DATA_FOLDER / "RD_003.csv")
    cases["RD_003/1"] = (DT_RD, [positions(df[df["ID"] == 1])])
    cases["RD_003/15"] = (DT_RD, [positions(df[df["ID"] == 15])])
    cases["RD_004"] = (DT_RD, [positions(pd.read_csv(DATA_FOLDER / "RD_004.csv"))])
    df = pd.read_csv(DATA_FOLDER / "15fps" / "260721.csv")
    frames = frame_index_15fps(DATA_FOLDER / "15fps" / "260721.txt")
    if len(frames) != len(df):
        raise ValueError("15 fps frame index length does not match the CSV")
    tracks = []
    for mid, rows in df.groupby("ID"):
        f = frames[rows.index.to_numpy()]
        track = np.zeros((f.max() - f.min() + 1, 3))
        track[f - f.min()] = positions(rows)
        tracks.append(track)
    cases["15fps"] = (DT_15FPS, tracks)
    return cases


def run_filter(track, dt, steps, tuning):
    """Feed one track to the UKF and return future positions after each frame."""
    forecasts = np.full((len(track), 3), np.nan)
    mussel = None
    for i, pos in enumerate(track):
        if mussel is None:
            if np.all(pos == 0):
                continue
            mussel = psi.predictionInstance((pos[0], pos[1], pos[2], 0),
                                            tuning, timestep=dt, num_timesteps=steps)
        else:
            mussel.x_history.append(pos[0])
            mussel.y_history.append(pos[1])
            mussel.z_history.append(pos[2])
        mussel.predictPositionUKF()
        forecasts[i] = (mussel.x_future_history[-1], mussel.y_future_history[-1],
                        mussel.z_future_history[-1])
    return forecasts


def strict_errors(track, forecasts, steps):
    """3D errors where the future target is a valid recorded measurement."""
    valid = ~np.all(track == 0, axis=1)
    last_valid = np.maximum.accumulate(np.where(valid, np.arange(len(track)), -1))
    k = np.arange(len(track) - steps)
    k = k[valid[k + steps] & ~np.isnan(forecasts[k, 0])]
    target = track[k + steps]
    model_err = np.linalg.norm(forecasts[k] - target, axis=1)
    hold_err = np.linalg.norm(track[last_valid[k]] - target, axis=1)
    return model_err, hold_err


def evaluate_strict(cases, tuning, steps):
    """Return RMSE, hold RMSE, percent improvement and sample count per case."""
    results = {}
    for name, (dt, tracks) in cases.items():
        model_errors, hold_errors = [], []
        for track in tracks:
            m, h = strict_errors(track, run_filter(track, dt, steps, tuning), steps)
            model_errors.append(m)
            hold_errors.append(h)
        model_errors = np.concatenate(model_errors)
        hold_errors = np.concatenate(hold_errors)
        rmse = float(np.sqrt(np.mean(model_errors ** 2)))
        hold_rmse = float(np.sqrt(np.mean(hold_errors ** 2)))
        pct = 100 * (1 - rmse / hold_rmse) if hold_rmse else float('nan')
        results[name] = (rmse, hold_rmse, pct, len(model_errors))
    return results


def evaluate_report_protocol(tuning, steps=HEADLINE_STEPS):
    """Use the historic report scoring protocol for the current UKF only."""
    results = {}
    for name in RD_DATASETS:
        track = pd.read_csv(DATA_FOLDER / f"{name}.csv")[["xpos", "ypos", "zpos"]].to_numpy(dtype=float)
        forecasts = np.nan_to_num(run_filter(track, DT_RD, steps, tuning))
        true, pred, hold = track[steps:], forecasts[:len(track) - steps], track[:len(track) - steps]
        valid = ~np.all(pred == 0, axis=1) & ~np.all(true == 0, axis=1)
        rmse = float(np.sqrt(np.mean(np.sum((pred[valid] - true[valid]) ** 2, axis=1))))
        hold_rmse = float(np.sqrt(np.mean(np.sum((hold[valid] - true[valid]) ** 2, axis=1))))
        results[name] = (rmse, hold_rmse, 100 * (1 - rmse / hold_rmse) if hold_rmse else float('nan'))
    return results


def print_strict_table(strict, case_names):
    print("\nSTRICT protocol — improvement over last valid position (ZOH), %")
    for steps in HORIZON_STEPS:
        print(f"\n{steps} frames ahead")
        print(f"  {'variant':28s}" + ''.join(f'{name:>12s}' for name in case_names))
        for variant in VARIANTS:
            values = [strict[variant][steps][name][2] for name in case_names]
            print(f'  {variant:28s}' + ''.join(f'{v:+11.1f}%' for v in values))
    print(f"\n3D RMSE at {HEADLINE_STEPS} frames ahead (mm)")
    for name in case_names:
        model, hold, _, n = strict['UKF'][HEADLINE_STEPS][name]
        print(f'  {name}: UKF={model * 1000:.2f}, ZOH={hold * 1000:.2f}, n={n}')


def print_report_table(report):
    print("\nReport-protocol evaluation (current UKF only)")
    for name, (rmse, hold, improvement) in report.items():
        print(f'  {name}: UKF={rmse:.5f} m, ZOH={hold:.5f} m, gain={improvement:+.1f}%')


def plot_results(strict, case_names):
    """Plot the active UKF's per-case results and variant worst-case scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    vals = [strict['UKF'][HEADLINE_STEPS][name][2] for name in case_names]
    ax1.bar(np.arange(len(case_names)), vals)
    ax1.set_xticks(np.arange(len(case_names)), case_names)
    ax1.set_ylabel('Improvement over ZOH (%)')
    ax1.set_title(f'UKF, {HEADLINE_STEPS} frames ahead')
    ax1.axhline(0, color='black', linewidth=.8)
    ax1.grid(True, axis='y', alpha=.25)
    for variant in VARIANTS:
        worst = [min(strict[variant][steps][name][2] for name in case_names)
                 for steps in HORIZON_STEPS]
        ax2.plot(HORIZON_STEPS, worst, marker='o', label=variant)
    ax2.axhline(0, color='black', linewidth=.8)
    ax2.set_xlabel('Forecast horizon (frames)')
    ax2.set_ylabel('Worst-case gain over ZOH (%)')
    ax2.set_xticks(HORIZON_STEPS)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=.25)
    fig.suptitle('UKF against zero-order hold')
    plt.tight_layout()
    plt.show()


def main():
    cases = load_test_cases()
    names = list(cases)
    strict = {variant: {steps: evaluate_strict(cases, tuning, steps) for steps in HORIZON_STEPS}
              for variant, tuning in VARIANTS.items()}
    report = evaluate_report_protocol(VARIANTS['UKF'])
    print_strict_table(strict, names)
    print_report_table(report)
    if SHOW_PLOTS:
        plot_results(strict, names)
    return strict, report


if __name__ == '__main__':
    main()
