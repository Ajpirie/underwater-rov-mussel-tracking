import numpy as np
import pandas as pd
import scipy as sp

import matplotlib

# MUST be set before importing pyplot to prevent macOS thread crashes
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from Data_Classes import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data_folder = "Data/Real/"
filenames = np.array(["RD_001", "RD_002", "RD_003", "RD_004"])
freq_limit = None  # Hz


class ppConfig:

    def __init__(self, output_filepath):
        self.output_filepath = output_filepath
        self.timeOffset = False
        self.ioAxes = True
        self.predictionErr = True
        self.comp3D = True


def matPrint(
    arr,
    orientation: str = "horizontal",
    precision: int = 3,
    show_imaginary: bool = True,
):
    """
    Prints a 2D array in a formatted manner with specified orientation and precision.

    Args:
        arr (array-like): The 2D array to be printed.
        orientation (str): The orientation of the output, either "horizontal" or "vertical". Default is "horizontal".
        precision (int): The number of decimal places to format the values. Default is 3.
        show_imaginary (bool): Whether to show the imaginary part of complex numbers. Default is True.
    """
    arr = np.array(arr)

    # Ensure 2D
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1) if orientation == "vertical" else arr.reshape(1, -1)

    # Apply orientation
    if orientation == "vertical":
        arr = arr.T
    elif orientation != "horizontal":
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    # Format values
    if not show_imaginary and np.iscomplexobj(arr):
        arr = np.real(arr)

    formatted = [[f"{val:.{precision}f}" for val in row] for row in arr]

    # Column widths
    col_widths = [max(len(row[i]) for row in formatted) for i in range(len(formatted[0]))]

    # Build rows
    rows = []
    for row in formatted:
        line = "  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
        rows.append(line)

    # Uniform width (strict)
    inner_width = max(len(r) for r in rows)

    n_rows = len(rows)

    # Print with exact width control
    for i, row in enumerate(rows):
        if i == 0:
            left, right = "⎡", "⎤"
        elif i == n_rows - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"

        # enforce exact width between brackets
        print(f"{left} {row:<{inner_width}} {right}")


def compareFreqSpectra():
    labels = []
    raw_data = []

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for index, filename in enumerate(filenames):
        file = data_folder + filename + ".csv"
        extracted_data = pd.read_csv(file)

        def process(col):
            freq = np.abs(sp.fft.fft(col.values))
            freq = freq[: len(freq) // 2]  # positive frequencies only
            if freq_limit is not None:
                freq = freq[:freq_limit]
            return freq

        raw_data.append(
            [
                process(extracted_data["xpos"]),
                process(extracted_data["ypos"]),
                process(extracted_data["zpos"]),
            ]
        )
        labels.append(filename)

    n_datasets = len(raw_data)
    offset_step = 0.3  # horizontal nudge between datasets

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axis_titles = [
        "X Position Frequency Spectrum",
        "Y Position Frequency Spectrum",
        "Z Position Frequency Spectrum",
    ]

    for ax_idx, (ax, title) in enumerate(zip(axes, axis_titles)):
        yMax = 0

        for i, (data, plot_label) in enumerate(zip(raw_data, labels)):
            freq_data = data[ax_idx]
            color = colors[i % len(colors)]

            # Each dataset gets its own x positions with a small offset
            n = len(freq_data)
            x_positions = np.arange(n) + i * offset_step

            markerline, stemlines, baseline = ax.stem(x_positions, freq_data, label=plot_label)
            # Apply consistent colour to all stem parts
            plt.setp(stemlines, color=color, linewidth=0.8)
            plt.setp(markerline, color=color, markersize=4)
            plt.setp(baseline, visible=False)

            yMax = max(yMax, np.max(freq_data))

        ax.set_ylim(0, yMax * 1.1)
        ax.grid(True, alpha=0.4)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plotInputVOutputAxes(
    ax1,
    ax2,
    ax3,
    xPos: np.ndarray,
    xPos_Future: np.ndarray,
    yPos: np.ndarray,
    yPos_Future: np.ndarray,
    zPos: np.ndarray,
    zPos_Future: np.ndarray,
    time: np.ndarray,
    time_future: np.ndarray,
):
    """
    Draws the X/Y/Z position-over-time comparisons onto three pre-created axes
    (rather than making its own figure), so it can be combined with other
    plots/animations on a single shared figure. All three axes share the same
    y-axis scale (derived from the combined range of every plotted value),
    so motion magnitude can be compared directly across X, Y, and Z.
    """

    def get_ylim(data, margin=0.1):
        lo, hi = np.min(data), np.max(data)
        pad = (hi - lo) * margin if (hi - lo) > 0 else 1.0
        return lo - pad, hi + pad

    # Shared y-limit derived from all plotted position data (actual + predicted,
    # across X/Y/Z) so every subplot uses the same scale.
    all_positions = np.concatenate([xPos, xPos_Future, yPos, yPos_Future, zPos, zPos_Future])
    shared_ylim = get_ylim(all_positions)

    ax1.set_xlim(0, time[-1] * 1.05)
    ax1.set_ylim(*shared_ylim)
    ax1.grid(True)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("X Position (m)\n\n <-- Left/Right -->")
    ax1.set_title("X Position Over Time")
    ax1.plot(time, xPos, label="Actual", color="blue")
    ax1.plot(time_future, xPos_Future, label="Predicted", color="red")
    ax1.legend()

    # Plot Y Position Over Time
    ax2.set_xlim(0, time[-1] * 1.05)
    ax2.set_ylim(*shared_ylim)
    ax2.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y Position (m)\n\n <-- Forward/Backward -->")
    ax2.set_title("Y Position Over Time")
    ax2.plot(time, yPos, label="Actual", color="blue")
    ax2.plot(time_future, yPos_Future, label="Predicted", color="red")
    ax2.legend()

    # Plot Z Position Over Time
    ax3.set_xlim(0, time[-1] * 1.05)
    ax3.set_ylim(*shared_ylim)
    ax3.grid(True)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Z Position (m)\n\n <-- Up/Down -->")
    ax3.set_title("Z Position Over Time")
    ax3.plot(time, zPos, label="Actual", color="blue")
    ax3.plot(time_future, zPos_Future, label="Predicted", color="red")
    ax3.legend()


def plotPredictionError(
    xPos: np.ndarray,
    xPos_Future: np.ndarray,
    yPos: np.ndarray,
    yPos_Future: np.ndarray,
    zPos: np.ndarray,
    zPos_Future: np.ndarray,
    time: np.ndarray,
    predict_num_timesteps: int,
    time_step: float,
    buff_len: int,
):
    """
    Directly compares each predicted position against the REAL measured position at the
    time it was predicting for (index-shifted by predict_num_timesteps), instead of only
    relying on eyeballing two time-shifted curves. Produces:
      1) prediction error vs time, per axis + combined L2 norm
      2) a parity plot (predicted vs actual) with a "perfect prediction" reference line

    Also prints RMSE for the model and for a naive "zero-order hold" baseline (predicting
    that position simply won't change), so you have a reference point for what "no real
    predictive skill" would look like.
    """

    N = len(xPos)
    valid_len = N - predict_num_timesteps
    if valid_len <= 0:
        print("Not enough data to evaluate predictions: recording is shorter than the prediction horizon.")
        return

    # Ground truth at the time each prediction was actually targeting #
    xTrue, yTrue, zTrue = (
        xPos[predict_num_timesteps:],
        yPos[predict_num_timesteps:],
        zPos[predict_num_timesteps:],
    )
    xPred, yPred, zPred = (
        xPos_Future[:valid_len],
        yPos_Future[:valid_len],
        zPos_Future[:valid_len],
    )
    t_eval = time[predict_num_timesteps:]

    # Drop the warm-up rows where the buffer wasn't full yet (predictor returns (0,0,0)) #
    warm_mask = ~((xPred == 0) & (yPred == 0) & (zPred == 0))
    xTrue, yTrue, zTrue = xTrue[warm_mask], yTrue[warm_mask], zTrue[warm_mask]
    xPred, yPred, zPred = xPred[warm_mask], yPred[warm_mask], zPred[warm_mask]
    t_eval = t_eval[warm_mask]

    if len(t_eval) == 0:
        print("No valid (post-warm-up) predictions available to evaluate.")
        return

    xErr, yErr, zErr = xPred - xTrue, yPred - yTrue, zPred - zTrue
    l2Err = np.sqrt(xErr**2 + yErr**2 + zErr**2)

    # --- Print RMSE summary, with a naive "hold current position" baseline for reference ---
    rmse = np.sqrt(np.mean(l2Err**2))

    xNaive, yNaive, zNaive = (
        xPos[:valid_len][warm_mask],
        yPos[:valid_len][warm_mask],
        zPos[:valid_len][warm_mask],
    )
    naive_err = np.sqrt((xNaive - xTrue) ** 2 + (yNaive - yTrue) ** 2 + (zNaive - zTrue) ** 2)
    naive_rmse = np.sqrt(np.mean(naive_err**2))

    print(f"-- Buffer Length: {buff_len} --\n")
    print(f"Prediction RMSE at {predict_num_timesteps * time_step:.1f}s horizon: {rmse:.4f} m")
    print(f"Naive 'hold position' baseline RMSE:            {naive_rmse:.4f} m")
    if naive_rmse > 0:
        print(f"Model improves on naive baseline by: {(1 - rmse / naive_rmse) * 100:.1f}%")
    print("\n")


def plot3DDataComparison(
    ax,
    fig,
    xPos_1: np.ndarray,
    yPos_1: np.ndarray,
    zPos_1: np.ndarray,
    xPos_2: np.ndarray,
    yPos_2: np.ndarray,
    zPos_2: np.ndarray,
    plot1_label: str,
    plt2_label: str,
):
    """
    Draws the animated 3D real-vs-predicted comparison onto a pre-created 3D axis
    (rather than making its own figure), so it can be combined with other
    plots on a single shared figure/window.
    """
    TRAIL_LENGTH = 20
    PADDING = 0.1

    def combined_limits(arr1, arr2):
        lo = min(np.min(arr1), np.min(arr2))
        hi = max(np.max(arr1), np.max(arr2))
        pad = (hi - lo) * PADDING if (hi - lo) > 0 else 1.0
        return lo - pad, hi + pad

    # Remap: X=left/right, Y=depth (into screen), Z=up/down — same convention as original
    x_min, x_max = combined_limits(xPos_1, xPos_2)
    y_min, y_max = combined_limits(zPos_1, zPos_2)
    z_min, z_max = combined_limits(yPos_1, yPos_2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X")
    ax.xaxis.labelpad = 30
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    ax.view_init(elev=20, azim=-75)
    ax.tick_params(axis="x", labelrotation=65)
    for label in ax.get_xticklabels():
        label.set_verticalalignment("top")
        label.set_horizontalalignment("right")

    # Dataset 1 — blue
    (point1,) = ax.plot([], [], [], color="tab:blue", marker="o", label=plot1_label, zorder=5)
    (trail1,) = ax.plot([], [], [], color="tab:blue", alpha=0.4, linewidth=1.5, zorder=4)

    # Dataset 2 — red
    (point2,) = ax.plot([], [], [], color="tab:red", marker="o", label=plt2_label, zorder=5)
    (trail2,) = ax.plot([], [], [], color="tab:red", alpha=0.4, linewidth=1.5, zorder=4)

    ax.grid(True)
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    def init():
        for artist in [point1, trail1, point2, trail2]:
            artist.set_data([], [])
            artist.set_3d_properties([])
        time_text.set_text("")
        return point1, trail1, point2, trail2, time_text

    def update(frame):
        # Clamp frame index independently so mismatched-length datasets are handled gracefully
        f1 = min(frame, len(xPos_1) - 1)
        f2 = min(frame, len(xPos_2) - 1)
        ts1 = max(0, f1 - TRAIL_LENGTH)
        ts2 = max(0, f2 - TRAIL_LENGTH)

        # Dataset 1 — remap axes: plot(x, z, y)
        point1.set_data([xPos_1[f1]], [zPos_1[f1]])
        point1.set_3d_properties([yPos_1[f1]])
        trail1.set_data(xPos_1[ts1 : f1 + 1], zPos_1[ts1 : f1 + 1])
        trail1.set_3d_properties(yPos_1[ts1 : f1 + 1])

        # Dataset 2 — remap axes: plot(x, z, y)
        point2.set_data([xPos_2[f2]], [zPos_2[f2]])
        point2.set_3d_properties([yPos_2[f2]])
        trail2.set_data(xPos_2[ts2 : f2 + 1], zPos_2[ts2 : f2 + 1])
        trail2.set_3d_properties(yPos_2[ts2 : f2 + 1])

        time_text.set_text(f"Frame: {frame}")
        return point1, trail1, point2, trail2, time_text

    anim = ani.FuncAnimation(
        fig,
        update,
        frames=max(len(xPos_1), len(xPos_2)),
        init_func=init,
        interval=50,
        blit=True,
    )
    return anim


def createCombinedFigure(cfg, mussel_ID: int, buff_len: int):
    """
    Creates a single figure sized and laid out to hold every plot/animation
    requested in cfg for one mussel, so results appear together in one window
    instead of separate pop-ups.

    Layout:
        - ioAxes only:  3 stacked subplots (X, Y, Z position over time)
        - comp3D only:  1 large 3D subplot
        - both:         3 stacked subplots on the left, 3D subplot spanning
                        the full height on the right

    Args:
        cfg: The ppConfig-like object with ioAxes/comp3D flags.
        mussel_ID (int): ID of the mussel, used in the figure title.
        buff_len (int): Buffer length used for the prediction, used in the figure title.

    Returns:
        tuple: (fig, axes) where axes is a dict that may contain keys
        "x", "y", "z" (2D axes) and/or "3d" (Axes3D), depending on cfg.
        Returns (None, {}) if neither ioAxes nor comp3D is requested.
    """
    show_axes = cfg.ioAxes
    show_3d = cfg.comp3D

    if not show_axes and not show_3d:
        return None, {}

    axes = {}

    # Extra vertical gap between the stacked X/Y/Z axes so each subplot's title
    # doesn't collide with the axis labels/ticks of the plot above it.
    IO_AXES_HSPACE = 0.6

    if show_axes and show_3d:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1], hspace=IO_AXES_HSPACE)
        axes["x"] = fig.add_subplot(gs[0, 0])
        axes["y"] = fig.add_subplot(gs[1, 0])
        axes["z"] = fig.add_subplot(gs[2, 0])
        axes["3d"] = fig.add_subplot(gs[:, 1], projection="3d")
    elif show_axes:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13), gridspec_kw={"hspace": IO_AXES_HSPACE})
        axes["x"], axes["y"], axes["z"] = ax1, ax2, ax3
    else:  # show_3d only
        fig = plt.figure(figsize=(10, 8))
        axes["3d"] = fig.add_subplot(111, projection="3d")

    fig.suptitle(
        f"Prediction Results For Mussel: {mussel_ID} | Buffer Length = {buff_len}",
        fontweight="bold",
        fontsize=16,
    )

    return fig, axes


def postProcessing(cfg, result: PredictionResult):
    """
    Runs the configured post-processing steps for a single mussel's prediction result.

    Every requested graph/animation (position-over-time axes and the 3D real-vs-predicted
    comparison) is drawn onto one shared figure/window via createCombinedFigure, so they
    can all be viewed together rather than popping up as separate windows. The RMSE summary
    (predictionErr) is printed to the console rather than plotted, so it isn't part of the
    figure.
    """

    mussel_ID = result.mussel_id

    time_Full = result.time

    xPos_Full = result.x
    yPos_Full = result.y
    zPos_Full = result.z

    xPos_Future = result.x_pred
    yPos_Future = result.y_pred
    zPos_Future = result.z_pred

    num_time_steps = result.prediction_steps
    buffer_length = result.buffer_length
    time_step = result.timestep

    if cfg.timeOffset:
        time_future = time_Full + (num_time_steps * time_step)

    else:
        time_future = time_Full

    if cfg.predictionErr:
        plotPredictionError(
            xPos_Full,
            xPos_Future,
            yPos_Full,
            yPos_Future,
            zPos_Full,
            zPos_Future,
            time_Full,
            num_time_steps,
            time_step,
            buffer_length,
        )

    # Build one combined figure/window holding every requested graph/animation for this mussel #
    fig, axes = createCombinedFigure(cfg, mussel_ID, buffer_length)

    if fig is None:
        return None

    anim = None

    if cfg.ioAxes:
        plotInputVOutputAxes(
            axes["x"],
            axes["y"],
            axes["z"],
            xPos_Full,
            xPos_Future,
            yPos_Full,
            yPos_Future,
            zPos_Full,
            zPos_Future,
            time_Full,
            time_future,
        )

    if cfg.comp3D:
        anim = plot3DDataComparison(
            axes["3d"],
            fig,
            xPos_Full,
            yPos_Full,
            zPos_Full,
            xPos_Future,
            yPos_Future,
            zPos_Future,
            "Real Position",
            "Predicted Position",
        )

    # NOTE: plt.tight_layout() is intentionally avoided here — matplotlib warns that it
    # isn't reliable on figures containing 3D axes, and can end up shrinking the hspace
    # we deliberately set in createCombinedFigure, reintroducing title overlap. Reserving
    # the top margin directly keeps the spacing we configured intact.
    fig.subplots_adjust(top=0.90)
    plt.show()

    return anim


def postProcessingIndividuals(cfg, results_list: list):
    """
    Creates a single Tkinter window with a dropdown menu to select and view
    each individual mussel's tracking data, avoiding multiple popups.
    """
    if not results_list:
        return

    root = tk.Tk()
    root.title("Individual Mussel Predictions")

    # Create control frame
    ctrl_frame = ttk.Frame(root)
    ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(ctrl_frame, text="Select Detected Mussel ID:").pack(side=tk.LEFT, padx=5)

    mussel_ids = [str(r.mussel_id) for r in results_list]
    selected_mussel = tk.StringVar(value=mussel_ids[0])

    dropdown = ttk.Combobox(ctrl_frame, textvariable=selected_mussel, values=mussel_ids, state="readonly")
    dropdown.pack(side=tk.LEFT, padx=5)

    # Create canvas frame
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # State dictionary to hold mutable references
    state = {"canvas": None, "anim": None}

    def draw_selected(*args):
        m_id = int(selected_mussel.get())
        result = next(r for r in results_list if r.mussel_id == m_id)

        # Cleanup previous plot to prevent memory leaks
        if state["canvas"] is not None:
            state["canvas"].get_tk_widget().destroy()
            state["canvas"] = None

        if state["anim"] is not None:
            # Add a check to ensure the event_source exists before stopping
            if state["anim"].event_source is not None:
                state["anim"].event_source.stop()
            state["anim"] = None

        plt.close("all")

        # Build new figure
        fig, axes = createCombinedFigure(cfg, result.mussel_id, result.buffer_length)
        if fig is None:
            return

        time_future = result.time + (result.prediction_steps * result.timestep) if cfg.timeOffset else result.time

        if cfg.ioAxes:
            plotInputVOutputAxes(
                axes["x"],
                axes["y"],
                axes["z"],
                result.x,
                result.x_pred,
                result.y,
                result.y_pred,
                result.z,
                result.z_pred,
                result.time,
                time_future,
            )

        if cfg.comp3D:
            anim = plot3DDataComparison(
                axes["3d"],
                fig,
                result.x,
                result.y,
                result.z,
                result.x_pred,
                result.y_pred,
                result.z_pred,
                "Real Position",
                "Predicted Position",
            )
            state["anim"] = anim

        fig.subplots_adjust(top=0.90)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        state["canvas"] = canvas

    selected_mussel.trace_add("write", draw_selected)

    # Initial plot
    draw_selected()

    # Block script execution until this window is closed by the user
    root.mainloop()
