import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import scipy as sp

def plot3DData(xPos_buffer: np.ndarray, yPos_buffer: np.ndarray, zPos_buffer: np.ndarray = None):
    use_3d = zPos_buffer is not None
    TRAIL_LENGTH = 20
    PADDING = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') if use_3d else fig.add_subplot(111)

    def tight_limits(arr):
        lo, hi = np.min(arr), np.max(arr)
        pad = (hi - lo) * PADDING if (hi - lo) > 0 else 1.0
        return lo - pad, hi + pad

    if use_3d:
        # Remap: X=left/right, Y=depth (into screen), Z=up/down
        # In matplotlib 3D: ax.X=left/right, ax.Y=depth, ax.Z=up/down
        # So pass data as: plot(xPos, zPos, yPos) → (left/right, depth, up/down)
        x_min, x_max = tight_limits(xPos_buffer)  # left/right
        y_min, y_max = tight_limits(zPos_buffer)  # depth (into screen)
        z_min, z_max = tight_limits(yPos_buffer)  # up/down

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_xlabel('X')
        ax.xaxis.labelpad = 30  # increase until it clears the tick labels
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        # Set view so Y axis goes into the screen, Z is up
        ax.view_init(elev=20, azim=-75)
        ax.tick_params(axis='x', labelrotation=65)
        for label in ax.get_xticklabels():
            label.set_verticalalignment('top')
            label.set_horizontalalignment('right')

        point, = ax.plot([], [], [], label='Mussel Position', marker='o', zorder=5)
        trail, = ax.plot([], [], [], color='tab:blue', alpha=0.4, linewidth=1.5, zorder=4)

        # Viewer marker: placed at the front of the Y axis (near viewer), centred on X and Z
        viewer_x = (x_min + x_max) / 2
        viewer_y = y_min - (y_max - y_min) * 0.15   # slightly in front of the plot
        viewer_z = (z_min + z_max) / 2
        ax.scatter([viewer_x], [viewer_y], [viewer_z],
                   marker='X', s=200, color='red', zorder=10)
        # Line from viewer toward the data to show line of sight
        ax.plot([viewer_x, viewer_x], [viewer_y, y_min],
                [viewer_z, viewer_z],
                color='red', alpha=0.3, linestyle='--', linewidth=1)
        ax.text(viewer_x, viewer_y, viewer_z, '', color='red', fontsize=8)

    else:
        x_min, x_max = tight_limits(xPos_buffer)
        y_min, y_max = tight_limits(yPos_buffer)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    ax.grid(True)
    time_text = (
        ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        if use_3d else
        ax.text(0.05, 0.95, '', transform=ax.transAxes)
    )
    ax.legend(loc='upper right')

    def init():
        if use_3d:
            point.set_data([], [])
            point.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])
        else:
            point.set_data([], [])
            trail.set_data([], [])
        time_text.set_text('')
        return point, trail, time_text

    def update(frame):
        trail_start = max(0, frame - TRAIL_LENGTH)

        if use_3d:
            # Remap data axes to plot axes: plot(x, z, y)
            x  = xPos_buffer[frame]
            px = zPos_buffer[frame]   # depth axis
            z  = yPos_buffer[frame]

            trail_x  = xPos_buffer[trail_start:frame + 1]
            trail_px = zPos_buffer[trail_start:frame + 1]
            trail_z  = yPos_buffer[trail_start:frame + 1]

            point.set_data([x], [px])
            point.set_3d_properties([z])
            trail.set_data(trail_x, trail_px)
            trail.set_3d_properties(trail_z)
        else:
            x, y = xPos_buffer[frame], yPos_buffer[frame]
            trail_x = xPos_buffer[trail_start:frame + 1]
            trail_y = yPos_buffer[trail_start:frame + 1]
            point.set_data([x], [y])
            trail.set_data(trail_x, trail_y)

        time_text.set_text(f'Frame: {frame}')
        return point, trail, time_text

    anim = ani.FuncAnimation(
        fig, update, frames=len(xPos_buffer),
        init_func=init, interval=50, blit=True
    )
    return anim

def plotFrequencySpectrum(x_freq_buffer: np.ndarray, x_title: str, y_freq_buffer: np.ndarray, y_title: str, z_freq_buffer: np.ndarray = None, z_title: str = "Z Position Frequency Spectrum"):
    """
    Plots the frequency spectrum of the input data from the buffer arrays.
    
    args:
        x_freq_buffer (np.ndarray): A numpy array containing the x position frequency data.
        x_title (str): The title for the x position frequency spectrum plot.
        y_freq_buffer (np.ndarray): A numpy array containing the y position frequency data.
        y_title (str): The title for the y position frequency spectrum plot.
        z_freq_buffer (np.ndarray, optional): A numpy array containing the z position frequency data.
        z_title (str): The title for the z position frequency spectrum plot.
    """
    if z_freq_buffer is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot X Position Frequency Spectrum
    ax1.set_xlim(-3, (len(x_freq_buffer) // 2) + 1)
    ax1.set_ylim(0, np.max(np.abs(x_freq_buffer)) * 1.1)
    ax1.grid(True)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(x_title)
    ax1.stem(np.abs(x_freq_buffer), label='X Position Frequency Spectrum')
    ax1.legend()

    # Plot Y Position Frequency Spectrum
    ax2.set_xlim(-3, (len(y_freq_buffer) // 2) + 1)
    ax2.set_ylim(0, np.max(np.abs(y_freq_buffer)) * 1.1)
    ax2.grid(True)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(y_title)
    ax2.stem(np.abs(y_freq_buffer), label='Y Position Frequency Spectrum')
    ax2.legend()

    # Plot Z Position Frequency Spectrum if provided
    if z_freq_buffer is not None:
        ax3.set_xlim(-3, (len(z_freq_buffer) // 2) + 1)
        ax3.set_ylim(0, np.max(np.abs(z_freq_buffer)) * 1.1)
        ax3.grid(True)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(z_title)
        ax3.stem(np.abs(z_freq_buffer), label='Z Position Frequency Spectrum')
        ax3.legend()
    
    plt.tight_layout()
    plt.show()

def matPrint(arr, orientation: str = "horizontal", precision: int = 3, show_imaginary: bool = True):
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

def get_ylim(data, margin=0.1):
    lo, hi = np.min(data), np.max(data)
    pad = (hi - lo) * margin  # padding based on range, not value magnitude
    return lo - pad, hi + pad

def plotInputVOutputAxes(xPos: np.ndarray, xPos_Future: np.ndarray, yPos: np.ndarray, yPos_Future: np.ndarray,  zPos: np.ndarray, zPos_Future: np.ndarray,  time: np.ndarray, time_future: np.ndarray):


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.set_xlim(0, time[-1] * 1.05)
    ax1.set_ylim(*get_ylim(xPos))
    ax1.grid(True)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)\n\n <-- Left/Right -->')
    ax1.set_title('X Position Over Time')
    ax1.plot(time, xPos, label='Actual', color='blue')
    ax1.plot(time_future, xPos_Future, label='Predicted', color='red')

    # Plot Y Position Frequency Spectrum
    ax2.set_xlim(0, time[-1] * 1.05)
    ax2.set_ylim(*get_ylim(yPos))
    ax2.grid(True)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)\n\n <-- Forward/Backward -->')
    ax2.set_title('Y Position Over Time')
    ax2.plot(time, yPos, label='Actual', color='blue')
    ax2.plot(time_future, yPos_Future, label='Predicted', color='red')

    # Plot Z Position Frequency Spectrum if provided
    ax3.set_xlim(0, time[-1] * 1.05)
    ax3.set_ylim(*get_ylim(zPos))
    ax3.grid(True)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (m)\n\n <-- Up/Down -->')
    ax3.set_title('Z Position Over Time')
    ax3.plot(time, zPos, label='Actual', color='blue')
    ax3.plot(time_future, zPos_Future, label='Predicted', color='red')
    
    plt.tight_layout()
    plt.show()

def plot3DDataComparison(xPos_1: np.ndarray, yPos_1: np.ndarray, zPos_1: np.ndarray, xPos_2: np.ndarray, yPos_2: np.ndarray, zPos_2: np.ndarray, plot1_label: str, plt2_label: str):
    TRAIL_LENGTH = 20
    PADDING = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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

    ax.set_xlabel('X')
    ax.xaxis.labelpad = 30
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.view_init(elev=20, azim=-75)
    ax.tick_params(axis='x', labelrotation=65)
    for label in ax.get_xticklabels():
        label.set_verticalalignment('top')
        label.set_horizontalalignment('right')

    # Dataset 1 — blue
    point1, = ax.plot([], [], [], color='tab:blue', marker='o', label=plot1_label, zorder=5)
    trail1, = ax.plot([], [], [], color='tab:blue', alpha=0.4, linewidth=1.5, zorder=4)

    # Dataset 2 — red
    point2, = ax.plot([], [], [], color='tab:red', marker='o', label=plt2_label, zorder=5)
    trail2, = ax.plot([], [], [], color='tab:red', alpha=0.4, linewidth=1.5, zorder=4)

    ax.grid(True)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    def init():
        for artist in [point1, trail1, point2, trail2]:
            artist.set_data([], [])
            artist.set_3d_properties([])
        time_text.set_text('')
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
        trail1.set_data(xPos_1[ts1:f1 + 1], zPos_1[ts1:f1 + 1])
        trail1.set_3d_properties(yPos_1[ts1:f1 + 1])

        # Dataset 2 — remap axes: plot(x, z, y)
        point2.set_data([xPos_2[f2]], [zPos_2[f2]])
        point2.set_3d_properties([yPos_2[f2]])
        trail2.set_data(xPos_2[ts2:f2 + 1], zPos_2[ts2:f2 + 1])
        trail2.set_3d_properties(yPos_2[ts2:f2 + 1])

        time_text.set_text(f'Frame: {frame}')
        return point1, trail1, point2, trail2, time_text

    anim = ani.FuncAnimation(
        fig, update, frames=max(len(xPos_1), len(xPos_2)),
        init_func=init, interval=50, blit=True
    )
    return anim

