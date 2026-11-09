import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

data_folder = "Data/Real/"
filenames = np.array(["RD_001", 
                      "RD_002"])
freq_limit = None  # Hz

def compareFreqSpectra():
    labels = []
    raw_data = []

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for index, filename in enumerate(filenames):
        file = data_folder + filename + ".csv"
        extracted_data = pd.read_csv(file)

        def process(col):
            freq = np.abs(sp.fft.fft(col.values))
            freq = freq[:len(freq) // 2]  # positive frequencies only
            if freq_limit is not None:
                freq = freq[:freq_limit]
            return freq

        raw_data.append([
            process(extracted_data['xpos']),
            process(extracted_data['ypos']),
            process(extracted_data['zpos']),
        ])
        labels.append(filename)

    n_datasets = len(raw_data)
    offset_step = 0.3  # horizontal nudge between datasets

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axis_titles = [
        'X Position Frequency Spectrum',
        'Y Position Frequency Spectrum',
        'Z Position Frequency Spectrum',
    ]

    for ax_idx, (ax, title) in enumerate(zip(axes, axis_titles)):
        yMax = 0

        for i, (data, plot_label) in enumerate(zip(raw_data, labels)):
            freq_data = data[ax_idx]
            color = colors[i % len(colors)]

            # Each dataset gets its own x positions with a small offset
            n = len(freq_data)
            x_positions = np.arange(n) + i * offset_step

            markerline, stemlines, baseline = ax.stem(
                x_positions, freq_data, label=plot_label
            )
            # Apply consistent colour to all stem parts
            plt.setp(stemlines, color=color, linewidth=0.8)
            plt.setp(markerline, color=color, markersize=4)
            plt.setp(baseline, visible=False)

            yMax = max(yMax, np.max(freq_data))

        ax.set_ylim(0, yMax * 1.1)
        ax.grid(True, alpha=0.4)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compareFreqSpectra()