import math
import numpy as np
import pandas as pd
import scipy as sp
import csv
import matplotlib.pyplot as mpl
import Test_Verification.Plotting as plt


def generateTestData(N, delta_t, filename):
    """
    Generates synthetic buoy motion data with multiple wave components,
    drift, and noise, then writes to a CSV file.

    Args:
        N (int): Number of data points
        delta_t (float): Time step between samples
        filename (str): Output CSV file name

    Returns:
        t (np.array): Time array
        x (np.array): x position
        y (np.array): y position
    """
    try:
        assert delta_t >= 0.001
    except AssertionError:
        raise ValueError("delta_t must be at least 0.001")

    # Time vector
    t = np.arange(0, N * delta_t, delta_t)

    # Wave frequencies (Hz)
    frequencies = [0.05, 0.12, 0.2]

    # Amplitudes
    amplitudes_x = [2.0, 1.0, 0.5]
    amplitudes_y = [1.5, 0.8, 0.3]

    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, len(frequencies))

    # Initialize signals
    x = np.zeros(N)
    y = np.zeros(N)

    # Generate wave motion
    for i, f in enumerate(frequencies):
        omega = 2 * np.pi * f
        x += amplitudes_x[i] * np.sin(omega * t + phases[i])
        y += amplitudes_y[i] * np.cos(omega * t + phases[i])

    # Add drift (current)
    x += 0.01 * t
    y += -0.005 * t

    # Add noise
    noise_level = 0.2
    x += noise_level * np.random.randn(N)
    y += noise_level * np.random.randn(N)

    # Write to CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "xPosition", "yPosition"])
        for i in range(N):
            writer.writerow([(int(t[i] * 1000) / 1000), x[i], y[i]])

    return


def fetchData(file_path: str, N: int = None, time_step: float = 0.125):
    """
    Retrieves time, xPos, and yPos data of selected mussel from file_path CSV and returns buffer arrays for processing.

    Args:
        file_path (str): The path to the CSV file containing the data.
        N (int): The number of data points to include in the buffer.
        time_step (float): The time interval between data points.

    Returns:
        buffer_data (tuple): A tuple containing two numpy arrays: xPos_buffer, and yPos_buffer.
        delta_t (float): The time interval between data points.
    """

    # Extract data from the filepath.
    extracted_data = pd.read_csv(file_path)
    xPos = extracted_data["xpos"].values
    yPos = extracted_data["ypos"].values
    zPos = extracted_data["zpos"].values

    if N is None:
        N = len(xPos)  # Use all data points if N is not specified

    time = np.arange(0, N * time_step, time_step)  # Initialize time array with zeros
    print(f"Data extracted from {file_path}:\nX Position\nY Position\nZ Position")

    # Check enough data supplied to fill the buffer arrays.
    print(f"Checking if there are at least {N} data points in the file...")
    try:
        assert len(time) >= N
    except AssertionError:
        raise ValueError(
            f"Not enough data points in the file to fill the buffer arrays. Required: {N}, Found: {len(time)}"
        )

    # Create buffer arrays to specified length (N) for x and y position data.
    print(f"Creating buffer arrays of length {N} for x and y position data...")
    xPos_buffer = np.zeros(N)
    yPos_buffer = np.zeros(N)
    zPos_buffer = np.zeros(N)

    # Fill the buffer arrays with the first N data points from the extracted data.
    xPos_buffer[:] = xPos[:N]
    yPos_buffer[:] = yPos[:N]
    zPos_buffer[:] = zPos[:N]

    # Calculate the time interval (delta_t_test & delta_t_avg) between data points.
    delta_t_test: float = time[1] - time[0]  # Assuming uniform time intervals
    delta_t_avg: float = np.mean(np.diff(time))  # Average time interval via diff and mean

    # Check if the time intervals are consistent.
    print("Checking if time intervals between data points are consistent...")
    try:
        assert np.allclose(delta_t_avg, delta_t_test)
    except AssertionError:
        raise ValueError(
            f"Inconsistent time intervals between data points. Average delta_t: {delta_t_avg:.3f} seconds, Test delta_t: {delta_t_test:.3f} seconds"
        )

    print(
        f"Data successfully fetched and processed from {file_path}.\nBuffer arrays created with length {N}.\nAverage time interval (delta_t_avg): {delta_t_avg:.3f} seconds."
    )

    return (xPos_buffer, yPos_buffer, zPos_buffer), delta_t_avg


def transformData(xPos_buffer: np.ndarray, yPos_buffer: np.ndarray, zPos_buffer: np.ndarray):
    """
    Performs discrete Fourier transform on the x, y, and z position data in the buffer arrays to analyze frequency components.

    Args:
        xPos_buffer (np.ndarray): A numpy array containing the x position data.
        yPos_buffer (np.ndarray): A numpy array containing the y position data.
        zPos_buffer (np.ndarray): A numpy array containing the z position data.

    Returns:
        Pos_freq (tuple): A tuple containing the frequency components for x, y, and z position data as numpy arrays: (xPos_freq, yPos_freq, zPos_freq).
    """
    # Perform discrete Fourier transform on the x, y, and z position data in the buffer arrays.
    print("Performing discrete Fourier transform on the x, y, and z position data...")
    xPos_freq = sp.fft.fft(xPos_buffer)
    yPos_freq = sp.fft.fft(yPos_buffer)
    zPos_freq = sp.fft.fft(zPos_buffer)

    xPos_freq[0] = 0  # Set the DC component to zero to focus on oscillatory behavior
    yPos_freq[0] = 0
    zPos_freq[0] = 0

    print("Fourier transform completed successfully.")
    return (xPos_freq, yPos_freq, zPos_freq)


def predictFuturePosition(
    xPos_freq: np.ndarray,
    yPos_freq: np.ndarray,
    zPos_freq: np.ndarray,
    delta_t: float,
    prediction_period: float = 10.0,
    threshold: float = 10.0,
):
    """
    Predicts future positions of the mussel based on the frequency components obtained from the Fourier transform.

    Args:
        xPos_freq (np.ndarray): A numpy array containing the frequency components for x position data.
        yPos_freq (np.ndarray): A numpy array containing the frequency components for y position data.
        zPos_freq (np.ndarray): A numpy array containing the frequency components for z position data.
        delta_t (float): The time interval between data points.
        prediction_period (float): The time from current when the prediction should be made. Default is 10.0 seconds.
    Returns:
        future_positions (tuple): future_xPos, future_yPos, and future_zPos, the predicted future positions for x, y, and z respectively.
    """
    # Calculate the number of future time steps to predict based on the prediction period and delta_t.
    num_future_steps = int(prediction_period / delta_t)

    # Create time vector for future predictions
    future_time = np.arange(0, num_future_steps * delta_t, delta_t)

    xThreshold = threshold * np.max(np.abs(xPos_freq)) / 100.0
    yThreshold = threshold * np.max(np.abs(yPos_freq)) / 100.0
    zThreshold = threshold * np.max(np.abs(zPos_freq)) / 100.0

    xPos_freq_simplified = np.where(np.abs(xPos_freq) > xThreshold, xPos_freq, 0)
    yPos_freq_simplified = np.where(np.abs(yPos_freq) > yThreshold, yPos_freq, 0)
    zPos_freq_simplified = np.where(np.abs(zPos_freq) > zThreshold, zPos_freq, 0)

    # Reconstruct the signal from the frequency components for x, y, and z position data.
    print("Reconstructing the signal from the frequency components for x, y, and z position data...")
    future_xPos = sp.fft.ifft(xPos_freq_simplified)[:num_future_steps]
    future_yPos = sp.fft.ifft(yPos_freq_simplified)[:num_future_steps]
    future_zPos = sp.fft.ifft(zPos_freq_simplified)[:num_future_steps]

    print(f"Future position prediction completed successfully for a prediction period of {prediction_period} seconds.")
    return future_xPos[-1], future_yPos[-1], future_zPos[-1]


if __name__ == "__main__":
    file_path = "Data/Real/RD_001.csv"  # Update with your actual file path
    N = 100  # Number of data points to include in the buffer
    # generateTestData(N, 0.01, file_path)  # Generate test data and save to CSV

    buffer_data, delta_t = fetchData(file_path)

    anim = plt.plot3DData(buffer_data[0], buffer_data[1], buffer_data[2])
    mpl.show()
    Pos_freq = transformData(buffer_data[0], buffer_data[1], buffer_data[2])
    plt.matPrint(Pos_freq, orientation="vertical", precision=3)
    # plt.plotFrequencySpectrum(Pos_freq[0], "X Position Frequency Spectrum", Pos_freq[1], "Y Position Frequency Spectrum", Pos_freq[2], "Z Position Frequency Spectrum")
    future_positions = predictFuturePosition(Pos_freq[0], Pos_freq[1], Pos_freq[2], delta_t)
    plt.matPrint(future_positions, orientation="vertical", precision=3, show_imaginary=False)
