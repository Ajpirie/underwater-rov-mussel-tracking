from dataclasses import dataclass
import numpy as np


@dataclass
class PredictionResult:
    mussel_id: int

    time: np.ndarray

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    x_pred: np.ndarray
    y_pred: np.ndarray
    z_pred: np.ndarray

    prediction_steps: int
    buffer_length: int
    timestep: float
