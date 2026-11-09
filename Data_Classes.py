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


@dataclass
class LullStatus:
    """
    Result of a single LullDetector.update() call.

    Describes whether a tracked point (e.g. a mussel-cluster centroid) is
    currently in a calm ("lull") window of otherwise wave-driven
    oscillatory motion, so a docking/collection manoeuvre can be timed to
    coincide with reduced motion instead of chasing a constantly-swaying
    target.
    """

    is_calm: bool  # debounced calm/energetic classification
    calm_streak: int  # consecutive frames currently below the calm threshold
    energetic_streak: int  # consecutive frames currently above the calm threshold
    speed: float  # current short-window speed estimate (m/s)
    baseline_speed: float  # slow-moving baseline speed estimate (m/s) -- tracks prevailing sea state
    ratio: float  # speed / baseline_speed -- below calm_ratio means "calm"


@dataclass
class LullStatus:
    """
    Result of a single LullDetector.update() call.

    Whether a tracked point (e.g. a mussel-cluster centroid) is currently
    moving slower than a fixed speed threshold, held for a minimum
    duration -- a calm ("lull") window in otherwise wave-driven motion,
    so a docking/collection manoeuvre can be timed to coincide with it.
    """

    is_calm: bool  # speed has stayed below the threshold for min_calm_seconds
    speed: float  # current windowed speed (m/s)
    calm_streak: int  # consecutive frames speed has been below the threshold
