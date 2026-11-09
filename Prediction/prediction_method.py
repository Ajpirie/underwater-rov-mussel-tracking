"""Per-mussel UKF motion prediction.

The six-state filter estimates [x, y, z, vx, vy, vz] from measured positions.
It uses damped velocity, analytic process noise, innovation gating and
adaptive process-noise scaling. Each mussel has its own timing and tuning.

Create predictionInstance((x, y, z, ID), timestep=dt, num_timesteps=steps).
Call predictPositionUKF(elapsed_s=seconds) for irregular observation intervals;
omit elapsed_s to use the nominal timestep. Future positions are appended to
x_future_history, y_future_history and z_future_history.

The motion model, uncertainty equations and default tuning are unchanged.
"""

import functools
from dataclasses import dataclass, replace
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform

# STATE AND MEASUREMENT DIMENSIONS FOR UKF #
state_dim = 6  # [x, y, z, vx, vy, vz]
meas_dim = 3  # [x, y, z]


###############################################################
#   UKF TUNING
###############################################################


@dataclass
class UKFTuning:
    """
    Tuning parameters for the UKF, in physical units so they stay
    valid when the frame rate changes. The defaults were chosen with
    Test_Verification/UKF_Model_Exploration.py; the benchmark shows how
    sensitive the results are to each one.
    """

    # MOTION MODEL #
    velocity_tau: float | None = 1.5  # velocity decay time constant (s); None = constant velocity
    accel_noise: float = 1e-3  # initial white-noise acceleration PSD driving velocity (m^2/s^3)
    position_noise: float = 1e-3  # initial random-walk PSD on position (m^2/s) -- slow drift of the line

    # MEASUREMENT NOISE #
    meas_std: tuple = (1.4e-3, 0.8e-3, 2.0e-3)  # x, y, z standard deviation (m); depth is noisiest

    # INITIAL UNCERTAINTY #
    init_pos_std: float = 0.01  # m
    init_vel_std: float = 0.1  # m/s

    # OUTLIER GATING (None disables) #
    gate_chi2: float | None = 25.0  # reject if Mahalanobis distance^2 exceeds this (3 dof)...
    gate_min_dist: float = 0.03  # ...and the measurement is also this far from the prediction (m)
    reacquire_frames: int = 2  # consecutive rejected measurements before re-initialising...
    reacquire_spread: float = 0.02  # ...provided they all lie within this distance of each other (m)

    # ADAPTIVE PROCESS NOISE (None disables) #
    adapt_rate: float | None = 0.05  # step size of the log(Q scale) update per measurement
    adapt_after: int = 3  # measurements after (re)initialisation before adapting (P0 dominates first)
    q_scale_min: float = 1e-2  # Q scale bounds -> accel_noise stays within [1e-5, 10] m^2/s^3
    q_scale_max: float = 1e4
    nis_clip: float = 10.0  # cap on NIS / meas_dim per update, so one bad frame can't blow Q up


default_tuning = UKFTuning()


###############################################################
#   UKF PREDICTION CLASS
###############################################################


class predictionInstance:
    """
    Class for individual instances of UKFs, defines initialisation and prediction functions
    """

    def __init__(
        self,
        mussel_data: tuple = (0, 0, 0, 0),
        tuning: UKFTuning = None,
        *,
        timestep: float,
        num_timesteps: int,
    ):
        """
        Initializes a prediction instance for a mussel.

        Args:
            mussel_data (tuple): A tuple containing the initial positions and mussel ID: (x: float, y: float, z: float, ID: int).
            tuning (UKFTuning): Filter tuning. Defaults to default_tuning.
        """

        self.ID = mussel_data[3]  # ID of tracked mussel
        self.tuning = UKFTuning() if tuning is None else replace(tuning)

        # Measurement history
        self.x_history = [mussel_data[0]]
        self.y_history = [mussel_data[1]]
        self.z_history = [mussel_data[2]]

        # Prediction history
        self.x_future_history = []
        self.y_future_history = []
        self.z_future_history = []

        # Gating and adaptive process noise bookkeeping
        self.initialised = False  # set once the state has been started at a real (non-dropout) measurement
        self.q_scale = 1.0  # current multiplier on Q, learned from the innovations
        self.updates_since_init = 0  # accepted measurements since the last (re)initialisation
        self.rejected_meas = []  # consecutive measurements rejected by the gate
        
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("Timestep must be positive")

        if not isinstance(num_timesteps, int) or num_timesteps < 1:
            raise ValueError("Prediction steps must be a positive integer")

        self.timestep = float(timestep)
        self.num_timesteps = num_timesteps

        self.initUKF()  # UKF object for prediction

    def initUKF(self):

        # Initialize the UKF with appropriate parameters

        # define the sigma points for the UKF using MerweScaledSigmaPoints to handle the non-linearities in the state transition and measurement functions
        sigma_points = MerweScaledSigmaPoints(n=state_dim, alpha=0.1, beta=2.0, kappa=0)

        # Initialize the UKF with the defined parameters
        self.ukf = UKF(
            dim_x=state_dim,
            dim_z=meas_dim,
            dt=self.timestep,
            fx=functools.partial(_stateTransition, tau=self.tuning.velocity_tau),
            hx=_retrievePredictedPosition,
            points=sigma_points,
        )

        # Measurement noise covariance from the measured per-axis tracker noise
        self.ukf.R = np.diag(np.square(self.tuning.meas_std))

        # Process noise for one timestep, before adaptive scaling
        self.Q_base = _processNoise(self.timestep, self.tuning)
        self._process_dt = self.timestep
        self.ukf.Q = self.Q_base * self.q_scale

        # The state is started at the first real measurement by the first predictPositionUKF() call, so a
        # track that begins on a dropout frame doesn't start the filter at (0, 0, 0)
        self._resetState((0, 0, 0))

    def predictPositionUKF(self, elapsed_s=None):
        """Update one observation and forecast; elapsed_s overrides the nominal frame period."""
        dt = self.timestep if elapsed_s is None else float(elapsed_s)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Elapsed time must be a positive finite number")

        # Latest measurement (current position), all zeros = dropout
        current_meas = np.array([self.x_history[-1], self.y_history[-1], self.z_history[-1]])
        meas_valid = not np.all(current_meas == 0)

        if not self.initialised:
            # Start the filter at the first real measurement. That measurement is the initial state, so it
            # is not also applied as an update (which would count it twice)
            if meas_valid:
                self._resetState(current_meas)
                self.initialised = True

        else:
            # Recompute Q for the actual elapsed time; the default path is unchanged.
            if dt != self._process_dt:
                self.Q_base = _processNoise(dt, self.tuning)
                self._process_dt = dt
            self.ukf.Q = self.Q_base * self.q_scale
            if dt == self.timestep:
                self.ukf.predict()
            else:
                self.ukf.predict(dt=dt)

            # filterpy's update() reuses the sigma points from predict(), which were drawn before Q was added,
            # so the innovation covariance and gain would ignore this step's process noise. Redraw them from
            # the prior (x, P), as in Wan & van der Merwe's additive-noise UKF
            self.ukf.sigmas_f = self.ukf.points_fn.sigma_points(self.ukf.x, self.ukf.P)

            # Update the UKF with the current measurement only if it's not all zeros
            if meas_valid:
                self._updateWithMeasurement(current_meas)

        # Propagate the state mean forward to the prediction horizon. The real filter state is
        # not touched, so there is nothing to save and restore afterwards
        future_state = self.ukf.x.copy()

        for i in range(self.num_timesteps):
            future_state = self.ukf.fx(future_state, self.timestep)
            
        # retrieve predicted positon after iteration
        future_pos = self.ukf.hx(future_state)

        # Append predicted future position to history lists
        self.x_future_history.append(future_pos[0])
        self.y_future_history.append(future_pos[1])
        self.z_future_history.append(future_pos[2])

    def _updateWithMeasurement(self, meas: np.ndarray):
        """
        Gates one measurement, then either updates the UKF with it (adapting Q
        from the innovation) or rejects it as a tracking glitch.

        Args:
            meas (np.ndarray): Measured (x, y, z) position for this timestep.
        """

        # Predicted measurement and innovation covariance, from the predicted sigma points
        sigmas_h = np.array([self.ukf.hx(s) for s in self.ukf.sigmas_f])
        meas_pred, S = unscented_transform(sigmas_h, self.ukf.Wm, self.ukf.Wc, self.ukf.R)
        innovation = meas - meas_pred
        nis = float(innovation @ np.linalg.solve(S, innovation))  # normalised innovation squared

        if self._isOutlier(innovation, nis):
            self.rejected_meas.append(meas)

            # Several rejected measurements that agree with each other mean the mussel has
            # genuinely moved (or tracking re-acquired it), so restart the filter there
            if self._isReacquisition():
                self._resetState(meas)

            return

        self.rejected_meas = []
        self.ukf.update(meas)
        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T)  # keep P symmetric for the next sigma points
        self.updates_since_init += 1

        if self.tuning.adapt_rate is not None and self.updates_since_init > self.tuning.adapt_after:
            self._adaptProcessNoise(nis)

    def _isOutlier(self, innovation: np.ndarray, nis: float) -> bool:
        # A glitch has to be both statistically implausible and physically far from the prediction
        if self.tuning.gate_chi2 is None:
            return False

        return nis > self.tuning.gate_chi2 and np.linalg.norm(innovation) > self.tuning.gate_min_dist

    def _isReacquisition(self) -> bool:
        if len(self.rejected_meas) < self.tuning.reacquire_frames:
            return False

        recent = np.array(self.rejected_meas[-self.tuning.reacquire_frames :])
        spread = np.max(np.linalg.norm(recent - recent[-1], axis=1))

        return spread <= self.tuning.reacquire_spread

    def _adaptProcessNoise(self, nis: float):
        # For a consistent filter E[NIS] = meas_dim. Nudge log(q_scale) up when innovations are
        # bigger than the filter expects and down when they are smaller
        step = self.tuning.adapt_rate * (min(nis / meas_dim, self.tuning.nis_clip) - 1.0)
        self.q_scale = float(np.clip(self.q_scale * np.exp(step), self.tuning.q_scale_min, self.tuning.q_scale_max))

    def _resetState(self, position: tuple):
        # (Re)initialise at a measured position with zero velocity. q_scale is kept, since the
        # sea state doesn't change just because tracking was lost
        self.ukf.x = np.array([position[0], position[1], position[2], 0, 0, 0])
        self.ukf.P = np.diag([self.tuning.init_pos_std**2] * 3 + [self.tuning.init_vel_std**2] * 3)

        self.updates_since_init = 0
        self.rejected_meas = []


###############################################################
#   HELPER FUNCTIONS FOR PREDICTION CLASS
###############################################################


def _stateTransition(state: np.ndarray, dt: float, tau: float = None):

    # Damped velocity modelling: velocity decays as e^(-dt / tau), and position integrates that decaying
    # velocity, giving new_pos = old_pos + velocity * tau * (1 - e^(-dt / tau)).
    # tau = None is the constant velocity limit: new_state = old_state + state_velocity * dt

    if tau is None:
        decay = 1.0
        gain = dt
    else:
        decay = np.exp(-dt / tau)
        gain = tau * (1 - decay)

    F = np.array(
        [
            [1, 0, 0, gain, 0, 0],
            [0, 1, 0, 0, gain, 0],
            [0, 0, 1, 0, 0, gain],
            [0, 0, 0, decay, 0, 0],
            [0, 0, 0, 0, decay, 0],
            [0, 0, 0, 0, 0, decay],
        ]
    )

    return np.dot(F, state)


def _retrievePredictedPosition(state: np.ndarray):
    # Extract the position components from the state vector

    return np.array([state[0], state[1], state[2]])


def _processNoise(dt: float, tuning: UKFTuning):
    """
    Exact discrete-time process noise for one timestep of the damped velocity
    model (white-noise acceleration of PSD accel_noise driving a velocity that
    decays with time constant velocity_tau), plus a random walk of PSD
    position_noise on position. Matches the Van Loan matrix-exponential
    discretisation to machine precision.

    Args:
        dt (float): Timestep (s).
        tuning (UKFTuning): Supplies velocity_tau, accel_noise and position_noise.

    Returns:
        np.ndarray: 6x6 Q matrix for the [x, y, z, vx, vy, vz] state layout.
    """

    q = tuning.accel_noise

    if tuning.velocity_tau is None:
        # Constant velocity (continuous white-noise acceleration)
        q_pp = q * dt**3 / 3
        q_pv = q * dt**2 / 2
        q_vv = q * dt
    else:
        a = 1 / tuning.velocity_tau
        e = np.exp(-a * dt)
        q_pp = q / a**2 * (dt - 2 * (1 - e) / a + (1 - e**2) / (2 * a))
        q_pv = q / a * ((1 - e) / a - (1 - e**2) / (2 * a))
        q_vv = q * (1 - e**2) / (2 * a)

    q_pp += tuning.position_noise * dt

    # Generate a 2x2 Q matrix for a single axis (position, velocity)
    q_axis = np.array([[q_pp, q_pv], [q_pv, q_vv]])

    # Kronecker with I3 places each axis' (position, velocity) block at [x, y, z, vx, vy, vz] indices,
    # i.e. the state's actual layout (the original block_diag assumed [x, vx, y, vy, z, vz])
    return np.kron(q_axis, np.eye(3))


###############################################################
#   HELPER FUNCTIONS FOR PREDICTION ROUTINE
###############################################################
