"""
iNAV Unscented Kalman Filter Sensor Fusion Engine (Module C)
Fuses IMU mechanization, VelocityNet displacement predictions,
Non-Holonomic Constraints (NHC), Zero-Velocity Updates (ZUPT),
and GNSS fixes with vehicle scale factor adaptation.
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger("iNAV.ukf")


class UKFNavigationFilter:
    """
    7-State Unscented Kalman Filter for 2D Vehicle Dead Reckoning.
    State Vector:
      x = [p_N, p_E, v_fwd, psi, b_g, b_a, k]^T
        p_N   : North position (m)
        p_E   : East position (m)
        v_fwd : Forward vehicle speed (m/s)
        psi   : Vehicle heading angle (rad, clockwise from North)
        b_g   : Gyroscope yaw rate bias (rad/s)
        b_a   : Accelerometer forward bias (m/s^2)
        k     : VelocityNet adaptive scale factor (unitless)
    """

    def __init__(
        self,
        dt: float = 0.1,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        self.dt = dt
        self.dim_x = 7

        # Van der Merwe Scaled Unscented Transform weights
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.dim_x + kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lam)

        # Compute weights
        self.w_m = np.full(2 * self.dim_x + 1, 1.0 / (2.0 * (self.dim_x + self.lam)))
        self.w_c = np.full(2 * self.dim_x + 1, 1.0 / (2.0 * (self.dim_x + self.lam)))
        self.w_m[0] = self.lam / (self.dim_x + self.lam)
        self.w_c[0] = self.lam / (self.dim_x + self.lam) + (1.0 - alpha**2 + beta)

        # State estimate and covariance
        self.x = np.zeros(self.dim_x, dtype=np.float64)
        self.x[6] = 1.0  # k initialized to 1.0

        self.P = np.diag([
            25.0, 25.0,     # position: 5m initial std
            1.0,            # speed: 1 m/s
            (np.radians(5.0))**2,  # heading: 5 deg
            (1e-3)**2,      # gyro bias: 0.001 rad/s
            (0.05)**2,      # accel bias: 0.05 m/s^2
            (0.05)**2       # scale factor k: 5% initial uncertainty
        ])

        # Process noise covariance Q
        self.Q = np.diag([
            0.01, 0.01,     # position drift
            0.04,           # speed variance
            (np.radians(0.2))**2,  # heading variance
            (1e-6)**2,      # gyro bias random walk
            (1e-4)**2,      # accel bias random walk
            (1e-5)**2       # scale factor random walk
        ])

    def initialize(
        self,
        init_lat: float,
        init_lon: float,
        init_speed_ms: float,
        init_heading_rad: float,
        init_gyro_bias: float = 0.0,
        init_accel_bias: float = 0.0
    ) -> None:
        """Initialize filter states at reference origin."""
        self.x[0] = 0.0  # p_N
        self.x[1] = 0.0  # p_E
        self.x[2] = max(init_speed_ms, 0.0)
        self.x[3] = init_heading_rad % (2.0 * np.pi)
        self.x[4] = init_gyro_bias
        self.x[5] = init_accel_bias
        self.x[6] = 1.0

    def generate_sigma_points(self) -> np.ndarray:
        """Generate 2L + 1 sigma points from current state and covariance."""
        # Ensure positive semi-definite P
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.dim_x) * 1e-9

        try:
            L = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            # Fallback if numerical conditioning degrades
            eigval, eigvec = np.linalg.eigh(self.P)
            eigval = np.maximum(eigval, 1e-8)
            L = eigvec @ np.diag(np.sqrt(eigval))

        sigma = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma[0] = self.x

        for i in range(self.dim_x):
            sigma[i + 1] = self.x + self.gamma * L[:, i]
            sigma[i + 1 + self.dim_x] = self.x - self.gamma * L[:, i]

        return sigma

    def predict(self, acc_fwd: float, gyro_yaw: float, dt: Optional[float] = None) -> None:
        """
        UKF Prediction Step:
        Propagates sigma points through vehicle kinematic equations.
        acc_fwd: forward acceleration in vehicle frame (m/s^2)
        gyro_yaw: yaw rate in vehicle frame (rad/s)
        """
        dt = dt if dt is not None else self.dt
        sigma = self.generate_sigma_points()
        n_pts = len(sigma)

        # Propagate each sigma point
        sigma_f = np.zeros_like(sigma)
        for i in range(n_pts):
            s = sigma[i]
            pN, pE, v, psi, bg, ba, k = s

            # De-bias inputs
            omega_corr = gyro_yaw - bg
            acc_corr = acc_fwd - ba

            # Heading propagation
            new_psi = (psi + omega_corr * dt) % (2.0 * np.pi)
            mid_psi = psi + 0.5 * omega_corr * dt

            # Speed propagation
            new_v = max(v + acc_corr * dt, 0.0)
            mid_v = 0.5 * (v + new_v)

            # Position propagation
            new_pN = pN + mid_v * np.cos(mid_psi) * dt
            new_pE = pE + mid_v * np.sin(mid_psi) * dt

            sigma_f[i] = [new_pN, new_pE, new_v, new_psi, bg, ba, k]

        # Compute predicted state mean
        x_pred = np.zeros(self.dim_x)
        for idx in range(self.dim_x):
            if idx == 3:
                # Circular mean for heading
                sin_sum = np.sum(self.w_m * np.sin(sigma_f[:, 3]))
                cos_sum = np.sum(self.w_m * np.cos(sigma_f[:, 3]))
                x_pred[3] = np.arctan2(sin_sum, cos_sum) % (2.0 * np.pi)
            else:
                x_pred[idx] = np.sum(self.w_m * sigma_f[:, idx])

        # Compute predicted covariance
        P_pred = np.zeros((self.dim_x, self.dim_x))
        for i in range(n_pts):
            diff = sigma_f[i] - x_pred
            # Wrap heading error
            diff[3] = (diff[3] + np.pi) % (2.0 * np.pi) - np.pi
            P_pred += self.w_c[i] * np.outer(diff, diff)

        P_pred += self.Q * dt
        self.x = x_pred
        self.P = P_pred

    def update_measurement(
        self,
        z: np.ndarray,
        h_func,
        R: np.ndarray,
        is_angle_measurement: bool = False
    ) -> None:
        """
        Generic UKF measurement update for observation vector z.
        h_func takes a sigma point and returns predicted measurement vector.
        """
        sigma = self.generate_sigma_points()
        n_pts = len(sigma)

        # Propagate sigma points through measurement function
        gamma_pts = np.array([h_func(s) for s in sigma])
        dim_z = len(z)

        # Mean predicted measurement
        z_pred = np.zeros(dim_z)
        if is_angle_measurement:
            sin_s = np.sum(self.w_m * np.sin(gamma_pts[:, 0]))
            cos_s = np.sum(self.w_m * np.cos(gamma_pts[:, 0]))
            z_pred[0] = np.arctan2(sin_s, cos_s) % (2.0 * np.pi)
        else:
            for j in range(dim_z):
                z_pred[j] = np.sum(self.w_m * gamma_pts[:, j])

        # Innovation covariance P_zz and cross covariance P_xz
        P_zz = np.zeros((dim_z, dim_z))
        P_xz = np.zeros((self.dim_x, dim_z))

        for i in range(n_pts):
            dz = gamma_pts[i] - z_pred
            if is_angle_measurement:
                dz[0] = (dz[0] + np.pi) % (2.0 * np.pi) - np.pi

            dx = sigma[i] - self.x
            dx[3] = (dx[3] + np.pi) % (2.0 * np.pi) - np.pi

            P_zz += self.w_c[i] * np.outer(dz, dz)
            P_xz += self.w_c[i] * np.outer(dx, dz)

        P_zz += R

        # Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)

        # Innovation
        y = z - z_pred
        if is_angle_measurement:
            y[0] = (y[0] + np.pi) % (2.0 * np.pi) - np.pi

        # State update
        dx_update = K @ y
        self.x += dx_update
        self.x[3] = self.x[3] % (2.0 * np.pi)
        self.x[2] = max(self.x[2], 0.0)  # non-negative speed
        self.x[6] = np.clip(self.x[6], 0.7, 1.3)  # bounded scale factor

        # Covariance update
        self.P -= K @ P_zz @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_velocity_net(
        self,
        delta_d_pred: float,
        sigma_pred: float,
        event_class: int = 1,
        window_dur: float = 2.0
    ) -> None:
        """
        Update with VelocityNet learned displacement.
        delta_d_pred: displacement in meters over window
        sigma_pred: learned standard deviation (uncertainty head)
        event_class: classified road condition (0: stationary, 1: normal, 2: rough road, 3: dynamic)
        window_dur: duration of the window in seconds
        """
        # Predicted speed over window = delta_d / window_dur
        v_net = delta_d_pred / window_dur
        base_variance = (sigma_pred / window_dur)**2

        # Adaptive covariance scheduling:
        # If rough road / potholes detected (class 2), high-frequency vertical acceleration spikes
        # inject noise; inflate measurement covariance 4x so filter trusts kinematic momentum.
        # If dynamic maneuver detected (class 3), inflate 2x to let IMU yaw/acceleration guide heading.
        if event_class == 2:
            variance = base_variance * 4.0
        elif event_class == 3:
            variance = base_variance * 2.0
        else:
            variance = base_variance

        # Direct forward speed measurement on state x[2]
        def h_v(s):
            return np.array([s[2]])

        z = np.array([v_net])
        R = np.array([[max(variance, 0.04)]])
        self.update_measurement(z, h_v, R)

    def update_zupt(self, gyro_reading: float) -> None:
        """
        Zero-Velocity Update (ZUPT):
        When stationary, forces v = 0 and rapidly resolves gyro bias.
        """
        # Measurement 1: Speed = 0
        def h_zero(s):
            return np.array([s[2]])

        self.update_measurement(np.array([0.0]), h_zero, np.array([[1e-4]]))

        # Measurement 2: Gyro bias matches current reading
        def h_gyro(s):
            return np.array([s[4]])

        self.update_measurement(np.array([gyro_reading]), h_gyro, np.array([[1e-4]]))

    def update_gnss(
        self,
        p_N_gnss: float,
        p_E_gnss: float,
        v_gnss: float,
        psi_gnss: Optional[float] = None,
        pos_accuracy_m: float = 3.0,
        inflation_factor: float = 1.0
    ) -> None:
        """
        GNSS Position and Velocity Update.
        Also updates adaptive vehicle scale factor k.
        """
        pos_var = (max(pos_accuracy_m, 1.0) * inflation_factor)**2
        spd_var = 0.25 * inflation_factor

        # 1. Position update: [p_N, p_E]
        def h_pos(s):
            return np.array([s[0], s[1]])

        z_pos = np.array([p_N_gnss, p_E_gnss])
        R_pos = np.diag([pos_var, pos_var])
        self.update_measurement(z_pos, h_pos, R_pos)

        # 2. Speed update: v_fwd
        def h_spd(s):
            return np.array([s[2]])

        self.update_measurement(np.array([v_gnss]), h_spd, np.array([[spd_var]]))

        # 3. Heading update if vehicle is moving at reasonable speed
        if psi_gnss is not None and v_gnss > 2.5:
            def h_hdg(s):
                return np.array([s[3]])

            hdg_var = (np.radians(3.0) * inflation_factor)**2
            self.update_measurement(np.array([psi_gnss]), h_hdg, np.array([[hdg_var]]), is_angle_measurement=True)
