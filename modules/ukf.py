"""
iNAV Unscented Kalman Filter Sensor Fusion Engine (Module C)
Fuses IMU mechanization, VelocityNet displacement predictions,
Non-Holonomic Constraints (NHC), Zero-Velocity Updates (ZUPT),
and GNSS fixes with vehicle scale factor adaptation.
"""

import logging
import math
from typing import Optional, Tuple, Dict, Any

import numpy as np

from modules.sensor_types import GnssSample, GnssValidity

logger = logging.getLogger("iNAV.ukf")

UKF_EARTH_RADIUS = 6371000.0
UKF_NIS_GATE_2D = 9.21                 # Chi-Square 2-DOF 99% threshold
UKF_NIS_GATE_1D = 6.635                # Chi-Square 1-DOF 99% threshold
UKF_MIN_HEADING_SPEED = 2.5            # Minimum speed (m/s) for course/heading update
UKF_DEFAULT_SPEED_SIGMA = 0.5          # Baseline 1-sigma speed noise (m/s)
UKF_DEFAULT_HEADING_SIGMA_DEG = 3.0    # Baseline 1-sigma course noise (deg)


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
        self.ref_lat = init_lat
        self.ref_lon = init_lon
        self.is_initialized = True

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
        is_angle_measurement: bool = False,
        nis_gate: Optional[float] = None
    ) -> Tuple[bool, float]:
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

        # Innovation
        y = z - z_pred
        if is_angle_measurement:
            y[0] = (y[0] + np.pi) % (2.0 * np.pi) - np.pi

        # Compute Normalized Innovation Squared (NIS)
        P_zz_inv = np.linalg.inv(P_zz)
        nis = float(y.T @ P_zz_inv @ y)

        if nis_gate is not None and nis > nis_gate:
            return False, nis

        # Kalman gain
        K = P_xz @ P_zz_inv

        # State update
        dx_update = K @ y
        self.x += dx_update
        self.x[3] = self.x[3] % (2.0 * np.pi)
        self.x[2] = max(self.x[2], 0.0)  # non-negative speed
        self.x[6] = np.clip(self.x[6], 0.7, 1.3)  # bounded scale factor

        # Covariance update
        self.P -= K @ P_zz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, nis

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

    def compute_pos_nis(
        self,
        p_N: float,
        p_E: float,
        accuracy_m: float,
        r_scale: float = 1.0
    ) -> float:
        """
        Computes 2D position Normalized Innovation Squared (NIS) without state update.
        """
        sigma_pos = max(accuracy_m, 1.0)
        R_diag = (sigma_pos**2) * max(r_scale, 1.0)

        S00 = self.P[0, 0] + R_diag
        S01 = self.P[0, 1]
        S10 = self.P[1, 0]
        S11 = self.P[1, 1] + R_diag

        det = S00 * S11 - S01 * S10
        if det < 1e-12:
            return 999999.0
        inv_det = 1.0 / det

        S_inv00 = S11 * inv_det
        S_inv01 = -S01 * inv_det
        S_inv10 = -S10 * inv_det
        S_inv11 = S00 * inv_det

        y0 = p_N - self.x[0]
        y1 = p_E - self.x[1]

        return float(y0 * (S_inv00 * y0 + S_inv01 * y1) + y1 * (S_inv10 * y0 + S_inv11 * y1))

    def update_gnss_pos(
        self,
        p_N: float,
        p_E: float,
        accuracy_m: float,
        r_scale: float = 1.0
    ) -> Tuple[bool, float]:
        """
        2D GNSS Position Measurement Update with Innovation Gating (NIS).
        Gate: NIS <= 9.21 (Chi-square 2-DOF 99%)
        """
        sigma_pos = max(accuracy_m, 1.0)
        R_diag = (sigma_pos**2) * max(r_scale, 1.0)
        R = np.diag([R_diag, R_diag])

        def h_pos(s):
            return np.array([s[0], s[1]])

        z = np.array([p_N, p_E])
        accepted, nis = self.update_measurement(z, h_pos, R, nis_gate=UKF_NIS_GATE_2D)
        return accepted, nis

    def update_gnss_speed(
        self,
        v_gnss: float,
        sigma_v: float = UKF_DEFAULT_SPEED_SIGMA,
        r_scale: float = 1.0
    ) -> Tuple[bool, float]:
        """
        GNSS Ground Speed Update with Innovation Gating (NIS).
        Gate: NIS <= 6.635 (Chi-square 1-DOF 99%)
        """
        R_val = (sigma_v**2) * max(r_scale, 1.0)
        R = np.array([[R_val]])

        def h_spd(s):
            return np.array([s[2]])

        z = np.array([v_gnss])
        accepted, nis = self.update_measurement(z, h_spd, R, nis_gate=UKF_NIS_GATE_1D)
        return accepted, nis

    def update_gnss_course(
        self,
        psi_gnss_rad: float,
        v_gnss: float,
        sigma_psi_rad: float = np.radians(UKF_DEFAULT_HEADING_SIGMA_DEG),
        r_scale: float = 1.0
    ) -> Tuple[bool, float]:
        """
        GNSS Course / Heading Update with Speed Gating & Innovation Gating.
        Gate: v_gnss > 2.5 m/s and NIS <= 6.635 (Chi-square 1-DOF 99%)
        """
        if v_gnss <= UKF_MIN_HEADING_SPEED:
            return False, 0.0

        R_val = (sigma_psi_rad**2) * max(r_scale, 1.0)
        R = np.array([[R_val]])

        def h_hdg(s):
            return np.array([s[3]])

        z = np.array([psi_gnss_rad])
        accepted, nis = self.update_measurement(z, h_hdg, R, is_angle_measurement=True, nis_gate=UKF_NIS_GATE_1D)
        return accepted, nis

    def update_gnss_sample(
        self,
        sample: GnssSample,
        health_mgr: Any
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Canonical GNSS sample update integrated with GNSS Health Manager.
        """
        results = {"pos": False, "speed": False, "course": False}

        if not health_mgr.validate_sample(sample):
            health_mgr.on_measurement_rejected()
            return False, results

        d_lat = np.radians(sample.latitude_deg - self.ref_lat)
        d_lon = np.radians(sample.longitude_deg - self.ref_lon)
        ref_lat_rad = np.radians(self.ref_lat)
        p_N = UKF_EARTH_RADIUS * d_lat
        p_E = UKF_EARTH_RADIUS * d_lon * np.cos(ref_lat_rad)

        if int(health_mgr.state) == 0:  # PURE_DR
            health_mgr.start_quarantine()

        candidate_nis = self.compute_pos_nis(p_N, p_E, sample.horizontal_accuracy_m)
        if candidate_nis > health_mgr.config.nis_pos_2d_threshold:
            health_mgr.on_measurement_rejected()
            return False, results

        can_update, r_scale = health_mgr.process_candidate(sample)
        if not can_update:
            return False, results

        p_ok, _ = self.update_gnss_pos(p_N, p_E, sample.horizontal_accuracy_m, r_scale)
        results["pos"] = p_ok
        if not p_ok:
            health_mgr.on_measurement_rejected()
            return False, results

        if sample.validity & GnssValidity.SPEED_VALID:
            s_ok, _ = self.update_gnss_speed(sample.speed_mps, UKF_DEFAULT_SPEED_SIGMA, r_scale)
            results["speed"] = s_ok

        if sample.validity & GnssValidity.BEARING_VALID:
            bearing_rad = np.radians(sample.bearing_deg)
            c_ok, _ = self.update_gnss_course(bearing_rad, sample.speed_mps, np.radians(UKF_DEFAULT_HEADING_SIGMA_DEG), r_scale)
            results["course"] = c_ok

        return True, results

    def update_gnss(
        self,
        p_N_gnss: float,
        p_E_gnss: float,
        v_gnss: float,
        psi_gnss: Optional[float] = None,
        pos_accuracy_m: float = 3.0,
        r_scale: float = 1.0
    ) -> bool:
        """
        GNSS Position and Velocity Update with NIS gating.
        """
        p_ok, _ = self.update_gnss_pos(p_N_gnss, p_E_gnss, pos_accuracy_m, r_scale)
        if not p_ok:
            return False

        self.update_gnss_speed(v_gnss, UKF_DEFAULT_SPEED_SIGMA, r_scale)
        if psi_gnss is not None:
            self.update_gnss_course(psi_gnss, v_gnss, np.radians(UKF_DEFAULT_HEADING_SIGMA_DEG), r_scale)
        return True

    def update_map_match(
        self,
        p_N_match: float,
        p_E_match: float,
        psi_road: float,
        confidence: float,
        is_heading_valid: bool = False
    ) -> Tuple[bool, float, Optional[float]]:
        """
        Probabilistic Map Measurement Update (Pillar 5).
        Fuses 1D cross-track road position constraint and optional road heading into the 7-state UKF.
        Features:
        - 1D cross-track normal constraint: preserves along-track variance and VelocityNet speed
        - Confidence-scaled measurement covariance: R_ct = (sigma_base / confidence)^2
        - 1-DOF Chi-Square NIS innovation gating (threshold 6.635)
        - Optional road-aligned heading constraint when speed > 2.5 m/s, confidence > 0.7, |d_hdg| < 15 deg
        """
        if not self.is_initialized:
            return False, 0.0, None

        if confidence < 0.25:
            return False, 0.0, None

        # Unit normal vector perpendicular to road: n = [-sin(psi_road), cos(psi_road)]
        n_N = -math.sin(psi_road)
        n_E =  math.cos(psi_road)

        # Cross-track residual along normal vector
        y_ct = float(n_N * (p_N_match - self.x[0]) + n_E * (p_E_match - self.x[1]))

        sigma_base = 2.5
        sigma_map = sigma_base / max(confidence, 0.25)
        R_ct = float(sigma_map**2)

        # 1-DOF innovation variance
        S = float(n_N**2 * self.P[0, 0] + 2.0 * n_N * n_E * self.P[0, 1] + n_E**2 * self.P[1, 1] + R_ct)
        if S < 1e-12:
            return False, 0.0, None

        nis = float((y_ct**2) / S)

        # 1-DOF Chi-Square NIS gating (threshold 6.635 at p=0.01)
        if nis > UKF_NIS_GATE_1D:
            return False, nis, None

        # Kalman gain strictly for position states
        K_N = float((self.P[0, 0] * n_N + self.P[0, 1] * n_E) / S)
        K_E = float((self.P[1, 0] * n_N + self.P[1, 1] * n_E) / S)

        self.x[0] += K_N * y_ct
        self.x[1] += K_E * y_ct

        # Covariance update strictly along road normal
        K_vec = np.zeros(self.dim_x)
        K_vec[0] = K_N
        K_vec[1] = K_E
        self.P -= S * np.outer(K_vec, K_vec)
        self.P = 0.5 * (self.P + self.P.T)
        for i in range(self.dim_x):
            self.P[i, i] = max(self.P[i, i], 1e-8)

        hdg_nis = None
        if is_heading_valid and self.x[2] > UKF_MIN_HEADING_SPEED and confidence > 0.7:
            d_hdg = abs((self.x[3] - psi_road + np.pi) % (2.0 * np.pi) - np.pi)
            if d_hdg < np.radians(15.0):
                hdg_sigma = np.radians(UKF_DEFAULT_HEADING_SIGMA_DEG) / max(confidence, 0.5)
                R_hdg = np.array([[hdg_sigma**2]])

                def h_hdg(s):
                    return np.array([s[3]])

                z_hdg = np.array([psi_road])
                hdg_accepted, hdg_nis = self.update_measurement(
                    z_hdg, h_hdg, R_hdg, is_angle_measurement=True, nis_gate=UKF_NIS_GATE_1D
                )

        return True, nis, hdg_nis

