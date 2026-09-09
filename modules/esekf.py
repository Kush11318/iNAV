"""
iNAV 15-State Error-State Kalman Filter (ES-EKF) (Pillar 4)
Minimal Tangent-Space Error-State Formulation:
- Nominal state (16D): Position (3), Velocity (3), Unit Quaternion (4), Accel Bias (3), Gyro Bias (3)
- Minimal Error state (15D): δx = [δp, δv, δθ, δa_b, δω_b]^T in tangent space
- True error-state reset (δx -> 0) after injection
- Online NIS Chi-Square Outlier Gating (rejects multipath / jump fixes)
- Joseph-Form Covariance Propagation: P <- (I - KH) P (I - KH)^T + K R K^T
"""

import logging
from typing import Optional, Tuple, Dict
import numpy as np

logger = logging.getLogger("iNAV.esekf")

# Chi-square 99% confidence critical values for degrees of freedom 1..6
CHI2_99_THRESHOLDS = {
    1: 6.635,
    2: 9.210,
    3: 11.345,
    4: 13.277,
    5: 15.086,
    6: 16.812
}


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix [v]_x of a 3D vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R_b_to_n.
    """
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qw*qz), 2.0*(qx*qz + qw*qy)],
        [2.0*(qx*qy + qw*qz), 1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qw*qx)],
        [2.0*(qx*qz - qw*qy), 2.0*(qy*qz + qw*qx), 1.0 - 2.0*(qx**2 + qy**2)]
    ], dtype=np.float64)


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions q and r [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw
    ], dtype=np.float64)


class ESEKFNavigationFilter:
    """
    15-State Error-State Extended Kalman Filter for 3D/2D Vehicular Navigation.
    Nominal States:
      p   : 3D position [North, East, Down] (m)
      v   : 3D velocity [v_N, v_E, v_D] (m/s)
      q   : 4D unit quaternion [qw, qx, qy, qz] (body to NED)
      a_b : 3D accelerometer bias (m/s^2)
      w_b : 3D gyroscope bias (rad/s)
    
    Error States:
      δx = [δp (0:3), δv (3:6), δθ (6:9), δa_b (9:12), δw_b (12:15)]^T
    """

    def __init__(
        self,
        dt: float = 0.1,
        gravity: float = 9.80665,
        sigma_a: float = 0.06,           # Accel measurement noise std (m/s^2) for smartphone MEMS
        sigma_g: float = 0.004,          # Gyro measurement noise std (rad/s) (~0.23 deg/s)
        sigma_ba: float = 1.0e-4,        # Accel bias random walk (m/s^3 / sqrt(Hz))
        sigma_bg: float = 1.0e-5,        # Gyro bias random walk (rad/s^2 / sqrt(Hz))
        drive_inflation: float = 2.5     # In-situ operational vehicle vibration inflation
    ):
        self.dt = dt
        self.g_ned = np.array([0.0, 0.0, gravity], dtype=np.float64)
        self.dim_error = 15

        self.sigma_a = sigma_a
        self.sigma_g = sigma_g
        self.sigma_ba = sigma_ba
        self.sigma_bg = sigma_bg
        self.drive_inflation = drive_inflation

        # Nominal states
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.a_b = np.zeros(3, dtype=np.float64)
        self.w_b = np.zeros(3, dtype=np.float64)
        self.freeze_gyro_bias: bool = False

        # 15x15 Error state covariance P
        self.P = np.eye(self.dim_error, dtype=np.float64)
        self._init_covariance()

        # 15x15 Discretized Process noise covariance Q
        self.Q = self.compute_discrete_Q(self.dt)

    def compute_discrete_Q(self, dt: float) -> np.ndarray:
        """
        Discretized process noise covariance Q following exact physical scaling laws:
        Q_d = diag(0_3, σ_a^2 * Δt^2 * I_3, σ_g^2 * Δt^2 * I_3, σ_ab^2 * Δt * I_3, σ_wb^2 * Δt * I_3)

        * Additive Measurement Noise (Δt^2 Scaling):
          Per-sample accelerometer noise standard deviation σ_a perturbs velocity over step Δt by
          σ_a * Δt, injecting a variance of σ_a^2 * Δt^2. Gyro orientation error follows σ_g^2 * Δt^2.
        * Bias Random Walk (Δt Scaling):
          Sensor bias random walk processes accumulate variance linearly with time, scaling strictly
          as σ_ab^2 * Δt and σ_wb^2 * Δt.
        * In-situ Operational Noise Inflation:
          Accounts for high-frequency engine harmonics and road surface irregularities.
        """
        Q_d = np.zeros((self.dim_error, self.dim_error), dtype=np.float64)
        s_a_eff = self.sigma_a * self.drive_inflation
        s_g_eff = self.sigma_g * self.drive_inflation

        var_v = (s_a_eff ** 2) * (dt ** 2)
        var_theta = (s_g_eff ** 2) * (dt ** 2)
        var_ba = (self.sigma_ba ** 2) * dt
        var_bg = (self.sigma_bg ** 2) * dt

        Q_d[3:6, 3:6] = np.eye(3) * var_v
        Q_d[6:9, 6:9] = np.eye(3) * var_theta
        Q_d[9:12, 9:12] = np.eye(3) * var_ba
        Q_d[12:15, 12:15] = np.eye(3) * var_bg
        return Q_d

    def _init_covariance(self):
        self.P[0:3, 0:3] = np.eye(3) * 25.0       # 5m position std
        self.P[3:6, 3:6] = np.eye(3) * 1.0        # 1 m/s velocity std
        self.P[6:9, 6:9] = np.eye(3) * (np.radians(5.0))**2  # 5 deg attitude std
        self.P[9:12, 9:12] = np.eye(3) * (0.05)**2   # accel bias 0.05 m/s^2
        self.P[12:15, 12:15] = np.eye(3) * (1e-3)**2 # gyro bias 0.001 rad/s

    def _init_process_noise(self):
        self.Q = self.compute_discrete_Q(self.dt)

    def initialize(
        self,
        init_lat_ned: float,
        init_lon_ned: float,
        init_speed_ms: float,
        init_heading_rad: float
    ):
        """Initialize filter states at reference origin."""
        self.p = np.array([init_lat_ned, init_lon_ned, 0.0], dtype=np.float64)
        self.v = np.array([
            init_speed_ms * np.cos(init_heading_rad),
            init_speed_ms * np.sin(init_heading_rad),
            0.0
        ], dtype=np.float64)

        # Initial attitude from yaw
        cy = np.cos(init_heading_rad * 0.5)
        sy = np.sin(init_heading_rad * 0.5)
        self.q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
        self.a_b = np.zeros(3, dtype=np.float64)
        self.w_b = np.zeros(3, dtype=np.float64)
        self._init_covariance()

    def predict(self, acc_meas: np.ndarray, gyro_meas: np.ndarray, dt: Optional[float] = None):
        """
        High-rate Strapdown Mechanization and Error Covariance Propagation.
        acc_meas: (3,) raw accelerometer in body frame (m/s^2)
        gyro_meas: (3,) raw gyroscope in body frame (rad/s)
        """
        dt = dt if dt is not None else self.dt

        # Unbiased body-frame specific force and angular rate
        f_b = acc_meas - self.a_b
        w_b_unbiased = gyro_meas - self.w_b

        # Current rotation matrix R_b_to_n
        R = quat_to_rot_matrix(self.q)

        # 1. Propagate nominal velocity and position
        f_n = R @ f_b + self.g_ned
        self.p += self.v * dt + 0.5 * f_n * (dt ** 2)
        self.v += f_n * dt

        # 2. Propagate nominal quaternion attitude (zeroth-order quaternion integrator)
        angle = np.linalg.norm(w_b_unbiased) * dt
        if angle > 1e-8:
            axis = w_b_unbiased / np.linalg.norm(w_b_unbiased)
            dq = np.array([np.cos(angle * 0.5), *(axis * np.sin(angle * 0.5))], dtype=np.float64)
        else:
            dq = np.array([1.0, 0.5 * w_b_unbiased[0] * dt, 0.5 * w_b_unbiased[1] * dt, 0.5 * w_b_unbiased[2] * dt])
            dq /= np.linalg.norm(dq)

        self.q = quat_multiply(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # 3. Form Error-State Transition Matrix F_x (15x15)
        # δp_dot = δv
        # δv_dot = -R * [f_b]_x * δθ - R * δa_b
        # δθ_dot = -[w_b_unbiased]_x * δθ - δw_b
        F_x = np.eye(self.dim_error, dtype=np.float64)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R @ skew_symmetric(f_b) * dt
        F_x[3:6, 9:12] = -R * dt
        F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(w_b_unbiased) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        # Propagate error covariance with exact continuous-to-discrete scaled Q_d
        Q_d = self.compute_discrete_Q(dt)
        self.P = F_x @ self.P @ F_x.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)

    def update_error_state(
        self,
        y: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        chi2_gate: bool = True
    ) -> bool:
        """
        Generic Measurement Update with Joseph-Form Covariance and NIS Chi-Square Gating.
        y: Innovation vector (dim_z,)
        H: Measurement Jacobian matrix (dim_z, 15)
        R: Measurement noise covariance (dim_z, dim_z)
        Returns:
          True if update accepted, False if rejected by NIS outlier gate.
        """
        dim_z = len(y)
        S = H @ self.P @ H.T + R  # Innovation covariance (dim_z, dim_z)

        # Online NIS Chi-Square Outlier Gating
        try:
            S_inv = np.linalg.inv(S)
            nis = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance matrix in ES-EKF; skipping update.")
            return False

        threshold = CHI2_99_THRESHOLDS.get(dim_z, 3.0 * dim_z)
        if chi2_gate and (nis > threshold):
            logger.debug(f"NIS gate triggered: NIS={nis:.2f} > threshold={threshold:.2f} (dim={dim_z}); update rejected.")
            return False

        # Kalman Gain
        K = self.P @ H.T @ S_inv  # (15, dim_z)

        # Error State Correction
        delta_x = K @ y  # (15,)

        # Nominal State Injection
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]

        # Quaternion error injection: q <- q * [1, 0.5 * δθ]
        d_theta = delta_x[6:9]
        dq = np.array([1.0, 0.5 * d_theta[0], 0.5 * d_theta[1], 0.5 * d_theta[2]], dtype=np.float64)
        dq /= np.linalg.norm(dq)
        self.q = quat_multiply(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # Bias corrections
        self.a_b += delta_x[9:12]
        if not self.freeze_gyro_bias:
            self.w_b += delta_x[12:15]

        # Joseph-Form Covariance Update:
        # P <- (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(self.dim_error) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Enforce positive semi-definite symmetry
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def predict_vehicular(self, acc_fwd: float, gyro_yaw: float, dt: Optional[float] = None):
        """
        Vehicular 2D Road-Constrained Mechanization with Non-Holonomic Constraints (NHC).
        Eliminates vertical gravity double-integration divergence while propagating
        error covariance across [delta_p, delta_v, delta_theta, delta_ab, delta_wb].
        """
        dt = dt if dt is not None else self.dt

        # Unbiased inputs
        fwd_corr = acc_fwd - self.a_b[0]
        yaw_corr = gyro_yaw - self.w_b[2]

        # Current heading from quaternion
        R = quat_to_rot_matrix(self.q)
        psi = np.arctan2(R[1, 0], R[0, 0])

        # Propagate heading
        mid_psi = psi + 0.5 * yaw_corr * dt
        new_psi = (psi + yaw_corr * dt) % (2.0 * np.pi)

        # Update quaternion with pure yaw rotation
        cy = np.cos(new_psi * 0.5)
        sy = np.sin(new_psi * 0.5)
        self.q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)

        # Propagate forward speed (clamped non-negative)
        v_curr = float(np.linalg.norm(self.v[:2]))
        v_next = max(v_curr + fwd_corr * dt, 0.0)
        v_mid = 0.5 * (v_curr + v_next)

        # Update nominal velocity and position in horizontal plane
        self.v[0] = v_next * np.cos(new_psi)
        self.v[1] = v_next * np.sin(new_psi)
        self.v[2] = 0.0  # NHC: zero vertical velocity

        self.p[0] += v_mid * np.cos(mid_psi) * dt
        self.p[1] += v_mid * np.sin(mid_psi) * dt
        self.p[2] = 0.0

        # Propagate 15x15 error covariance
        R_mat = quat_to_rot_matrix(self.q)
        f_b = np.array([fwd_corr, 0.0, 0.0])
        w_b = np.array([0.0, 0.0, yaw_corr])

        F_x = np.eye(self.dim_error, dtype=np.float64)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R_mat @ skew_symmetric(f_b) * dt
        F_x[3:6, 9:12] = -R_mat * dt
        F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(w_b) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        # Propagate 15x15 error covariance with exact continuous-to-discrete scaled Q_d
        Q_d = self.compute_discrete_Q(dt)
        self.P = F_x @ self.P @ F_x.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)

    def update_gnss_pos(self, p_N: float, p_E: float, accuracy_m: float = 3.0) -> bool:
        """GNSS 2D Horizontal Position Update with NIS Gating."""
        z = np.array([p_N, p_E])
        h_x = self.p[0:2]
        y = z - h_x

        H = np.zeros((2, self.dim_error))
        H[0, 0] = 1.0  # d p_N
        H[1, 1] = 1.0  # d p_E

        var = max(accuracy_m, 1.0) ** 2
        R = np.diag([var, var])
        return self.update_error_state(y, H, R, chi2_gate=True)

    def update_forward_speed(self, speed_ms: float, variance: float = 0.1) -> bool:
        """
        VelocityNet / SpectraNet forward speed update.
        Projects vehicle forward axis [1, 0, 0] in body frame to navigation frame.
        """
        R = quat_to_rot_matrix(self.q)
        v_current = float(np.linalg.norm(self.v[:2]))
        y = np.array([speed_ms - v_current])

        H = np.zeros((1, self.dim_error))
        H[0, 3] = R[0, 0]
        H[0, 4] = R[1, 0]

        R_cov = np.array([[max(variance, 0.04)]])
        accepted = self.update_error_state(y, H, R_cov, chi2_gate=True)
        if accepted:
            # Re-align nominal velocity vector with updated speed magnitude
            new_v = max(float(np.linalg.norm(self.v[:2])), 0.0)
            psi = np.arctan2(R[1, 0], R[0, 0])
            self.v[0] = new_v * np.cos(psi)
            self.v[1] = new_v * np.sin(psi)
            self.v[2] = 0.0
        return accepted

    def update_zupt(self) -> bool:
        """Zero-Velocity Update (ZUPT) when vehicle is confirmed stationary."""
        y = -self.v[:2]
        H = np.zeros((2, self.dim_error))
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        R = np.eye(2) * 1e-4
        self.v[:] = 0.0
        return self.update_error_state(y, H, R, chi2_gate=False)

    def get_heading_rad(self) -> float:
        """Extract heading angle (yaw from North, clockwise in radians)."""
        R = quat_to_rot_matrix(self.q)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return float(yaw % (2.0 * np.pi))

    def get_speed_ms(self) -> float:
        """Current 3D velocity magnitude."""
        return float(np.linalg.norm(self.v))

    def update_road_constraint(
        self,
        road_pN: float,
        road_pE: float,
        road_heading_deg: float,
        speed_ms: float,
        sigma_lat: float = 1.0,
        sigma_long: float = 20.0
    ) -> bool:
        """
        Pillar 5 Closed-Loop Map-Aided EKF Measurement Update.
        Constructs an anisotropic measurement covariance matrix R_road rotated into the
        road link frame:
          R_road = R_z(psi_road) * diag(sigma_long^2, sigma_lat^2) * R_z(psi_road)^T
        - sigma_lat = 1.0m (tight: locks cross-track error to lane centerline)
        - sigma_long = 20.0m (loose: prevents clamping forward travel speed)
        - Dynamic heading observation update scaled by vehicle speed (sigma_psi = 2.0 deg at highway speed).
        Protected by online NIS Chi-Square gating to reject unmapped off-ramps or lane departures.
        """
        psi_rad = np.radians(road_heading_deg)
        cos_p = np.cos(psi_rad)
        sin_p = np.sin(psi_rad)

        # 2D Position Rotation Matrix (from road frame [along, cross] to NED [N, E])
        R_rot = np.array([
            [cos_p, -sin_p],
            [sin_p,  cos_p]
        ], dtype=np.float64)

        # Anisotropic covariance in road frame: loose along-track, tight cross-track
        var_long = sigma_long ** 2
        var_lat = sigma_lat ** 2
        R_road_local = np.diag([var_long, var_lat])
        R_pos_ned = R_rot @ R_road_local @ R_rot.T

        # 1. Anisotropic Position update
        z_pos = np.array([road_pN, road_pE], dtype=np.float64)
        y_pos = z_pos - self.p[0:2]
        H_pos = np.zeros((2, self.dim_error), dtype=np.float64)
        H_pos[0, 0] = 1.0
        H_pos[1, 1] = 1.0
        accepted_pos = self.update_error_state(y_pos, H_pos, R_pos_ned, chi2_gate=True)

        # 2. Dynamic Heading update scaled by vehicle speed
        if speed_ms > 8.0:  # > 30 km/h: tightly align vehicle heading to road link
            sigma_psi = np.radians(2.0)
        else:
            sigma_psi = np.radians(10.0)

        current_heading = self.get_heading_rad()
        diff_hdg = (psi_rad - current_heading + np.pi) % (2.0 * np.pi) - np.pi
        y_hdg = np.array([diff_hdg], dtype=np.float64)

        H_hdg = np.zeros((1, self.dim_error), dtype=np.float64)
        H_hdg[0, 8] = 1.0  # delta_theta_z (yaw error in tangent space)
        R_hdg = np.array([[sigma_psi ** 2]], dtype=np.float64)
        self.update_error_state(y_hdg, H_hdg, R_hdg, chi2_gate=True)

        return accepted_pos

    def update_nhc(self, var_lat: float = 0.1, var_vert: float = 0.05) -> bool:
        """
        Non-Holonomic Constraints (NHC) (§2):
        Ground vehicles cannot slide sideways or fly:
          v_y^b ≈ 0,  v_z^b ≈ 0
        Keeps velocity vector strictly aligned with vehicle chassis.
        """
        R = quat_to_rot_matrix(self.q)
        v_b = R.T @ self.v
        y = -v_b[1:3]

        H = np.zeros((2, self.dim_error), dtype=np.float64)
        H[0, 3:6] = R[:, 1]  # d(v_y^b)/d(v_ned)
        H[1, 3:6] = R[:, 2]  # d(v_z^b)/d(v_ned)

        R_cov = np.diag([var_lat, var_vert])
        return self.update_error_state(y, H, R_cov, chi2_gate=False)

    def update_road_curvature(
        self,
        gyro_meas_z: float,
        speed_ms: float,
        curvature_kappa: float,
        var: float = 1e-4
    ) -> bool:
        """
        Road Link Curvature (κ = 1/R) Observation Update (§3).
        Breaks the unobservability of b_ωz during long blackouts:
          δz_κ = ω_meas_z - (v_fwd * κ) = b_ωz + n_ω
        Directly measures and calibrates gyroscope yaw bias during curved highway driving.
        """
        w_expected = speed_ms * curvature_kappa
        innov = gyro_meas_z - w_expected - self.w_b[2]
        y = np.array([innov], dtype=np.float64)

        H = np.zeros((1, self.dim_error), dtype=np.float64)
        H[0, 14] = 1.0  # delta_wb_z (gyro bias yaw component)
        R_cov = np.array([[max(var, 1e-6)]], dtype=np.float64)

        # Temporarily enable bias update for this explicit curvature observation
        prev_freeze = self.freeze_gyro_bias
        self.freeze_gyro_bias = False
        accepted = self.update_error_state(y, H, R_cov, chi2_gate=True)
        self.freeze_gyro_bias = prev_freeze
        return accepted
