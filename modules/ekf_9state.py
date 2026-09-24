"""
Phase 12: 9-State Velocity-State Extended Kalman Filter (EKF)

State Vector:
  x = [pE, pN, pU, vE, vN, vU, psi, ba, bg]^T in ENU coordinates.
    pE, pN, pU : World-frame position (meters, East-North-Up)
    vE, vN, vU : World-frame velocity (m/s, East-North-Up)
    psi        : Vehicle heading angle (radians, clockwise from North)
    ba         : Body/longitudinal accelerometer bias (m/s^2)
    bg         : Yaw gyroscope bias (rad/s)

This is an experimental filter strictly isolated from the production 7-state UKF.
Tests the architectural hypothesis:
  Does explicit world-frame velocity [vE, vN, vU] combined with GNSS velocity initialization,
  Phase 11 AI forward-velocity updates, and full Kalman NHC reduce longitudinal drift?
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class EKF9StateNavigationFilter:
    def __init__(
        self,
        dt: float = 0.1,
        heading_convention: str = "clockwise_from_north"
    ):
        self.dt = dt
        self.dim_x = 9
        self.heading_convention = heading_convention  # "clockwise_from_north" or "cartesian_from_east"

        # State estimate x: [pE, pN, pU, vE, vN, vU, psi, ba, bg]
        self.x = np.zeros(self.dim_x, dtype=np.float64)

        # State covariance P
        self.P = np.diag([
            25.0, 25.0, 25.0,            # position (5m std)
            1.0, 1.0, 1.0,               # velocity (1 m/s std)
            (np.radians(5.0))**2,        # heading (5 deg std)
            (0.05)**2,                   # accel bias (0.05 m/s^2 std)
            (1e-3)**2                    # gyro bias (0.001 rad/s std)
        ]).astype(np.float64)

        # Process noise covariance Q (continuous-time spectral density scaled by dt)
        self.Q = np.diag([
            0.01, 0.01, 0.01,            # position noise
            0.09, 0.09, 0.04,            # velocity process noise (accel disturbance)
            (np.radians(0.2))**2,        # heading gyro noise
            (1e-4)**2,                   # accel bias random walk
            (1e-6)**2                    # gyro bias random walk
        ]).astype(np.float64)

        # Diagnostics & counters
        self.ai_updates_count = 0
        self.ai_rejected_count = 0
        self.nhc_updates_count = 0
        self.nhc_rejected_count = 0
        self.zupt_updates_count = 0

        self.last_ai_innov = 0.0
        self.last_ai_nis = 0.0
        self.last_nhc_innov = np.zeros(2)
        self.last_nhc_nis = 0.0

        self.ref_lat: float = 0.0
        self.ref_lon: float = 0.0
        self.is_initialized: bool = False

    def initialize(
        self,
        init_lat: float,
        init_lon: float,
        init_speed_ms: float,
        init_heading_deg: float,
        init_vE: Optional[float] = None,
        init_vN: Optional[float] = None,
        init_vU: Optional[float] = 0.0,
        init_gyro_bias: float = 0.0,
        init_accel_bias: float = 0.0
    ) -> None:
        """Initialize filter states at local reference origin."""
        self.ref_lat = init_lat
        self.ref_lon = init_lon
        self.is_initialized = True

        self.x = np.zeros(self.dim_x, dtype=np.float64)
        # Position [0, 1, 2] = [0, 0, 0] at local origin

        hdg_rad = math.radians(init_heading_deg) % (2.0 * math.pi)

        if self.heading_convention == "clockwise_from_north":
            self.x[6] = hdg_rad
            if init_vE is not None and init_vN is not None:
                self.x[3] = init_vE
                self.x[4] = init_vN
            else:
                self.x[3] = init_speed_ms * math.sin(hdg_rad)
                self.x[4] = init_speed_ms * math.cos(hdg_rad)
        else:
            # Cartesian angle from East: theta = pi/2 - psi_nav
            theta = (math.pi / 2.0 - hdg_rad) % (2.0 * math.pi)
            self.x[6] = theta
            if init_vE is not None and init_vN is not None:
                self.x[3] = init_vE
                self.x[4] = init_vN
            else:
                self.x[3] = init_speed_ms * math.cos(theta)
                self.x[4] = init_speed_ms * math.sin(theta)

        self.x[5] = init_vU if init_vU is not None else 0.0
        self.x[7] = init_accel_bias
        self.x[8] = init_gyro_bias

        # Initial covariance
        self.P = np.diag([
            25.0, 25.0, 25.0,
            1.0, 1.0, 1.0,
            (np.radians(5.0))**2,
            (0.05)**2,
            (1e-3)**2
        ]).astype(np.float64)

    def predict(self, acc_fwd: float, gyro_yaw: float, dt: Optional[float] = None) -> None:
        """
        IMU Propagation:
          omega_z = gyro_yaw - bg
          a_forward = acc_fwd - ba
          Rotate forward acceleration into ENU using psi
          Propagate states and covariance
        """
        dt = dt if dt is not None else self.dt

        pE, pN, pU, vE, vN, vU, psi, ba, bg = self.x

        # De-biased inputs
        omega_z = gyro_yaw - bg
        a_fwd = acc_fwd - ba

        if self.heading_convention == "clockwise_from_north":
            # psi: clockwise from North (psi_nav)
            # Turning rate: gyro_yaw is positive clockwise
            psi_mid = psi + 0.5 * omega_z * dt
            psi_next = (psi + omega_z * dt) % (2.0 * math.pi)

            # Acceleration in ENU:
            # East = sin(psi), North = cos(psi)
            aE = a_fwd * math.sin(psi_mid)
            aN = a_fwd * math.cos(psi_mid)
            aU = 0.0

            # State transition Jacobian F
            F = np.eye(self.dim_x, dtype=np.float64)
            F[0, 3] = dt  # pE wrt vE
            F[1, 4] = dt  # pN wrt vN
            F[2, 5] = dt  # pU wrt vU

            # vE wrt psi and ba
            F[3, 6] = a_fwd * math.cos(psi_mid) * dt
            F[3, 7] = -math.sin(psi_mid) * dt

            # vN wrt psi and ba
            F[4, 6] = -a_fwd * math.sin(psi_mid) * dt
            F[4, 7] = -math.cos(psi_mid) * dt

            # psi wrt bg
            F[6, 8] = -dt

        else:
            # Cartesian angle from East: theta
            # Turning clockwise decreases theta
            theta = psi
            theta_mid = theta - 0.5 * omega_z * dt
            psi_next = (theta - omega_z * dt) % (2.0 * math.pi)

            # Acceleration in ENU:
            # East = cos(theta), North = sin(theta)
            aE = a_fwd * math.cos(theta_mid)
            aN = a_fwd * math.sin(theta_mid)
            aU = 0.0

            F = np.eye(self.dim_x, dtype=np.float64)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt

            F[3, 6] = -a_fwd * math.sin(theta_mid) * dt
            F[3, 7] = -math.cos(theta_mid) * dt

            F[4, 6] = a_fwd * math.cos(theta_mid) * dt
            F[4, 7] = -math.sin(theta_mid) * dt

            F[6, 8] = dt  # derivative wrt bg

        # Midpoint velocity and position integration
        vE_next = vE + aE * dt
        vN_next = vN + aN * dt
        vU_next = vU + aU * dt

        vE_mid = 0.5 * (vE + vE_next)
        vN_mid = 0.5 * (vN + vN_next)
        vU_mid = 0.5 * (vU + vU_next)

        pE_next = pE + vE_mid * dt
        pN_next = pN + vN_mid * dt
        pU_next = pU + vU_mid * dt

        self.x = np.array([
            pE_next, pN_next, pU_next,
            vE_next, vN_next, vU_next,
            psi_next,
            ba, bg
        ], dtype=np.float64)

        # Covariance propagation
        P_next = F @ self.P @ F.T + self.Q * dt
        self.P = 0.5 * (P_next + P_next.T) + np.eye(self.dim_x) * 1e-9

    def update_gnss_position(
        self,
        pE_gnss: float,
        pN_gnss: float,
        pU_gnss: float = 0.0,
        sigma_pos: float = 3.0,
        nis_gate: float = 11.34  # 3-DOF 99%
    ) -> Tuple[bool, float]:
        """GNSS Position Measurement Update: z = [pE, pN, pU]."""
        z = np.array([pE_gnss, pN_gnss, pU_gnss], dtype=np.float64)
        h_x = self.x[0:3]

        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.eye(3, dtype=np.float64) * (sigma_pos ** 2)
        return self._kalman_update(z, h_x, H, R, nis_gate=nis_gate)

    def update_gnss_velocity(
        self,
        vE_gnss: float,
        vN_gnss: float,
        vU_gnss: float = 0.0,
        sigma_vel: float = 0.3,
        nis_gate: float = 11.34  # 3-DOF 99%
    ) -> Tuple[bool, float]:
        """GNSS Direct Velocity Measurement Update: z = [vE, vN, vU]."""
        z = np.array([vE_gnss, vN_gnss, vU_gnss], dtype=np.float64)
        h_x = self.x[3:6]

        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        R = np.eye(3, dtype=np.float64) * (sigma_vel ** 2)
        return self._kalman_update(z, h_x, H, R, nis_gate=nis_gate)

    def update_gnss_heading(
        self,
        psi_gnss_deg: float,
        sigma_hdg_deg: float = 3.0,
        nis_gate: float = 6.635  # 1-DOF 99%
    ) -> Tuple[bool, float]:
        """GNSS Course/Heading Update: z = psi_gnss."""
        hdg_rad = math.radians(psi_gnss_deg) % (2.0 * math.pi)
        if self.heading_convention == "cartesian_from_east":
            z_val = (math.pi / 2.0 - hdg_rad) % (2.0 * math.pi)
        else:
            z_val = hdg_rad

        z = np.array([z_val], dtype=np.float64)
        h_x = np.array([self.x[6]], dtype=np.float64)

        # Innovation wrapping
        innov = (z[0] - h_x[0] + math.pi) % (2.0 * math.pi) - math.pi

        H = np.zeros((1, self.dim_x), dtype=np.float64)
        H[0, 6] = 1.0

        R = np.array([[math.radians(sigma_hdg_deg) ** 2]], dtype=np.float64)

        S = H @ self.P @ H.T + R
        S_inv = 1.0 / S[0, 0]
        nis = float(innov * S_inv * innov)

        if nis > nis_gate:
            return False, nis

        K = (self.P @ H.T) * S_inv  # (9, 1)
        dx = K[:, 0] * innov

        self.x = self.x + dx
        self.x[6] = self.x[6] % (2.0 * math.pi)

        # Joseph form covariance update
        I_KH = np.eye(self.dim_x) - K @ H
        P_up = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (P_up + P_up.T) + np.eye(self.dim_x) * 1e-9
        return True, nis

    def update_ai_forward_velocity(
        self,
        v_ai: float,
        sigma_ai: float,
        nis_gate: float = 6.635  # 1-DOF 99%
    ) -> Tuple[bool, float]:
        """
        Phase 11 AI Forward Velocity Update:
          z = v_ai
          h(x) = predicted forward velocity from (vE, vN, psi)
        """
        pE, pN, pU, vE, vN, vU, psi, ba, bg = self.x

        if self.heading_convention == "clockwise_from_north":
            # v_forward = vE*sin(psi) + vN*cos(psi)
            v_fwd = vE * math.sin(psi) + vN * math.cos(psi)
            H = np.zeros((1, self.dim_x), dtype=np.float64)
            H[0, 3] = math.sin(psi)
            H[0, 4] = math.cos(psi)
            H[0, 6] = vE * math.cos(psi) - vN * math.sin(psi)  # d(v_fwd)/d(psi)
        else:
            # v_forward = vE*cos(psi) + vN*sin(psi)
            v_fwd = vE * math.cos(psi) + vN * math.sin(psi)
            H = np.zeros((1, self.dim_x), dtype=np.float64)
            H[0, 3] = math.cos(psi)
            H[0, 4] = math.sin(psi)
            H[0, 6] = -vE * math.sin(psi) + vN * math.cos(psi)

        z = np.array([v_ai], dtype=np.float64)
        h_x = np.array([v_fwd], dtype=np.float64)
        R = np.array([[max(sigma_ai, 0.1) ** 2]], dtype=np.float64)

        accepted, nis = self._kalman_update(z, h_x, H, R, nis_gate=nis_gate)
        self.last_ai_innov = float(z[0] - h_x[0])
        self.last_ai_nis = float(nis)

        if accepted:
            self.ai_updates_count += 1
        else:
            self.ai_rejected_count += 1
        return accepted, nis

    def update_nhc(
        self,
        sigma_lat: float = 0.5,
        sigma_vert: float = 0.3,
        nis_gate: float = 9.21  # 2-DOF 99%
    ) -> Tuple[bool, float]:
        """
        Full Non-Holonomic Constraint (NHC) as a real Kalman Measurement:
          z = [0, 0] (lateral velocity = 0, vertical velocity = 0)
          h(x) = [v_lat, v_vert]
        """
        pE, pN, pU, vE, vN, vU, psi, ba, bg = self.x

        if self.heading_convention == "clockwise_from_north":
            # Lateral velocity (leftward): v_lat = -vE*cos(psi) + vN*sin(psi)
            v_lat = -vE * math.cos(psi) + vN * math.sin(psi)
            v_vert = vU

            H = np.zeros((2, self.dim_x), dtype=np.float64)
            H[0, 3] = -math.cos(psi)
            H[0, 4] = math.sin(psi)
            H[0, 6] = vE * math.sin(psi) + vN * math.cos(psi)  # d(v_lat)/d(psi) = v_forward
            H[1, 5] = 1.0
        else:
            # Literal prompt formula: v_lat = -sin(psi)*vE + cos(psi)*vN
            v_lat = -math.sin(psi) * vE + math.cos(psi) * vN
            v_vert = vU

            H = np.zeros((2, self.dim_x), dtype=np.float64)
            H[0, 3] = -math.sin(psi)
            H[0, 4] = math.cos(psi)
            H[0, 6] = -math.cos(psi) * vE - math.sin(psi) * vN
            H[1, 5] = 1.0

        z = np.array([0.0, 0.0], dtype=np.float64)
        h_x = np.array([v_lat, v_vert], dtype=np.float64)
        R = np.diag([max(sigma_lat, 0.05) ** 2, max(sigma_vert, 0.05) ** 2])

        accepted, nis = self._kalman_update(z, h_x, H, R, nis_gate=nis_gate)
        self.last_nhc_innov = z - h_x
        self.last_nhc_nis = float(nis)

        if accepted:
            self.nhc_updates_count += 1
        else:
            self.nhc_rejected_count += 1
        return accepted, nis

    def update_zupt(
        self,
        sigma_zupt: float = 0.05,
        nis_gate: float = 11.34  # 3-DOF 99%
    ) -> Tuple[bool, float]:
        """Stationary Zero-Velocity Update (ZUPT): z = [0, 0, 0], h(x) = [vE, vN, vU]."""
        z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        h_x = self.x[3:6]

        H = np.zeros((3, self.dim_x), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        R = np.eye(3, dtype=np.float64) * (sigma_zupt ** 2)
        accepted, nis = self._kalman_update(z, h_x, H, R, nis_gate=nis_gate)
        if accepted:
            self.zupt_updates_count += 1
        return accepted, nis

    def _kalman_update(
        self,
        z: np.ndarray,
        h_x: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        nis_gate: Optional[float] = None
    ) -> Tuple[bool, float]:
        """General linear Kalman measurement update with Joseph form covariance."""
        innov = z - h_x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        nis = float(innov.T @ S_inv @ innov)
        if nis_gate is not None and nis > nis_gate:
            return False, nis

        K = self.P @ H.T @ S_inv  # (dim_x, m)
        dx = K @ innov

        self.x = self.x + dx
        self.x[6] = self.x[6] % (2.0 * math.pi)

        # Joseph form covariance update: P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(self.dim_x) - K @ H
        P_up = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (P_up + P_up.T) + np.eye(self.dim_x) * 1e-9

        return True, nis

    def get_forward_speed(self) -> float:
        """Returns scalar forward speed from current state."""
        vE, vN = self.x[3], self.x[4]
        psi = self.x[6]
        if self.heading_convention == "clockwise_from_north":
            return float(vE * math.sin(psi) + vN * math.cos(psi))
        else:
            return float(vE * math.cos(psi) + vN * math.sin(psi))

    def get_lateral_speed(self) -> float:
        """Returns scalar body lateral speed from current state."""
        vE, vN = self.x[3], self.x[4]
        psi = self.x[6]
        if self.heading_convention == "clockwise_from_north":
            return float(-vE * math.cos(psi) + vN * math.sin(psi))
        else:
            return float(-vE * math.sin(psi) + vN * math.cos(psi))

    def get_heading_deg(self) -> float:
        """Returns heading in degrees clockwise from North [0, 360)."""
        psi = self.x[6]
        if self.heading_convention == "clockwise_from_north":
            return float(math.degrees(psi) % 360.0)
        else:
            # theta = pi/2 - psi_nav => psi_nav = pi/2 - theta
            return float(math.degrees(math.pi / 2.0 - psi) % 360.0)
