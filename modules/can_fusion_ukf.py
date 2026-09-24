"""
Experimental Module: CAN / Wheel-Speed & Differential-Yaw Fusion UKF (Phase 16A & 16B-2)

Strictly isolated experimental module.
Inherits from UKFNavigationFilter (modules/ukf.py) without modifying production code.
Adds Kalman measurement updates for:
  1. Forward vehicle CAN / wheel speed (Phase 16A)
  2. Differential rear-wheel yaw rate (Phase 16B-2)
"""

import math
from typing import Optional, Tuple, List
import numpy as np

from modules.ukf import UKFNavigationFilter


class CANFusionUKF(UKFNavigationFilter):
    """
    7-State Unscented Kalman Filter extended with Vehicle CAN Speed & Differential Yaw Fusion.
    State Vector:
      x = [p_N, p_E, v_fwd, psi, b_g, b_a, k]^T
        p_N   : North position (m)
        p_E   : East position (m)
        v_fwd : Forward vehicle speed (m/s)       <-- observed by CAN speed
        psi   : Vehicle heading angle (rad)        <-- coupled to b_g
        b_g   : Gyroscope yaw rate bias (rad/s)   <-- observed by wheel differential yaw
        b_a   : Accelerometer forward bias (m/s^2)
        k     : VelocityNet adaptive scale factor (unitless, frozen/independent of CAN)
    """

    def __init__(
        self,
        dt: float = 0.1,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        init_gyro_bias_unc: float = 0.02  # Initial gyro bias uncertainty (rad/s) for bias observability
    ):
        super().__init__(dt=dt, alpha=alpha, beta=beta, kappa=kappa)
        self.init_gyro_bias_unc = init_gyro_bias_unc
        self.last_valid_can_speed: Optional[float] = None
        self.total_can_updates = 0
        self.rejected_can_updates = 0

        # Yaw update diagnostic counters
        self.total_yaw_updates = 0
        self.accepted_yaw_updates = 0
        self.rejected_yaw_updates = 0
        self.rejection_reasons = {
            "nan_inf": 0,
            "bounds": 0,
            "slip_differential": 0,
            "nis_gate": 0
        }
        self.yaw_nis_history: List[float] = []

        self.total_map_updates = 0
        self.accepted_map_updates = 0
        self.rejected_map_updates = 0
        self.heading_gate_rejections = 0
        self.map_rejection_reasons = {
            "confidence": 0,
            "nis_gate": 0,
            "heading_gate": 0,
            "other": 0
        }
        self.map_nis_history: List[float] = []

        # Map heading diagnostic counters (Phase 17B)
        self.total_map_heading_updates = 0
        self.accepted_map_heading_updates = 0
        self.rejected_map_heading_updates = 0
        self.map_heading_nis_history: List[float] = []

    def initialize(
        self,
        init_lat: float,
        init_lon: float,
        init_speed_ms: float,
        init_heading_rad: float,
        init_gyro_bias: float = 0.0,
        init_accel_bias: float = 0.0,
        allow_bias_learning: bool = False
    ) -> None:
        """Initialize filter states at reference origin."""
        super().initialize(
            init_lat=init_lat,
            init_lon=init_lon,
            init_speed_ms=init_speed_ms,
            init_heading_rad=init_heading_rad,
            init_gyro_bias=init_gyro_bias,
            init_accel_bias=init_accel_bias
        )
        if allow_bias_learning:
            # Enable realistic gyro bias uncertainty so differential wheel yaw can calibrate b_g
            self.P[4, 4] = self.init_gyro_bias_unc**2
            self.Q[4, 4] = (1e-4)**2  # Gyro bias random walk diffusion

    def update_can_speed(
        self,
        v_can: float,
        r_var: float = 0.0325,
        nis_gate: float = 16.0,
        max_accel_ms2: float = 8.0
    ) -> Tuple[bool, float]:
        """
        Updates the forward velocity state x[2] using calibrated CAN / wheel speed.
        Measurement model: z = x[2] + nu, nu ~ N(0, r_var)
        """
        if np.isnan(v_can) or np.isinf(v_can):
            self.rejected_can_updates += 1
            return False, float("nan")

        if v_can < -0.1 or v_can > 70.0:
            self.rejected_can_updates += 1
            return False, 999.0

        v_meas = max(0.0, float(v_can))

        if self.last_valid_can_speed is not None:
            dv = abs(v_meas - self.last_valid_can_speed)
            acc = dv / max(self.dt, 1e-3)
            if acc > max_accel_ms2:
                self.rejected_can_updates += 1
                return False, 888.0

        def h_speed(s: np.ndarray) -> np.ndarray:
            return np.array([s[2]], dtype=np.float64)

        z = np.array([v_meas], dtype=np.float64)
        R = np.array([[max(r_var, 0.01)]], dtype=np.float64)

        applied, nis = self.update_measurement(
            z=z,
            h_func=h_speed,
            R=R,
            is_angle_measurement=False,
            nis_gate=nis_gate
        )

        if applied:
            self.last_valid_can_speed = v_meas
            self.total_can_updates += 1
        else:
            self.rejected_can_updates += 1

        return applied, nis

    def update_wheel_yaw(
        self,
        z_wheel_yaw: float,
        gyro_reading: float,
        r_yaw: float = 0.002213,
        nis_gate: float = 16.0,
        max_yaw_rate_rads: float = 1.5  # ~86 deg/s physical turn limit
    ) -> Tuple[bool, float]:
        """
        Updates the filter from rear-wheel differential yaw rate.

        Measurement model:
            h(x) = omega_gyro - x[4]  (since x[4] = b_g, predicted yaw rate is gyro - b_g)
            z = z_wheel_yaw = (v_RR - v_RL) / B_eff

        Innovation:
            y = z_wheel_yaw - (omega_gyro - b_g)
        """
        self.total_yaw_updates += 1

        # 1. Numerical validity
        if np.isnan(z_wheel_yaw) or np.isinf(z_wheel_yaw) or np.isnan(gyro_reading) or np.isinf(gyro_reading):
            self.rejected_yaw_updates += 1
            self.rejection_reasons["nan_inf"] += 1
            return False, float("nan")

        # 2. Physical turn rate bounds
        if abs(z_wheel_yaw) > max_yaw_rate_rads:
            self.rejected_yaw_updates += 1
            self.rejection_reasons["bounds"] += 1
            return False, 999.0

        # 3. Measurement function h(s) = gyro_reading - s[4]
        def h_yaw(s: np.ndarray) -> np.ndarray:
            return np.array([gyro_reading - s[4]], dtype=np.float64)

        z = np.array([z_wheel_yaw], dtype=np.float64)
        R = np.array([[max(r_yaw, 1e-5)]], dtype=np.float64)

        applied, nis = self.update_measurement(
            z=z,
            h_func=h_yaw,
            R=R,
            is_angle_measurement=False,
            nis_gate=nis_gate
        )

        if applied:
            self.accepted_yaw_updates += 1
            self.yaw_nis_history.append(float(nis))
        else:
            self.rejected_yaw_updates += 1
            self.rejection_reasons["nis_gate"] += 1

        return applied, nis

    def update_map_match(
        self,
        p_N_match: float,
        p_E_match: float,
        psi_road: float,
        confidence: float,
        is_heading_valid: bool = False,
        sigma_heading_rad: Optional[float] = None,
        heading_nis_gate: float = 6.635,
        max_d_hdg_deg: float = 15.0,
        theta_max_deg: Optional[float] = None
    ) -> Tuple[bool, float, Optional[float]]:
        """
        1D road-normal map matching update with optional road-heading constraint,
        algebraic Joseph-form covariance stabilization, and diagnostic accounting.
        """
        self.total_map_updates += 1
        if not self.is_initialized:
            self.rejected_map_updates += 1
            self.map_rejection_reasons["other"] += 1
            return False, 0.0, None

        if confidence < 0.25:
            self.rejected_map_updates += 1
            self.map_rejection_reasons["confidence"] += 1
            return False, 0.0, None

        # Phase 18B: Heading-consistency gate
        if theta_max_deg is not None:
            d_psi = abs((self.x[3] - psi_road + np.pi) % (2.0 * np.pi) - np.pi)
            if d_psi > np.radians(theta_max_deg):
                self.rejected_map_updates += 1
                self.map_rejection_reasons["heading_gate"] += 1
                self.heading_gate_rejections += 1
                return False, 0.0, None

        # 1. Unit normal vector perpendicular to road: n = [-sin(psi_road), cos(psi_road)]
        n_N = -math.sin(psi_road)
        n_E =  math.cos(psi_road)

        # Cross-track residual along normal vector
        y_ct = float(n_N * (p_N_match - self.x[0]) + n_E * (p_E_match - self.x[1]))

        sigma_base = 2.5
        sigma_map = sigma_base / max(confidence, 0.25)
        R_ct = float(sigma_map**2)

        # 1-DOF innovation variance
        H = np.zeros((1, self.dim_x))
        H[0, 0] = n_N
        H[0, 1] = n_E

        S = float((H @ self.P @ H.T)[0, 0] + R_ct)
        if S < 1e-12:
            self.rejected_map_updates += 1
            self.map_rejection_reasons["other"] += 1
            return False, 0.0, None

        nis = float((y_ct**2) / S)

        # 1-DOF Chi-Square NIS gating (threshold 6.635 at p=0.01)
        if nis > 6.635:
            self.rejected_map_updates += 1
            self.map_rejection_reasons["nis_gate"] += 1
            return False, nis, None

        # Full Kalman gain for all states correlated with position
        K = (self.P @ H.T) / S  # shape (dim_x, 1)

        self.x += (K * y_ct).flatten()
        self.x[3] = self.x[3] % (2.0 * np.pi)

        # Joseph form covariance update strictly guarantees positive semi-definiteness
        I_KH = np.eye(self.dim_x) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + (K * R_ct) @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        for i in range(self.dim_x):
            self.P[i, i] = max(self.P[i, i], 1e-8)

        self.accepted_map_updates += 1
        self.map_nis_history.append(float(nis))

        # 2. Optional Controlled Road-Heading Constraint (Phase 17B)
        hdg_nis = None
        if is_heading_valid and self.x[2] > 2.5 and confidence > 0.7:
            self.total_map_heading_updates += 1
            d_hdg = abs((self.x[3] - psi_road + np.pi) % (2.0 * np.pi) - np.pi)
            if d_hdg < np.radians(max_d_hdg_deg):
                if sigma_heading_rad is not None:
                    hdg_sigma = float(sigma_heading_rad)
                else:
                    hdg_sigma = float(np.radians(5.0) / max(confidence, 0.5))
                R_hdg = float(hdg_sigma**2)

                # Linear Kalman update with wrapped angular innovation
                y_hdg = float((psi_road - self.x[3] + np.pi) % (2.0 * np.pi) - np.pi)
                S_hdg = float(self.P[3, 3] + R_hdg)
                nis_h = float((y_hdg**2) / S_hdg)

                if nis_h <= heading_nis_gate:
                    K_h = self.P[:, 3] / S_hdg  # shape (dim_x,)
                    self.x += K_h * y_hdg
                    self.x[3] = self.x[3] % (2.0 * np.pi)

                    # Joseph form covariance update for heading
                    H_h = np.zeros((1, self.dim_x))
                    H_h[0, 3] = 1.0
                    K_h_col = K_h[:, np.newaxis]
                    I_KH_h = np.eye(self.dim_x) - K_h_col @ H_h
                    self.P = I_KH_h @ self.P @ I_KH_h.T + (K_h_col * R_hdg) @ K_h_col.T
                    self.P = 0.5 * (self.P + self.P.T)
                    for i in range(self.dim_x):
                        self.P[i, i] = max(self.P[i, i], 1e-8)

                    self.accepted_map_heading_updates += 1
                    self.map_heading_nis_history.append(nis_h)
                    hdg_nis = nis_h
                else:
                    self.rejected_map_heading_updates += 1
            else:
                self.rejected_map_heading_updates += 1

        return True, nis, hdg_nis


def compute_pre_outage_can_calibration(
    pre_gps_speed: np.ndarray,
    pre_wheel_rl: np.ndarray,
    pre_wheel_rr: np.ndarray,
    nominal_r_eff: float = 0.2776,
    min_speed_ms: float = 2.5,
    min_samples: int = 10
) -> Tuple[float, str, float]:
    """
    Estimates effective wheel radius R_eff strictly from pre-outage data
    (when phone GNSS is available and healthy).
    """
    omega_rear = 0.5 * (pre_wheel_rl + pre_wheel_rr)
    valid = (pre_gps_speed > min_speed_ms) & (omega_rear > 5.0) & (~np.isnan(pre_gps_speed))

    if np.sum(valid) >= min_samples:
        r_ratios = pre_gps_speed[valid] / omega_rear[valid]
        r_eff = float(np.median(r_ratios))
        r_eff = float(np.clip(r_eff, 0.25, 0.31))
        std_r = float(np.std(r_ratios))
        return r_eff, "pre_outage_gps_calibrated", std_r
    else:
        return nominal_r_eff, "nominal_fallback", 0.0
