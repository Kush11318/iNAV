"""
iNAV Auto-Alignment & Coordinate Transformation Engine (Module A)
Aligns smartphone body frame (b) to vehicle frame (v):
- Static phase: Gravity vector -> Pitch and Roll, static gyro bias estimation
- Dynamic phase: Forward acceleration PCA during driving -> Yaw angle offset
- Continuous monitoring: Re-mount / orientation perturbation detection
"""

import logging
from typing import Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger("iNAV.alignment")


class AlignmentEngine:
    """
    Automated Phone-to-Vehicle Frame Calibration Engine.
    Converts measurements from phone body frame (b) to vehicle frame (v):
      x_v: Forward
      y_v: Right
      z_v: Down
    """

    def __init__(
        self,
        gravity_norm: float = 9.80665,
        remount_threshold_deg: float = 5.0,
        stationary_acc_std_thresh: float = 0.25,
        stationary_gyro_std_thresh: float = 0.05
    ):
        self.gravity_norm = gravity_norm
        self.remount_threshold_deg = remount_threshold_deg
        self.stationary_acc_std_thresh = stationary_acc_std_thresh
        self.stationary_gyro_std_thresh = stationary_gyro_std_thresh

        # Rotation matrix R_b_to_v (3x3)
        self.R_b_to_v: np.ndarray = np.eye(3, dtype=np.float64)

        # Estimated biases
        self.gyro_bias: np.ndarray = np.zeros(3, dtype=np.float64)
        self.accel_bias: np.ndarray = np.zeros(3, dtype=np.float64)

        # State flags
        self.static_calibrated: bool = False
        self.dynamic_calibrated: bool = False
        self.confidence_score: float = 0.0  # 0.0 to 1.0

        # Reference unit vertical vector in body frame
        self.u_z_body: np.ndarray = np.array([0.0, 0.0, 1.0])

        # Running gravity filter for disturbance detection
        self.running_gravity: Optional[np.ndarray] = None
        self.remount_detected: bool = False

    def is_stationary(self, acc_window: np.ndarray, gyro_window: np.ndarray) -> bool:
        """
        Check if phone is stationary based on accelerometer and gyroscope variance.
        acc_window: shape (N, 3), gyro_window: shape (N, 3)
        """
        acc_std = np.mean(np.std(acc_window, axis=0))
        gyro_std = np.mean(np.std(gyro_window, axis=0))
        return (acc_std < self.stationary_acc_std_thresh) and (gyro_std < self.stationary_gyro_std_thresh)

    def calibrate_static(self, acc_stationary: np.ndarray, gyro_stationary: np.ndarray) -> bool:
        """
        Static Alignment Phase:
        Uses mean gravity vector to determine vehicle vertical axis (Down)
        and estimates static gyroscope sensor bias.
        """
        if len(acc_stationary) < 10:
            return False

        # Average measured acceleration during rest is reaction force to gravity: a_b = -g_b
        mean_acc = np.mean(acc_stationary, axis=0)
        norm_acc = np.linalg.norm(mean_acc)

        # Acceleration-Magnitude Threshold Gating (Section 2):
        # Reject and freeze alignment if active vehicle acceleration is present
        if abs(norm_acc - self.gravity_norm) > 0.5:
            logger.debug(f"Dynamic motion detected (|norm - g| = {abs(norm_acc - self.gravity_norm):.2f} > 0.5 m/s^2); gravity update frozen.")
            return False

        # Down unit vector in body frame: u_z = - mean_acc / norm
        self.u_z_body = -mean_acc / norm_acc

        # Gyro bias is simply the average output while stationary
        self.gyro_bias = np.mean(gyro_stationary, axis=0)

        # Initial partial rotation matrix (aligning z only, identity horizontal)
        z = self.u_z_body
        # Choose arbitrary orthogonal vector in body frame
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        y = np.cross(z, ref)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)

        self.R_b_to_v = np.vstack([x, y, z])
        self.static_calibrated = True
        self.confidence_score = 0.5
        self.running_gravity = mean_acc.copy()

        logger.info(f"Static calibration complete. Vertical axis: {self.u_z_body}, Gyro bias: {self.gyro_bias}")
        return True

    def calibrate_dynamic(
        self,
        acc_driving: np.ndarray,
        gyro_driving: np.ndarray,
        speed_delta: Optional[np.ndarray] = None
    ) -> bool:
        """
        Dynamic Alignment Phase:
        Applies PCA on horizontal acceleration during acceleration/braking
        to find the vehicle's forward heading axis.
        """
        if not self.static_calibrated or len(acc_driving) < 30:
            return False

        # Remove static gravity component
        u_z = self.u_z_body

        # Project acceleration onto horizontal plane: a_h = a - (a . u_z) * u_z
        dots = np.sum(acc_driving * u_z, axis=1, keepdims=True)
        a_horizontal = acc_driving - dots * u_z

        # Perform PCA via Singular Value Decomposition / Covariance
        cov = np.cov(a_horizontal, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # The principal component (largest eigenvalue) corresponds to forward/backward axis
        u_x = eigenvectors[:, np.argmax(eigenvalues)]
        u_x /= np.linalg.norm(u_x)

        # Method 2 (Centripetal Acceleration Correlation: a_lat ≈ v * ω_z):
        # When vehicle curves, check correlation between cross-axis and gyro vertical yaw rate
        u_y_candidate = np.cross(u_z, u_x)
        u_y_candidate /= np.linalg.norm(u_y_candidate)
        a_lat = np.dot(a_horizontal, u_y_candidate)
        w_z = np.dot(gyro_driving, u_z)  # Gyro yaw component around gravity axis

        centripetal_corr = float(np.mean(a_lat * w_z))

        # Resolve forward vs backward direction ambiguity
        if speed_delta is not None and len(speed_delta) == len(acc_driving):
            proj = np.dot(a_horizontal, u_x)
            alignment_sign = np.sum(proj * speed_delta)
            if alignment_sign < 0:
                u_x = -u_x
        elif abs(centripetal_corr) > 0.05:
            # Under right turn (w_z > 0), centripetal acceleration points left (a_lat < 0)
            # If corr > 0, the lateral axis is inverted, flip u_x to correct coordinate frame
            if centripetal_corr > 0:
                u_x = -u_x
        else:
            # Fallback heuristic: positive longitudinal acceleration represents forward driving
            if np.mean(np.dot(a_horizontal, u_x)) < 0:
                u_x = -u_x

        # Ensure u_x is strictly perpendicular to u_z
        u_x = u_x - np.dot(u_x, u_z) * u_z
        u_x /= np.linalg.norm(u_x)

        # Right vector: u_y = u_z x u_x
        u_y = np.cross(u_z, u_x)
        u_y /= np.linalg.norm(u_y)

        # Full rotation matrix R_b_to_v
        self.R_b_to_v = np.vstack([u_x, u_y, u_z])
        self.dynamic_calibrated = True
        self.confidence_score = 1.0

        logger.info(f"Dynamic calibration complete. Forward axis determined with confidence {self.confidence_score:.2f}")
        return True

    def calibrate_vbo(
        self,
        acc_driving: np.ndarray,
        gnss_vel_ned: np.ndarray,
        min_speed_ms: float = 3.0
    ) -> bool:
        """
        Method 3: Pre-Outage GNSS Velocity Vector Alignment (VBO - Gold Standard).
        Compares GNSS horizontal velocity vector with IMU body-frame motion during
        healthy open-sky driving (v > 3 m/s, duration >= 3s).
        Bypasses in-cabin magnetic distortion completely.
        """
        if not self.static_calibrated or len(acc_driving) < 30 or len(gnss_vel_ned) < 30:
            return False

        speeds = np.linalg.norm(gnss_vel_ned[:, :2], axis=1)
        valid_mask = speeds >= min_speed_ms
        if np.sum(valid_mask) < 25:
            return False

        u_z = self.u_z_body
        dots = np.sum(acc_driving * u_z, axis=1, keepdims=True)
        a_horizontal = acc_driving - dots * u_z

        # Acceleration projected onto forward direction
        cov = np.cov(a_horizontal[valid_mask], rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        u_x = eigenvectors[:, np.argmax(eigenvalues)]
        u_x = u_x - np.dot(u_x, u_z) * u_z
        u_x /= np.linalg.norm(u_x)

        # Sign resolution: correlate forward projection with acceleration magnitude from GNSS speed
        gnss_speed_diff = np.diff(speeds[valid_mask], prepend=speeds[valid_mask][0])
        proj = np.dot(a_horizontal[valid_mask], u_x)
        if np.sum(proj * gnss_speed_diff) < 0:
            u_x = -u_x

        u_y = np.cross(u_z, u_x)
        u_y /= np.linalg.norm(u_y)

        self.R_b_to_v = np.vstack([u_x, u_y, u_z])
        self.dynamic_calibrated = True
        self.confidence_score = 1.0
        logger.info(f"GNSS VBO alignment locked. R_b_to_v converged (confidence: 1.00)")
        return True

    def check_remount(self, current_acc: np.ndarray, alpha: float = 0.05) -> bool:
        """
        Continuous re-mount & disturbance detector.
        Tracks gravity vector jump: if orientation abruptly changes > threshold,
        flags re-mount condition.
        """
        if self.running_gravity is None:
            self.running_gravity = current_acc.copy()
            return False

        # Low-pass filter running gravity estimate
        self.running_gravity = (1.0 - alpha) * self.running_gravity + alpha * current_acc
        norm_g = np.linalg.norm(self.running_gravity)

        if norm_g < 1e-3:
            return False

        current_u_z = -self.running_gravity / norm_g
        dot_product = np.clip(np.dot(current_u_z, self.u_z_body), -1.0, 1.0)
        angle_diff_deg = np.degrees(np.arccos(dot_product))

        if angle_diff_deg > self.remount_threshold_deg:
            self.remount_detected = True
            self.confidence_score = max(self.confidence_score - 0.5, 0.2)
            logger.warning(f"Re-mount / disturbance detected! Orientation shifted by {angle_diff_deg:.1f}°")
            return True

        self.remount_detected = False
        return False

    def transform_imu(
        self,
        acc_b: np.ndarray,
        gyro_b: np.ndarray,
        remove_gravity: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform raw body-frame accelerometer and gyroscope readings into vehicle frame:
          acc_v = R_b_to_v * acc_b - [0, 0, g]^T (if remove_gravity)
          gyro_v = R_b_to_v * (gyro_b - bias_b)
        Supports either single sample (3,) or batch (N, 3).
        """
        is_single = (acc_b.ndim == 1)
        if is_single:
            acc_b = acc_b.reshape(1, 3)
            gyro_b = gyro_b.reshape(1, 3)

        # De-bias gyro in body frame
        gyro_clean = gyro_b - self.gyro_bias

        # Rotate to vehicle frame
        acc_v = np.dot(acc_b, self.R_b_to_v.T)
        gyro_v = np.dot(gyro_clean, self.R_b_to_v.T)

        if remove_gravity:
            # Subtract gravity in vehicle frame (gravity points down along +z_v)
            acc_v[:, 2] -= self.gravity_norm

        if is_single:
            return acc_v[0], gyro_v[0]
        return acc_v, gyro_v
