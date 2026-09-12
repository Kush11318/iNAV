"""
iNAV Unified Alignment & Calibration Subsystem (Module A - Phase 3).

Establishes strict mathematical and operational parity with C++ inav_alignment.hpp.
Transforms sensor samples from PHONE_BODY to VEHICLE_FRD.

Canonical Frame Conventions:
  PHONE_BODY:
    +X : Phone right edge
    +Y : Phone top edge (display up)
    +Z : Outward from display (screen normal)
  VEHICLE_FRD:
    +X : Vehicle Forward
    +Y : Vehicle Right
    +Z : Vehicle Down (Right-handed, positive clockwise yaw around +Z)

Gravity & Linear Acceleration Convention:
  At rest in PHONE_BODY (flat face-up): specific force is [0, 0, +9.80665] m/s^2.
  In VEHICLE_FRD: specific force at rest is [0, 0, -9.80665] m/s^2.
  Linear acceleration detrending: a_lin = a_meas_v - [0, 0, -9.80665] = [ax, ay, az + 9.80665].
  At rest, a_lin = [0, 0, 0] m/s^2.
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np

from modules.sensor_types import (
    SensorFrame,
    Vec3,
    ImuSample,
    GnssSample,
    ImuValidity,
    GnssValidity,
    units
)

logger = logging.getLogger("iNAV.alignment")


class AlignmentState(IntEnum):
    UNINITIALIZED = 0
    STATIC_LEVELING = 1
    STATIC_ALIGNED = 2
    DYNAMIC_ALIGNING = 3
    FULL_ALIGNED = 4
    REMOUNT_DETECTED = 5


@dataclass
class AlignmentConfig:
    gravity_norm: float = 9.80665

    # Stationarity Gate Thresholds
    stationary_accel_std: float = 0.10   # m/s^2 maximum std across stationary window
    stationary_gyro_std: float = 0.02    # rad/s maximum std across stationary window
    stationary_speed_max: float = 0.20   # m/s maximum GNSS speed for stationary confirmation
    min_static_samples: int = 20         # Samples required for static gravity leveling

    # Dynamic PCA Excitation Thresholds
    pca_min_samples: int = 30            # Samples required for dynamic PCA
    pca_min_accel_std: float = 0.40      # m/s^2 minimum horizontal accel variability
    pca_min_eigen_ratio: float = 4.0     # lambda1 / lambda2 ratio to prevent isotropic noise lock

    # Remount Detection Thresholds
    remount_angle_thresh_deg: float = 5.0  # Angular gravity shift to flag remount
    remount_debounce_count: int = 5        # Consecutive epochs above threshold before remount trip


@dataclass
class AlignmentResult:
    R_b_to_v: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    gyro_bias_body: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    accel_bias_body: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    confidence: float = 0.0
    state: AlignmentState = AlignmentState.UNINITIALIZED
    remount_flag: bool = False

    @property
    def is_aligned(self) -> bool:
        return self.state in (AlignmentState.STATIC_ALIGNED, AlignmentState.FULL_ALIGNED)


class AlignmentEngine:
    """
    Unified Alignment & Calibration Engine.
    Converts PHONE_BODY samples to VEHICLE_FRD.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.cfg = config if config is not None else AlignmentConfig()
        self.reset()

    def reset(self) -> None:
        self.result = AlignmentResult()
        # Default flat face-up (+Z down in FRD is -Z in phone body)
        self.u_z_body = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.acc_window: List[Vec3] = []
        self.gyro_window: List[Vec3] = []
        self.speed_window: List[float] = []

        self.running_gravity: Optional[np.ndarray] = None
        self.remount_counter = 0

    @property
    def R_b_to_v(self) -> np.ndarray:
        return self.result.R_b_to_v

    @property
    def gyro_bias(self) -> np.ndarray:
        return np.array([self.result.gyro_bias_body.x,
                         self.result.gyro_bias_body.y,
                         self.result.gyro_bias_body.z], dtype=np.float64)

    @property
    def confidence_score(self) -> float:
        return self.result.confidence

    def feed_imu(
        self,
        sample: ImuSample,
        gnss: Optional[GnssSample] = None
    ) -> None:
        """Feed a canonical IMU sample and optional GNSS fix into alignment pipeline."""
        if not sample.is_valid:
            return

        self.acc_window.append(sample.accel_mps2)
        self.gyro_window.append(sample.gyro_radps)
        if gnss is not None and gnss.has_basic_fix:
            self.speed_window.append(gnss.speed_mps)
        else:
            self.speed_window.append(-1.0)

        # Retain bounded buffer
        if len(self.acc_window) > 100:
            self.acc_window.pop(0)
            self.gyro_window.pop(0)
            self.speed_window.pop(0)

        # 1. Remount Monitoring if aligned
        if self.result.is_aligned:
            if self.check_remount(sample.accel_mps2):
                return

        # 2. Static Leveling Phase
        if self.result.state in (AlignmentState.UNINITIALIZED,
                                 AlignmentState.STATIC_LEVELING,
                                 AlignmentState.REMOUNT_DETECTED):
            if len(self.acc_window) >= self.cfg.min_static_samples:
                if self._check_stationarity(self.cfg.min_static_samples):
                    self._calibrate_static_from_window(self.cfg.min_static_samples)
                else:
                    self.result.state = AlignmentState.STATIC_LEVELING
            return

        # 3. Dynamic PCA Alignment Phase
        if self.result.state in (AlignmentState.STATIC_ALIGNED, AlignmentState.DYNAMIC_ALIGNING):
            if len(self.acc_window) >= self.cfg.pca_min_samples:
                self._calibrate_dynamic_from_window(self.cfg.pca_min_samples)

    def calibrate_static_buffer(
        self,
        acc_samples: Union[List[Vec3], np.ndarray],
        gyro_samples: Union[List[Vec3], np.ndarray]
    ) -> bool:
        """
        Explicitly perform stationary gravity leveling on provided sample buffers.
        Returns True if stationarity is verified and leveling succeeds.
        """
        acc_arr = self._to_numpy_array(acc_samples)
        gyro_arr = self._to_numpy_array(gyro_samples)

        if len(acc_arr) < self.cfg.min_static_samples or len(gyro_arr) < self.cfg.min_static_samples:
            return False

        a_mean = np.mean(acc_arr, axis=0)
        a_std = np.std(acc_arr, axis=0)
        g_mean = np.mean(gyro_arr, axis=0)
        g_std = np.std(gyro_arr, axis=0)

        norm_a = np.linalg.norm(a_mean)
        a_std_mag = np.linalg.norm(a_std)
        g_std_mag = np.linalg.norm(g_std)

        # Stationarity Gate
        if (a_std_mag > self.cfg.stationary_accel_std or
            g_std_mag > self.cfg.stationary_gyro_std or
            abs(norm_a - self.cfg.gravity_norm) > 0.5):
            return False

        self._apply_static_solution(a_mean, g_mean)
        return True

    def calibrate_dynamic_buffer(
        self,
        acc_samples: Union[List[Vec3], np.ndarray],
        gyro_samples: Union[List[Vec3], np.ndarray],
        speed_deltas: Optional[Union[List[float], np.ndarray]] = None
    ) -> bool:
        """
        Explicitly perform horizontal PCA forward alignment on driving buffers.
        """
        if not self.result.is_aligned:
            return False

        acc_arr = self._to_numpy_array(acc_samples)
        gyro_arr = self._to_numpy_array(gyro_samples)
        spd_arr = np.array(speed_deltas, dtype=np.float64) if speed_deltas is not None else None

        return self._apply_dynamic_pca(acc_arr, gyro_arr, spd_arr)

    def calibrate_static(
        self,
        acc_stationary: Union[List[Vec3], np.ndarray],
        gyro_stationary: Union[List[Vec3], np.ndarray]
    ) -> bool:
        """Alias for calibrate_static_buffer."""
        return self.calibrate_static_buffer(acc_stationary, gyro_stationary)

    def calibrate_dynamic(
        self,
        acc_driving: Union[List[Vec3], np.ndarray],
        gyro_driving: Union[List[Vec3], np.ndarray],
        speed_delta: Optional[Union[List[float], np.ndarray]] = None
    ) -> bool:
        """Alias for calibrate_dynamic_buffer."""
        return self.calibrate_dynamic_buffer(acc_driving, gyro_driving, speed_delta)

    def transform_imu(
        self,
        body_sample_or_acc: Union[ImuSample, np.ndarray],
        gyro_b: Optional[np.ndarray] = None
    ) -> Union[ImuSample, Tuple[np.ndarray, np.ndarray]]:
        """
        Transforms a raw PHONE_BODY ImuSample into a canonical VEHICLE_FRD ImuSample.
        If called with two numpy arrays (acc_b, gyro_b), returns (acc_v, gyro_v).
        """
        if isinstance(body_sample_or_acc, np.ndarray) and gyro_b is not None:
            return self.transform_imu_arrays(body_sample_or_acc, gyro_b)

        body_sample: ImuSample = body_sample_or_acc # type: ignore
        # Gyro: subtract body-frame bias, then rotate into vehicle frame
        gb = np.array([body_sample.gyro_radps.x - self.result.gyro_bias_body.x,
                       body_sample.gyro_radps.y - self.result.gyro_bias_body.y,
                       body_sample.gyro_radps.z - self.result.gyro_bias_body.z], dtype=np.float64)
        gv = self.result.R_b_to_v @ gb

        # Accel: rotate body specific force into vehicle frame
        ab = np.array([body_sample.accel_mps2.x,
                       body_sample.accel_mps2.y,
                       body_sample.accel_mps2.z], dtype=np.float64)
        av = self.result.R_b_to_v @ ab

        return ImuSample(
            timestamp_ns=body_sample.timestamp_ns,
            accel_mps2=Vec3(float(av[0]), float(av[1]), float(av[2])),
            gyro_radps=Vec3(float(gv[0]), float(gv[1]), float(gv[2])),
            frame=SensorFrame.VEHICLE_FRD,
            validity=body_sample.validity,
            is_stationary=body_sample.is_stationary
        )

    def transform_imu_arrays(
        self,
        acc_b: np.ndarray,
        gyro_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Array/matrix helper for batch replay backwards-compatibility."""
        is_single = (acc_b.ndim == 1)
        if is_single:
            acc_b = acc_b.reshape(1, 3)
            gyro_b = gyro_b.reshape(1, 3)

        bias = np.array([self.result.gyro_bias_body.x,
                         self.result.gyro_bias_body.y,
                         self.result.gyro_bias_body.z])
        gyro_clean = gyro_b - bias
        acc_v = acc_b @ self.result.R_b_to_v.T
        gyro_v = gyro_clean @ self.result.R_b_to_v.T

        if is_single:
            return acc_v[0], gyro_v[0]
        return acc_v, gyro_v

    def get_linear_accel(self, veh_sample: ImuSample) -> Vec3:
        """
        Detrends gravity in VEHICLE_FRD:
        Linear kinematic acceleration = a_meas_v - [0, 0, -g] = [ax, ay, az + g].
        At rest, a_lin is [0, 0, 0] m/s^2.
        """
        return Vec3(
            veh_sample.accel_mps2.x,
            veh_sample.accel_mps2.y,
            veh_sample.accel_mps2.z + self.cfg.gravity_norm
        )

    def check_remount(self, acc_b: Union[Vec3, np.ndarray]) -> bool:
        """
        Continuous debounced remount detection via low-pass gravity vector divergence.
        """
        a_vec = np.array([acc_b.x, acc_b.y, acc_b.z]) if isinstance(acc_b, Vec3) else np.asarray(acc_b, dtype=np.float64)

        alpha = 0.05
        if self.running_gravity is None:
            self.running_gravity = a_vec.copy()
            return False

        self.running_gravity = (1.0 - alpha) * self.running_gravity + alpha * a_vec
        norm_g = np.linalg.norm(self.running_gravity)
        if norm_g < 1e-3:
            return False

        u_z_curr = -self.running_gravity / norm_g
        dot_product = np.clip(np.dot(u_z_curr, self.u_z_body), -1.0, 1.0)
        angle_diff_deg = math.degrees(math.acos(dot_product))

        if angle_diff_deg > self.cfg.remount_angle_thresh_deg:
            self.remount_counter += 1
            if self.remount_counter >= self.cfg.remount_debounce_count:
                self.result.state = AlignmentState.REMOUNT_DETECTED
                self.result.remount_flag = True
                self.result.confidence = max(0.1, self.result.confidence - 0.5)
                logger.warning(f"Remount trip confirmed! Angular shift: {angle_diff_deg:.2f}°")
                return True
        else:
            if self.remount_counter > 0:
                self.remount_counter -= 1

        return False

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _check_stationarity(self, n: int) -> bool:
        if len(self.acc_window) < n or len(self.gyro_window) < n:
            return False

        # Speed check
        for spd in self.speed_window[-n:]:
            if spd >= 0.0 and spd > self.cfg.stationary_speed_max:
                return False

        acc_arr = self._to_numpy_array(self.acc_window[-n:])
        gyro_arr = self._to_numpy_array(self.gyro_window[-n:])

        a_mean = np.mean(acc_arr, axis=0)
        norm_a = np.linalg.norm(a_mean)
        a_std = np.std(acc_arr, axis=0)
        g_std = np.std(gyro_arr, axis=0)

        return (float(np.linalg.norm(a_std)) <= self.cfg.stationary_accel_std and
                float(np.linalg.norm(g_std)) <= self.cfg.stationary_gyro_std and
                abs(norm_a - self.cfg.gravity_norm) <= 0.5)

    def _calibrate_static_from_window(self, n: int) -> None:
        acc_arr = self._to_numpy_array(self.acc_window[-n:])
        gyro_arr = self._to_numpy_array(self.gyro_window[-n:])
        self._apply_static_solution(np.mean(acc_arr, axis=0), np.mean(gyro_arr, axis=0))

    def _apply_static_solution(self, a_mean: np.ndarray, g_mean: np.ndarray) -> None:
        norm_a = np.linalg.norm(a_mean)
        if norm_a < 1e-3:
            return

        # Down unit vector in body frame: u_z = - a_mean / norm
        self.u_z_body = -a_mean / norm_a
        self.result.gyro_bias_body = Vec3(float(g_mean[0]), float(g_mean[1]), float(g_mean[2]))

        ref = np.array([1.0, 0.0, 0.0]) if abs(self.u_z_body[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        y = np.cross(self.u_z_body, ref)
        y /= np.linalg.norm(y)
        x = np.cross(y, self.u_z_body)
        x /= np.linalg.norm(x)

        self.result.R_b_to_v = np.vstack([x, y, self.u_z_body])

        # Pitch & Roll relative to vehicle horizon
        self.result.pitch_deg = math.degrees(math.asin(np.clip(-self.u_z_body[0], -1.0, 1.0)))
        self.result.roll_deg = math.degrees(math.atan2(self.u_z_body[1], -self.u_z_body[2]))

        self.result.state = AlignmentState.STATIC_ALIGNED
        self.result.confidence = 0.50
        self.running_gravity = a_mean.copy()
        self.remount_counter = 0
        self.result.remount_flag = False

    def _calibrate_dynamic_from_window(self, n: int) -> None:
        acc_arr = self._to_numpy_array(self.acc_window[-n:])
        gyro_arr = self._to_numpy_array(self.gyro_window[-n:])
        self._apply_dynamic_pca(acc_arr, gyro_arr)

    def _apply_dynamic_pca(
        self,
        acc_samples: np.ndarray,
        gyro_samples: np.ndarray,
        speed_deltas: Optional[np.ndarray] = None
    ) -> bool:
        n = len(acc_samples)
        if n < self.cfg.pca_min_samples:
            return False

        u_z = self.u_z_body

        # Horizontal orthonormal basis
        ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(ref, u_z)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u_z, e1)
        e2 /= np.linalg.norm(e2)

        # Horizontal projection
        dots = np.sum(acc_samples * u_z, axis=1, keepdims=True)
        a_h = acc_samples - dots * u_z

        h1 = np.dot(a_h, e1)
        h2 = np.dot(a_h, e2)
        h_mat = np.column_stack([h1, h2])

        cov = np.cov(h_mat, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        lambda1 = float(eigenvalues[1])
        lambda2 = float(eigenvalues[0])

        horiz_std = math.sqrt(max(0.0, lambda1))
        if horiz_std < self.cfg.pca_min_accel_std:
            self.result.state = AlignmentState.DYNAMIC_ALIGNING
            return False

        eigen_ratio = lambda1 / max(lambda2, 1e-6)
        if eigen_ratio < self.cfg.pca_min_eigen_ratio:
            self.result.state = AlignmentState.DYNAMIC_ALIGNING
            return False

        v = eigenvectors[:, 1]
        u_x = v[0] * e1 + v[1] * e2
        u_x /= np.linalg.norm(u_x)

        # Sign Resolution
        sign_confidence = 0.0
        if speed_deltas is not None and len(speed_deltas) == n:
            proj = np.dot(acc_samples, u_x)
            if np.sum(proj * speed_deltas) < 0.0:
                u_x = -u_x
            sign_confidence += 0.40
        else:
            # Centripetal Acceleration Correlation: a_lat ~ v * omega_z
            u_y_cand = np.cross(u_z, u_x)
            u_y_cand /= np.linalg.norm(u_y_cand)
            a_lat = np.dot(acc_samples, u_y_cand)
            w_z = np.dot(gyro_samples, u_z)
            centripetal_corr = float(np.mean(a_lat * w_z))

            if abs(centripetal_corr) > 0.02:
                if centripetal_corr > 0.0:
                    u_x = -u_x
                sign_confidence += 0.35
            else:
                # Net longitudinal acceleration fallback
                if np.mean(np.dot(acc_samples, u_x)) < 0.0:
                    u_x = -u_x
                sign_confidence += 0.15

        u_y = np.cross(u_z, u_x)
        u_y /= np.linalg.norm(u_y)
        u_x = np.cross(u_y, u_z)
        u_x /= np.linalg.norm(u_x)

        self.result.R_b_to_v = np.vstack([u_x, u_y, u_z])
        self.result.state = AlignmentState.FULL_ALIGNED
        self.result.yaw_deg = math.degrees(math.atan2(self.result.R_b_to_v[0, 1], self.result.R_b_to_v[0, 0]))

        ratio_score = np.clip((eigen_ratio - self.cfg.pca_min_eigen_ratio) / 10.0, 0.0, 0.30)
        std_score = np.clip((horiz_std - self.cfg.pca_min_accel_std) / 1.0, 0.0, 0.20)
        self.result.confidence = float(np.clip(0.50 + ratio_score + std_score + sign_confidence, 0.50, 1.00))

        return True

    def _to_numpy_array(self, samples: Union[List[Vec3], np.ndarray]) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            return samples
        return np.array([[s.x, s.y, s.z] for s in samples], dtype=np.float64)
