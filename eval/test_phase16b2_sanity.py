"""
Phase 16B-2: Differential Wheel-Yaw Sanity & Observability Test Suite

Verifies:
  A. Zero innovation -> dx ~ 0.
  B. Positive yaw innovation -> correction in correct direction.
  C. Negative yaw innovation -> symmetric correction.
  D. Critical Bias-Observability Test:
     True yaw rate = 0, Gyro = +0.05 rad/s, Initial b_g = 0, Wheel yaw = 0.
     Repeated updates drive b_g -> +0.05 rad/s and omega_corr -> 0.
  E. Covariance remains positive definite.
  F. Missing/invalid wheel yaw falls back gracefully.
  G. Baseline parity: Experiment A matches Phase 16A control to numerical tolerance.
  H. Straight-line test: near-zero wheel yaw does not cause artificial heading drift.
"""

import math
import unittest
import numpy as np

from modules.ukf import UKFNavigationFilter
from modules.can_fusion_ukf import CANFusionUKF


class TestPhase16B2Sanity(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.init_lat = 52.4025
        self.init_lon = -1.5035
        self.init_speed = 15.0
        self.init_hdg = math.radians(45.0)
        self.r_yaw = 0.002213  # (0.04704 rad/s)^2

    def test_A_zero_innovation(self):
        """A. Zero innovation: wheel yaw = gyro yaw - b_g -> dx ~ 0."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.02
        b_g = ukf.x[4]  # 0.0
        z_wheel_yaw = gyro_reading - b_g  # exactly matching

        x_prior = ukf.x.copy()
        applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=z_wheel_yaw, gyro_reading=gyro_reading, r_yaw=self.r_yaw)

        self.assertTrue(applied)
        self.assertLess(nis, 1e-5)
        np.testing.assert_allclose(ukf.x, x_prior, atol=1e-6)

    def test_B_positive_yaw_innovation(self):
        """B. Positive yaw innovation: wheel yaw > predicted yaw rate -> b_g decreases, omega_corr increases."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.0
        # Wheel claims turning rate of +0.05 rad/s, while gyro reading is 0.0
        applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=0.05, gyro_reading=gyro_reading, r_yaw=self.r_yaw)
        self.assertTrue(applied)
        # Innovation y = z - (gyro - b_g) = +0.05.
        # Since h(s) = gyro - s[4], dx[4] should be negative (b_g decreases so omega_corr = gyro - b_g increases to match wheel)
        self.assertLess(ukf.x[4], 0.0)
        omega_corr = gyro_reading - ukf.x[4]
        self.assertGreater(omega_corr, 0.0)

    def test_C_negative_yaw_innovation(self):
        """C. Negative yaw innovation -> symmetric correction."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.0
        applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=-0.05, gyro_reading=gyro_reading, r_yaw=self.r_yaw)
        self.assertTrue(applied)
        self.assertGreater(ukf.x[4], 0.0)
        omega_corr = gyro_reading - ukf.x[4]
        self.assertLess(omega_corr, 0.0)

    def test_D_critical_bias_observability(self):
        """
        D. CRITICAL BIAS-OBSERVABILITY TEST (Mandatory)
        True: yaw rate = 0
        Gyro: gyro = +0.05 rad/s
        Initial: b_g = 0
        Wheel: wheel yaw ~ 0
        Repeated updates must drive b_g -> +0.05 rad/s and omega_corr -> 0.
        """
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.05
        z_wheel_yaw = 0.0

        for step in range(100):  # 10 seconds of updates at 10 Hz
            ukf.predict(acc_fwd=0.0, gyro_yaw=gyro_reading, dt=self.dt)
            applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=z_wheel_yaw, gyro_reading=gyro_reading, r_yaw=self.r_yaw)
            self.assertTrue(applied)

        final_bg = ukf.x[4]
        omega_corr = gyro_reading - final_bg

        # Verify b_g approached +0.05 rad/s closely (within 10%)
        self.assertAlmostEqual(final_bg, 0.05, delta=0.005)
        # Verify corrected yaw rate is near zero (within 0.005 rad/s ~ 0.28 deg/s)
        self.assertAlmostEqual(omega_corr, 0.0, delta=0.005)

    def test_E_covariance_positive_definite(self):
        """E. P matrix remains positive definite across 100 mixed updates."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        np.random.seed(42)
        for _ in range(100):
            ukf.predict(acc_fwd=0.1, gyro_yaw=0.02, dt=self.dt)
            ukf.update_can_speed(v_can=15.0 + np.random.normal(0, 0.2))
            ukf.update_wheel_yaw(z_wheel_yaw=0.02 + np.random.normal(0, 0.04), gyro_reading=0.02, r_yaw=self.r_yaw)

            eigs = np.linalg.eigvals(ukf.P)
            self.assertTrue(np.all(eigs > 0.0), f"Non-positive eigenvalue: {eigs.min()}")

    def test_F_no_wheel_fallback(self):
        """F. Missing or invalid wheel yaw falls back gracefully."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        x_before = ukf.x.copy()
        applied_nan, _ = ukf.update_wheel_yaw(z_wheel_yaw=float("nan"), gyro_reading=0.02)
        self.assertFalse(applied_nan)
        np.testing.assert_array_equal(ukf.x, x_before)

        applied_inf, _ = ukf.update_wheel_yaw(z_wheel_yaw=float("inf"), gyro_reading=0.02)
        self.assertFalse(applied_inf)
        np.testing.assert_array_equal(ukf.x, x_before)

        applied_bound, _ = ukf.update_wheel_yaw(z_wheel_yaw=5.0, gyro_reading=0.02)  # > 1.5 rad/s impossible
        self.assertFalse(applied_bound)
        np.testing.assert_array_equal(ukf.x, x_before)

    def test_G_baseline_parity(self):
        """G. Experiment A must match existing Phase 16A control to numerical tolerance."""
        ukf_std = UKFNavigationFilter(dt=self.dt)
        ukf_can = CANFusionUKF(dt=self.dt)

        ukf_std.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)
        ukf_can.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=False)

        for step in range(30):
            ukf_std.predict(acc_fwd=0.1, gyro_yaw=0.02)
            ukf_can.predict(acc_fwd=0.1, gyro_yaw=0.02)
            if step % 5 == 0:
                ukf_std.update_velocity_net(delta_d_pred=3.0, sigma_pred=0.4, event_class=1)
                ukf_can.update_velocity_net(delta_d_pred=3.0, sigma_pred=0.4, event_class=1)

            np.testing.assert_allclose(ukf_std.x, ukf_can.x, atol=1e-12)
            np.testing.assert_allclose(ukf_std.P, ukf_can.P, atol=1e-12)

    def test_H_straight_line_stability(self):
        """H. Straight-line test: near-zero wheel yaw must not cause artificial heading drift."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        init_hdg_deg = math.degrees(ukf.x[3])
        for step in range(100):  # 10s straight line
            ukf.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=self.dt)
            ukf.update_can_speed(v_can=15.0)
            ukf.update_wheel_yaw(z_wheel_yaw=0.0, gyro_reading=0.0, r_yaw=self.r_yaw)

        final_hdg_deg = math.degrees(ukf.x[3])
        self.assertAlmostEqual(final_hdg_deg, init_hdg_deg, delta=0.01)


if __name__ == "__main__":
    unittest.main()
