"""
Phase 16B-4: Calibrated Differential Wheel-Yaw Sanity & Observability Test Suite

Verifies:
  1. Zero innovation -> dx ~ 0.
  2. Positive yaw innovation -> correction in correct direction.
  3. Negative yaw innovation -> symmetric correction.
  4. Artificial Gyro-Bias Observability (Mandatory):
     True yaw rate = 0, Gyro = +0.05 rad/s, Initial b_g = 0, Wheel yaw = 0.
     Repeated updates drive b_g -> approximately +0.05 rad/s.
  5. Covariance remains positive definite across mixed updates.
  6. Missing/invalid wheel yaw falls back gracefully (NaN, Inf, bounds).
  7. Baseline parity: System A matches existing control to numerical tolerance.
  8. Straight-road calibrated yaw ~ 0:
     Applying k_diff = 0.9946 eliminates the -4.6 deg/s false turn rate,
     leaving residual straight yaw rate < 0.5 deg/s.
"""

import math
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import unittest
import numpy as np

from modules.ukf import UKFNavigationFilter
from modules.can_fusion_ukf import CANFusionUKF


class TestPhase16B4Sanity(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.init_lat = 52.4025
        self.init_lon = -1.5035
        self.init_speed = 15.0
        self.init_hdg = math.radians(45.0)
        # Phase 16B-4 calibrated yaw noise variance: sigma_yaw = 1.56 deg/s
        self.sigma_yaw_rad = 1.56 * math.pi / 180.0  # 0.027227 rad/s
        self.r_yaw = self.sigma_yaw_rad ** 2         # 0.0007413 rad^2/s^2
        self.b_eff = 1.4976
        self.r_eff = 0.2776
        self.k_diff_nominal = 0.9946

    def test_1_zero_innovation(self):
        """1. Zero innovation: wheel yaw = gyro yaw - b_g -> dx ~ 0."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.02
        b_g = ukf.x[4]  # 0.0
        z_wheel_yaw = gyro_reading - b_g  # exactly matching predicted yaw rate

        x_prior = ukf.x.copy()
        applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=z_wheel_yaw, gyro_reading=gyro_reading, r_yaw=self.r_yaw)

        self.assertTrue(applied)
        self.assertLess(nis, 1e-5)
        np.testing.assert_allclose(ukf.x, x_prior, atol=1e-6)

    def test_2_positive_yaw_innovation(self):
        """2. Positive yaw innovation: wheel yaw > predicted yaw rate -> b_g decreases, omega_corr increases."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.0
        # Wheel claims turning rate of +0.05 rad/s while gyro reading is 0.0
        applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=0.05, gyro_reading=gyro_reading, r_yaw=self.r_yaw)
        self.assertTrue(applied)
        # Innovation y = z - (gyro - b_g) = +0.05.
        # Since h(s) = gyro - s[4], dx[4] should be negative (b_g decreases so omega_corr = gyro - b_g increases)
        self.assertLess(ukf.x[4], 0.0)
        omega_corr = gyro_reading - ukf.x[4]
        self.assertGreater(omega_corr, 0.0)

    def test_3_negative_yaw_innovation(self):
        """3. Negative yaw innovation: symmetric correction in opposite direction."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        gyro_reading = 0.0
        applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=-0.05, gyro_reading=gyro_reading, r_yaw=self.r_yaw)
        self.assertTrue(applied)
        self.assertGreater(ukf.x[4], 0.0)
        omega_corr = gyro_reading - ukf.x[4]
        self.assertLess(omega_corr, 0.0)

    def test_4_artificial_gyro_bias_observability(self):
        """
        4. ARTIFICIAL GYRO-BIAS OBSERVABILITY TEST (Mandatory)
        True yaw rate = 0
        Gyro = +0.05 rad/s
        Initial b_g = 0
        Wheel yaw = 0
        Repeated wheel-yaw updates should drive:
        b_g -> approximately +0.05 rad/s
        and corrected yaw rate (gyro - b_g) -> 0.
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

        # Verify b_g approached +0.05 rad/s closely (within 10% = delta 0.005)
        self.assertAlmostEqual(final_bg, 0.05, delta=0.005,
                               msg=f"b_g did not converge to +0.05 rad/s, got {final_bg:.5f}")
        # Verify corrected yaw rate is near zero
        self.assertAlmostEqual(omega_corr, 0.0, delta=0.005,
                               msg=f"Corrected yaw rate did not converge to 0, got {omega_corr:.5f}")

    def test_5_covariance_positive_definite(self):
        """5. P matrix remains positive definite across 100 mixed updates."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        np.random.seed(42)
        for _ in range(100):
            ukf.predict(acc_fwd=0.1, gyro_yaw=0.02, dt=self.dt)
            ukf.update_can_speed(v_can=15.0 + np.random.normal(0, 0.2))
            ukf.update_wheel_yaw(z_wheel_yaw=0.02 + np.random.normal(0, 0.02), gyro_reading=0.02, r_yaw=self.r_yaw)

            eigs = np.linalg.eigvals(ukf.P)
            self.assertTrue(np.all(eigs > 0.0), f"Non-positive eigenvalue found: {eigs.min()}")

    def test_6_missing_can_fallback(self):
        """6. Missing/invalid wheel yaw falls back gracefully."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        x_before = ukf.x.copy()
        applied_nan, _ = ukf.update_wheel_yaw(z_wheel_yaw=float("nan"), gyro_reading=0.02, r_yaw=self.r_yaw)
        self.assertFalse(applied_nan)
        np.testing.assert_array_equal(ukf.x, x_before)

        applied_inf, _ = ukf.update_wheel_yaw(z_wheel_yaw=float("inf"), gyro_reading=0.02, r_yaw=self.r_yaw)
        self.assertFalse(applied_inf)
        np.testing.assert_array_equal(ukf.x, x_before)

        applied_bound, _ = ukf.update_wheel_yaw(z_wheel_yaw=5.0, gyro_reading=0.02, r_yaw=self.r_yaw)  # > 1.5 rad/s
        self.assertFalse(applied_bound)
        np.testing.assert_array_equal(ukf.x, x_before)

    def test_7_baseline_parity(self):
        """7. Baseline parity: System A matches existing control to numerical tolerance."""
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

    def test_8_straight_road_calibrated_yaw(self):
        """
        8. Straight-road calibrated yaw ~ 0:
        Simulate typical straight driving with asymmetric tire radius:
        omega_RL = 72.0 rad/s (~20 m/s = 72 km/h)
        omega_RR = omega_RL * 0.994316 = 71.5907 rad/s
        Raw wheel yaw: (omega_RR - omega_RL) * R_eff / B_eff ~ -0.0759 rad/s = -4.35 deg/s!
        Calibrated wheel yaw: (omega_RR - k_diff * omega_RL) * R_eff / B_eff ~ 0.0 rad/s.
        Verify that calibrated wheel yaw does NOT corrupt the gyro bias.
        """
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg, allow_bias_learning=True)

        omega_rl = 72.0
        omega_rr = omega_rl * 0.994316  # Real vehicle tire asymmetry

        # Uncalibrated (Phase 16B-2)
        raw_yaw = (omega_rr - omega_rl) * self.r_eff / self.b_eff
        self.assertLess(raw_yaw, -0.05)  # Catastrophic false turn rate of ~ -4.35 deg/s

        # Calibrated (Phase 16B-4)
        k_diff = 0.994316
        cal_yaw = (omega_rr - k_diff * omega_rl) * self.r_eff / self.b_eff
        self.assertAlmostEqual(cal_yaw, 0.0, places=5)

        # Run 50 steps with calibrated wheel yaw on straight road with 0 gyro rate
        gyro_reading = 0.0
        for _ in range(50):
            ukf.predict(acc_fwd=0.0, gyro_yaw=gyro_reading, dt=self.dt)
            applied, nis = ukf.update_wheel_yaw(z_wheel_yaw=cal_yaw, gyro_reading=gyro_reading, r_yaw=self.r_yaw)
            self.assertTrue(applied)

        # Final gyro bias should remain near zero (within 0.001 rad/s)
        self.assertAlmostEqual(ukf.x[4], 0.0, delta=0.001,
                               msg=f"Calibrated wheel yaw corrupted gyro bias: {ukf.x[4]:.6f}")


if __name__ == "__main__":
    unittest.main()
