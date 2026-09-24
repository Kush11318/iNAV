"""
Unit & Parity Sanity Test Suite for CAN / Wheel-Speed Fusion UKF (Phase 16A)
Verifies:
  1. Zero innovation -> no unexpected state movement (numerical drift < 1e-6)
  2. Positive speed innovation -> velocity responds in correct direction
  3. Negative speed innovation -> velocity responds in correct direction
  4. Covariance remains positive definite after multiple updates
  5. Missing CAN (NaN / inf / dropouts) -> filter falls back cleanly to control propagation
  6. Discontinuous spike rejection -> acceleration gate triggers correctly
  7. Exact baseline reproducibility when CAN update is disabled
"""

import math
import unittest
import numpy as np

from modules.ukf import UKFNavigationFilter
from modules.can_fusion_ukf import CANFusionUKF, compute_pre_outage_can_calibration


class TestCANFusionSanity(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.init_lat = 52.4025
        self.init_lon = -1.5035
        self.init_speed = 15.0
        self.init_hdg = math.radians(45.0)

    def test_01_zero_innovation(self):
        """Zero innovation should produce negligible state change and near-zero NIS."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        x_prior = ukf.x.copy()
        applied, nis = ukf.update_can_speed(v_can=self.init_speed)

        self.assertTrue(applied)
        self.assertLess(nis, 1e-5)
        # Position, heading, biases, and scale should remain unchanged
        np.testing.assert_allclose(ukf.x, x_prior, atol=1e-6)

    def test_02_positive_innovation(self):
        """Positive speed innovation (measured > prior) should increase estimated velocity."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        applied, nis = ukf.update_can_speed(v_can=self.init_speed + 2.5)
        self.assertTrue(applied)
        self.assertGreater(ukf.x[2], self.init_speed)
        # Check that velocity gained approximately (1.0 / (1.0 + 0.0325)) * 2.5 ~ 2.42 m/s
        self.assertAlmostEqual(ukf.x[2], self.init_speed + 2.42, delta=0.1)

    def test_03_negative_innovation(self):
        """Negative speed innovation (measured < prior) should decrease estimated velocity."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        applied, nis = ukf.update_can_speed(v_can=self.init_speed - 2.5)
        self.assertTrue(applied)
        self.assertLess(ukf.x[2], self.init_speed)
        self.assertAlmostEqual(ukf.x[2], self.init_speed - 2.42, delta=0.1)

    def test_04_covariance_positive_definite(self):
        """P matrix must remain strictly symmetric positive definite after 50 updates."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        np.random.seed(42)
        for _ in range(50):
            # Predict step
            ukf.predict(acc_fwd=0.1, gyro_yaw=0.01)
            # CAN update with noisy observation
            noisy_v = ukf.x[2] + np.random.normal(0, 0.2)
            ukf.update_can_speed(v_can=noisy_v)

            # Check symmetry
            np.testing.assert_allclose(ukf.P, ukf.P.T, atol=1e-10)
            # Check eigenvalues strictly positive
            eigs = np.linalg.eigvals(ukf.P)
            self.assertTrue(np.all(eigs > 0.0), f"Negative eigenvalue found: {eigs.min()}")

    def test_05_missing_can_fallback(self):
        """Filter handles NaN, inf, or dropouts gracefully by rejecting and maintaining state."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        x_before = ukf.x.copy()
        applied_nan, _ = ukf.update_can_speed(v_can=float("nan"))
        self.assertFalse(applied_nan)
        np.testing.assert_array_equal(ukf.x, x_before)

        applied_inf, _ = ukf.update_can_speed(v_can=float("inf"))
        self.assertFalse(applied_inf)
        np.testing.assert_array_equal(ukf.x, x_before)

    def test_06_spike_rejection(self):
        """Sudden impossible speed jumps (|dv/dt| > 8 m/s^2) are rejected by acceleration gate."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        # Initial valid update
        ukf.update_can_speed(v_can=15.0)

        # Immediate jump from 15 m/s to 30 m/s in 0.1s (150 m/s^2 acceleration)
        applied_spike, nis = ukf.update_can_speed(v_can=30.0)
        self.assertFalse(applied_spike)
        self.assertEqual(ukf.rejected_can_updates, 1)

    def test_07_baseline_numerical_parity(self):
        """When CAN update is not called, CANFusionUKF exactly reproduces UKFNavigationFilter."""
        ukf_std = UKFNavigationFilter(dt=self.dt)
        ukf_can = CANFusionUKF(dt=self.dt)

        ukf_std.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)
        ukf_can.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        for step in range(30):
            acc = 0.2 * math.sin(step * 0.1)
            gyro = 0.05 * math.cos(step * 0.1)

            ukf_std.predict(acc_fwd=acc, gyro_yaw=gyro)
            ukf_can.predict(acc_fwd=acc, gyro_yaw=gyro)

            if step % 5 == 0:
                ukf_std.update_velocity_net(delta_d_pred=3.0, sigma_pred=0.4, event_class=1)
                ukf_can.update_velocity_net(delta_d_pred=3.0, sigma_pred=0.4, event_class=1)

            np.testing.assert_allclose(ukf_std.x, ukf_can.x, atol=1e-12)
            np.testing.assert_allclose(ukf_std.P, ukf_can.P, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
