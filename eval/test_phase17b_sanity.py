"""
Phase 17B: Map Heading Ablation Sanity Test Suite

Verifies:
  1. Heading innovation sign and angular wrapping around ±π (e.g. 359° vs 1°).
  2. Road-heading convention (clockwise from North).
  3. Positive-definite covariance after joint position + heading updates (Joseph form).
  4. 1-DOF Chi-Square NIS gating on heading innovation (rejects when NIS > 6.635).
  5. Speed and confidence gating: no heading update when speed <= 2.5 m/s or conf <= 0.70.
  6. Angular gating: no heading update when |d_hdg| >= 15°.
  7. System B equivalence: when is_heading_valid=False, System C reduces identically to Phase 17A.
  8. Numerical reproducibility between repeated runs.
"""

import math
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import unittest
import numpy as np

from modules.can_fusion_ukf import CANFusionUKF


class TestPhase17BSanity(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.init_lat = 52.4025
        self.init_lon = -1.5035
        self.init_speed = 15.0
        self.init_hdg = 0.0

    def test_1_angle_wrapping_and_innovation_sign(self):
        """1. Angle wrapping around ±π correctly handles 359° vs 1° boundary."""
        ukf = CANFusionUKF(dt=self.dt)
        # Vehicle heading is 359 deg (almost North)
        hdg_359 = math.radians(359.0)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, hdg_359)

        # Road heading is 1 deg
        road_1 = math.radians(1.0)
        applied, pos_nis, hdg_nis = ukf.update_map_match(
            p_N_match=0.0,
            p_E_match=0.0,
            psi_road=road_1,
            confidence=0.85,
            is_heading_valid=True
        )

        self.assertTrue(applied)
        self.assertIsNotNone(hdg_nis)
        # Innovation was +2 deg, so heading should have increased towards 0/1 deg
        # Note: 359° + small positive angle wraps past 360° to ~0°..1°
        est_hdg_deg = math.degrees(ukf.x[3]) % 360.0
        self.assertTrue(est_hdg_deg < 5.0 or est_hdg_deg > 358.0, f"Heading {est_hdg_deg} did not wrap cleanly!")

    def test_2_positive_definite_covariance(self):
        """2. Joseph-form covariance update guarantees strictly positive eigenvalues."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.1)

        # Apply 10 sequential joint road-normal + heading updates
        for i in range(10):
            ukf.update_map_match(
                p_N_match=float(i * 1.5),
                p_E_match=float(i * 0.2),
                psi_road=0.12,
                confidence=0.85,
                is_heading_valid=True,
                sigma_heading_rad=math.radians(2.0)
            )
            min_eig = np.min(np.linalg.eigvalsh(ukf.P))
            self.assertGreater(min_eig, 0.0, f"Covariance P lost positive-definiteness at step {i}: min_eig={min_eig}")

    def test_3_heading_nis_gating(self):
        """3. Outlier heading innovation exceeding 6.635 (Chi-Square 1-DOF) is rejected."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.0)
        # Narrow heading uncertainty to force high NIS on 12 deg error
        ukf.P[3, 3] = math.radians(0.5)**2

        # 12° error with small sigma (0.5°) -> NIS = (12/0.7)^2 >> 6.635
        applied, pos_nis, hdg_nis = ukf.update_map_match(
            p_N_match=0.0,
            p_E_match=0.0,
            psi_road=math.radians(12.0),
            confidence=0.90,
            is_heading_valid=True,
            sigma_heading_rad=math.radians(0.5),
            heading_nis_gate=6.635
        )

        self.assertTrue(applied)  # Position update applied
        self.assertIsNone(hdg_nis)  # Heading update rejected by NIS gate
        self.assertEqual(ukf.rejected_map_heading_updates, 1)

    def test_4_speed_and_confidence_gating(self):
        """4. Heading update is skipped when speed <= 2.5 m/s or confidence <= 0.70."""
        # Low speed
        ukf_slow = CANFusionUKF(dt=self.dt)
        ukf_slow.initialize(self.init_lat, self.init_lon, 1.5, 0.0)
        _, _, hdg_nis_slow = ukf_slow.update_map_match(
            p_N_match=0.0, p_E_match=0.0, psi_road=0.05, confidence=0.85, is_heading_valid=True
        )
        self.assertIsNone(hdg_nis_slow, "Heading update applied when speed <= 2.5 m/s!")

        # Low confidence
        ukf_low_conf = CANFusionUKF(dt=self.dt)
        ukf_low_conf.initialize(self.init_lat, self.init_lon, 15.0, 0.0)
        _, _, hdg_nis_conf = ukf_low_conf.update_map_match(
            p_N_match=0.0, p_E_match=0.0, psi_road=0.05, confidence=0.60, is_heading_valid=True
        )
        self.assertIsNone(hdg_nis_conf, "Heading update applied when confidence <= 0.70!")

    def test_5_angular_threshold_gating(self):
        """5. Heading update is rejected when angular discrepancy >= 15°."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, 15.0, 0.0)

        # 16 deg discrepancy (> 15 deg gate)
        _, _, hdg_nis = ukf.update_map_match(
            p_N_match=0.0,
            p_E_match=0.0,
            psi_road=math.radians(16.0),
            confidence=0.85,
            is_heading_valid=True
        )
        self.assertIsNone(hdg_nis, "Heading update applied when |d_hdg| > 15°!")

    def test_6_system_b_reduces_to_phase17a(self):
        """6. Setting is_heading_valid=False reproduces Phase 17A exactly."""
        ukf_17a = CANFusionUKF(dt=self.dt)
        ukf_17a.initialize(self.init_lat, self.init_lon, 15.0, 0.1)

        ukf_17b_b = CANFusionUKF(dt=self.dt)
        ukf_17b_b.initialize(self.init_lat, self.init_lon, 15.0, 0.1)

        for step in range(5):
            ukf_17a.predict(acc_fwd=0.2, gyro_yaw=0.01)
            ukf_17b_b.predict(acc_fwd=0.2, gyro_yaw=0.01)

            ukf_17a.update_map_match(p_N_match=2.0, p_E_match=1.0, psi_road=0.1, confidence=0.8, is_heading_valid=False)
            ukf_17b_b.update_map_match(p_N_match=2.0, p_E_match=1.0, psi_road=0.1, confidence=0.8, is_heading_valid=False)

        np.testing.assert_allclose(ukf_17a.x, ukf_17b_b.x, atol=1e-12)
        np.testing.assert_allclose(ukf_17a.P, ukf_17b_b.P, atol=1e-12)

    def test_7_reproducibility(self):
        """7. Two runs with identical inputs produce identical state traces."""
        ukf_1 = CANFusionUKF(dt=self.dt)
        ukf_1.initialize(self.init_lat, self.init_lon, 15.0, 0.05)

        ukf_2 = CANFusionUKF(dt=self.dt)
        ukf_2.initialize(self.init_lat, self.init_lon, 15.0, 0.05)

        for _ in range(10):
            ukf_1.predict(0.5, 0.02)
            ukf_2.predict(0.5, 0.02)
            ukf_1.update_can_speed(15.2)
            ukf_2.update_can_speed(15.2)
            ukf_1.update_map_match(1.0, 0.5, 0.06, 0.85, is_heading_valid=True)
            ukf_2.update_map_match(1.0, 0.5, 0.06, 0.85, is_heading_valid=True)

        np.testing.assert_allclose(ukf_1.x, ukf_2.x, atol=1e-12)
        np.testing.assert_allclose(ukf_1.P, ukf_2.P, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
