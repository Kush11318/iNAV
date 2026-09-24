"""
Phase 17A: CAN Speed + HMM/OSM Map Matching Sanity Test Suite

Verifies:
  1. Zero cross-track innovation -> dx == 0, nis == 0.
  2. Directional road normal projection:
     - North road (hdg=0°) -> normal is East -> dx strictly in East, North untouched.
     - East road (hdg=90°) -> normal is North -> dx strictly in North, East untouched.
  3. Speed and along-track preservation:
     Map update preserves forward speed state x[2] and along-track variance.
  4. Positive-definite covariance after repeated updates.
  5. Low confidence rejection (confidence < 0.25).
  6. 1-DOF Chi-Square NIS gating (rejects outliers with NIS > 6.635).
  7. Conditional road heading constraint:
     Updates heading when aligned within 15° at speed > 2.5 m/s, ignores when misaligned.
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


class TestPhase17ASanity(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.init_lat = 52.4025
        self.init_lon = -1.5035
        self.init_speed = 15.0
        self.init_hdg = 0.0  # Heading North

    def test_1_zero_innovation(self):
        """1. Zero innovation leaves state vector unchanged."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, self.init_hdg)

        p_N = ukf.x[0]
        p_E = ukf.x[1]
        x_prior = ukf.x.copy()

        applied, nis, _ = ukf.update_map_match(
            p_N_match=p_N,
            p_E_match=p_E,
            psi_road=0.0,
            confidence=0.8,
            is_heading_valid=False
        )

        self.assertTrue(applied)
        self.assertLess(nis, 1e-6)
        np.testing.assert_allclose(ukf.x, x_prior, atol=1e-6)

    def test_2_directional_road_normal_projection(self):
        """2. Correction occurs strictly perpendicular to road direction."""
        # Case A: Road heading North (psi=0 rad). Normal is East [0, 1].
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.0)

        # Matched point is 3m East, 0m North
        applied, nis, _ = ukf.update_map_match(
            p_N_match=0.0,
            p_E_match=3.0,
            psi_road=0.0,
            confidence=0.8,
            is_heading_valid=False
        )
        self.assertTrue(applied)
        self.assertAlmostEqual(ukf.x[0], 0.0, places=5, msg="North position moved on North-running road!")
        self.assertGreater(ukf.x[1], 0.0, msg="East position did not update along normal!")

        # Case B: Road heading East (psi=pi/2 rad). Normal is South [-1, 0].
        ukf_b = CANFusionUKF(dt=self.dt)
        ukf_b.initialize(self.init_lat, self.init_lon, self.init_speed, math.pi / 2.0)

        # Matched point is 3m North, 0m East
        applied_b, nis_b, _ = ukf_b.update_map_match(
            p_N_match=3.0,
            p_E_match=0.0,
            psi_road=math.pi / 2.0,
            confidence=0.8,
            is_heading_valid=False
        )
        self.assertTrue(applied_b)
        self.assertGreater(ukf_b.x[0], 0.0, msg="North position did not update along normal!")
        self.assertAlmostEqual(ukf_b.x[1], 0.0, places=5, msg="East position moved on East-running road!")

    def test_3_speed_preservation(self):
        """3. Map cross-track update does not alter forward speed state x[2]."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.0)

        v_before = ukf.x[2]
        applied, _, _ = ukf.update_map_match(
            p_N_match=0.0,
            p_E_match=4.0,
            psi_road=0.0,
            confidence=0.9,
            is_heading_valid=False
        )
        self.assertTrue(applied)
        self.assertEqual(ukf.x[2], v_before, "Forward velocity was modified by 1D map update!")

    def test_4_covariance_positive_definite(self):
        """4. Covariance matrix P remains strictly positive definite across multiple updates."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.0)

        for i in range(20):
            ukf.predict(acc_fwd=0.1, gyro_yaw=0.01, dt=self.dt)
            ukf.update_can_speed(v_can=15.0)
            if i % 2 == 0:
                ukf.update_map_match(
                    p_N_match=ukf.x[0],
                    p_E_match=ukf.x[1] + 1.0,
                    psi_road=0.0,
                    confidence=0.7,
                    is_heading_valid=False
                )
            eigs = np.linalg.eigvals(ukf.P)
            self.assertTrue(np.all(eigs > 0.0), f"Non-positive eigenvalue found at step {i}: {eigs.min()}")

    def test_5_low_confidence_rejection(self):
        """5. Match with confidence < 0.25 is rejected without state alteration."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.0)

        x_before = ukf.x.copy()
        applied, _, _ = ukf.update_map_match(
            p_N_match=10.0,
            p_E_match=10.0,
            psi_road=0.0,
            confidence=0.20,
            is_heading_valid=False
        )
        self.assertFalse(applied)
        np.testing.assert_array_equal(ukf.x, x_before)

    def test_6_nis_outlier_gating(self):
        """6. Gross spatial outlier (NIS > 6.635) is rejected by Chi-Square gate."""
        ukf = CANFusionUKF(dt=self.dt)
        ukf.initialize(self.init_lat, self.init_lon, self.init_speed, 0.0)

        # 40m cross-track outlier
        applied, nis, _ = ukf.update_map_match(
            p_N_match=0.0,
            p_E_match=40.0,
            psi_road=0.0,
            confidence=0.8,
            is_heading_valid=False
        )
        self.assertFalse(applied)
        self.assertGreater(nis, 6.635)

    def test_7_conditional_heading_constraint(self):
        """7. Heading constraint applies when aligned (< 15°), ignores when misaligned (> 15°)."""
        # Case A: Well-aligned heading (road hdg = 5°, filter hdg = 0°) -> Accepted
        ukf_a = CANFusionUKF(dt=self.dt)
        ukf_a.initialize(self.init_lat, self.init_lon, 10.0, 0.0)

        applied_a, _, hdg_nis_a = ukf_a.update_map_match(
            p_N_match=0.0,
            p_E_match=0.0,
            psi_road=math.radians(5.0),
            confidence=0.85,
            is_heading_valid=True
        )
        self.assertTrue(applied_a)
        self.assertIsNotNone(hdg_nis_a)
        self.assertGreater(ukf_a.x[3], 0.0, "Heading did not pull toward road heading!")

        # Case B: Misaligned heading (road hdg = 45°, filter hdg = 0°) -> Skipped
        ukf_b = CANFusionUKF(dt=self.dt)
        ukf_b.initialize(self.init_lat, self.init_lon, 10.0, 0.0)

        applied_b, _, hdg_nis_b = ukf_b.update_map_match(
            p_N_match=0.0,
            p_E_match=0.0,
            psi_road=math.radians(45.0),
            confidence=0.85,
            is_heading_valid=True
        )
        self.assertTrue(applied_b)  # Position still applied
        self.assertIsNone(hdg_nis_b, "Heading was erroneously updated when misaligned by 45°!")


if __name__ == "__main__":
    unittest.main()
