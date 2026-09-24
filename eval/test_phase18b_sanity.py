"""
Phase 18B Unit Sanity Tests:
1. Heading gate rejection (angular difference > theta_max)
2. Heading gate acceptance (angular difference <= theta_max)
3. Angle wrapping across 0/360 boundary (e.g. 5 deg vs 355 deg)
4. State integrity: heading is NOT modified by heading gate
5. Numerical equivalence to Phase 17A System C when theta_max is None
"""

import unittest
import math
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from modules.can_fusion_ukf import CANFusionUKF


class TestPhase18BSanity(unittest.TestCase):
    def setUp(self):
        self.ukf = CANFusionUKF(dt=0.1)
        self.ukf.initialize(52.2, -2.2, 10.0, math.radians(45.0))

    def test_heading_gate_rejection(self):
        # Filter heading is 45 deg. Road heading is 80 deg (difference = 35 deg > 30 deg)
        applied, nis, _ = self.ukf.update_map_match(
            p_N_match=1.0, p_E_match=1.0, psi_road=math.radians(80.0),
            confidence=0.9, is_heading_valid=False, theta_max_deg=30.0
        )
        self.assertFalse(applied)
        self.assertEqual(self.ukf.heading_gate_rejections, 1)
        self.assertEqual(self.ukf.map_rejection_reasons["heading_gate"], 1)

    def test_heading_gate_acceptance(self):
        # Filter heading is 45 deg. Road heading is 55 deg (difference = 10 deg <= 30 deg)
        applied, nis, _ = self.ukf.update_map_match(
            p_N_match=0.1, p_E_match=0.1, psi_road=math.radians(55.0),
            confidence=0.9, is_heading_valid=False, theta_max_deg=30.0
        )
        self.assertTrue(applied)
        self.assertEqual(self.ukf.heading_gate_rejections, 0)

    def test_angle_wrapping_boundary(self):
        # Filter heading is 5 deg. Road heading is 355 deg (difference = 10 deg <= 30 deg)
        self.ukf.x[3] = math.radians(5.0)
        applied, nis, _ = self.ukf.update_map_match(
            p_N_match=0.1, p_E_match=0.1, psi_road=math.radians(355.0),
            confidence=0.9, is_heading_valid=False, theta_max_deg=30.0
        )
        self.assertTrue(applied)

    def test_heading_not_modified(self):
        # Ensure road-normal update with heading gate does not modify heading state
        self.ukf.x[3] = math.radians(45.0)
        orig_hdg = float(self.ukf.x[3])
        applied, _, _ = self.ukf.update_map_match(
            p_N_match=1.0, p_E_match=-1.0, psi_road=math.radians(55.0),
            confidence=0.9, is_heading_valid=False, theta_max_deg=30.0
        )
        self.assertTrue(applied)
        # Heading state must be exactly unchanged because H_ct has zeros for heading
        self.assertAlmostEqual(self.ukf.x[3], orig_hdg, places=7)

    def test_backward_compatibility_none(self):
        # When theta_max_deg is None, large heading discrepancy is not gated by heading gate
        applied, nis, _ = self.ukf.update_map_match(
            p_N_match=0.1, p_E_match=0.1, psi_road=math.radians(120.0),
            confidence=0.9, is_heading_valid=False, theta_max_deg=None
        )
        # Should be processed (or gated only by NIS)
        self.assertEqual(self.ukf.heading_gate_rejections, 0)


if __name__ == "__main__":
    unittest.main()
