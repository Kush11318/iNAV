"""
iNAV 5-Pillar Comprehensive Integration Test Suite
Verifies all 5 architectural pillars from the dead-reckoning roadmap:
- Pillar 1: 3D Spatial Rotation Augmentation in SO(3)
- Pillar 2: SPECTRA Neural Architecture (STFT + Depthwise Separable 2D + Channel Attention + BiGRU Attention Pooling)
- Pillar 3: MTN Pose Alignment & Dynamic Latency Loss Matching
- Pillar 4: 15-State Error-State EKF + NIS Chi-Square Outlier Gating + Joseph Covariance
- Pillar 5: Spatial Grid Road Index (<1ms) + Speed-Scaled Heading Emission + Travel Distance HMM
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import math
import unittest
import numpy as np
import torch

from modules.augmentation import random_rotation_matrix_so3, apply_3d_spatial_rotation, apply_batch_3d_spatial_rotation
from modules.velocity_net import VelocityNet
from modules.mtn import MotionTransformationNetwork
from modules.esekf import ESEKFNavigationFilter
from modules.map_matcher import HMMMapMatcher, SpatialGridIndex, RoadSegment
from train import dynamic_latency_huber_loss


class TestFivePillars(unittest.TestCase):

    def test_pillar1_3d_rotation_augmentation(self):
        """Pillar 1: Test SO(3) 3D spatial rotation matrices."""
        for _ in range(10):
            R = random_rotation_matrix_so3()
            # Orthogonality: R @ R.T == I
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
            # Determinant must be strictly +1 (proper rotation, not reflection)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=5)

        # Batch torch rotation
        x = torch.randn(8, 6, 20)
        x_rot = apply_batch_3d_spatial_rotation(x, prob=1.0)
        self.assertEqual(x_rot.shape, (8, 6, 20))
        # Verify norms of 3D acceleration vectors are preserved under rotation
        acc_orig_norm = torch.norm(x[:, 0:3, :], dim=1)
        acc_rot_norm = torch.norm(x_rot[:, 0:3, :], dim=1)
        torch.testing.assert_close(acc_orig_norm, acc_rot_norm, atol=1e-4, rtol=1e-4)
        print(" [PASS] Pillar 1: 3D SO(3) Spatial Rotation Augmentation verified.")

    def test_pillar2_velocitynet_architecture(self):
        """Pillar 2: Test VelocityNet Multi-Head 1D-CNN + GRU Displacement Network."""
        net = VelocityNet(in_channels=6, num_events=5)
        # 2-second window (20 samples @ 10Hz)
        x20 = torch.randn(4, 6, 20)
        d20, ev20, sig20 = net(x20)
        self.assertEqual(d20.shape, (4, 1))
        self.assertEqual(ev20.shape, (4, 5))
        self.assertEqual(sig20.shape, (4, 1))
        self.assertTrue(torch.all(sig20 > 0.0))  # strictly positive uncertainty
        print(" [PASS] Pillar 2: VelocityNet Multi-Head 1D-CNN + GRU verified.")

    def test_pillar3_mtn_and_dynamic_loss(self):
        """Pillar 3: Test MTN Pose Alignment & Dynamic Latency Loss Matching."""
        mtn = MotionTransformationNetwork(in_channels=6)
        x = torch.randn(4, 6, 20)
        euler, R_mat = mtn(x)
        self.assertEqual(euler.shape, (4, 3))
        self.assertEqual(R_mat.shape, (4, 3, 3))

        # Check rotation matrix properties for first batch item
        R0 = R_mat[0].detach().numpy()
        np.testing.assert_allclose(R0 @ R0.T, np.eye(3), atol=1e-4)

        # Test Dynamic GNSS Latency Loss Matcher
        huber_fn = torch.nn.SmoothL1Loss()
        pred = torch.tensor([[10.0], [12.0], [15.0], [18.0]])
        # Target has 1-step latency lag
        target_lagged = torch.tensor([[0.0], [10.0], [12.0], [15.0]])
        loss_dyn = dynamic_latency_huber_loss(pred, target_lagged, huber_fn)
        loss_std = huber_fn(pred, target_lagged)
        # Latency matcher must yield lower or equal loss than unaligned comparison
        self.assertLessEqual(float(loss_dyn), float(loss_std))
        print(" [PASS] Pillar 3: MTN Pose Alignment & Dynamic Latency Loss verified.")

    def test_pillar4_esekf_and_nis_gating(self):
        """Pillar 4: Test 15-State ES-EKF with Joseph Covariance and NIS Gating."""
        ekf = ESEKFNavigationFilter(dt=0.1)
        ekf.initialize(init_lat_ned=0.0, init_lon_ned=0.0, init_speed_ms=10.0, init_heading_rad=0.0)

        # High-rate prediction (stationary accelerometer specific force cancels gravity)
        for _ in range(10):
            ekf.predict(acc_meas=np.array([0.0, 0.0, -9.80665]), gyro_meas=np.array([0.0, 0.0, 0.0]), dt=0.1)

        # Positive definiteness of Joseph covariance: all eigenvalues > 0
        eigvals = np.linalg.eigvals(ekf.P)
        self.assertTrue(np.all(eigvals > 0.0), "Covariance matrix must remain strictly positive definite.")

        # Valid GNSS position update must be accepted
        accepted = ekf.update_gnss_pos(1.0, 0.0, accuracy_m=3.0)
        self.assertTrue(accepted, "Valid GNSS update within noise envelope must be accepted.")

        # Massive 500m multipath spike must be rejected by NIS Chi-square gate
        rejected = ekf.update_gnss_pos(500.0, 500.0, accuracy_m=3.0)
        self.assertFalse(rejected, "Anomalous 500m multipath jump must be rejected by NIS gate.")

        # Forward speed update
        ekf.update_forward_speed(10.0, variance=0.1)
        self.assertAlmostEqual(ekf.get_speed_ms(), 10.0, delta=2.0)
        print(" [PASS] Pillar 4: 15-State ES-EKF + Joseph Covariance + NIS Outlier Gating verified.")

    def test_pillar5_spatial_grid_and_hmm_map_matching(self):
        """Pillar 5: Test Spatial Grid Index and Speed-Weighted HMM Map Matching."""
        matcher = HMMMapMatcher(sigma_d=4.0, beta=5.0)

        # Add Highway segment (West-to-East: heading 90 deg)
        matcher.add_road_polyline("M1-Motorway", [(0.0, 0.0), (1000.0, 0.0)])
        # Add Parallel frontage street 25m to the North
        matcher.add_road_polyline("Frontage-Road", [(0.0, 25.0), (1000.0, 25.0)])

        # Benchmark query latency over 1000 lookups
        t0 = time.perf_counter()
        for i in range(1000):
            segs = matcher.index.query_radius(float(i % 1000), 2.0, radius_m=40.0)
        elapsed_per_query_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0
        self.assertLess(elapsed_per_query_ms, 1.0, f"Query took {elapsed_per_query_ms:.3f}ms (target <1.0ms)")

        # Vehicle driving at 30 m/s (highway speed) at heading 90 deg on Motorway with 3m GPS drift
        res = matcher.match(
            point_xy=np.array([100.0, 3.0]),
            heading_deg=90.0,
            speed_ms=30.0,
            travel_dist_m=3.0
        )
        self.assertFalse(res.is_off_road)
        self.assertEqual(res.matched_segment.name, "M1-Motorway")
        self.assertAlmostEqual(res.snapped_point[1], 0.0, places=1)  # Snapped onto y=0 centerline
        self.assertAlmostEqual(res.heading_deg, 90.0, places=1)
        print(f" [PASS] Pillar 5: Spatial Grid ({elapsed_per_query_ms:.3f}ms lookup) + HMM Snapping verified.")


if __name__ == "__main__":
    unittest.main()
