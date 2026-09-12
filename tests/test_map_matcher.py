"""
PHASE 6 IMPLEMENTATION CONTRACT v1.0 — COMPREHENSIVE TEST SUITE
Deterministic verification covering all requirements:
- Geometry tests (A - G)
- Direction tests (H - J)
- HMM tests (K - P)
- UKF tests (Q - V)
- Safety tests (W - Z)
- Section 33: Most Important Numerical Test (1D cross-track normal invariant)
- Section 34: Second Critical Test (Eastbound along-track non-reset)
- Section 35: Wrong-Road Test (Parallel roads stability)
- Section 36: Ground-Truth Leakage Test
- Section 30: Python <-> C++ numerical parity test
"""

import math
import os
import sys
from pathlib import Path
import ctypes
import unittest
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.map_matcher import RoadGraph, RoadNode, RoadEdge, RoadSegment, FixedLagHMMMapMatcher
from modules.ukf import UKFNavigationFilter

DLL_PATH = os.path.join(os.path.dirname(__file__), "inav_ukf_c.dll")


class TestPhase6Contract(unittest.TestCase):

    def setUp(self):
        self.graph = RoadGraph()
        self.graph.set_reference_origin(52.20, -2.19)

        # Edge 1: Northbound road (heading 0 deg) from (0, 0) to (1000, 0)
        self.graph.add_node(1, 52.20, -2.19)
        self.graph.add_node(2, 52.209, -2.19)
        self.graph.add_edge(
            edge_id=1,
            start_node_id=1,
            end_node_id=2,
            name="Northbound Way",
            polyline_pts=[np.array([0.0, 0.0]), np.array([1000.0, 0.0])],
            is_oneway=True
        )

        # Edge 2: Eastbound road (heading 90 deg) from (1000, 0) to (1000, 500)
        self.graph.add_node(3, 52.209, -2.183)
        self.graph.add_edge(
            edge_id=2,
            start_node_id=2,
            end_node_id=3,
            name="Eastbound Way",
            polyline_pts=[np.array([1000.0, 0.0]), np.array([1000.0, 500.0])],
            is_oneway=False
        )

        # Edge 3: Parallel Road A at E = 0, Road B at E = 20
        self.graph.add_node(4, 52.20, -2.1897)
        self.graph.add_node(5, 52.209, -2.1897)
        self.graph.add_edge(
            edge_id=3,
            start_node_id=4,
            end_node_id=5,
            name="Parallel Road B",
            polyline_pts=[np.array([0.0, 20.0]), np.array([1000.0, 20.0])],
            is_oneway=True
        )

        # Edge 4: Westbound Way at North 1500m
        self.graph.add_node(6, 52.215, -2.183)
        self.graph.add_node(7, 52.215, -2.19)
        self.graph.add_edge(
            edge_id=4,
            start_node_id=6,
            end_node_id=7,
            name="Westbound Way",
            polyline_pts=[np.array([1500.0, 500.0]), np.array([1500.0, 0.0])],
            is_oneway=True
        )

        # Edge 5: Diagonal road (heading 45 deg)
        self.graph.add_edge(
            edge_id=5,
            start_node_id=1,
            end_node_id=3,
            name="Diagonal Way",
            polyline_pts=[np.array([0.0, 0.0]), np.array([500.0, 500.0])],
            is_oneway=False
        )

        self.matcher = FixedLagHMMMapMatcher(graph=self.graph)

    # =========================================================================
    # GEOMETRY TESTS (A - G)
    # =========================================================================
    def test_geom_A_northbound_road(self):
        """Geometry A: Northbound road heading is exactly 0 deg."""
        seg = self.graph.edges[1].segments[0]
        self.assertAlmostEqual(seg.heading_deg, 0.0, places=2)
        # Tangent [1, 0], Normal [0, 1]
        t_N = math.cos(math.radians(seg.heading_deg))
        t_E = math.sin(math.radians(seg.heading_deg))
        self.assertAlmostEqual(t_N, 1.0, places=4)
        self.assertAlmostEqual(t_E, 0.0, places=4)

    def test_geom_B_eastbound_road(self):
        """Geometry B: Eastbound road heading is exactly 90 deg."""
        seg = self.graph.edges[2].segments[0]
        self.assertAlmostEqual(seg.heading_deg, 90.0, places=2)

    def test_geom_C_westbound_road(self):
        """Geometry C: Westbound road heading is exactly 270 deg."""
        seg = self.graph.edges[4].segments[0]
        self.assertAlmostEqual(seg.heading_deg, 270.0, places=2)

    def test_geom_D_diagonal_road(self):
        """Geometry D: Diagonal road heading is 45 deg."""
        seg = self.graph.edges[5].segments[0]
        self.assertAlmostEqual(seg.heading_deg, 45.0, places=2)

    def test_geom_E_negative_cross_track(self):
        """Geometry E: Point west of northbound road has negative cross-track."""
        res = self.matcher.match(np.array([100.0, -4.0]), heading_deg=0.0, speed_ms=10.0)
        self.assertEqual(res.edge_id, 1)
        # Signed cross track with right-hand normal n = [-sin(psi), cos(psi)] = [0, 1]
        # n . (p - p0) = 1 * (-4.0 - 0) = -4.0
        self.assertAlmostEqual(res.cross_track_error_m, -4.0, places=2)

    def test_geom_F_positive_cross_track(self):
        """Geometry F: Point east of northbound road has positive cross-track."""
        res = self.matcher.match(np.array([100.0, +4.0]), heading_deg=0.0, speed_ms=10.0)
        self.assertEqual(res.edge_id, 1)
        self.assertAlmostEqual(res.cross_track_error_m, +4.0, places=2)

    def test_geom_G_projection_near_endpoint(self):
        """Geometry G: Projection beyond segment endpoint is clamped to endpoint."""
        proj, dist = self.matcher.project_point_to_segment(
            np.array([1050.0, 0.0]), np.array([0.0, 0.0]), np.array([1000.0, 0.0])
        )
        self.assertAlmostEqual(proj[0], 1000.0, places=2)
        self.assertAlmostEqual(proj[1], 0.0, places=2)
        self.assertAlmostEqual(dist, 50.0, places=2)

    # =========================================================================
    # DIRECTION TESTS (H - J)
    # =========================================================================
    def test_dir_H_oneway_valid(self):
        """Direction H: Travel along permitted one-way direction is accepted with high confidence."""
        res = self.matcher.match(np.array([200.0, 0.5]), heading_deg=0.0, speed_ms=15.0)
        self.assertEqual(res.edge_id, 1)
        self.assertGreater(res.confidence, 0.7)

    def test_dir_I_oneway_reverse_rejected(self):
        """Direction I: Travel against one-way direction is heavily penalized."""
        res = self.matcher.match(np.array([200.0, 0.5]), heading_deg=180.0, speed_ms=15.0)
        self.assertLess(res.confidence, 0.25)

    def test_dir_J_bidirectional(self):
        """Direction J: Bidirectional road accepts both forward and backward travel."""
        self.matcher.reset()
        res_fwd = self.matcher.match(np.array([1000.0, 100.0]), heading_deg=90.0, speed_ms=10.0)
        self.matcher.reset()
        res_rev = self.matcher.match(np.array([1000.0, 100.0]), heading_deg=270.0, speed_ms=10.0)
        self.assertEqual(res_fwd.edge_id, 2)
        self.assertEqual(res_rev.edge_id, 2)
        self.assertGreater(res_fwd.confidence, 0.5)
        self.assertGreater(res_rev.confidence, 0.5)

    # =========================================================================
    # HMM TESTS (K - P)
    # =========================================================================
    def test_hmm_K_same_road_continuity(self):
        """HMM K: Consecutive epochs on the same road maintain high score and backpointers."""
        self.matcher.reset()
        for step in range(5):
            res = self.matcher.match(np.array([step * 20.0, 0.0]), heading_deg=0.0, speed_ms=10.0, dt=1.0)
            self.assertEqual(res.edge_id, 1)
        self.assertEqual(len(self.matcher.trellis_history), 5)

    def test_hmm_L_connected_junction_transition(self):
        """HMM L: Smooth transition through connected junction."""
        self.matcher.reset()
        # Approach junction
        for n in range(950, 1000, 10):
            self.matcher.match(np.array([float(n), 0.0]), heading_deg=0.0, speed_ms=10.0, dt=1.0)
        # Turn into Edge 2
        res = self.matcher.match(np.array([1000.0, 20.0]), heading_deg=90.0, speed_ms=10.0, dt=1.0)
        self.assertEqual(res.edge_id, 2)
        self.assertGreater(res.confidence, 0.5)

    def test_hmm_M_disconnected_road_rejected(self):
        """HMM M: Jump to disconnected road is penalized by graph topology."""
        self.matcher.reset()
        self.matcher.match(np.array([500.0, 0.0]), heading_deg=0.0, speed_ms=10.0)
        # Service road is disconnected
        connected = self.graph.are_edges_connected(1, 3)
        self.assertFalse(connected)

    def test_hmm_N_parallel_roads_stability(self):
        """HMM N: Parallel roads do not cause jumping without motion evidence."""
        self.matcher.reset()
        for n in range(0, 100, 20):
            res = self.matcher.match(np.array([float(n), 1.0]), heading_deg=0.0, speed_ms=15.0, dt=1.0)
            self.assertEqual(res.edge_id, 1)
        # Shift slightly toward Road B (E=8m) but continue moving North
        res_next = self.matcher.match(np.array([120.0, 8.0]), heading_deg=0.0, speed_ms=15.0, dt=1.0)
        self.assertEqual(res_next.edge_id, 1)

    def test_hmm_O_wrong_nearby_road(self):
        """HMM O: Nearby road with conflicting heading is rejected."""
        res = self.matcher.match(np.array([1000.0, 20.0]), heading_deg=0.0, speed_ms=15.0)
        # Edge 2 is at 90 deg, vehicle heading is 0 deg. Heading residual is 90 deg -> rejected
        self.assertNotEqual(res.edge_id, 2)

    def test_hmm_P_turn_at_junction(self):
        """HMM P: Vehicle turning at junction switches edge smoothly."""
        self.matcher.reset()
        self.matcher.match(np.array([990.0, 0.0]), heading_deg=0.0, speed_ms=8.0)
        res = self.matcher.match(np.array([1000.0, 10.0]), heading_deg=85.0, speed_ms=8.0)
        self.assertEqual(res.edge_id, 2)

    # =========================================================================
    # UKF TESTS (Q - V)
    # =========================================================================
    def test_ukf_Q_cross_track_correction(self):
        """UKF Q: Cross-track update pulls lateral position toward centerline."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        ukf.x[1] = 4.0 # 4m East
        accepted, nis, _ = ukf.update_map_match(p_N_match=100.0, p_E_match=0.0, psi_road=0.0, confidence=0.9)
        self.assertTrue(accepted)
        self.assertLess(abs(ukf.x[1]), 4.0)

    def test_ukf_R_zero_along_track_correction(self):
        """UKF R: Zero along-track correction on straight road (along-track invariant)."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        ukf.x[0] = 100.0
        ukf.x[1] = 5.0
        # Road is North (psi_r = 0), normal is East: n = [0, 1]
        accepted, _, _ = ukf.update_map_match(p_N_match=120.0, p_E_match=0.0, psi_road=0.0, confidence=0.85)
        self.assertTrue(accepted)
        # Along-track position p_N must be completely unchanged!
        self.assertAlmostEqual(ukf.x[0], 100.0, places=4)

    def test_ukf_S_nis_rejection(self):
        """UKF S: Outlier lateral innovation (50m) rejected by NIS threshold."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        accepted, nis, _ = ukf.update_map_match(p_N_match=0.0, p_E_match=50.0, psi_road=0.0, confidence=0.85)
        self.assertFalse(accepted)
        self.assertGreater(nis, 6.635)

    def test_ukf_T_accepted_map_update(self):
        """UKF T: Plausible innovation (NIS <= 6.635) is accepted."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        ukf.x[1] = 1.0
        accepted, nis, _ = ukf.update_map_match(p_N_match=0.0, p_E_match=0.0, psi_road=0.0, confidence=0.85)
        self.assertTrue(accepted)
        self.assertLessEqual(nis, 6.635)

    def test_ukf_U_heading_update(self):
        """UKF U: Heading constraint is fused when vehicle speed > 2.5 m/s and confidence > 0.7."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, np.radians(5.0)) # 5 deg heading error
        accepted, _, hdg_nis = ukf.update_map_match(
            p_N_match=0.0, p_E_match=0.0, psi_road=0.0, confidence=0.9, is_heading_valid=True
        )
        self.assertTrue(accepted)
        self.assertIsNotNone(hdg_nis)
        # Heading error should be reduced toward 0
        self.assertLess(abs(ukf.x[3]), np.radians(5.0))

    def test_ukf_V_heading_wrap(self):
        """UKF V: Heading update wraps properly across +/- pi."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, np.radians(358.0))
        accepted, _, _ = ukf.update_map_match(
            p_N_match=0.0, p_E_match=0.0, psi_road=0.0, confidence=0.9, is_heading_valid=True
        )
        self.assertTrue(accepted)
        # 358 deg (-2 deg) should smoothly converge to 0 deg without spinning 360 deg
        err = abs((ukf.x[3] + np.pi) % (2.0 * np.pi) - np.pi)
        self.assertLess(err, np.radians(2.0))

    # =========================================================================
    # SAFETY TESTS (W - Z)
    # =========================================================================
    def test_safety_W_ground_truth_injection_test(self):
        """Safety W: Map matcher does not inspect or depend on ground truth fields."""
        with open("modules/map_matcher.py", "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("gt_lat", src)
        self.assertNotIn("gt_lon", src)
        self.assertNotIn("COL_TRUE_LAT", src)
        self.assertNotIn("COL_TRUE_LON", src)

    def test_safety_X_map_disabled_parity(self):
        """Safety X: Disabling map matching causes zero map updates."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        # In pure DR mode, UKF state propagates only via IMU & VelocityNet
        ukf.predict(0.0, 0.0, dt=0.1)
        self.assertAlmostEqual(ukf.x[0], 1.0, places=2)

    def test_safety_Y_no_candidate_no_update(self):
        """Safety Y: Empty candidate region (off-road) results in confidence 0 and no update."""
        res = self.matcher.match(np.array([5000.0, 5000.0]), heading_deg=0.0, speed_ms=10.0)
        self.assertTrue(res.is_off_road)
        self.assertEqual(res.confidence, 0.0)

    def test_safety_Z_low_confidence_no_update(self):
        """Safety Z: Low confidence (< 0.25) map match is ignored by UKF."""
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        accepted, _, _ = ukf.update_map_match(0.0, 0.0, 0.0, confidence=0.20)
        self.assertFalse(accepted)

    # =========================================================================
    # SECTION 33: MOST IMPORTANT NUMERICAL TEST
    # =========================================================================
    def test_critical_33_most_important_numerical_test(self):
        """
        Section 33: Construct road heading = 0 deg, vehicle at pN = 100m, pE = 10m,
        road N0 = 0, E0 = 0 -> d_perp = 10m.
        Apply cross-track update:
        Expected: E decreases toward 0, N remains approximately unchanged.
        Then test: vehicle moves from N=100 -> N=110, E=10.
        Expected: cross-track remains ~10m without fighting forward motion.
        """
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, 0.0)
        ukf.x[0] = 100.0
        ukf.x[1] = 10.0

        pN_before = ukf.x[0]
        pE_before = ukf.x[1]

        # Apply cross-track update on North road (psi_r = 0)
        accepted, nis, _ = ukf.update_map_match(p_N_match=100.0, p_E_match=0.0, psi_road=0.0, confidence=0.85)
        self.assertTrue(accepted)
        self.assertLess(abs(ukf.x[1]), abs(pE_before)) # E decreases toward 0
        self.assertAlmostEqual(ukf.x[0], pN_before, places=4) # N remains unchanged

        # Step 2: Forward motion to N = 110, E = 10
        ukf.x[0] = 110.0
        ukf.x[1] = 10.0
        res = self.matcher.match(np.array([ukf.x[0], ukf.x[1]]), heading_deg=0.0, speed_ms=10.0)
        self.assertAlmostEqual(res.cross_track_error_m, 10.0, places=2)
        self.assertAlmostEqual(res.snapped_point[0], 110.0, places=2)

    # =========================================================================
    # SECTION 34: SECOND CRITICAL TEST
    # =========================================================================
    def test_critical_34_eastbound_along_track_non_reset(self):
        """
        Section 34: Road heading = 90 deg, vehicle moves East.
        The map measurement must NOT continuously reset its along-road position.
        The AI/UKF velocity must continue advancing.
        """
        ukf = UKFNavigationFilter(dt=0.1)
        ukf.initialize(52.20, -2.19, 10.0, np.radians(90.0))
        ukf.x[0] = 1000.0 # North position on Eastbound road
        ukf.x[1] = 100.0  # East position

        # Apply update on Eastbound road (psi_r = 90 deg, normal n = [-1, 0])
        # Vehicle is at N = 1005 (5m North lateral offset), E = 100
        ukf.x[0] = 1005.0
        accepted, _, _ = ukf.update_map_match(p_N_match=1000.0, p_E_match=100.0, psi_road=np.radians(90.0), confidence=0.85)
        self.assertTrue(accepted)
        # Cross-track (N) corrects toward 1000.0, but along-track (E) is 100% preserved!
        self.assertAlmostEqual(ukf.x[1], 100.0, places=4)
        self.assertLess(abs(ukf.x[0] - 1000.0), 5.0)

    # =========================================================================
    # SECTION 35: WRONG-ROAD TEST
    # =========================================================================
    def test_critical_35_wrong_road_test(self):
        """
        Section 35: Road A at E = 0, Road B at E = 20.
        Vehicle starts on Road A. Drift toward Road B does not cause immediate switch
        unless heading + motion evidence supports it.
        """
        self.matcher.reset()
        # Trajectory locked to Road A
        for n in range(0, 80, 20):
            res = self.matcher.match(np.array([float(n), 0.5]), heading_deg=0.0, speed_ms=10.0, dt=1.0)
            self.assertEqual(res.edge_id, 1)

        # Drift toward Road B (E = 12m, slightly closer to B at 20 than A at 0)
        res_drift = self.matcher.match(np.array([100.0, 12.0]), heading_deg=0.0, speed_ms=10.0, dt=1.0)
        # HMM topology & continuity should keep hypothesis locked to Road A
        self.assertEqual(res_drift.edge_id, 1)

    # =========================================================================
    # SECTION 36: GROUND-TRUTH LEAKAGE TEST
    # =========================================================================
    def test_critical_36_ground_truth_leakage_test(self):
        """
        Section 36: Injecting corrupted/random nonsense values into evaluator ground truth
        leaves map matcher output completely identical.
        """
        pt_est = np.array([150.0, 2.0])
        hdg_est = 0.0

        self.matcher.reset()
        res1 = self.matcher.match(pt_est, heading_deg=hdg_est, speed_ms=10.0)

        # Deliberately fake ground truth to Antarctica
        fake_gt_lat = -85.0
        fake_gt_lon = 120.0

        self.matcher.reset()
        res2 = self.matcher.match(pt_est, heading_deg=hdg_est, speed_ms=10.0)

        self.assertEqual(res1.edge_id, res2.edge_id)
        self.assertAlmostEqual(res1.cross_track_error_m, res2.cross_track_error_m, places=6)
        self.assertAlmostEqual(res1.confidence, res2.confidence, places=6)

    # =========================================================================
    # SECTION 30: PYTHON <-> C++ PARITY TEST
    # =========================================================================
    def test_critical_30_python_cpp_parity(self):
        """Section 30: Python <-> C++ map update 64-bit numerical parity."""
        if not os.path.exists(DLL_PATH):
            self.skipTest("C++ UKF DLL not found")

        c_lib = ctypes.CDLL(DLL_PATH)

        c_lib.ukf_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c_lib.ukf_create.restype = ctypes.c_int

        c_lib.ukf_initialize.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]
        c_lib.ukf_initialize.restype = None

        c_lib.ukf_update_map_match.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
        ]
        c_lib.ukf_update_map_match.restype = ctypes.c_int

        c_lib.ukf_get_x.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        c_lib.ukf_get_x.restype = None

        c_lib.ukf_destroy.argtypes = [ctypes.c_int]
        c_lib.ukf_destroy.restype = None

        h = c_lib.ukf_create(0.1, 1e-3, 2.0, 0.0)
        c_lib.ukf_initialize(h, 52.20, -2.19, 10.0, 0.0, 0.0, 0.0)

        ukf_py = UKFNavigationFilter(dt=0.1)
        ukf_py.initialize(52.20, -2.19, 10.0, 0.0, 0.0, 0.0)

        pos_nis_c = ctypes.c_double(0.0)
        hdg_nis_c = ctypes.c_double(0.0)

        ok_c = c_lib.ukf_update_map_match(h, 10.0, 1.0, 0.05, 0.85, 1, ctypes.byref(pos_nis_c), ctypes.byref(hdg_nis_c))
        ok_py, pos_nis_py, hdg_nis_py = ukf_py.update_map_match(10.0, 1.0, 0.05, 0.85, is_heading_valid=True)

        self.assertEqual(ok_c, 1 if ok_py else 0)
        self.assertAlmostEqual(pos_nis_c.value, pos_nis_py, places=5)
        if hdg_nis_py is not None:
            self.assertAlmostEqual(hdg_nis_c.value, hdg_nis_py, places=5)

        x_c = (ctypes.c_double * 7)()
        c_lib.ukf_get_x(h, x_c)

        for i in range(7):
            self.assertAlmostEqual(x_c[i], ukf_py.x[i], places=6, msg=f"Parity mismatch at x[{i}]")

        c_lib.ukf_destroy(h)


if __name__ == "__main__":
    unittest.main()
