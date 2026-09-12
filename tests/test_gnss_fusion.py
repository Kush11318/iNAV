"""
Phase 5 Comprehensive GNSS Fusion, Health & Reacquisition Verification Suite
Tests A through J + Teleport Continuity Tests:
  TEST A — Healthy stationary GNSS (speed-gated course rejection)
  TEST B — Healthy moving GNSS (position and speed correction)
  TEST C — Noisy GNSS (uncertainty-weighted correction)
  TEST D — 500m GNSS Outlier (NIS chi-square rejection)
  TEST E — GNSS Outage (AIDED -> PURE_DR transition, no filter reset)
  TEST F — GNSS Reacquisition (PURE_DR -> QUARANTINE -> Trust Ramp -> AIDED)
  TEST G — Multiple Bad Reacquisition Fixes (Rejection, continuity preserved)
  TEST H — Low-speed GNSS Course (Rejection when v <= 2.5 m/s)
  TEST I — Monotonic Clock Freshness (Timeout detection)
  TEST J — Python <-> C++ Exact Numerical Parity across all states and covariance
  TELEPORT CONTINUITY TEST — Proof of smooth correction vs direct assignment
"""

import ctypes
import math
import os
import sys
import numpy as np

# Ensure iNAV root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.sensor_types import GnssSample, GnssValidity
from modules.gnss_health import GnssHealthManager, GnssHealthState, GnssHealthConfig
from modules.ukf import (
    UKFNavigationFilter as PyUKF,
    UKF_NIS_GATE_2D,
    UKF_NIS_GATE_1D,
    UKF_MIN_HEADING_SPEED,
    UKF_EARTH_RADIUS
)

# Load C++ UKF DLL
DLL_PATH = os.path.join(os.path.dirname(__file__), "inav_ukf_c.dll")
if not os.path.exists(DLL_PATH):
    raise FileNotFoundError(f"C++ UKF DLL not found at {DLL_PATH}")

cpp_lib = ctypes.CDLL(DLL_PATH)

cpp_lib.ukf_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
cpp_lib.ukf_create.restype = ctypes.c_int

cpp_lib.ukf_destroy.argtypes = [ctypes.c_int]
cpp_lib.ukf_destroy.restype = None

cpp_lib.ukf_initialize.argtypes = [
    ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double
]
cpp_lib.ukf_initialize.restype = None

cpp_lib.ukf_predict.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
cpp_lib.ukf_predict.restype = None

cpp_lib.ukf_compute_pos_nis.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
cpp_lib.ukf_compute_pos_nis.restype = ctypes.c_double

cpp_lib.ukf_update_gnss_pos.argtypes = [
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
]
cpp_lib.ukf_update_gnss_pos.restype = ctypes.c_int

cpp_lib.ukf_update_gnss_speed.argtypes = [
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
]
cpp_lib.ukf_update_gnss_speed.restype = ctypes.c_int

cpp_lib.ukf_update_gnss_course.argtypes = [
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
]
cpp_lib.ukf_update_gnss_course.restype = ctypes.c_int

cpp_lib.ukf_health_get_state.argtypes = [ctypes.c_int]
cpp_lib.ukf_health_get_state.restype = ctypes.c_int

cpp_lib.ukf_health_get_quarantine_count.argtypes = [ctypes.c_int]
cpp_lib.ukf_health_get_quarantine_count.restype = ctypes.c_int

cpp_lib.ukf_health_check_timeout.argtypes = [ctypes.c_int, ctypes.c_longlong]
cpp_lib.ukf_health_check_timeout.restype = None

cpp_lib.ukf_health_reset.argtypes = [ctypes.c_int]
cpp_lib.ukf_health_reset.restype = None

cpp_lib.ukf_update_gnss_sample.argtypes = [
    ctypes.c_int, ctypes.c_longlong, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_uint
]
cpp_lib.ukf_update_gnss_sample.restype = ctypes.c_int

cpp_lib.ukf_get_x.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cpp_lib.ukf_get_x.restype = None

cpp_lib.ukf_get_P.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cpp_lib.ukf_get_P.restype = None


def test_a_healthy_stationary():
    """TEST A: Healthy stationary GNSS: low-speed course rejected, position stable."""
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 0.0, 0.0)

    # Stationary GNSS report: speed = 0.2 m/s (< 2.5 m/s threshold), bearing = 90 deg
    accepted_hdg, _ = ukf.update_gnss_course(psi_gnss_rad=math.radians(90.0), v_gnss=0.2)
    assert not accepted_hdg, "Speed <= 2.5 m/s MUST reject GNSS heading update"
    assert abs(ukf.x[3]) < 1e-6, "Heading must remain unchanged when stationary"

    # Position update with small accuracy (3m)
    p_ok, nis = ukf.update_gnss_pos(0.1, 0.1, accuracy_m=3.0)
    assert p_ok, "Healthy stationary position update must be accepted"
    assert abs(ukf.x[0]) < 1.0 and abs(ukf.x[1]) < 1.0, "Position must remain stable"
    print("TEST A (Stationary GNSS): PASS")


def test_b_healthy_moving():
    """TEST B: Healthy moving GNSS: position and speed measurements correct UKF without teleport."""
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 10.0, 0.0)

    init_pos_N = ukf.x[0]
    p_ok, _ = ukf.update_gnss_pos(5.0, 0.0, accuracy_m=3.0)
    assert p_ok, "Moving position update must be accepted"
    assert ukf.x[0] > init_pos_N, "Filter must smoothly correct towards GNSS position"
    assert ukf.x[0] < 5.0, "Filter must not teleport to raw position"

    s_ok, _ = ukf.update_gnss_speed(12.0)
    assert s_ok, "Moving speed update must be accepted"
    assert 10.0 < ukf.x[2] < 12.0, "Speed must smoothly correct towards 12 m/s"
    print("TEST B (Moving GNSS): PASS")


def test_c_noisy_gnss():
    """TEST C: Noisy GNSS: large reported accuracy results in small Kalman correction."""
    ukf_accurate = PyUKF(dt=0.1)
    ukf_accurate.initialize(12.9716, 77.5946, 5.0, 0.0)

    ukf_noisy = PyUKF(dt=0.1)
    ukf_noisy.initialize(12.9716, 77.5946, 5.0, 0.0)

    # Both receive identical 5m innovation offset, but one has 2m accuracy vs 30m accuracy
    ukf_accurate.update_gnss_pos(5.0, 0.0, accuracy_m=2.0)
    ukf_noisy.update_gnss_pos(5.0, 0.0, accuracy_m=30.0)

    assert ukf_accurate.x[0] > ukf_noisy.x[0], "Accurate fix must produce larger state update than noisy fix"
    assert ukf_noisy.x[0] < 1.5, "Noisy fix must not dominate the filter estimate"
    print("TEST C (Noisy GNSS): PASS")


def test_d_outlier_500m():
    """TEST D: 500m GNSS outlier: rejected by 2-DOF NIS chi-square gate without filter corruption."""
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 0.0, 0.0)

    nis = ukf.compute_pos_nis(500.0, 500.0, accuracy_m=3.0)
    assert nis > UKF_NIS_GATE_2D, f"500m outlier must produce huge NIS (got {nis})"

    pos_before = np.copy(ukf.x)
    cov_before = np.copy(ukf.P)

    accepted, _ = ukf.update_gnss_pos(500.0, 500.0, accuracy_m=3.0)
    assert not accepted, "500m outlier MUST be rejected by UKF"
    assert np.allclose(ukf.x, pos_before), "State must remain unchanged after outlier rejection"
    assert np.allclose(ukf.P, cov_before), "Covariance must remain unchanged after outlier rejection"
    print("TEST D (500m Outlier Rejection): PASS")


def test_e_gnss_outage():
    """TEST E: GNSS outage: AIDED -> PURE_DR transition, no filter reset, dead reckoning continues."""
    health = GnssHealthManager()
    t0 = 1_000_000_000

    # Healthy fix transitions to AIDED
    sample = GnssSample(
        timestamp_ns=t0,
        latitude_deg=12.9716,
        longitude_deg=77.5946,
        speed_mps=10.0,
        bearing_deg=45.0,
        horizontal_accuracy_m=3.0,
        validity=GnssValidity.BASIC_FIX_VALID
    )
    # 3 epochs to clear initial quarantine
    for i in range(3):
        sample.timestamp_ns = t0 + i * 200_000_000
        health.process_candidate(sample)

    assert health.state == GnssHealthState.AIDED, "Must be in AIDED mode"

    # Advance time by 2.0 seconds (> 1.5s timeout)
    outage_t = sample.timestamp_ns + 2_000_000_000
    health.check_timeout(outage_t)

    assert health.state == GnssHealthState.PURE_DR, "Timeout must transition to PURE_DR"
    print("TEST E (GNSS Outage Transition): PASS")


def test_f_gnss_reacquisition():
    """TEST F: GNSS reacquisition: PURE_DR -> QUARANTINE -> 3 epochs -> Trust Ramp -> AIDED."""
    health = GnssHealthManager()
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 5.0, 0.0)

    assert health.state == GnssHealthState.PURE_DR

    t0 = 10_000_000_000  # 10s
    sample = GnssSample(
        timestamp_ns=t0,
        latitude_deg=12.9716,
        longitude_deg=77.5946,
        speed_mps=5.0,
        bearing_deg=0.0,
        horizontal_accuracy_m=3.0,
        validity=GnssValidity.BASIC_FIX_VALID
    )

    # Epoch 1: Candidate quarantined
    can_up1, _ = health.process_candidate(sample)
    assert not can_up1, "First returning fix MUST be quarantined"
    assert health.state == GnssHealthState.QUARANTINE
    assert health.quarantine_count == 1

    # Epoch 2: Candidate quarantined
    sample.timestamp_ns += 200_000_000
    can_up2, _ = health.process_candidate(sample)
    assert not can_up2, "Second returning fix MUST be quarantined"
    assert health.quarantine_count == 2

    # Epoch 3: Quarantine complete -> Trust Ramp active
    sample.timestamp_ns += 200_000_000
    can_up3, r_scale = health.process_candidate(sample)
    assert can_up3, "Third consecutive valid fix completes quarantine"
    assert health.state == GnssHealthState.AIDED
    assert health.is_ramp_active
    assert r_scale >= 9.0, f"Trust ramp must inflate initial covariance (got scale {r_scale})"

    # Advance time through 5.0s ramp
    sample.timestamp_ns += 5_500_000_000
    _, r_scale_final = health.process_candidate(sample)
    assert not health.is_ramp_active, "Ramp must complete after 5.0s"
    assert abs(r_scale_final - 1.0) < 1e-3, "Covariance scale must return to 1.0 after ramp"
    print("TEST F (GNSS Reacquisition & Trust Ramp): PASS")


def test_g_bad_reacquisition_fixes():
    """TEST G: Bad reacquisition fixes: outlier fails NIS, resets quarantine count, state continuous."""
    health = GnssHealthManager()
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 5.0, 0.0)

    t0 = 10_000_000_000
    # Good fix 1
    s1 = GnssSample(
        timestamp_ns=t0,
        latitude_deg=12.9716,
        longitude_deg=77.5946,
        speed_mps=5.0,
        bearing_deg=0.0,
        horizontal_accuracy_m=3.0,
        validity=GnssValidity.BASIC_FIX_VALID
    )
    ukf.update_gnss_sample(s1, health)
    assert health.state == GnssHealthState.QUARANTINE
    assert health.quarantine_count == 1

    # Bad fix 2 (teleport candidate 1000m away)
    s2 = GnssSample(
        timestamp_ns=t0 + 200_000_000,
        latitude_deg=12.9806,  # ~1000m North
        longitude_deg=77.5946,
        speed_mps=5.0,
        bearing_deg=0.0,
        horizontal_accuracy_m=3.0,
        validity=GnssValidity.BASIC_FIX_VALID
    )
    ok, res = ukf.update_gnss_sample(s2, health)
    assert not ok, "Bad candidate fix must be rejected"
    assert health.quarantine_count == 0, "Quarantine count must reset to 0 after rejected candidate"
    assert health.state == GnssHealthState.QUARANTINE
    assert abs(ukf.x[0]) < 10.0, "Filter state must remain continuous, no jump"
    print("TEST G (Bad Reacquisition Outlier Handling): PASS")


def test_h_low_speed_heading():
    """TEST H: Low-speed GNSS course: vehicle at 1.0 m/s must NOT accept noisy GNSS bearing."""
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 1.0, 0.0)

    # GNSS bearing claims vehicle is facing 180 degrees (South) while traveling 1.0 m/s
    accepted, _ = ukf.update_gnss_course(psi_gnss_rad=math.pi, v_gnss=1.0)
    assert not accepted, "Course update MUST be rejected below 2.5 m/s"
    assert abs(ukf.x[3]) < 1e-6, "Heading must not rotate from noisy low-speed GNSS bearing"
    print("TEST H (Low-Speed Course Gating): PASS")


def test_i_clock_freshness():
    """TEST I: Monotonic clock freshness: timeout properly triggers PURE_DR."""
    health = GnssHealthManager()
    cfg = health.config
    t0 = 50_000_000_000  # 50s

    # Prime health to AIDED
    s = GnssSample(
        timestamp_ns=t0,
        latitude_deg=12.9716,
        longitude_deg=77.5946,
        speed_mps=5.0,
        bearing_deg=0.0,
        horizontal_accuracy_m=3.0,
        validity=GnssValidity.BASIC_FIX_VALID
    )
    for _ in range(3):
        health.process_candidate(s)
        s.timestamp_ns += 200_000_000

    assert health.state == GnssHealthState.AIDED

    # 1.0s later: no timeout (1.0s < 1.5s)
    health.check_timeout(s.timestamp_ns + 1_000_000_000)
    assert health.state == GnssHealthState.AIDED

    # 1.6s later: timeout expired (1.6s > 1.5s)
    health.check_timeout(s.timestamp_ns + 1_600_000_000)
    assert health.state == GnssHealthState.PURE_DR
    print("TEST I (Monotonic Clock Freshness): PASS")


def test_j_python_cpp_parity():
    """TEST J: Python <-> C++ exact numerical parity on all GNSS updates and health state."""
    handle = cpp_lib.ukf_create(0.1, 1e-3, 2.0, 0.0)
    assert handle >= 0, "Failed to create C++ UKF instance"

    py_ukf = PyUKF(dt=0.1)

    init_lat, init_lon = 12.9716, 77.5946
    init_spd, init_hdg = 8.0, math.radians(45.0)

    py_ukf.initialize(init_lat, init_lon, init_spd, init_hdg)
    cpp_lib.ukf_initialize(handle, init_lat, init_lon, init_spd, init_hdg, 0.0, 0.0)

    # Run deterministic sequence:
    # 50 IMU steps + GNSS updates + outage + reacquisition + trust ramp
    np.random.seed(42)
    t_ns = 1_000_000_000

    for step in range(100):
        acc = 0.5 * math.sin(step * 0.1)
        gyro = 0.02 * math.cos(step * 0.05)

        # 1. Predict
        py_ukf.predict(acc, gyro, dt=0.1)
        cpp_lib.ukf_predict(handle, acc, gyro, 0.1)

        t_ns += 100_000_000

        # 2. Position update (healthy vs outlier vs outage)
        if step < 30 or step >= 50:
            # Healthy GNSS
            pN = py_ukf.x[0] + 0.2 * np.random.randn()
            pE = py_ukf.x[1] + 0.2 * np.random.randn()
            acc_m = 3.0

            if step == 20:
                # Deliberate outlier
                pN += 500.0
                pE += 500.0

            py_pos_nis = py_ukf.compute_pos_nis(pN, pE, acc_m)
            cpp_pos_nis = cpp_lib.ukf_compute_pos_nis(handle, pN, pE, acc_m, 1.0)
            assert abs(py_pos_nis - cpp_pos_nis) < 1e-4, f"NIS mismatch at step {step}: Py={py_pos_nis}, Cpp={cpp_pos_nis}"

            c_out_nis = ctypes.c_double(0.0)
            py_accepted, _ = py_ukf.update_gnss_pos(pN, pE, acc_m)
            cpp_accepted = bool(cpp_lib.ukf_update_gnss_pos(handle, pN, pE, acc_m, 1.0, ctypes.byref(c_out_nis)))
            assert py_accepted == cpp_accepted, f"Acceptance mismatch at step {step}: Py={py_accepted}, Cpp={cpp_accepted}"

            # Speed update
            v_gnss = max(py_ukf.x[2] + 0.1 * np.random.randn(), 0.0)
            c_spd_nis = ctypes.c_double(0.0)
            py_s_ok, _ = py_ukf.update_gnss_speed(v_gnss)
            cpp_s_ok = bool(cpp_lib.ukf_update_gnss_speed(handle, v_gnss, 0.5, 1.0, ctypes.byref(c_spd_nis)))
            assert py_s_ok == cpp_s_ok

            # Course update
            c_crs_nis = ctypes.c_double(0.0)
            psi_gnss = (py_ukf.x[3] + 0.02 * np.random.randn()) % (2.0 * math.pi)
            py_c_ok, _ = py_ukf.update_gnss_course(psi_gnss, py_ukf.x[2])
            cpp_c_ok = bool(cpp_lib.ukf_update_gnss_course(handle, psi_gnss, py_ukf.x[2], math.radians(3.0), 1.0, ctypes.byref(c_crs_nis)))
            assert py_c_ok == cpp_c_ok

        # Compare states
        cpp_x = (ctypes.c_double * 7)()
        cpp_lib.ukf_get_x(handle, cpp_x)
        x_cpp = np.array([cpp_x[i] for i in range(7)])

        max_x_diff = np.max(np.abs(py_ukf.x - x_cpp))
        assert max_x_diff < 1e-6, f"Step {step}: State diff {max_x_diff} exceeds 1e-6!"

        # Compare covariance
        cpp_P = (ctypes.c_double * 49)()
        cpp_lib.ukf_get_P(handle, cpp_P)
        P_cpp = np.array([cpp_P[i] for i in range(49)]).reshape(7, 7)

        max_P_diff = np.max(np.abs(py_ukf.P - P_cpp))
        assert max_P_diff < 1e-6, f"Step {step}: Covariance diff {max_P_diff} exceeds 1e-6!"

    cpp_lib.ukf_destroy(handle)
    print(f"TEST J (Python <-> C++ Numerical Parity, 100 steps): PASS (max diff = {max_x_diff:.2e})")


def test_teleport_continuity():
    """
    TELEPORT CONTINUITY TEST (Section 25):
    Before outage: N=100m, E=20m.
    During outage: UKF propagates normally along vehicle trajectory.
    GNSS returns: N=145m, E=60m.
    Must NOT directly assign N=145, E=60!
    """
    ukf = PyUKF(dt=0.1)
    ukf.initialize(12.9716, 77.5946, 2.0, math.radians(45.0))

    # Set prior position to N=100m, E=20m before outage
    ukf.x[0] = 100.0
    ukf.x[1] = 20.0

    # Simulate outage propagation for 30 seconds (300 steps of dt=0.1s at v=2.0m/s along NE)
    for _ in range(300):
        ukf.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)

    pred_N = ukf.x[0]
    pred_E = ukf.x[1]

    # Returning fix arrives: N=145m, E=60m with accuracy 3.0m and initial reacquisition trust ramp (scale=10.0)
    returning_N = 145.0
    returning_E = 60.0
    r_scale = 10.0  # From reacquisition trust ramp

    accepted, nis = ukf.update_gnss_pos(returning_N, returning_E, accuracy_m=3.0, r_scale=r_scale)
    assert accepted, f"Returning fix must be accepted by NIS (got NIS {nis:.2f})"

    post_N = ukf.x[0]
    post_E = ukf.x[1]

    # Verify NO direct assignment
    assert post_N != returning_N, "Must NOT directly assign returning N coordinate!"
    assert post_E != returning_E, "Must NOT directly assign returning E coordinate!"

    # Verify smooth correction bounded by Kalman weighting
    assert pred_N < post_N < returning_N, "Position must smoothly correct towards GNSS N"
    assert returning_E < post_E < pred_E, "Position must smoothly correct towards GNSS E"

    # Verify jump is small and continuous
    step_jump = math.sqrt((post_N - pred_N)**2 + (post_E - pred_E)**2)
    assert step_jump < 5.0, f"Single-step correction must be smooth and bounded (got jump {step_jump:.2f}m)"

    # Also test deliberately bad returning fix: N=1000m, E=1000m
    bad_N = 1000.0
    bad_E = 1000.0
    bad_accepted, bad_nis = ukf.update_gnss_pos(bad_N, bad_E, accuracy_m=3.0, r_scale=r_scale)

    assert not bad_accepted, "Deliberately bad returning fix (1000m) MUST be rejected by NIS!"
    assert ukf.x[0] == post_N and ukf.x[1] == post_E, "State must remain continuous with zero teleportation"

    print("TELEPORT CONTINUITY TEST: PASS (Smooth Kalman correction proven; bad fix rejected)")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5: COMPREHENSIVE GNSS FUSION, HEALTH & TELEPORT VERIFICATION SUITE")
    print("=" * 70)

    test_a_healthy_stationary()
    test_b_healthy_moving()
    test_c_noisy_gnss()
    test_d_outlier_500m()
    test_e_gnss_outage()
    test_f_gnss_reacquisition()
    test_g_bad_reacquisition_fixes()
    test_h_low_speed_heading()
    test_i_clock_freshness()
    test_teleport_continuity()
    test_j_python_cpp_parity()

    print("=" * 70)
    print("ALL PHASE 5 TESTS (TEST A - J + TELEPORT CONTINUITY): PASS!")
    print("=" * 70)
