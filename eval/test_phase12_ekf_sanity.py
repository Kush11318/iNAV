"""
Phase 12: Section 17 Implementation Checks (A through H)
Tests the 9-State EKF implementation for:
  A. Numerical Jacobian Verification
  B. GNSS Velocity Update
  C. NHC Innovation under lateral errors
  D. NHC cross-covariance coupling to psi
  E. AI forward velocity update modifying vE/vN
  F. No artificial state clamping
  G. Positive semidefiniteness of covariance P
  H. Synthetic 10s trajectory simulation
"""

import sys
import math
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.ekf_9state import EKF9StateNavigationFilter


def test_check_a_numerical_jacobians():
    print(">>> CHECK A: Numerical Jacobian Verification...")
    eps = 1e-6

    # Test both conventions
    for conv in ["clockwise_from_north", "cartesian_from_east"]:
        ekf = EKF9StateNavigationFilter(dt=0.1, heading_convention=conv)
        ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=20.0, init_heading_deg=35.0)

        # 1. AI Forward Velocity Jacobian
        x0 = ekf.x.copy()
        # Analytical H
        if conv == "clockwise_from_north":
            H_fwd_ana = np.zeros((1, 9))
            H_fwd_ana[0, 3] = math.sin(x0[6])
            H_fwd_ana[0, 4] = math.cos(x0[6])
            H_fwd_ana[0, 6] = x0[3] * math.cos(x0[6]) - x0[4] * math.sin(x0[6])
        else:
            H_fwd_ana = np.zeros((1, 9))
            H_fwd_ana[0, 3] = math.cos(x0[6])
            H_fwd_ana[0, 4] = math.sin(x0[6])
            H_fwd_ana[0, 6] = -x0[3] * math.sin(x0[6]) + x0[4] * math.cos(x0[6])

        # Numerical H via central differences
        H_fwd_num = np.zeros((1, 9))
        for i in range(9):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += eps
            xm[i] -= eps
            ekf.x = xp
            yp = ekf.get_forward_speed()
            ekf.x = xm
            ym = ekf.get_forward_speed()
            H_fwd_num[0, i] = (yp - ym) / (2.0 * eps)
        ekf.x = x0.copy()

        err_fwd = np.max(np.abs(H_fwd_ana - H_fwd_num))
        assert err_fwd < 1e-4, f"[{conv}] AI Velocity Jacobian mismatch: {err_fwd:.2e}"

        # 2. NHC Jacobian
        if conv == "clockwise_from_north":
            H_nhc_ana = np.zeros((2, 9))
            H_nhc_ana[0, 3] = -math.cos(x0[6])
            H_nhc_ana[0, 4] = math.sin(x0[6])
            H_nhc_ana[0, 6] = x0[3] * math.sin(x0[6]) + x0[4] * math.cos(x0[6])
            H_nhc_ana[1, 5] = 1.0
        else:
            H_nhc_ana = np.zeros((2, 9))
            H_nhc_ana[0, 3] = -math.sin(x0[6])
            H_nhc_ana[0, 4] = math.cos(x0[6])
            H_nhc_ana[0, 6] = -math.cos(x0[6]) * x0[3] - math.sin(x0[6]) * x0[4]
            H_nhc_ana[1, 5] = 1.0

        H_nhc_num = np.zeros((2, 9))
        for i in range(9):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += eps
            xm[i] -= eps
            ekf.x = xp
            yp = np.array([ekf.get_lateral_speed(), ekf.x[5]])
            ekf.x = xm
            ym = np.array([ekf.get_lateral_speed(), ekf.x[5]])
            H_nhc_num[:, i] = (yp - ym) / (2.0 * eps)
        ekf.x = x0.copy()

        err_nhc = np.max(np.abs(H_nhc_ana - H_nhc_num))
        assert err_nhc < 1e-4, f"[{conv}] NHC Jacobian mismatch: {err_nhc:.2e}"

    print("  [PASS] Analytical Jacobians match numerical derivatives to < 1e-4 across all conventions.")


def test_check_b_gnss_velocity_update():
    print(">>> CHECK B: GNSS Velocity Update...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=10.0, init_heading_deg=0.0)

    # Initially moving North: vE = 0, vN = 10, vU = 0
    init_vE, init_vN = ekf.x[3], ekf.x[4]

    # Apply GNSS velocity observation: vE = 0.5, vN = 10.8 (within 3-DOF 99% gate)
    accepted, nis = ekf.update_gnss_velocity(vE_gnss=0.5, vN_gnss=10.8, vU_gnss=0.0, sigma_vel=0.3)
    assert accepted, f"GNSS velocity update rejected (NIS: {nis})"

    new_vE, new_vN = ekf.x[3], ekf.x[4]
    assert new_vE > init_vE, "vE did not increase towards GNSS observation"
    assert new_vN > init_vN, "vN did not increase towards GNSS observation"
    print(f"  [PASS] GNSS velocity update shifted state: ({init_vE:.2f}, {init_vN:.2f}) -> ({new_vE:.2f}, {new_vN:.2f})")


def test_check_c_nhc_innovation():
    print(">>> CHECK C: NHC Innovation during Lateral Error...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=20.0, init_heading_deg=0.0)

    # Heading North: vE=0, vN=20. Induce lateral skid: set vE = 5.0 m/s
    ekf.x[3] = 5.0
    v_lat = ekf.get_lateral_speed()
    assert abs(v_lat) > 1.0, f"Lateral speed should be non-zero during skid (got {v_lat})"

    accepted, nis = ekf.update_nhc(sigma_lat=0.5, sigma_vert=0.3)
    assert accepted, "NHC update unexpectedly rejected"
    assert abs(ekf.last_nhc_innov[0]) > 1.0, "NHC innovation should be non-zero"
    print(f"  [PASS] Lateral skid {v_lat:.2f} m/s produced valid NHC innovation {ekf.last_nhc_innov[0]:.2f}")


def test_check_d_nhc_heading_coupling():
    print(">>> CHECK D: NHC Cross-Covariance Modifying Heading (psi)...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=20.0, init_heading_deg=0.0)

    # Propagate through an accelerating turn to build cross-covariance between velocity and heading
    for _ in range(10):
        ekf.predict(acc_fwd=2.0, gyro_yaw=np.radians(10.0), dt=0.1)

    # Check that P has off-diagonal covariance between vE/vN and psi
    cov_vE_psi = ekf.P[3, 6]
    cov_vN_psi = ekf.P[4, 6]
    assert abs(cov_vE_psi) > 1e-6 or abs(cov_vN_psi) > 1e-6, "P has no velocity-heading cross covariance"

    psi_before = ekf.x[6]
    # Induce small lateral skid and apply NHC
    ekf.x[3] += 1.5
    ekf.update_nhc(sigma_lat=0.2, sigma_vert=0.2)
    psi_after = ekf.x[6]

    d_psi_deg = math.degrees(psi_after - psi_before)
    assert abs(d_psi_deg) > 1e-4, "NHC update failed to adjust heading through cross-covariance"
    print(f"  [PASS] NHC update modified heading via cross-covariance by {d_psi_deg:+.4f} deg")


def test_check_e_ai_velocity_update():
    print(">>> CHECK E: AI Forward Velocity Update Modifying vE and vN...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    # Heading North-East (45 deg)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=10.0, init_heading_deg=45.0)

    vE_0, vN_0 = ekf.x[3], ekf.x[4]
    fwd_speed_0 = ekf.get_forward_speed()

    # Apply AI forward speed update of 11.5 m/s (acceleration event within gate)
    accepted, nis = ekf.update_ai_forward_velocity(v_ai=11.5, sigma_ai=0.5)
    assert accepted, f"AI velocity update rejected (NIS: {nis})"

    vE_1, vN_1 = ekf.x[3], ekf.x[4]
    fwd_speed_1 = ekf.get_forward_speed()

    assert fwd_speed_1 > fwd_speed_0, "Forward speed did not increase towards AI measurement"
    assert vE_1 > vE_0, "vE did not increase along 45-deg heading"
    assert vN_1 > vN_0, "vN did not increase along 45-deg heading"
    print(f"  [PASS] AI speed update shifted forward speed: {fwd_speed_0:.2f} -> {fwd_speed_1:.2f} m/s (vE: {vE_0:.2f}->{vE_1:.2f}, vN: {vN_0:.2f}->{vN_1:.2f})")


def test_check_f_no_state_clamping():
    print(">>> CHECK F: Verify No State Clamping (except 2pi angle wrap)...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=10.0, init_heading_deg=0.0)

    # Let the velocity grow or turn without hard artificial upper bounds
    ekf.x[3] = 120.0  # 120 m/s
    ekf.predict(acc_fwd=10.0, gyro_yaw=0.0, dt=0.1)
    assert ekf.x[3] >= 120.0, "State was artificially clamped!"
    print("  [PASS] No hard artificial clamps found on velocity or position.")


def test_check_g_positive_semidefinite_covariance():
    print(">>> CHECK G: Positive Semidefiniteness of Covariance P...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=15.0, init_heading_deg=30.0)

    # Run 50 steps of predictions and alternating updates
    for k in range(50):
        ekf.predict(acc_fwd=1.0 * math.sin(k * 0.1), gyro_yaw=0.05 * math.cos(k * 0.1), dt=0.1)
        if k % 2 == 0:
            ekf.update_nhc(sigma_lat=0.5, sigma_vert=0.3)
        if k % 5 == 0:
            ekf.update_ai_forward_velocity(v_ai=15.0 + k * 0.1, sigma_ai=0.5)

        # Symmetry check
        sym_err = np.max(np.abs(ekf.P - ekf.P.T))
        assert sym_err < 1e-7, f"Covariance P is not symmetric (error: {sym_err})"

        # Eigenvalue check
        eigvals = np.linalg.eigvalsh(ekf.P)
        min_eig = np.min(eigvals)
        assert min_eig > 0.0, f"Covariance P has non-positive eigenvalue: {min_eig}"

    print(f"  [PASS] Covariance P remained strictly symmetric and positive definite (min eig: {min_eig:.2e})")


def test_check_h_synthetic_sanity_outage():
    print(">>> CHECK H: Synthetic 10s Outage Sanity Test...")
    ekf = EKF9StateNavigationFilter(dt=0.1)
    ekf.initialize(init_lat=52.0, init_lon=0.0, init_speed_ms=20.0, init_heading_deg=90.0)  # Heading East

    steps = 100  # 10s @ 10Hz
    for k in range(steps):
        ekf.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        # Apply NHC at 10Hz
        ekf.update_nhc(sigma_lat=0.5, sigma_vert=0.3)
        # Apply AI velocity at 2Hz (every 5 steps)
        if k % 5 == 0:
            ekf.update_ai_forward_velocity(v_ai=20.0, sigma_ai=0.4)

    # After 10s at 20 m/s heading East, should be ~200m East, ~0m North
    pE = ekf.x[0]
    pN = ekf.x[1]
    fwd_spd = ekf.get_forward_speed()
    assert abs(pE - 200.0) < 5.0, f"Position East deviated significantly: {pE:.2f}m (expected ~200m)"
    assert abs(pN) < 2.0, f"Position North drifted on straight East track: {pN:.2f}m (expected ~0m)"
    assert abs(fwd_spd - 20.0) < 1.0, f"Forward speed deviated: {fwd_spd:.2f} m/s"

    print(f"  [PASS] 10s synthetic outage completed: pE={pE:.2f}m (exp ~200m), pN={pN:.2f}m, speed={fwd_spd:.2f}m/s")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PHASE 12 SECTION 17 IMPLEMENTATION CHECKS (A-H)")
    print("=" * 80)
    test_check_a_numerical_jacobians()
    test_check_b_gnss_velocity_update()
    test_check_c_nhc_innovation()
    test_check_d_nhc_heading_coupling()
    test_check_e_ai_velocity_update()
    test_check_f_no_state_clamping()
    test_check_g_positive_semidefinite_covariance()
    test_check_h_synthetic_sanity_outage()
    print("=" * 80)
    print("ALL SECTION 17 CHECKS PASSED PERFECTLY!")
    print("=" * 80)
