"""
Synthetic Physics & Numerical Stability Test Suite for C++ 7-State UKF
Re-running Tests A-F from Phase 4 Audit plus Tests G-H:
Verifies both C++ UKF physical behavior and consistency with Python reference.
"""

import math
import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.test_parity_ukf import CppUKFWrapper
from modules.ukf import UKFNavigationFilter as PyUKF


def run_synthetic_tests():
    print("=" * 80)
    print("PHASE 4: SYNTHETIC PHYSICS & NUMERICAL STABILITY SUITE (C++ 7-STATE UKF)")
    print("=" * 80)

    results = {}

    # TEST A: Straight Constant Motion
    # 10 m/s for 10 seconds (100 steps of dt=0.1) -> Expected displacement ~ 100.0 m North
    cpp_a = CppUKFWrapper(dt=0.1)
    cpp_a.initialize(0.0, 0.0, 10.0, 0.0)
    py_a = PyUKF(dt=0.1)
    py_a.initialize(0.0, 0.0, 10.0, 0.0)

    for _ in range(100):
        cpp_a.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        py_a.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)

    x_cpp_a = cpp_a.get_x()
    diff_a = abs(x_cpp_a[0] - py_a.x[0])
    pass_a = abs(x_cpp_a[0] - 100.0) < 1.0 and abs(x_cpp_a[1]) < 1e-3 and diff_a < 1e-6
    results["Test A (Straight Motion)"] = pass_a
    print(f"TEST A (Straight 10m/s, 10s):")
    print(f"  C++   : p_N={x_cpp_a[0]:.4f}m (exp 100.0), p_E={x_cpp_a[1]:.4f}m (exp 0.0)")
    print(f"  Python: p_N={py_a.x[0]:.4f}m, p_E={py_a.x[1]:.4f}m")
    print(f"  Diff  : {diff_a:.2e}m [{'PASS' if pass_a else 'FAIL'}]")

    # TEST B: Constant-Rate Turn
    # Speed = 10 m/s, yaw_rate = 0.1 rad/s for 15.70796s (90 deg turn, quarter circle)
    # Radius R = v / omega = 10 / 0.1 = 100 m.
    cpp_b = CppUKFWrapper(dt=0.05)
    cpp_b.initialize(0.0, 0.0, 10.0, 0.0)
    py_b = PyUKF(dt=0.05)
    py_b.initialize(0.0, 0.0, 10.0, 0.0)
    steps_b = int(round((math.pi / 2.0 / 0.1) / 0.05))  # 314 steps

    for _ in range(steps_b):
        cpp_b.predict(acc_fwd=0.0, gyro_yaw=0.1, dt=0.05)
        py_b.predict(acc_fwd=0.0, gyro_yaw=0.1, dt=0.05)

    x_cpp_b = cpp_b.get_x()
    diff_b = math.hypot(x_cpp_b[0] - py_b.x[0], x_cpp_b[1] - py_b.x[1])
    pass_b = abs(x_cpp_b[0] - 100.0) < 1.0 and abs(x_cpp_b[1] - 100.0) < 1.0 and diff_b < 1e-6
    results["Test B (Constant Turn)"] = pass_b
    print(f"TEST B (Quarter Circle R=100m):")
    print(f"  C++   : p_N={x_cpp_b[0]:.2f}m, p_E={x_cpp_b[1]:.2f}m, psi={math.degrees(x_cpp_b[3]):.2f}deg")
    print(f"  Python: p_N={py_b.x[0]:.2f}m, p_E={py_b.x[1]:.2f}m, psi={math.degrees(py_b.x[3]):.2f}deg")
    print(f"  Diff  : {diff_b:.2e}m [{'PASS' if pass_b else 'FAIL'}]")

    # TEST C: Pure Stationary
    cpp_c = CppUKFWrapper(dt=0.1)
    cpp_c.initialize(0.0, 0.0, 0.0, 0.5)
    py_c = PyUKF(dt=0.1)
    py_c.initialize(0.0, 0.0, 0.0, 0.5)

    for _ in range(50):
        cpp_c.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        cpp_c.update_zupt(0.0)
        py_c.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        py_c.update_zupt(0.0)

    x_cpp_c = cpp_c.get_x()
    diff_c = math.hypot(x_cpp_c[0] - py_c.x[0], x_cpp_c[1] - py_c.x[1])
    drift_cpp = math.hypot(x_cpp_c[0], x_cpp_c[1])
    pass_c = x_cpp_c[2] < 1e-6 and drift_cpp < 0.02 and diff_c < 1e-4
    results["Test C (Stationary)"] = pass_c
    print(f"TEST C (Pure Stationary 5s):")
    print(f"  C++   : PosDrift={drift_cpp:.6f}m, Speed={x_cpp_c[2]:.6e}m/s")
    print(f"  Python: PosDrift={math.hypot(py_c.x[0], py_c.x[1]):.6f}m, Speed={py_c.x[2]:.6e}m/s")
    print(f"  Diff  : {diff_c:.2e}m [{'PASS' if pass_c else 'FAIL'}]")

    # TEST D: Heading Wrap Across 0 / 2pi
    cpp_d = CppUKFWrapper(dt=0.1)
    py_d = PyUKF(dt=0.1)
    psi_start = math.radians(359.0)
    cpp_d.initialize(0.0, 0.0, 5.0, psi_start)
    py_d.initialize(0.0, 0.0, 5.0, psi_start)
    w_turn = math.radians(20.0)  # +2 deg over 0.1s -> 1 deg

    cpp_d.predict(acc_fwd=0.0, gyro_yaw=w_turn, dt=0.1)
    py_d.predict(acc_fwd=0.0, gyro_yaw=w_turn, dt=0.1)

    x_cpp_d = cpp_d.get_x()
    diff_d = abs(x_cpp_d[3] - py_d.x[3])
    pass_d = abs(math.degrees(x_cpp_d[3]) - 1.0) < 1e-3 and diff_d < 1e-6
    results["Test D (Heading Wrap)"] = pass_d
    print(f"TEST D (Heading Wrap 359 -> 1 deg):")
    print(f"  C++   : psi={math.degrees(x_cpp_d[3]):.4f} deg (exp 1.0 deg)")
    print(f"  Python: psi={math.degrees(py_d.x[3]):.4f} deg")
    print(f"  Diff  : {diff_d:.2e} rad [{'PASS' if pass_d else 'FAIL'}]")

    # TEST E: VelocityNet Measurement Update
    cpp_e = CppUKFWrapper(dt=0.1)
    cpp_e.initialize(0.0, 0.0, 5.0, 0.0)
    py_e = PyUKF(dt=0.1)
    py_e.initialize(0.0, 0.0, 5.0, 0.0)

    cpp_e.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
    py_e.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)

    cpp_e.update_velocity_net(delta_d=30.0, sigma=0.1, event_class=1, window_dur=2.0)  # 15 m/s
    py_e.update_velocity_net(delta_d_pred=30.0, sigma_pred=0.1, event_class=1, window_dur=2.0)

    x_cpp_e = cpp_e.get_x()
    diff_e = abs(x_cpp_e[2] - py_e.x[2])
    pass_e = x_cpp_e[2] > 14.0 and diff_e < 1e-6
    results["Test E (VelocityNet Update)"] = pass_e
    print(f"TEST E (VelocityNet Update):")
    print(f"  C++   : v_updated={x_cpp_e[2]:.4f} m/s (prior 5.0 m/s, meas 15.0 m/s)")
    print(f"  Python: v_updated={py_e.x[2]:.4f} m/s")
    print(f"  Diff  : {diff_e:.2e} m/s [{'PASS' if pass_e else 'FAIL'}]")

    # TEST F: Covariance Sanity (500 steps)
    cpp_f = CppUKFWrapper(dt=0.1)
    cpp_f.initialize(0.0, 0.0, 10.0, 0.0)
    py_f = PyUKF(dt=0.1)
    py_f.initialize(0.0, 0.0, 10.0, 0.0)

    for k in range(500):
        acc = 0.1 * math.sin(k * 0.1)
        gyr = 0.05 * math.cos(k * 0.1)
        cpp_f.predict(acc_fwd=acc, gyro_yaw=gyr, dt=0.1)
        py_f.predict(acc_fwd=acc, gyro_yaw=gyr, dt=0.1)
        if k % 10 == 0:
            cpp_f.update_velocity_net(delta_d=20.0, sigma=0.5, event_class=1, window_dur=2.0)
            py_f.update_velocity_net(delta_d_pred=20.0, sigma_pred=0.5, event_class=1, window_dur=2.0)

    P_cpp_f = cpp_f.get_P()
    sym_err = np.max(np.abs(P_cpp_f - P_cpp_f.T))
    min_eig = np.min(np.linalg.eigvalsh(P_cpp_f))
    cond_num = np.max(np.linalg.eigvalsh(P_cpp_f)) / min_eig
    pass_f = sym_err < 1e-10 and min_eig > 0.0 and np.all(np.isfinite(P_cpp_f))
    results["Test F (Covariance Stability)"] = pass_f
    print(f"TEST F (Covariance Stability 500 steps):")
    print(f"  C++ Symmetry Error    : {sym_err:.2e}")
    print(f"  C++ Minimum Eigenvalue: {min_eig:.4e} (> 0.0)")
    print(f"  C++ Condition Number  : {cond_num:.2e} [{'PASS' if pass_f else 'FAIL'}]")

    # TEST G: Gyro Residual Bias Contract (no double subtraction)
    # Phase 3 Alignment removes static bias. Initial UKF bg = 0.
    # Residual bias in run: 0.003 rad/s.
    cpp_g = CppUKFWrapper(dt=0.1)
    cpp_g.initialize(0.0, 0.0, 0.0, 0.0, gyro_bias=0.0, accel_bias=0.0)
    py_g = PyUKF(dt=0.1)
    py_g.initialize(0.0, 0.0, 0.0, 0.0, init_gyro_bias=0.0, init_accel_bias=0.0)

    res_bias = 0.003
    for _ in range(100):
        cpp_g.predict(acc_fwd=0.0, gyro_yaw=res_bias, dt=0.1)
        cpp_g.update_zupt(gyro_reading=res_bias)
        py_g.predict(acc_fwd=0.0, gyro_yaw=res_bias, dt=0.1)
        py_g.update_zupt(gyro_reading=res_bias)

    x_cpp_g = cpp_g.get_x()
    diff_g = abs(x_cpp_g[4] - py_g.x[4])
    pass_g = diff_g < 1e-6 and (0.0 < x_cpp_g[4] <= res_bias)
    results["Test G (Gyro Residual Bias)"] = pass_g
    print(f"TEST G (Gyro Residual Bias Tracking, 100 ZUPTs):")
    print(f"  True residual rate: {res_bias:.4f} rad/s")
    print(f"  C++ estimated bg  : {x_cpp_g[4]:.5f} rad/s")
    print(f"  Python estimated  : {py_g.x[4]:.5f} rad/s")
    print(f"  Diff              : {diff_g:.2e} rad/s [{'PASS' if pass_g else 'FAIL'}]")

    # TEST H: Accelerometer Bias Consistency
    cpp_h = CppUKFWrapper(dt=0.1)
    cpp_h.initialize(0.0, 0.0, 10.0, 0.0, gyro_bias=0.0, accel_bias=0.0)
    py_h = PyUKF(dt=0.1)
    py_h.initialize(0.0, 0.0, 10.0, 0.0, init_gyro_bias=0.0, init_accel_bias=0.0)

    for k in range(100):
        cpp_h.predict(acc_fwd=0.05, gyro_yaw=0.0, dt=0.1)
        py_h.predict(acc_fwd=0.05, gyro_yaw=0.0, dt=0.1)
        if k % 20 == 0 and k > 0:
            cpp_h.update_velocity_net(delta_d=20.0, sigma=0.2, event_class=1, window_dur=2.0)
            py_h.update_velocity_net(delta_d_pred=20.0, sigma_pred=0.2, event_class=1, window_dur=2.0)

    x_cpp_h = cpp_h.get_x()
    diff_h = np.abs(x_cpp_h - py_h.x)
    # Angle wrap diff for heading
    diff_h[3] = abs((x_cpp_h[3] - py_h.x[3] + math.pi) % (2.0 * math.pi) - math.pi)
    max_diff_h = np.max(diff_h)
    pass_h = max_diff_h < 1e-6 and np.all(np.isfinite(x_cpp_h)) and x_cpp_h[2] >= 0.0
    results["Test H (Accel Bias Consistency)"] = pass_h
    print(f"TEST H (Accel Bias Consistency):")
    print(f"  C++   : Speed={x_cpp_h[2]:.2f}m/s, ba={x_cpp_h[5]:.4f}m/s^2")
    print(f"  Python: Speed={py_h.x[2]:.2f}m/s, ba={py_h.x[5]:.4f}m/s^2")
    print(f"  Diff  : {max_diff_h:.2e} [{'PASS' if pass_h else 'FAIL'}]")

    print("\n" + "=" * 80)
    all_pass = all(results.values())
    for name, status in results.items():
        print(f"  {name:35s}: {'PASS' if status else 'FAIL'}")
    print("=" * 80)
    print(f"OVERALL SYNTHETIC TESTS (A-H) VERDICT: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)
    assert all_pass, "Synthetic physics tests failed!"


if __name__ == "__main__":
    run_synthetic_tests()
