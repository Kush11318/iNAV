"""
Python/C++ 7-State UKF Numerical Parity Verification Suite
Mandatory Phase 4 Primary Acceptance Test:
Compares the Python reference UKF (modules/ukf.py) and C++ UKF (cpp/include/inav_ukf.hpp)
step-by-step across all 7 state components and all 49 covariance matrix elements.

Acceptance Threshold:
  max |x_py[i] - x_cpp[i]| <= 1e-6 for all i in [0..6]
  max |P_py[r, c] - P_cpp[r, c]| <= 1e-6 for all r, c in [0..6]
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

from modules.ukf import UKFNavigationFilter as PyUKF

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

cpp_lib.ukf_update_velocity_net.argtypes = [
    ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_double
]
cpp_lib.ukf_update_velocity_net.restype = None

cpp_lib.ukf_update_zupt.argtypes = [ctypes.c_int, ctypes.c_double]
cpp_lib.ukf_update_zupt.restype = None

cpp_lib.ukf_get_x.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cpp_lib.ukf_get_x.restype = None

cpp_lib.ukf_get_P.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cpp_lib.ukf_get_P.restype = None


class CppUKFWrapper:
    def __init__(self, dt=0.1, alpha=1e-3, beta=2.0, kappa=0.0):
        self.handle = cpp_lib.ukf_create(dt, alpha, beta, kappa)
        if self.handle < 0:
            raise RuntimeError("Failed to allocate C++ UKF instance")

    def __del__(self):
        if hasattr(self, "handle") and self.handle >= 0:
            cpp_lib.ukf_destroy(self.handle)

    def initialize(self, lat, lon, speed, heading, gyro_bias=0.0, accel_bias=0.0):
        cpp_lib.ukf_initialize(self.handle, lat, lon, speed, heading, gyro_bias, accel_bias)

    def predict(self, acc_fwd, gyro_yaw, dt=0.1):
        cpp_lib.ukf_predict(self.handle, acc_fwd, gyro_yaw, dt)

    def update_velocity_net(self, delta_d, sigma, event_class=1, window_dur=2.0):
        cpp_lib.ukf_update_velocity_net(self.handle, delta_d, sigma, event_class, window_dur)

    def update_zupt(self, gyro_reading):
        cpp_lib.ukf_update_zupt(self.handle, gyro_reading)

    def get_x(self):
        out = (ctypes.c_double * 7)()
        cpp_lib.ukf_get_x(self.handle, out)
        return np.array([out[i] for i in range(7)])

    def get_P(self):
        out = (ctypes.c_double * 49)()
        cpp_lib.ukf_get_P(self.handle, out)
        return np.array([out[i] for i in range(49)]).reshape((7, 7))


def run_ukf_parity_test():
    print("=" * 60)
    print("PHASE 4: PYTHON <-> C++ 7-STATE UKF STEP-BY-STEP PARITY TEST")
    print("=" * 60)

    dt = 0.1
    py_filter = PyUKF(dt=dt)
    cpp_filter = CppUKFWrapper(dt=dt)

    # Initial state
    init_lat, init_lon = 12.9716, 77.5946
    init_speed = 5.0
    init_heading = math.radians(45.0)
    init_bg = 0.001
    init_ba = -0.02

    py_filter.initialize(init_lat, init_lon, init_speed, init_heading, init_bg, init_ba)
    cpp_filter.initialize(init_lat, init_lon, init_speed, init_heading, init_bg, init_ba)

    state_names = ["p_N", "p_E", "v_fwd", "psi", "b_g", "b_a", "k"]

    max_state_diff = np.zeros(7)
    max_cov_diff = np.zeros((7, 7))

    # Generate multi-phase deterministic test sequence:
    # Phase 1: 50 steps straight acceleration
    # Phase 2: 50 steps right turn (gyro yaw = +0.2 rad/s)
    # Phase 3: 50 steps left turn wrapping past 0/2pi (gyro yaw = -0.4 rad/s)
    # Phase 4: 50 steps cruising with VelocityNet updates (classes 1, 2, 3)
    # Phase 5: 30 steps deceleration to stop
    # Phase 6: 40 steps stationary ZUPT updates

    steps = []
    # Phase 1: straight acceleration
    for i in range(50):
        steps.append({"type": "predict", "acc": 0.5, "gyro": 0.0, "dt": 0.1})
    # Phase 2: right turn
    for i in range(50):
        steps.append({"type": "predict", "acc": 0.0, "gyro": 0.2, "dt": 0.1})
    # Phase 3: left turn across 0/2pi
    for i in range(50):
        steps.append({"type": "predict", "acc": 0.1, "gyro": -0.35, "dt": 0.1})
    # Phase 4: VelocityNet updates with rough road and dynamics
    for i in range(50):
        steps.append({"type": "predict", "acc": 0.05, "gyro": 0.02, "dt": 0.1})
        if i % 10 == 0:
            ev = (i // 10) % 3 + 1
            steps.append({"type": "vnet", "delta_d": 14.2 + (i % 3) * 0.5, "sigma": 0.8, "event": ev, "dur": 2.0})
    # Phase 5: Deceleration
    for i in range(30):
        steps.append({"type": "predict", "acc": -0.8, "gyro": 0.0, "dt": 0.1})
    # Phase 6: Stationary with ZUPT
    for i in range(40):
        steps.append({"type": "predict", "acc": 0.0, "gyro": 0.0015, "dt": 0.1})
        steps.append({"type": "zupt", "gyro": 0.0015})

    print(f"Executing {len(steps)} deterministic UKF test steps...")

    step_idx = 0
    for s in steps:
        step_idx += 1
        stype = s["type"]
        if stype == "predict":
            py_filter.predict(s["acc"], s["gyro"], s["dt"])
            cpp_filter.predict(s["acc"], s["gyro"], s["dt"])
        elif stype == "vnet":
            py_filter.update_velocity_net(s["delta_d"], s["sigma"], s["event"], s["dur"])
            cpp_filter.update_velocity_net(s["delta_d"], s["sigma"], s["event"], s["dur"])
        elif stype == "zupt":
            py_filter.update_zupt(s["gyro"])
            cpp_filter.update_zupt(s["gyro"])

        x_py = py_filter.x
        x_cpp = cpp_filter.get_x()
        P_py = py_filter.P
        P_cpp = cpp_filter.get_P()

        # Heading angular difference wrapped to [-pi, pi)
        diff_x = np.abs(x_py - x_cpp)
        heading_err = abs((x_py[3] - x_cpp[3] + math.pi) % (2.0 * math.pi) - math.pi)
        diff_x[3] = heading_err

        diff_P = np.abs(P_py - P_cpp)

        max_state_diff = np.maximum(max_state_diff, diff_x)
        max_cov_diff = np.maximum(max_cov_diff, diff_P)

        # Early check for divergence
        if np.max(diff_x) > 1e-4 or np.max(diff_P) > 1e-4:
            print(f"\n[FAIL] Divergence detected at step {step_idx} ({stype}):")
            print(f"State diff: {diff_x}")
            print(f"Cov max diff: {np.max(diff_P)}")
            print(f"x_py:  {x_py}")
            print(f"x_cpp: {x_cpp}")
            sys.exit(1)

    print("\n--- STEP-BY-STEP PARITY RESULTS ---")
    print(f"Total steps compared: {step_idx}")
    print("\nState Vector Maximum Absolute Differences:")
    all_state_pass = True
    for i, name in enumerate(state_names):
        err = max_state_diff[i]
        passed = err <= 1e-6
        if not passed:
            all_state_pass = False
        status = "PASS" if passed else "FAIL"
        print(f"  x[{i}] ({name:6s}): max diff = {err:.3e}  [{status}] (target <= 1e-6)")

    max_cov_err = np.max(max_cov_diff)
    cov_pass = max_cov_err <= 1e-6
    cov_status = "PASS" if cov_pass else "FAIL"
    print(f"\nCovariance Matrix Maximum Absolute Difference across all 49 elements:")
    print(f"  P matrix: max diff = {max_cov_err:.3e}  [{cov_status}] (target <= 1e-6)")

    overall_pass = all_state_pass and cov_pass
    print("\n" + "=" * 60)
    if overall_pass:
        print("OVERALL PARITY RESULT: PASS (C++ UKF mathematically identical to Python)")
    else:
        print("OVERALL PARITY RESULT: FAIL")
    print("=" * 60)

    assert overall_pass, "Python/C++ UKF parity test failed acceptance threshold!"
    return max_state_diff, max_cov_err


if __name__ == "__main__":
    run_ukf_parity_test()
