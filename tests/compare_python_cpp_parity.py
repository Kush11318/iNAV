import math
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.ukf import UKFNavigationFilter
from eval.replay import run_inav_cpp_kinematic_pipeline
import pandas as pd

def compare_python_vs_cpp():
    print("=" * 80)
    print("PHASE 4 AUDIT: PYTHON UKF vs C++ KINEMATIC INTEGRATOR PARITY COMPARISON")
    print("=" * 80)

    # Synthetic driving sequence:
    # 0 - 2s: stationary (ZUPT)
    # 2 - 10s: accelerate and drive forward at 10 m/s
    # 10 - 15s: turn right at 0.1 rad/s (90 deg turn)
    # 15 - 20s: drive straight
    # 20 - 25s: brake to stop
    dt = 0.1
    t = np.arange(0.0, 25.0, dt)
    n = len(t)

    # Synthetic speed profile
    true_speed = np.zeros(n)
    true_yaw_rate = np.zeros(n)

    for i, ti in enumerate(t):
        if 2.0 <= ti < 10.0:
            true_speed[i] = 10.0
        elif 10.0 <= ti < 15.0:
            true_speed[i] = 10.0
            true_yaw_rate[i] = (math.pi / 2.0) / 5.0 # 0.314 rad/s (90 deg in 5s)
        elif 15.0 <= ti < 20.0:
            true_speed[i] = 10.0
        elif 20.0 <= ti <= 25.0:
            true_speed[i] = max(0.0, 10.0 - 2.0 * (ti - 20.0))

    # Run Python UKF
    py_ukf = UKFNavigationFilter(dt=dt)
    py_ukf.initialize(0.0, 0.0, init_speed_ms=0.0, init_heading_rad=0.0)

    py_pN = np.zeros(n)
    py_pE = np.zeros(n)
    py_spd = np.zeros(n)
    py_hdg = np.zeros(n)

    for i in range(n):
        sp = true_speed[i]
        wz = true_yaw_rate[i]

        py_ukf.predict(acc_fwd=0.0, gyro_yaw=wz, dt=dt)

        if sp == 0.0:
            py_ukf.update_zupt(wz)
        else:
            # VelocityNet simulated measurement every 0.5s (5 steps)
            if i % 5 == 0:
                py_ukf.update_velocity_net(delta_d_pred=sp * 2.0, sigma_pred=0.2, event_class=1, window_dur=2.0)

        py_pN[i] = py_ukf.x[0]
        py_pE[i] = py_ukf.x[1]
        py_spd[i] = py_ukf.x[2]
        py_hdg[i] = py_ukf.x[3]

    # Run C++ Kinematic Model (replicated directly in Python from inav_filter.hpp)
    cpp_pN = np.zeros(n)
    cpp_pE = np.zeros(n)
    cpp_spd = np.zeros(n)
    cpp_hdg = np.zeros(n)

    v_c = 0.0
    psi_c = 0.0
    pN_c = 0.0
    pE_c = 0.0
    bg_c = 0.0

    for i in range(n):
        sp = true_speed[i]
        wz = true_yaw_rate[i]

        # C++ predict
        omega_corr = wz - bg_c
        psi_c = (psi_c + omega_corr * dt) % (2.0 * math.pi)

        if sp == 0.0:
            v_c = 0.0
            bg_c = 0.98 * bg_c + 0.02 * wz
        else:
            if i % 5 == 0:
                v_net = (sp * 2.0) / 2.0
                base_var = (0.2 / 2.0)**2
                R_meas = max(base_var, 0.04)
                P_vv = 0.25
                K = P_vv / (P_vv + R_meas)
                v_c = max(0.0, v_c + K * (v_net - v_c))

        disp = v_c * dt
        pN_c += disp * math.cos(psi_c)
        pE_c += disp * math.sin(psi_c)

        cpp_pN[i] = pN_c
        cpp_pE[i] = pE_c
        cpp_spd[i] = v_c
        cpp_hdg[i] = psi_c

    diff_pos = np.hypot(py_pN - cpp_pN, py_pE - cpp_pE)
    diff_spd = np.abs(py_spd - cpp_spd)
    diff_hdg = np.abs(np.arctan2(np.sin(py_hdg - cpp_hdg), np.cos(py_hdg - cpp_hdg)))

    print(f"Final Position (Python UKF) : p_N = {py_pN[-1]:.2f} m, p_E = {py_pE[-1]:.2f} m")
    print(f"Final Position (C++ Integr.): p_N = {cpp_pN[-1]:.2f} m, p_E = {cpp_pE[-1]:.2f} m")
    print(f"Final Position Difference   : {diff_pos[-1]:.2f} m")
    print(f"Max Position Divergence     : {np.max(diff_pos):.2f} m")
    print(f"Max Speed Divergence        : {np.max(diff_spd):.2f} m/s")
    print(f"Max Heading Divergence      : {math.degrees(np.max(diff_hdg)):.2f} deg")

    print("-" * 80)
    print("PARITY VERDICT: ALGORITHMICALLY INCOMPATIBLE")
    print("- Python executes a true 7-state UKF with joint (p, v, psi, bg, ba, k) covariance.")
    print("- C++ executes a deterministic kinematic integrator with zero covariance matrix.")
    print("=" * 80)

if __name__ == "__main__":
    compare_python_vs_cpp()
