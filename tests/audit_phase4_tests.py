import math
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.ukf import UKFNavigationFilter

def run_synthetic_tests():
    print("=" * 80)
    print("PHASE 4 AUDIT: SYNTHETIC MATHEMATICAL TESTS (PYTHON UKF)")
    print("=" * 80)

    # TEST A: Straight Constant Motion
    # 10 m/s for 10 seconds (100 steps of dt=0.1) -> Expected displacement = 100.0 m North
    ukf_a = UKFNavigationFilter(dt=0.1)
    ukf_a.initialize(0.0, 0.0, init_speed_ms=10.0, init_heading_rad=0.0)
    for _ in range(100):
        ukf_a.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
    
    err_a_N = abs(ukf_a.x[0] - 100.0)
    err_a_E = abs(ukf_a.x[1] - 0.0)
    print(f"TEST A (Straight 10m/s, 10s): p_N={ukf_a.x[0]:.3f}m (exp 100.0), p_E={ukf_a.x[1]:.3f}m (exp 0.0) | Error: {err_a_N:.4f}m")

    # TEST B: Constant-Rate Turn
    # Speed = 10 m/s, yaw_rate = 0.1 rad/s for 15.70796s (90 deg turn, quarter circle)
    # Radius R = v / omega = 10 / 0.1 = 100 m.
    # Expected final position: p_N = R * sin(pi/2) = 100.0 m, p_E = R * (1 - cos(pi/2)) = 100.0 m
    ukf_b = UKFNavigationFilter(dt=0.05)
    ukf_b.initialize(0.0, 0.0, init_speed_ms=10.0, init_heading_rad=0.0)
    steps_b = int(round((math.pi / 2.0 / 0.1) / 0.05)) # 314 steps
    for _ in range(steps_b):
        ukf_b.predict(acc_fwd=0.0, gyro_yaw=0.1, dt=0.05)
    
    err_b_N = abs(ukf_b.x[0] - 100.0)
    err_b_E = abs(ukf_b.x[1] - 100.0)
    err_b_psi = abs(ukf_b.x[3] - math.pi / 2.0)
    print(f"TEST B (Quarter Circle R=100m): p_N={ukf_b.x[0]:.2f}m (exp 100), p_E={ukf_b.x[1]:.2f}m (exp 100), psi={math.degrees(ukf_b.x[3]):.1f}deg (exp 90) | PosErr: {math.hypot(err_b_N, err_b_E):.3f}m")

    # TEST C: Pure Stationary
    # At rest, speed = 0, gyro = 0 for 50 steps
    ukf_c = UKFNavigationFilter(dt=0.1)
    ukf_c.initialize(0.0, 0.0, init_speed_ms=0.0, init_heading_rad=0.5)
    for _ in range(50):
        ukf_c.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        ukf_c.update_zupt(0.0)
    
    pos_drift_c = math.hypot(ukf_c.x[0], ukf_c.x[1])
    print(f"TEST C (Pure Stationary 5s): Drift={pos_drift_c:.6f}m, Speed={ukf_c.x[2]:.6f}m/s")

    # TEST D: Heading Wrap Across 0 / 2pi
    # Start at psi = 359 deg (6.2657 rad), rotate +2 deg (0.0349 rad)
    # Expected = 1 deg (0.01745 rad)
    ukf_d = UKFNavigationFilter(dt=0.1)
    psi_start = math.radians(359.0)
    ukf_d.initialize(0.0, 0.0, init_speed_ms=5.0, init_heading_rad=psi_start)
    w_turn = math.radians(20.0) # 20 deg/s for 0.1s = +2 deg
    ukf_d.predict(acc_fwd=0.0, gyro_yaw=w_turn, dt=0.1)
    
    exp_psi = math.radians(1.0)
    err_psi = abs(ukf_d.x[3] - exp_psi)
    print(f"TEST D (Heading Wrap 359 -> 1 deg): Result={math.degrees(ukf_d.x[3]):.2f} deg (exp 1.00 deg) | Error: {math.degrees(err_psi):.2e} deg")

    # TEST E: Measurement Update (VelocityNet)
    # Speed is initially 5.0 m/s. Measurement arrives: v = 15.0 m/s with small noise
    ukf_e = UKFNavigationFilter(dt=0.1)
    ukf_e.initialize(0.0, 0.0, init_speed_ms=5.0, init_heading_rad=0.0)
    ukf_e.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
    v_before = ukf_e.x[2]
    ukf_e.update_velocity_net(delta_d_pred=30.0, sigma_pred=0.1, event_class=1, window_dur=2.0) # 15 m/s
    v_after = ukf_e.x[2]
    print(f"TEST E (VelocityNet Update): v_init={v_before:.2f} m/s, Meas=15.0 m/s -> Updated v={v_after:.2f} m/s")

    # TEST F: Covariance Sanity (500 steps)
    ukf_f = UKFNavigationFilter(dt=0.1)
    ukf_f.initialize(0.0, 0.0, init_speed_ms=10.0, init_heading_rad=0.0)
    for k in range(500):
        ukf_f.predict(acc_fwd=0.1 * math.sin(k*0.1), gyro_yaw=0.05 * math.cos(k*0.1), dt=0.1)
        if k % 10 == 0:
            ukf_f.update_velocity_net(delta_d_pred=20.0, sigma_pred=0.5, event_class=1, window_dur=2.0)
    
    P = ukf_f.P
    sym_err = np.max(np.abs(P - P.T))
    eigvals = np.linalg.eigvalsh(P)
    min_eig = np.min(eigvals)
    print(f"TEST F (Covariance Sanity after 500 steps):")
    print(f"       Symmetry Error ||P - P^T||_inf: {sym_err:.2e}")
    print(f"       Minimum Eigenvalue            : {min_eig:.4e} (Must be > 0)")
    print(f"       Condition Number              : {np.max(eigvals)/min_eig:.2e}")
    print("=" * 80)

if __name__ == "__main__":
    run_synthetic_tests()
