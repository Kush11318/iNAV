"""
Phase 13A: Synthetic & Empirical Sanity Test Suite
Tests:
  A. Constant-speed cruise (High-speed motorway cruise)
  B. Accelerate (Forward acceleration)
  C. Brake (Deceleration from highway speed)
  D. Cruise -> Brake -> Cruise (Speed regime change from 22.8 m/s -> 9.5 m/s -> 12.4 m/s)
  E. Stationary (0.0 m/s ZUPT verification)

Verifies:
  - v_ref evolves correctly
  - no negative speed
  - no numerical instability
  - uncertainty grows sensibly with outage duration
  - UKF does not diverge or produce NaNs
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
from modules.phase13a_velocity_reference import Phase13AVelocityReference
from modules.ukf import UKFNavigationFilter
from modules.alignment import AlignmentEngine, AlignmentState


def run_synthetic_tests():
    print("=" * 80)
    print("PHASE 13A SANITY CHECKS (A through E)")
    print("=" * 80)

    model_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at {model_path}")
        return False

    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]

    # Load dataset for realistic IMU statistics
    data_path = config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet"
    df_m = pd.read_parquet(data_path)
    for c in cols:
        df_m[c] = df_m[c].interpolate().bfill().ffill().fillna(0.0)
    imu_m = df_m[cols].values.astype(np.float32)
    spd_m = df_m[config.COL_TRUE_SPEED_MS].values

    all_passed = True

    # -------------------------------------------------------------
    # Test A: Constant-Speed Cruise
    # -------------------------------------------------------------
    print("\n--- Test A: Constant-Speed Cruise (~23 m/s) ---")
    idx_a = 2960
    win_cruise = imu_m[idx_a : idx_a + 40]
    init_v_a = float(spd_m[idx_a + 40])
    
    tracker = Phase13AVelocityReference()
    tracker.initialize(init_v_a, initial_sigma=0.2)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v_a, init_heading_rad=0.0)

    for step in range(20):  # 10 seconds
        for _ in range(5):
            ukf.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        v_ref, sig, ev = tracker.update_with_model(model, win_cruise, dt_elapsed=0.5)
        ukf.update_velocity_net(delta_d_pred=v_ref * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Initial speed: {init_v_a:.2f} m/s -> Final v_ref: {v_ref:.2f} m/s, UKF speed: {ukf.x[2]:.2f} m/s")
    passed_a = (abs(v_ref - init_v_a) < 3.0) and not np.isnan(ukf.x[0]) and (v_ref >= 0.0)
    print(f"Test A Passed: {passed_a}")
    all_passed &= passed_a

    # -------------------------------------------------------------
    # Test B: Accelerate
    # -------------------------------------------------------------
    print("\n--- Test B: Accelerate ---")
    # Real acceleration segment from motorway dataset
    acc_indices = np.where((spd_m[40:] - spd_m[:-40]) > 4.0)[0]
    idx_b = int(acc_indices[0]) if len(acc_indices) > 0 else 100
    win_acc = imu_m[idx_b : idx_b + 40]
    init_v_b = float(spd_m[idx_b])
    final_v_b = float(spd_m[idx_b + 40])

    tracker = Phase13AVelocityReference()
    tracker.initialize(init_v_b, initial_sigma=0.2)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v_b, init_heading_rad=0.0)

    for step in range(8):  # 4 seconds
        for _ in range(5):
            ukf.predict(acc_fwd=1.0, gyro_yaw=0.0, dt=0.1)
        v_ref, sig, ev = tracker.update_with_model(model, win_acc, dt_elapsed=0.5)
        ukf.update_velocity_net(delta_d_pred=v_ref * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Initial: {init_v_b:.2f} m/s -> Final v_ref: {v_ref:.2f} m/s, GT target: {final_v_b:.2f} m/s")
    passed_b = (v_ref >= init_v_b) and not np.isnan(ukf.x[0])
    print(f"Test B Passed: {passed_b}")
    all_passed &= passed_b

    # -------------------------------------------------------------
    # Test C: Brake
    # -------------------------------------------------------------
    print("\n--- Test C: Brake ---")
    # Real braking segment from motorway dataset
    brk_indices = np.where((spd_m[:-40] - spd_m[40:]) > 5.0)[0]
    idx_c = int(brk_indices[0]) if len(brk_indices) > 0 else 280
    win_brake = imu_m[idx_c : idx_c + 40]
    init_v_c = float(spd_m[idx_c])
    final_v_c = float(spd_m[idx_c + 40])

    tracker = Phase13AVelocityReference()
    tracker.initialize(init_v_c, initial_sigma=0.2)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v_c, init_heading_rad=0.0)

    for step in range(8):  # 4 seconds
        for _ in range(5):
            ukf.predict(acc_fwd=-1.5, gyro_yaw=0.0, dt=0.1)
        v_ref, sig, ev = tracker.update_with_model(model, win_brake, dt_elapsed=0.5)
        ukf.update_velocity_net(delta_d_pred=v_ref * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Initial: {init_v_c:.2f} m/s -> Final v_ref: {v_ref:.2f} m/s, GT target: {final_v_c:.2f} m/s")
    passed_c = (v_ref < init_v_c) and (v_ref >= 0.0) and not np.isnan(ukf.x[0])
    print(f"Test C Passed: {passed_c}")
    all_passed &= passed_c

    # -------------------------------------------------------------
    # Test D: Cruise -> Brake -> Cruise (Speed Regime Change)
    # -------------------------------------------------------------
    print("\n--- Test D: Cruise -> Brake -> Cruise (Speed Regime Change from sync_vw11) ---")
    df_vw = pd.read_parquet(config.SYNC_PROCESSED_DIR / "sync_vw11.parquet")
    for c in cols:
        df_vw[c] = df_vw[c].interpolate().bfill().ffill().fillna(0.0)
    imu_vw = df_vw[cols].values.astype(np.float32)
    spd_vw = df_vw[config.COL_TRUE_SPEED_MS].values

    # vw11 Outage 3 covers indices 3137 to 3437 (30 seconds, 22.8 m/s down to 12 m/s)
    idx_d_start = 3137
    init_v_d = float(spd_vw[idx_d_start])
    
    tracker = Phase13AVelocityReference()
    tracker.initialize(init_v_d, initial_sigma=0.2)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v_d, init_heading_rad=0.0)

    for k in range(0, 300, 5):  # 30 seconds
        win_d = imu_vw[idx_d_start + k - 39 : idx_d_start + k + 1]
        v_ref, sig, ev = tracker.update_with_model(model, win_d, dt_elapsed=0.5)
        ukf.update_velocity_net(delta_d_pred=v_ref * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    true_v_d = float(spd_vw[idx_d_start + 300])
    print(f"Started at: {init_v_d:.2f} m/s -> After 30s: v_ref={v_ref:.2f} m/s, True={true_v_d:.2f} m/s (Phase 11 frozen would stay at {init_v_d:.2f} m/s)")
    # Must track the speed drop without snapping back to 22.8 m/s
    passed_d = (v_ref < 16.0) and (abs(v_ref - true_v_d) < 4.0)
    print(f"Test D Passed: {passed_d}")
    all_passed &= passed_d

    # -------------------------------------------------------------
    # Test E: Stationary (0 m/s)
    # -------------------------------------------------------------
    print("\n--- Test E: Stationary (0 m/s ZUPT verification) ---")
    idx_e = 701
    win_stat = imu_m[idx_e : idx_e + 40]
    
    tracker = Phase13AVelocityReference()
    tracker.initialize(0.0, initial_sigma=0.2)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=0.0, init_heading_rad=0.0)

    for step in range(20):  # 10s
        for _ in range(5):
            ukf.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        v_ref, sig, ev = tracker.update_with_model(model, win_stat, dt_elapsed=0.5)
        if v_ref < 0.15 or ev == 0:
            ukf.update_zupt(gyro_reading=0.0)
        else:
            ukf.update_velocity_net(delta_d_pred=v_ref * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Result: Final v_ref = {v_ref:.2f} m/s, UKF speed = {ukf.x[2]:.4f} m/s, P[2,2] = {ukf.P[2,2]:.2e}")
    passed_e = (v_ref == 0.0) and (abs(ukf.x[2]) < 1e-3)
    print(f"Test E Passed: {passed_e}")
    all_passed &= passed_e

    # Uncertainty growth verification
    print(f"\nUncertainty growth check: Final sigma = {sig:.2f} m/s (Started at 0.2 m/s)")
    passed_sig = (sig >= 0.2)
    all_passed &= passed_sig

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 5 SANITY CHECKS PASSED SUCCESSFULLY!")
    else:
        print("SOME SANITY CHECKS FAILED!")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_synthetic_tests()
    sys.exit(0 if success else 1)
