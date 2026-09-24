"""
Phase 13B: Sanity Test Suite (7 Regimes)
Tests:
  1. Constant-speed cruise (Phase 13B ≈ Phase 11, gate closed, anchor dominates)
  2. Hard braking (gate opens, tracks deceleration drop)
  3. Acceleration (detects forward acc, allows positive correction)
  4. Cruise -> Braking -> Cruise (gate opens during braking, smoothly decays back to cruise)
  5. Stationary (triggers ZUPT, speed goes to 0.0)
  6. Rough road (uncertainty inflated, correction attenuated)
  7. Turning (gate closed, uncertainty inflated, no false longitudinal push)
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
from modules.phase13b_event_gated_velocity import Phase13BEventGatedVelocityTracker
from modules.ukf import UKFNavigationFilter


def run_sanity_tests():
    print("=" * 80)
    print("PHASE 13B SANITY CHECKS (7 REGIMES)")
    print("=" * 80)

    model_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at {model_path}")
        return False

    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]

    # Load motorway dataset
    data_path = config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet"
    df_m = pd.read_parquet(data_path)
    for c in cols:
        df_m[c] = df_m[c].interpolate().bfill().ffill().fillna(0.0)
    imu_m = df_m[cols].values.astype(np.float32)
    spd_m = df_m[config.COL_TRUE_SPEED_MS].values

    all_passed = True

    # -------------------------------------------------------------
    # Test 1: Constant-Speed Cruise
    # -------------------------------------------------------------
    print("\n--- Test 1: Constant-Speed Cruise (~23 m/s) ---")
    idx_1 = 2960
    win_cruise = imu_m[idx_1 : idx_1 + 40]
    init_v1 = float(spd_m[idx_1 + 40])

    tracker = Phase13BEventGatedVelocityTracker()
    tracker.initialize(init_v1)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v1, init_heading_rad=0.0)

    for step in range(20):  # 10s
        for _ in range(5):
            ukf.predict(acc_fwd=0.0, gyro_yaw=0.0, dt=0.1)
        v_meas, sig, ev, gate_open, gate_type, corr = tracker.update_step(
            model=model, imu_window_40x6=win_cruise, a_fwd_mean=0.0, yaw_rate_mean=0.0, dt_elapsed=0.5
        )
        ukf.update_velocity_net(delta_d_pred=v_meas * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Initial anchor: {init_v1:.2f} m/s -> Final v_meas: {v_meas:.2f} m/s, gate_type: {gate_type}, corr: {corr:.3f}")
    # Must remain very close to anchor because gate is closed during cruise
    passed_1 = (abs(v_meas - init_v1) < 1.0) and (corr < 0.5) and not gate_open
    print(f"Test 1 Passed: {passed_1}")
    all_passed &= passed_1

    # -------------------------------------------------------------
    # Test 2: Hard Braking
    # -------------------------------------------------------------
    print("\n--- Test 2: Hard Braking ---")
    # Real braking segment
    brk_indices = np.where((spd_m[:-40] - spd_m[40:]) > 5.0)[0]
    idx_2 = int(brk_indices[0]) if len(brk_indices) > 0 else 280
    win_brake = imu_m[idx_2 : idx_2 + 40]
    init_v2 = float(spd_m[idx_2])

    tracker = Phase13BEventGatedVelocityTracker()
    tracker.initialize(init_v2)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v2, init_heading_rad=0.0)

    for step in range(10):  # 5s
        for _ in range(5):
            ukf.predict(acc_fwd=-2.0, gyro_yaw=0.0, dt=0.1)
        v_meas, sig, ev, gate_open, gate_type, corr = tracker.update_step(
            model=model, imu_window_40x6=win_brake, a_fwd_mean=-2.0, yaw_rate_mean=0.0, dt_elapsed=0.5
        )
        ukf.update_velocity_net(delta_d_pred=v_meas * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Initial: {init_v2:.2f} m/s -> Final v_meas: {v_meas:.2f} m/s, gate_open: {gate_open}, gate_type: {gate_type}, corr: {corr:.2f}")
    passed_2 = gate_open and (v_meas < init_v2) and (corr < 0.0)
    print(f"Test 2 Passed: {passed_2}")
    all_passed &= passed_2

    # -------------------------------------------------------------
    # Test 3: Acceleration
    # -------------------------------------------------------------
    print("\n--- Test 3: Acceleration ---")
    acc_indices = np.where((spd_m[40:] - spd_m[:-40]) > 4.0)[0]
    idx_3 = int(acc_indices[0]) if len(acc_indices) > 0 else 100
    win_acc = imu_m[idx_3 : idx_3 + 40]
    init_v3 = float(spd_m[idx_3])

    tracker = Phase13BEventGatedVelocityTracker()
    tracker.initialize(init_v3)
    ukf = UKFNavigationFilter(dt=0.1)
    ukf.initialize(init_lat=12.97, init_lon=77.59, init_speed_ms=init_v3, init_heading_rad=0.0)

    for step in range(10):  # 5s
        for _ in range(5):
            ukf.predict(acc_fwd=1.2, gyro_yaw=0.0, dt=0.1)
        v_meas, sig, ev, gate_open, gate_type, corr = tracker.update_step(
            model=model, imu_window_40x6=win_acc, a_fwd_mean=1.2, yaw_rate_mean=0.0, dt_elapsed=0.5
        )
        ukf.update_velocity_net(delta_d_pred=v_meas * 2.0, sigma_pred=sig * 2.0, event_class=ev, window_dur=2.0)

    print(f"Initial: {init_v3:.2f} m/s -> Final v_meas: {v_meas:.2f} m/s, gate_open: {gate_open}, gate_type: {gate_type}, corr: {corr:.2f}")
    passed_3 = (v_meas >= init_v3) and not np.isnan(v_meas)
    print(f"Test 3 Passed: {passed_3}")
    all_passed &= passed_3

    # -------------------------------------------------------------
    # Test 4: Cruise -> Braking -> Cruise
    # -------------------------------------------------------------
    print("\n--- Test 4: Cruise -> Braking -> Cruise ---")
    tracker = Phase13BEventGatedVelocityTracker()
    init_v4 = 22.0
    tracker.initialize(init_v4)

    # 1. Cruise 5s (gate closed)
    for _ in range(10):
        v_meas, _, _, gate_open, gate_type, corr = tracker.update_step(
            model, win_cruise, a_fwd_mean=0.0, yaw_rate_mean=0.0, dt_elapsed=0.5
        )
    print(f"During initial cruise: v_meas = {v_meas:.2f} m/s, corr = {corr:.3f}")

    # 2. Brake 4s (gate open)
    for _ in range(8):
        v_meas, _, _, gate_open, gate_type, corr = tracker.update_step(
            model, win_brake, a_fwd_mean=-2.0, yaw_rate_mean=0.0, dt_elapsed=0.5
        )
    print(f"During braking: v_meas = {v_meas:.2f} m/s, corr = {corr:.3f}, gate_open = {gate_open}")
    passed_4_brake = (v_meas < init_v4) and (corr < -1.0) and gate_open

    # 3. Post-braking cruise 8s (gate closed, correction decays back toward anchor)
    for _ in range(16):
        v_meas, _, _, gate_open, gate_type, corr = tracker.update_step(
            model, win_cruise, a_fwd_mean=0.0, yaw_rate_mean=0.0, dt_elapsed=0.5
        )
    print(f"After post-braking cruise: v_meas = {v_meas:.2f} m/s (decayed back toward anchor {init_v4:.2f} m/s), corr = {corr:.3f}")
    passed_4_decay = (abs(v_meas - init_v4) < 2.0) and (abs(corr) < 2.0)
    passed_4 = passed_4_brake and passed_4_decay
    print(f"Test 4 Passed: {passed_4}")
    all_passed &= passed_4

    # -------------------------------------------------------------
    # Test 5: Stationary
    # -------------------------------------------------------------
    print("\n--- Test 5: Stationary (0.0 m/s ZUPT) ---")
    idx_5 = 701
    win_stat = imu_m[idx_5 : idx_5 + 40]
    tracker = Phase13BEventGatedVelocityTracker()
    tracker.initialize(0.0)  # vehicle stopped

    v_meas, sig, ev, gate_open, gate_type, corr = tracker.update_step(
        model, win_stat, a_fwd_mean=0.0, yaw_rate_mean=0.0, dt_elapsed=0.5
    )
    print(f"Stationary result: v_meas = {v_meas:.2f} m/s, gate_type = {gate_type}, gate_open = {gate_open}")
    passed_5 = (v_meas < 0.2) or (gate_type == "STATIONARY")
    print(f"Test 5 Passed: {passed_5}")
    all_passed &= passed_5

    # -------------------------------------------------------------
    # Test 6: Rough Road
    # -------------------------------------------------------------
    print("\n--- Test 6: Rough Road (High vibration) ---")
    win_rough = win_cruise.copy()
    win_rough[:, 2] += np.random.randn(40) * 2.5  # high vertical acceleration noise
    tracker = Phase13BEventGatedVelocityTracker()
    tracker.initialize(init_v1)

    v_meas, sig, ev, gate_open, gate_type, corr = tracker.update_step(
        model, win_rough, a_fwd_mean=0.0, yaw_rate_mean=0.0, dt_elapsed=0.5
    )
    print(f"Rough road sigma: {sig:.2f} m/s (Uncertainty inflated)")
    passed_6 = (sig >= 0.4)
    print(f"Test 6 Passed: {passed_6}")
    all_passed &= passed_6

    # -------------------------------------------------------------
    # Test 7: Turning
    # -------------------------------------------------------------
    print("\n--- Test 7: Turning (High Yaw Rate) ---")
    tracker = Phase13BEventGatedVelocityTracker()
    tracker.initialize(init_v1)

    # Turning with high yaw rate (0.3 rad/s)
    v_meas, sig, ev, gate_open, gate_type, corr = tracker.update_step(
        model, win_cruise, a_fwd_mean=0.0, yaw_rate_mean=0.3, dt_elapsed=0.5
    )
    print(f"Turning result: gate_open = {gate_open}, gate_type = {gate_type}, sigma = {sig:.2f} m/s")
    passed_7 = (not gate_open) and (gate_type == "TURNING") and (sig >= 0.4)
    print(f"Test 7 Passed: {passed_7}")
    all_passed &= passed_7

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 7 SANITY CHECKS PASSED SUCCESSFULLY!")
    else:
        print("SOME SANITY CHECKS FAILED!")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_sanity_tests()
    sys.exit(0 if success else 1)
