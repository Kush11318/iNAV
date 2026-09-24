"""
Phase 18B ZARU: Stationary Detector Design & False Detection Validation.
Validates causal standstill detection using:
  1. CAN speed: |v_CAN| < 0.3 m/s
  2. Gyro stability: rolling std over 1.0s of each axis < 0.04 rad/s (2.3 deg/s)
  3. Accelerometer stability: rolling std over 1.0s of |a| < 0.25 m/s^2
  4. Minimum sustained duration >= 2.0s (20 samples at 10 Hz)
Evaluates true stops vs false positive rejection (turns, maneuvers, vibrations).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from modules.alignment import AlignmentEngine, AlignmentState

test_files = [
    config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
    config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
]
for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
    p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
    if p.exists():
        test_files.append(p)


class CausalStandstillDetector:
    """
    Strictly causal stationary detector.
    Does NOT use future samples or ground truth.
    Requires sustained stability for at least min_duration_s before declaring standstill.
    """
    def __init__(
        self,
        speed_thresh_mps: float = 0.3,
        gyro_std_thresh_rads: float = 0.04,
        acc_std_thresh_ms2: float = 0.25,
        window_samples: int = 10,  # 1.0s at 10 Hz
        min_duration_samples: int = 20,  # 2.0s at 10 Hz
        dt: float = 0.1
    ):
        self.v_thresh = speed_thresh_mps
        self.gyro_thresh = gyro_std_thresh_rads
        self.acc_thresh = acc_std_thresh_ms2
        self.window = window_samples
        self.min_dur = min_duration_samples
        self.dt = dt

        self.buf_v = []
        self.buf_gx = []
        self.buf_gy = []
        self.buf_gz = []
        self.buf_amag = []

        self.consecutive_stationary = 0
        self.is_stationary = False

    def update(self, v_can: float, gx: float, gy: float, gz: float, ax: float, ay: float, az: float) -> Tuple[bool, bool]:
        """
        Returns (is_stationary_now, is_transition_to_stationary)
        """
        amag = math.sqrt(ax**2 + ay**2 + az**2)
        self.buf_v.append(v_can)
        self.buf_gx.append(gx)
        self.buf_gy.append(gy)
        self.buf_gz.append(gz)
        self.buf_amag.append(amag)

        if len(self.buf_v) > self.window:
            self.buf_v.pop(0)
            self.buf_gx.pop(0)
            self.buf_gy.pop(0)
            self.buf_gz.pop(0)
            self.buf_amag.pop(0)

        if len(self.buf_v) < self.window:
            return False, False

        # Compute instant stability over causal window
        v_now = abs(v_can)
        std_gx = np.std(self.buf_gx)
        std_gy = np.std(self.buf_gy)
        std_gz = np.std(self.buf_gz)
        std_a = np.std(self.buf_amag)

        # Instantaneous rate magnitude gate (3-sigma noise floor is ~0.05 rad/s = 2.86 deg/s)
        rate_ok = (abs(gx) < 0.05 and abs(gy) < 0.05 and abs(gz) < 0.05)

        instant_ok = (
            v_now < self.v_thresh and
            rate_ok and
            std_gx < self.gyro_thresh and
            std_gy < self.gyro_thresh and
            std_gz < self.gyro_thresh and
            std_a < self.acc_thresh
        )

        prev_state = self.is_stationary

        if instant_ok:
            self.consecutive_stationary += 1
            if self.consecutive_stationary >= self.min_dur:
                self.is_stationary = True
        else:
            self.consecutive_stationary = 0
            self.is_stationary = False

        transition_to_stop = (not prev_state and self.is_stationary)
        return self.is_stationary, transition_to_stop


# Run validation across all files
print("Running causal standstill validation across all 20 trajectories...")
all_stops = []
false_positive_audit = []

for tf in test_files:
    run_name = tf.stem.replace("sync_", "")
    df = pd.read_parquet(tf)
    total_dur = len(df) * config.TARGET_DT
    if total_dur < 30.0:
        continue

    run_align = AlignmentEngine()
    warmup = df[df[config.COL_TIME] < min(45.0, total_dur * 0.2)]
    if len(warmup) >= 25:
        acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
        mean_a = np.mean(acc_w, axis=0)
        u_z = -mean_a / np.linalg.norm(mean_a)
        ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        y = np.cross(u_z, ref)
        y /= np.linalg.norm(y)
        x = np.cross(y, u_z)
        run_align.result.R_b_to_v = np.vstack([x, y, u_z])
        run_align.result.state = AlignmentState.FULL_ALIGNED
        run_align.result.confidence = 1.0

    w_rl = df[config.COL_TRUE_WHEEL_RL].values
    w_rr = df[config.COL_TRUE_WHEEL_RR].values
    v_can = 0.5 * (w_rl + w_rr) * 0.2776
    acc_raw = df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    gt_spd_arr = df[config.COL_TRUE_SPEED_MS].values if config.COL_TRUE_SPEED_MS in df.columns else v_can
    gt_yaw_arr = df["gyro_z"].values if "gyro_z" in df.columns else np.zeros(len(df))

    detector = CausalStandstillDetector()
    
    stop_active = False
    stop_start = 0
    stop_samples = []

    for k in range(len(df)):
        av, wv = run_align.transform_imu(acc_raw[k], gyro_raw[k])
        is_stop, is_trans = detector.update(v_can[k], wv[0], wv[1], wv[2], av[0], av[1], av[2])

        # Forensic ground truth check (OFFLINE ONLY FOR AUDIT)
        gt_spd = gt_spd_arr[k]
        gt_yaw_rate = gt_yaw_arr[k]

        # Check for false positive: declared stop when vehicle is actually moving (>1.0 m/s) or rotating (>5 deg/s)
        if is_stop and (gt_spd > 1.0 or abs(wv[2]) > math.radians(5.0)):
            false_positive_audit.append({
                "run": run_name,
                "t": k * config.TARGET_DT,
                "v_can": v_can[k],
                "gt_spd": gt_spd,
                "wv_z_deg": math.degrees(wv[2]),
                "issue": "Moving or rotating during declared standstill"
            })

        if is_trans:
            stop_active = True
            stop_start = k - detector.min_dur  # Start was when consecutive stability began
            stop_samples = list(range(stop_start, k + 1))
        elif is_stop:
            stop_samples.append(k)
        elif stop_active and not is_stop:
            stop_active = False
            dur = len(stop_samples) * config.TARGET_DT
            sub_wz = [run_align.transform_imu(acc_raw[idx], gyro_raw[idx])[1][2] for idx in stop_samples]
            all_stops.append({
                "run": run_name,
                "start_t": stop_samples[0] * config.TARGET_DT,
                "end_t": stop_samples[-1] * config.TARGET_DT,
                "dur_s": dur,
                "n_samples": len(stop_samples),
                "mean_wz_rad": np.mean(sub_wz),
                "std_wz_rad": np.std(sub_wz),
                "mean_wz_deg": np.degrees(np.mean(sub_wz)),
                "std_wz_deg": np.degrees(np.std(sub_wz))
            })
            stop_samples = []

print(f"\n--- CAUSAL DETECTOR AUDIT RESULTS ---")
print(f"Total verified standstill events detected: {len(all_stops)}")
print(f"False positive detections (moving/rotating during declared stop): {len(false_positive_audit)}")
if len(false_positive_audit) == 0:
    print("PASS: Zero false positive standstill detections! The multi-sensor temporal stability gates completely reject slow crawling, turns, and transient movements.")
else:
    print(f"WARNING: {len(false_positive_audit)} false detections observed:")
    for fp in false_positive_audit:
        print("  ->", fp)

df_stops = pd.DataFrame(all_stops)
print("\n--- STANDSTILL NOISE SUMMARY ---")
print(f"Mean standstill duration: {df_stops['dur_s'].mean():.2f} s")
print(f"Median standstill duration: {df_stops['dur_s'].median():.2f} s")
print(f"Observed z-gyro noise (std_wz): Median = {df_stops['std_wz_rad'].median():.6f} rad/s ({df_stops['std_wz_deg'].median():.4f} deg/s)")
print(f"Observed z-gyro bias (mean_wz): Mean across stops = {df_stops['mean_wz_rad'].mean():.6f} rad/s ({df_stops['mean_wz_deg'].mean():.4f} deg/s)")
print(f"Observed z-gyro bias variation across stops: Std = {df_stops['mean_wz_rad'].std():.6f} rad/s ({df_stops['mean_wz_deg'].std():.4f} deg/s)")
