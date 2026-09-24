"""
Phase 18B ZARU Step 1: Data Discovery & Stationary Event Identification.
Scans all 20 IO-VNBD benchmark runs to discover stationary intervals,
measure actual smartphone gyro noise and bias stability during stops,
and inspect pre-outage stop coverage for the 56 benchmark outages.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from modules.alignment import AlignmentEngine, AlignmentState
from modules.can_fusion_ukf import compute_pre_outage_can_calibration

test_files = [
    config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
    config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
]
for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
    p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
    if p.exists():
        test_files.append(p)

print(f"Loaded {len(test_files)} trajectories. Inspecting stationary events...")

# First, inspect raw distributions during low CAN speed across all files
low_speed_samples = []
stationary_events = []

for tf in test_files:
    run_name = tf.stem.replace("sync_", "")
    df = pd.read_parquet(tf)
    total_dur = len(df) * config.TARGET_DT
    if total_dur < 30.0:
        continue

    # Alignment engine to transform to vehicle frame
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

    # CAN speed computation
    w_rl = df[config.COL_TRUE_WHEEL_RL].values
    w_rr = df[config.COL_TRUE_WHEEL_RR].values
    v_can = 0.5 * (w_rl + w_rr) * 0.2776  # nominal radius

    acc_raw = df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    
    # Transform to vehicle frame
    acc_v = np.zeros_like(acc_raw)
    gyro_v = np.zeros_like(gyro_raw)
    for i in range(len(df)):
        av, wv = run_align.transform_imu(acc_raw[i], gyro_raw[i])
        acc_v[i] = av
        gyro_v[i] = wv

    acc_mag = np.linalg.norm(acc_v, axis=1)
    
    # Rolling statistics over 1.0s window (10 samples at 10 Hz)
    window = 10
    gyro_std_x = pd.Series(gyro_v[:, 0]).rolling(window, min_periods=window).std().values
    gyro_std_y = pd.Series(gyro_v[:, 1]).rolling(window, min_periods=window).std().values
    gyro_std_z = pd.Series(gyro_v[:, 2]).rolling(window, min_periods=window).std().values
    acc_mag_std = pd.Series(acc_mag).rolling(window, min_periods=window).std().values

    # Candidate condition: CAN speed < 0.3 m/s
    cand_mask = (v_can < 0.3) & (abs(w_rl - w_rr) < 2.0)
    
    # Identify contiguous intervals of candidate stops
    in_stop = False
    stop_start = 0
    
    for i in range(len(df)):
        if cand_mask[i] and not in_stop:
            in_stop = True
            stop_start = i
        elif not cand_mask[i] and in_stop:
            in_stop = False
            stop_len = i - stop_start
            stop_dur = stop_len * config.TARGET_DT
            if stop_dur >= 1.0:  # Check candidates >= 1s
                sub_v_can = v_can[stop_start:i]
                sub_gz = gyro_v[stop_start:i, 2]
                sub_gx = gyro_v[stop_start:i, 0]
                sub_gy = gyro_v[stop_start:i, 1]
                sub_amag = acc_mag[stop_start:i]
                
                std_gz = np.std(sub_gz)
                std_gx = np.std(sub_gx)
                std_gy = np.std(sub_gy)
                std_amag = np.std(sub_amag)
                mean_gz = np.mean(sub_gz)
                mean_v = np.mean(sub_v_can)
                
                # Check strict stability criteria for true standstill (>= 2.0s duration, gyro noise < 0.05 rad/s)
                is_accepted = (stop_dur >= 2.0) and (std_gz < 0.04) and (std_amag < 0.25)
                
                stationary_events.append({
                    "run": run_name,
                    "start_idx": stop_start,
                    "end_idx": i,
                    "start_t": stop_start * config.TARGET_DT,
                    "end_t": i * config.TARGET_DT,
                    "dur_s": stop_dur,
                    "mean_v_can": mean_v,
                    "max_v_can": np.max(sub_v_can),
                    "mean_gz_rad": mean_gz,
                    "std_gz_rad": std_gz,
                    "std_gx_rad": std_gx,
                    "std_gy_rad": std_gy,
                    "std_amag": std_amag,
                    "mean_amag": np.mean(sub_amag),
                    "is_accepted": is_accepted
                })

df_events = pd.DataFrame(stationary_events)
print(f"\nTotal candidate stationary events (>= 1.0s, v_can < 0.3m/s): {len(df_events)}")
accepted = df_events[df_events["is_accepted"]].copy()
print(f"Accepted standstill events (>= 2.0s, std_gz < 0.04 rad/s, std_amag < 0.25 m/s^2): {len(accepted)}")

if len(accepted) > 0:
    print("\n--- STANDSTILL GYROSCOPE NOISE STATISTICS (ACCEPTED EVENTS) ---")
    print(f"Mean std_gz: {accepted['std_gz_rad'].mean():.6f} rad/s ({np.degrees(accepted['std_gz_rad'].mean()):.4f} deg/s)")
    print(f"Median std_gz: {accepted['std_gz_rad'].median():.6f} rad/s ({np.degrees(accepted['std_gz_rad'].median()):.4f} deg/s)")
    print(f"Min std_gz: {accepted['std_gz_rad'].min():.6f} rad/s ({np.degrees(accepted['std_gz_rad'].min()):.4f} deg/s)")
    print(f"Max std_gz: {accepted['std_gz_rad'].max():.6f} rad/s ({np.degrees(accepted['std_gz_rad'].max()):.4f} deg/s)")
    print(f"Mean duration: {accepted['dur_s'].mean():.2f} s (range: {accepted['dur_s'].min():.1f}s to {accepted['dur_s'].max():.1f}s)")
    print(f"Mean z-gyro value (bias estimate across stops): {accepted['mean_gz_rad'].mean():.6f} rad/s ({np.degrees(accepted['mean_gz_rad'].mean()):.4f} deg/s)")
    print(f"Std of z-gyro mean across stops (bias variation): {accepted['mean_gz_rad'].std():.6f} rad/s ({np.degrees(accepted['mean_gz_rad'].std()):.4f} deg/s)")

# Now check coverage of the 56 benchmark outages
print("\n--- CHECKING 56 OUTAGE SCHEDULE PRE-OUTAGE STOP COVERAGE ---")
outage_coverage = []
for tf in test_files:
    run_name = tf.stem.replace("sync_", "")
    df = pd.read_parquet(tf)
    total_dur = len(df) * config.TARGET_DT
    if total_dur < 30.0:
        continue
    schedule = generate_outage_schedule(total_dur, run_id=run_name)
    if not schedule:
        continue
    df_sim, df_outages = inject_outages(df, schedule)
    
    run_accepted = accepted[accepted["run"] == run_name]
    
    for _, row in df_outages.iterrows():
        oid = int(row["outage_id"])
        dur = int(row["duration_s"])
        mask = (df_sim["outage_id"] == oid)
        outage_start_idx = mask.idxmax()
        outage_start_t = outage_start_idx * config.TARGET_DT
        outage_end_t = outage_start_t + dur
        
        # Pre-outage stops: any stop ending before outage_start_t
        pre_stops = run_accepted[run_accepted["end_t"] <= outage_start_t]
        has_pre_stop = len(pre_stops) > 0
        last_stop_age = (outage_start_t - pre_stops["end_t"].max()) if has_pre_stop else None
        
        # In-outage stops: any stop overlapping with the outage
        in_stops = run_accepted[(run_accepted["start_t"] < outage_end_t) & (run_accepted["end_t"] > outage_start_t)]
        has_in_stop = len(in_stops) > 0
        
        outage_coverage.append({
            "scenario": f"{run_name}_o{oid}",
            "run": run_name,
            "outage_id": oid,
            "dur_s": dur,
            "has_pre_stop": has_pre_stop,
            "n_pre_stops": len(pre_stops),
            "last_stop_age_s": last_stop_age,
            "has_in_stop": has_in_stop,
            "n_in_stops": len(in_stops)
        })

df_cov = pd.DataFrame(outage_coverage)
n_with_pre = df_cov["has_pre_stop"].sum()
n_with_in = df_cov["has_in_stop"].sum()
print(f"Total benchmark outages: {len(df_cov)}")
print(f"Outages with >= 1 PRE-OUTAGE standstill event: {n_with_pre} / {len(df_cov)} ({n_with_pre/len(df_cov)*100:.1f}%)")
print(f"Outages with >= 1 DURING-OUTAGE standstill event: {n_with_in} / {len(df_cov)} ({n_with_in/len(df_cov)*100:.1f}%)")

# Inspect the 7 severe Phase 18A failure cases
fail_keys = ["vw14b_o5", "vw4_o1", "vw11_o3", "vw8_o1", "vw6_o2", "vw8_o3", "vw7_o1"]
print("\n--- PRE-OUTAGE STOP COVERAGE FOR THE 7 SEVERE PHASE 18A FAILURES ---")
for fk in fail_keys:
    match = df_cov[df_cov["scenario"] == fk]
    if len(match) > 0:
        r = match.iloc[0]
        age_str = f"{r['last_stop_age_s']:.1f}s ago" if r['has_pre_stop'] else "NONE"
        in_str = f"Yes ({r['n_in_stops']} stops)" if r['has_in_stop'] else "No"
        print(f"[{fk:<15}] Dur={r['dur_s']:3d}s | Pre-Outage Stop: {str(r['has_pre_stop']):<5} (last stop {age_str}) | In-Outage Stop: {in_str}")
