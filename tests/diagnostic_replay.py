import math
import numpy as np
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eval.replay import run_inav_ukf_pipeline, run_inav_cpp_kinematic_pipeline

def run_diagnostic_replay():
    print("=" * 80)
    print("PHASE 4 AUDIT: REAL IO-VNBD REPLAY DIAGNOSTIC (30-SECOND REPRESENTATIVE SEGMENT)")
    print("=" * 80)

    parquet_path = "data/sample_test_trajectory.parquet"
    if not os.path.exists(parquet_path):
        print(f"[SKIP] Parquet not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path).iloc[:300].copy() # 30s at 10 Hz
    init_lat = float(df['gps_lat'].dropna().iloc[0])
    init_lon = float(df['gps_lon'].dropna().iloc[0])
    init_speed = float(df['gps_speed_ms'].dropna().iloc[0])
    init_heading = float(df['gps_bearing_deg'].dropna().iloc[0])

    print(f"Segment Length: {len(df)} epochs (30.0 s)")
    print(f"Initial State: Lat={init_lat:.6f}, Lon={init_lon:.6f}, Speed={init_speed:.2f} m/s, Heading={init_heading:.1f} deg")

    # Run Python UKF Pipeline
    ukf_lat, ukf_lon, ukf_spd = run_inav_ukf_pipeline(
        df_outage=df,
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed,
        init_heading_deg=init_heading
    )

    # Run C++ Kinematic Pipeline
    cpp_lat, cpp_lon, cpp_spd = run_inav_cpp_kinematic_pipeline(
        df_outage=df,
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed,
        init_heading_deg=init_heading
    )

    # Compute distance difference between UKF and C++
    R_earth = 6371000.0
    dlat = np.radians(ukf_lat - cpp_lat)
    dlon = np.radians(ukf_lon - cpp_lon)
    lat_mid = np.radians(init_lat)
    dx = dlon * R_earth * math.cos(lat_mid)
    dy = dlat * R_earth
    dist_diff = np.hypot(dx, dy)

    # Ground truth GPS distance
    gt_lat = df['gps_lat'].values
    gt_lon = df['gps_lon'].values
    gt_dx = np.radians(gt_lon - init_lon) * R_earth * math.cos(lat_mid)
    gt_dy = np.radians(gt_lat - init_lat) * R_earth
    gt_dist = np.hypot(gt_dx, gt_dy)

    err_ukf = np.hypot(
        np.radians(ukf_lon - gt_lon) * R_earth * math.cos(lat_mid),
        np.radians(ukf_lat - gt_lat) * R_earth
    )
    err_cpp = np.hypot(
        np.radians(cpp_lon - gt_lon) * R_earth * math.cos(lat_mid),
        np.radians(cpp_lat - gt_lat) * R_earth
    )

    print(f"Traveled Distance (Ground Truth) : {gt_dist[-1]:.2f} m")
    print(f"Final Position Error (Python UKF): {err_ukf[-1]:.2f} m ({err_ukf[-1]/max(gt_dist[-1], 1)*100:.2f}% drift)")
    print(f"Final Position Error (C++ Kinem.): {err_cpp[-1]:.2f} m ({err_cpp[-1]/max(gt_dist[-1], 1)*100:.2f}% drift)")
    print(f"Divergence between UKF and C++   : {dist_diff[-1]:.2f} m (Max: {np.max(dist_diff):.2f} m)")
    print(f"Mean Speed Difference            : {np.mean(np.abs(ukf_spd - cpp_spd)):.2f} m/s")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostic_replay()
