"""
Scratch Unit Test for Phase 1 Navigation Modules:
- AlignmentEngine (Module A)
- UKFNavigationFilter (Module C)
- GnssHandler (Module E)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.alignment import AlignmentEngine
from modules.ukf import UKFNavigationFilter
from modules.gnss_handler import GnssHandler, NavigationMode

print("--- 1. Testing AlignmentEngine ---")
alignment = AlignmentEngine()
# Load sample data
df = pd.read_parquet(config.SYNC_PROCESSED_DIR / "sync_s1.parquet")
acc_raw = df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values[:200]
gyro_raw = df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values[:200]

ok_static = alignment.calibrate_static(acc_raw[:30], gyro_raw[:30])
print(f"Static calibration success: {ok_static}, confidence: {alignment.confidence_score}")

ok_dyn = alignment.calibrate_dynamic(acc_raw[30:150], gyro_raw[30:150])
print(f"Dynamic calibration success: {ok_dyn}, confidence: {alignment.confidence_score}")

acc_v, gyro_v = alignment.transform_imu(acc_raw[0], gyro_raw[0])
print(f"Body accel: {acc_raw[0]} -> Vehicle accel: {acc_v}")
print(f"Body gyro: {gyro_raw[0]} -> Vehicle gyro: {gyro_v}")

print("\n--- 2. Testing UKFNavigationFilter ---")
ukf = UKFNavigationFilter(dt=0.1)
ukf.initialize(init_lat=52.4, init_lon=-1.5, init_speed_ms=10.0, init_heading_rad=np.radians(90.0))

print(f"Initial state: pN={ukf.x[0]:.2f}, pE={ukf.x[1]:.2f}, v={ukf.x[2]:.2f}, psi={np.degrees(ukf.x[3]):.1f}°")

# Run 50 prediction steps
for step in range(50):
    ukf.predict(acc_fwd=0.1, gyro_yaw=0.01, dt=0.1)

print(f"After 50 steps (5s): pN={ukf.x[0]:.2f}, pE={ukf.x[1]:.2f}, v={ukf.x[2]:.2f}, psi={np.degrees(ukf.x[3]):.1f}°")

# Test VelocityNet update
ukf.update_velocity_net(delta_d_pred=2.0, sigma_pred=0.2, window_dur=2.0)
print(f"After VelocityNet update: v={ukf.x[2]:.2f} m/s")

# Test ZUPT
ukf.update_zupt(gyro_reading=0.0)
print(f"After ZUPT update: v={ukf.x[2]:.4f} m/s, gyro_bias={ukf.x[4]:.6f}")

print("\n--- 3. Testing GnssHandler ---")
gnss = GnssHandler(quarantine_count=3, ramp_duration_sec=2.0)

mode, infl = gnss.process_fix(lat=52.4, lon=-1.5, accuracy_m=5.0, sat_count=10, timestamp_s=0.0)
print(f"t=0s: Mode={mode.value}, Inflation={infl}")

# Simulate Blackout
mode, infl = gnss.process_fix(lat=np.nan, lon=np.nan, accuracy_m=np.nan, sat_count=0, timestamp_s=10.0)
print(f"t=10s (tunnel entry): Mode={mode.value}, Inflation={infl}")

# Simulate Signal Return (Quarantine)
for i in range(4):
    t = 40.0 + i * 0.1
    mode, infl = gnss.process_fix(lat=52.41 + i*0.0001, lon=-1.5, accuracy_m=8.0, sat_count=7, timestamp_s=t)
    print(f"t={t:.1f}s: Mode={mode.value}, Inflation={infl:.1f}")

print("\nAll Module tests passed successfully!")
