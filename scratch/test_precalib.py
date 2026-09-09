import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd()))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from eval.baseline import local_xy_to_latlon
from modules.alignment import AlignmentEngine
from modules.ukf import UKFNavigationFilter
from modules.velocity_net import VelocityNetPredictor

sync_file = config.SYNC_PROCESSED_DIR / "sync_vta10.parquet"
df = pd.read_parquet(sync_file)
total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]
schedule = generate_outage_schedule(total_dur, durations=[10, 30, 60, 120])
df_sim, df_outages = inject_outages(df, schedule)

best_pt = config.MODELS_DIR / "velocity_net_best.pt"
predictor = VelocityNetPredictor(str(best_pt))

# Pre-outage warmup period (first 30s of run when GNSS is available)
warmup = df[df[config.COL_TIME] < 30.0]
acc_warmup = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
gyro_warmup = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

# Pre-calibrate AlignmentEngine once during GNSS warmup
global_align = AlignmentEngine()
# Use first 50 samples for static if stationary, or whole warmup
global_align.calibrate_static(acc_warmup[:50], gyro_warmup[:50])
global_align.calibrate_dynamic(acc_warmup, gyro_warmup)

# Pre-calibrate VelocityNet scale factor
cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
wins = [np.ascontiguousarray(warmup[cols].iloc[i:i+20].values, dtype=np.float32) for i in range(0, len(warmup)-20, 5)]
preds = [predictor.predict(w)[0] / 2.0 for w in wins]
trues = [warmup[config.COL_TRUE_SPEED_MS].iloc[i:i+20].mean() for i in range(0, len(warmup)-20, 5)]
ratios = [t / max(pr, 1.0) for t, pr in zip(trues, preds) if t > 3.0]
k_calib = float(np.median(ratios))
print(f"Pre-calibrated k={k_calib:.3f}")

def run_outage(df_sub, init_lat, init_lon, init_spd, init_hdg, predictor, k_scale, align):
    n = len(df_sub)
    dt = 0.1
    ukf = UKFNavigationFilter(dt=dt)
    ukf.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
    
    acc_raw = df_sub[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_sub[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    
    est_pN, est_pE, est_speed = np.zeros(n), np.zeros(n), np.zeros(n)
    window_buffer = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = align.transform_imu(a_b, w_b)
        
        # In this dataset, phone orientation relative to vehicle:
        # Let's inspect w_v yaw rate
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)
        
        imu_sample = np.concatenate([a_b, w_b])
        window_buffer.append(imu_sample)

        if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
            win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
            pred_d, ev_class, sigma = predictor.predict(win_arr)

            if ev_class == 0 or pred_d < 0.1:
                ukf.update_zupt(gyro_reading=w_v[2])
            else:
                v_meas = (pred_d * k_scale) / 2.0
                var = ((sigma * k_scale) / 2.0)**2
                def h_v(s):
                    return np.array([s[2]])
                ukf.update_measurement(np.array([v_meas]), h_v, np.array([[max(var, 0.04)]]))

        if len(window_buffer) > 40:
            window_buffer.pop(0)

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed

for _, row in df_outages.iterrows():
    oid = int(row["outage_id"])
    dur = int(row["duration_s"])
    mask = (df_sim["outage_id"] == oid)
    df_sub = df.loc[mask]
    init_idx = max(0, mask.idxmax() - 1)
    init_lat = float(df.loc[init_idx, config.COL_TRUE_LAT])
    init_lon = float(df.loc[init_idx, config.COL_TRUE_LON])
    init_spd = float(df.loc[init_idx, config.COL_TRUE_SPEED_MS])
    init_hdg = float(df.loc[init_idx, config.COL_TRUE_HEADING])

    gt_lat = df_sub[config.COL_TRUE_LAT].values
    gt_lon = df_sub[config.COL_TRUE_LON].values
    gt_hdg = df_sub[config.COL_TRUE_HEADING].values

    p_lat, p_lon, _ = run_outage(df_sub, init_lat, init_lon, init_spd, init_hdg, predictor, k_calib, global_align)
    score = score_outage_segment(p_lat, p_lon, gt_lat, gt_lon, row["distance_travelled_m"], dur, gt_hdg)
    print(f"{dur:3d}s | Dist: {score['distance_m']:6.1f}m | FPE: {score['fpe_m']:5.2f}m | Drift: {score['drift_percent']:5.2f}% | Along: {score['along_track_rms_m']:5.2f}m | Cross: {score['cross_track_rms_m']:5.2f}m")
