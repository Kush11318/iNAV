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

best_pt = config.MODELS_DIR / "velocity_net_best.pt"
predictor = VelocityNetPredictor(str(best_pt))

def evaluate_with_preoutage(fpath):
    df = pd.read_parquet(fpath)
    total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]
    schedule = generate_outage_schedule(total_dur, durations=config.OUTAGE_DURATIONS_SEC)
    if not schedule:
        return []

    df_sim, df_outages = inject_outages(df, schedule)

    # Global run-level alignment calibration using warmup (GNSS healthy)
    warmup = df[df[config.COL_TIME] < 30.0]
    acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_w = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    
    run_align = AlignmentEngine()
    # Gravity is the mean acceleration over warmup
    mean_a = np.mean(acc_w, axis=0)
    run_align.u_z_body = -mean_a / np.linalg.norm(mean_a)
    z = run_align.u_z_body
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, ref)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    run_align.R_b_to_v = np.vstack([x, y, z])
    run_align.static_calibrated = True
    run_align.calibrate_dynamic(acc_w, gyro_w)

    # Scale factor calibration
    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    wins = [np.ascontiguousarray(warmup[cols].iloc[i:i+20].values, dtype=np.float32) for i in range(0, len(warmup)-20, 5)]
    preds = [predictor.predict(w)[0] / 2.0 for w in wins]
    trues = [warmup[config.COL_TRUE_SPEED_MS].iloc[i:i+20].mean() for i in range(0, len(warmup)-20, 5)]
    ratios = [t / max(pr, 1.0) for t, pr in zip(trues, preds) if t > 3.0]
    k_calib = float(np.clip(np.median(ratios), 0.5, 4.0)) if ratios else 1.0

    scores = []
    for _, row in df_outages.iterrows():
        oid = int(row["outage_id"])
        dur = int(row["duration_s"])
        mask = (df_sim["outage_id"] == oid)
        df_sub = df.loc[mask]
        if len(df_sub) < 5:
            continue

        # Extract pre-outage state from 1.0s window strictly before outage start
        outage_start_idx = mask.idxmax()
        pre_idx_start = max(0, outage_start_idx - 10)
        pre_df = df.iloc[pre_idx_start:outage_start_idx]

        init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
        init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
        init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
        # Circular mean of pre-outage heading
        h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
        init_hdg = np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0

        n = len(df_sub)
        dt = 0.1
        ukf = UKFNavigationFilter(dt=dt)
        ukf.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))

        acc_raw = df_sub[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
        gyro_raw = df_sub[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

        est_pN, est_pE = np.zeros(n), np.zeros(n)
        window_buffer = []

        for k in range(n):
            a_b = acc_raw[k]
            w_b = gyro_raw[k]
            a_v, w_v = run_align.transform_imu(a_b, w_b)
            ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)
            imu_sample = np.concatenate([a_b, w_b])
            window_buffer.append(imu_sample)

            if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
                win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
                pred_d, ev_class, sigma = predictor.predict(win_arr)

                if ev_class == 0 or pred_d < 0.1:
                    ukf.update_zupt(gyro_reading=w_v[2])
                else:
                    scaled_d = pred_d * k_calib
                    scaled_sigma = max(sigma * k_calib, 0.2)
                    v_net = scaled_d / 2.0
                    var = (scaled_sigma / 2.0)**2
                    def h_v(s):
                        return np.array([s[2]])
                    ukf.update_measurement(np.array([v_net]), h_v, np.array([[max(var, 0.04)]]))

            if len(window_buffer) > 40:
                window_buffer.pop(0)

            est_pN[k] = ukf.x[0]
            est_pE[k] = ukf.x[1]

        p_lat, p_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
        sc = score_outage_segment(p_lat, p_lon, df_sub[config.COL_TRUE_LAT].values, df_sub[config.COL_TRUE_LON].values, row["distance_travelled_m"], dur, df_sub[config.COL_TRUE_HEADING].values)
        scores.append(sc)

    return scores

files = ["sync_vfa01.parquet", "sync_vta14.parquet", "sync_vta16.parquet"]
all_scores = []
for f in files:
    fpath = config.SYNC_PROCESSED_DIR / f
    scs = evaluate_with_preoutage(fpath)
    for s in scs:
        all_scores.append(s)
        print(f"[{f[:10]}] {s['duration_s']:3d}s | Dist: {s['distance_m']:6.1f}m | FPE: {s['fpe_m']:5.2f}m | Drift: {s['drift_percent']:5.2f}%")

df_res = pd.DataFrame(all_scores)
print("\nMedian Drift by Duration:")
for dur in config.OUTAGE_DURATIONS_SEC:
    sub = df_res[df_res["duration_s"] == dur]
    if len(sub) > 0:
        print(f"  {dur:3d}s: {sub['drift_percent'].median():5.2f}% (FPE: {sub['fpe_m'].median():5.2f}m)")
