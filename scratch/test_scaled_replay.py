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

def test_file(fname):
    df = pd.read_parquet(config.SYNC_PROCESSED_DIR / fname)
    total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]
    schedule = generate_outage_schedule(total_dur, durations=[10, 30, 60])
    df_sim, df_outages = inject_outages(df, schedule)

    # Pre-outage warmup calibration (first 30s)
    warmup = df[df[config.COL_TIME] < 30.0]
    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    wins = [np.ascontiguousarray(warmup[cols].iloc[i:i+20].values, dtype=np.float32) for i in range(0, len(warmup)-20, 5)]
    preds = [predictor.predict(w)[0] / 2.0 for w in wins]
    trues = [warmup[config.COL_TRUE_SPEED_MS].iloc[i:i+20].mean() for i in range(0, len(warmup)-20, 5)]
    ratios = [t / max(pr, 1.0) for t, pr in zip(trues, preds) if t > 3.0]
    k_calib = float(np.median(ratios)) if ratios else 1.0

    print(f"\nTesting {fname} (k_calib = {k_calib:.3f}):")
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

        n = len(df_sub)
        dt = 0.1
        alignment = AlignmentEngine()
        acc_raw = df_sub[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
        gyro_raw = df_sub[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
        calib_samples = min(50, n)
        alignment.calibrate_static(acc_raw[:calib_samples], gyro_raw[:calib_samples])
        alignment.calibrate_dynamic(acc_raw[:calib_samples], gyro_raw[:calib_samples])

        ukf = UKFNavigationFilter(dt=dt)
        ukf.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
        est_pN, est_pE, est_speed = np.zeros(n), np.zeros(n), np.zeros(n)
        window_buffer = []

        for k in range(n):
            a_b = acc_raw[k]
            w_b = gyro_raw[k]
            a_v, w_v = alignment.transform_imu(a_b, w_b)
            ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)
            imu_sample = np.concatenate([a_b, w_b])
            window_buffer.append(imu_sample)

            if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
                win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
                pred_d, ev_class, sigma = predictor.predict(win_arr)

                if ev_class == 0 or pred_d < 0.1:
                    ukf.update_zupt(gyro_reading=w_v[2])
                else:
                    ukf.update_velocity_net(delta_d_pred=pred_d * k_calib, sigma_pred=sigma * k_calib, window_dur=2.0)

            if len(window_buffer) > 40:
                window_buffer.pop(0)

            est_pN[k] = ukf.x[0]
            est_pE[k] = ukf.x[1]
            est_speed[k] = ukf.x[2]

        p_lat, p_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
        score = score_outage_segment(p_lat, p_lon, df_sub[config.COL_TRUE_LAT].values, df_sub[config.COL_TRUE_LON].values, row["distance_travelled_m"], dur, df_sub[config.COL_TRUE_HEADING].values)
        d = score["duration_s"]
        dist = score["distance_m"]
        fpe = score["fpe_m"]
        dr = score["drift_percent"]
        along = score["along_track_rms_m"]
        cross = score["cross_track_rms_m"]
        print(f"  Duration: {d:3d}s | Dist: {dist:6.1f}m | FPE: {fpe:5.2f}m | Drift: {dr:5.2f}% | Along: {along:5.2f}m | Cross: {cross:5.2f}m")

for f in ["sync_vta10.parquet", "sync_vta11.parquet", "sync_vta12.parquet"]:
    test_file(f)
