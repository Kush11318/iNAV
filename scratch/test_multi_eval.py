import sys
from pathlib import Path
import numpy as np
import pandas as pd
import glob

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

def evaluate_test_file(fpath):
    df = pd.read_parquet(fpath)
    total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]
    schedule = generate_outage_schedule(total_dur, durations=config.OUTAGE_DURATIONS_SEC)
    if not schedule:
        return []

    df_sim, df_outages = inject_outages(df, schedule)

    # Warmup calibration
    warmup = df[df[config.COL_TIME] < 30.0]
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
            est_speed[k] = ukf.x[2]

        p_lat, p_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
        sc = score_outage_segment(p_lat, p_lon, df_sub[config.COL_TRUE_LAT].values, df_sub[config.COL_TRUE_LON].values, row["distance_travelled_m"], dur, df_sub[config.COL_TRUE_HEADING].values)
        scores.append(sc)

    return scores

# Test on 6 diverse test files
files = [
    "sync_vta10.parquet", "sync_vta14.parquet", "sync_vta16.parquet",
    "sync_vta17.parquet", "sync_vfa01.parquet", "sync_vfa02.parquet"
]

all_sc = []
for fname in files:
    fpath = config.SYNC_PROCESSED_DIR / fname
    if not fpath.exists():
        continue
    scs = evaluate_test_file(fpath)
    print(f"\nResults for {fname}: {len(scs)} outages")
    for s in scs:
        d = s["duration_s"]
        dist = s["distance_m"]
        fpe = s["fpe_m"]
        dr = s["drift_percent"]
        print(f"  {d:3d}s | Dist: {dist:6.1f}m | FPE: {fpe:5.2f}m | Drift: {dr:5.2f}%")
        all_sc.append(s)

if all_sc:
    df_all = pd.DataFrame(all_sc)
    print("\n" + "="*50)
    print(f"OVERALL EVALUATION SUMMARY ({len(df_all)} outages):")
    for dur in config.OUTAGE_DURATIONS_SEC:
        sub = df_all[df_all["duration_s"] == dur]
        if len(sub) > 0:
            print(f"  Outage {dur:3d}s: Median Drift = {sub['drift_percent'].median():5.2f}%, Median FPE = {sub['fpe_m'].median():5.2f}m (n={len(sub)})")
    print(f"  TOTAL OVERALL MEDIAN DRIFT: {df_all['drift_percent'].median():5.2f}%")
