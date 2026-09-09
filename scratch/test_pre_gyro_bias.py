import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import config
from modules.alignment import AlignmentEngine
from modules.spectra_net import SpectraNetPredictor
from eval.replay import estimate_scale_factor_rls, run_inav_spectra_esekf_pipeline
from eval.score import score_outage_segment
from eval.outage_sim import generate_outage_schedule, inject_outages

def estimate_pre_outage_gyro_bias(df_history: pd.DataFrame, alignment: AlignmentEngine, dt: float = 0.1) -> float:
    if len(df_history) < 20 or config.COL_TRUE_HEADING not in df_history.columns:
        return 0.0

    hdg = np.radians(df_history[config.COL_TRUE_HEADING].values)
    # unwrap heading angles to avoid 0/360 wrap around spikes
    hdg_unwrapped = np.unwrap(hdg)
    yaw_rate_gnss = np.diff(hdg_unwrapped) / dt

    # vehicle frame gyro yaw
    gyro_raw = df_history[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values[:-1]
    acc_raw = df_history[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values[:-1]
    _, w_v = alignment.transform_imu(acc_raw, gyro_raw)
    gyro_yaw = w_v[:, 2]

    # difference: gyro_yaw - yaw_rate_gnss is the residual gyro bias
    diff = gyro_yaw - yaw_rate_gnss
    # Gate to remove transient steering jerks (e.g. abs(yaw_rate) < 0.2 rad/s)
    mask = np.abs(yaw_rate_gnss) < 0.15
    if np.sum(mask) > 10:
        bg_est = float(np.median(diff[mask]))
    else:
        bg_est = float(np.median(diff))
    # Clamp to realistic MEMS bounds (e.g. +/- 0.05 rad/s ~= +/- 2.8 deg/s)
    return float(np.clip(bg_est, -0.05, 0.05))

# Test on first parquet file
p = Path('data/processed/synchronized/sync_vw2.parquet')
df = pd.read_parquet(p)
total_sec = len(df) * config.TARGET_DT
sched = generate_outage_schedule(total_sec, durations=[180])
df_sim, df_outages = inject_outages(df, sched)

predictor = SpectraNetPredictor()

for _, row in df_outages.head(3).iterrows():
    oid = int(row['outage_id'])
    dur = int(row['duration_s'])
    mask = (df_sim['outage_id'] == oid)
    df_sub = df.loc[mask]
    outage_start_idx = mask.idxmax()

    pre_df = df.iloc[max(0, outage_start_idx - 300):outage_start_idx]
    init_lat = float(df.loc[outage_start_idx, config.COL_TRUE_LAT])
    init_lon = float(df.loc[outage_start_idx, config.COL_TRUE_LON])
    init_spd = float(df.loc[outage_start_idx, config.COL_TRUE_SPEED_MS])
    init_hdg = float(df.loc[outage_start_idx, config.COL_TRUE_HEADING])

    alignment = AlignmentEngine()
    alignment.calibrate_static(pre_df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values[:50],
                               pre_df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values[:50])
    alignment.calibrate_dynamic(pre_df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values,
                                pre_df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values)

    k_scale = estimate_scale_factor_rls(pre_df, predictor)
    bg_est = estimate_pre_outage_gyro_bias(pre_df, alignment)

    # 1. Run standard (w_b = 0)
    p_lat1, p_lon1, _ = run_inav_spectra_esekf_pipeline(df_sub, init_lat, init_lon, init_spd, init_hdg, predictor, alignment, k_scale=k_scale)
    score1 = score_outage_segment(p_lat1, p_lon1, df_sub[config.COL_TRUE_LAT].values, df_sub[config.COL_TRUE_LON].values, row['distance_travelled_m'], dur)

    # 2. Run with pre-outage gyro bias compensation
    # Patch esekf predict_vehicular to subtract bg_est from yaw
    def run_calib(df_sub, bg):
        n = len(df_sub)
        acc_raw = df_sub[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
        gyro_raw = df_sub[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
        from modules.esekf import ESEKFNavigationFilter
        from eval.baseline import local_xy_to_latlon
        es = ESEKFNavigationFilter(dt=0.1)
        es.initialize(0.0, 0.0, init_spd, np.radians(init_hdg))
        es.w_b[2] = bg  # Inject online pre-outage estimated gyro bias!
        
        est_pN, est_pE = np.zeros(n), np.zeros(n)
        window_buffer = []
        for k in range(n):
            a_v, w_v = alignment.transform_imu(acc_raw[k], gyro_raw[k])
            es.predict_vehicular(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=0.1)
            window_buffer.append(np.concatenate([acc_raw[k], gyro_raw[k]]))
            if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
                win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
                pred_d, ev_class, sigma = predictor.predict(win_arr)
                if ev_class == 0 or pred_d < 0.1:
                    es.update_zupt()
                else:
                    v_fwd = (pred_d * k_scale) / 2.0
                    var = max((sigma * k_scale / 2.0) ** 2, 0.04)
                    es.update_forward_speed(v_fwd, var)
            if len(window_buffer) > 40:
                window_buffer.pop(0)
            est_pN[k], est_pE[k] = es.p[0], es.p[1]
        return local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)

    p_lat2, p_lon2 = run_calib(df_sub, bg_est)
    score2 = score_outage_segment(p_lat2, p_lon2, df_sub[config.COL_TRUE_LAT].values, df_sub[config.COL_TRUE_LON].values, row['distance_travelled_m'], dur)

    print(f"\n==================== OUTAGE {oid} (180s) ====================")
    print(f" Estimated Pre-outage Yaw Bias bg: {bg_est:.5f} rad/s ({np.degrees(bg_est):.3f} deg/s)")
    print(f" Standard (bg=0)       : FPE = {score1['fpe_m']:.1f}m | Drift% = {score1['drift_percent']:.2f}% | ATE = {score1['ate_m']:.1f}m")
    print(f" Calibrated Gyro Bias : FPE = {score2['fpe_m']:.1f}m | Drift% = {score2['drift_percent']:.2f}% | ATE = {score2['ate_m']:.1f}m")
    diff_m = score1['fpe_m'] - score2['fpe_m']
    print(f" IMPROVEMENT          : {diff_m:.1f}m saved ({(diff_m / score1['fpe_m'])*100:.1f}% reduction!)")
