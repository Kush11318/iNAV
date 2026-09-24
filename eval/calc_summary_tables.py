import pandas as pd
import numpy as np

# Table 1: Stationary statistics
df_stops = pd.read_csv('eval/phase18b_standstill_events.csv')
print('=== TABLE 1: STANDSTILL STATS ===')
print(f'Total accepted events: {len(df_stops)}')
print(df_stops['dur_s'].describe())
print(f'Gyro X std: med={df_stops["std_wx_rad"].median()*57.2958:.4f} deg/s, mean={df_stops["std_wx_rad"].mean()*57.2958:.4f} deg/s')
print(f'Gyro Y std: med={df_stops["std_wy_rad"].median()*57.2958:.4f} deg/s, mean={df_stops["std_wy_rad"].mean()*57.2958:.4f} deg/s')
print(f'Gyro Z std: med={df_stops["std_wz_rad"].median()*57.2958:.4f} deg/s, mean={df_stops["std_wz_rad"].mean()*57.2958:.4f} deg/s')
print(f'Gyro Z mean: med={df_stops["mean_wz_rad"].median()*57.2958:.4f} deg/s, std across stops={df_stops["mean_wz_rad"].std()*57.2958:.4f} deg/s')
print(f'Accel std: med={df_stops["std_amag"].median():.4f} m/s^2, mean={df_stops["std_amag"].mean():.4f} m/s^2')
print(f'CAN speed max during stop: med={df_stops["max_v_can"].median():.4f} m/s, max={df_stops["max_v_can"].max():.4f} m/s')

# Table 2: Bias estimation calibration
df_calib = pd.read_csv('eval/phase18b_calibration_results.csv')
print('\n=== TABLE 2: BIAS ESTIMATION ===')
print(f'Total events: {len(df_calib)}')
print(f'Mean |delta_bg|: {df_calib["bias_corr_mag_deg"].mean():.4f} deg/s ({df_calib["bias_corr_mag_deg"].mean()/57.2958:.6f} rad/s)')
print(f'Median cov reduction factor: {df_calib["cov_reduction_factor"].median():.1f}x (mean: {df_calib["cov_reduction_factor"].mean():.1f}x)')
print(f'Mean sigma_bg before: {df_calib["sigma_bg_before_deg"].mean():.4f} deg/s, after: {df_calib["sigma_bg_after_deg"].mean():.4f} deg/s')

# Table 3: Heading drift
df_drift = pd.read_csv('eval/phase18b_post_stop_heading_drift.csv')
print('\n=== TABLE 3: HEADING DRIFT ===')
for h in [10, 30, 60, 120, 180]:
    colA, colB, colC = f'err_A_{h}s', f'err_B_{h}s', f'err_C_{h}s'
    sub = df_drift.dropna(subset=[colA, colB, colC])
    print(f'{h:3d}s (N={len(sub):2d}): A={sub[colA].median():.2f} deg | B={sub[colB].median():.2f} deg | C={sub[colC].median():.2f} deg | delta(B-A)={sub[colB].median()-sub[colA].median():+.2f} deg')

# Table 4: Benchmark
df_bench = pd.read_csv('eval/phase18b_zaru_benchmark_results.csv')
print('\n=== TABLE 4: NAVIGATION BENCHMARK ===')
for name, sub in [('ALL (N=56)', df_bench), ('WITH PRE-STOP (N=26)', df_bench[df_bench['has_pre_stop']]), ('WITHOUT PRE-STOP (N=30)', df_bench[~df_bench['has_pre_stop']])]:
    print(f'-- {name} --')
    for m in ['A', 'B', 'C']:
        med_fpe = sub[f'fpe_{m}'].median()
        mean_fpe = sub[f'fpe_{m}'].mean()
        max_fpe = sub[f'fpe_{m}'].max()
        med_drift = sub[f'drift_pct_{m}'].median()
        pct_under_10 = (sub[f'drift_pct_{m}'] < 10.0).mean() * 100.0
        along = sub[f'along_track_{m}'].median()
        cross = sub[f'cross_track_{m}'].median()
        hdg = sub[f'heading_err_{m}'].median()
        print(f'  Sys {m}: Med FPE={med_fpe:6.2f}m | Mean={mean_fpe:9.2f}m | Max={max_fpe:9.2f}m | Drift={med_drift:5.2f}% | <10%={pct_under_10:4.1f}% | Along={along:6.2f}m | Cross={cross:6.2f}m | Hdg={hdg:5.2f} deg')

# Breakdown by duration
print('\n-- BREAKDOWN BY DURATION --')
for dur in [30, 60, 120, 180]:
    sub = df_bench[df_bench['duration_s'] == dur]
    print(f'Duration {dur}s (N={len(sub)}):')
    for m in ['A', 'B', 'C']:
        print(f'  Sys {m}: Med FPE={sub[f"fpe_{m}"].median():6.2f}m | Med Drift={sub[f"drift_pct_{m}"].median():5.2f}% | Med Hdg={sub[f"heading_err_{m}"].median():5.2f} deg')
