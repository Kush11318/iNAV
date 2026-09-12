import pandas as pd
import numpy as np

df = pd.read_csv('eval/phase10_nav_results.csv')
print('Total outages:', len(df))

da_med = df['drift_a_pct'].median()
da_mean = df['drift_a_pct'].mean()
da_p25 = df['drift_a_pct'].quantile(0.25)
da_p75 = df['drift_a_pct'].quantile(0.75)

db_med = df['drift_b_pct'].median()
db_mean = df['drift_b_pct'].mean()
db_p25 = df['drift_b_pct'].quantile(0.25)
db_p75 = df['drift_b_pct'].quantile(0.75)

print(f"Model A: Median Drift = {da_med:.2f}%, Mean Drift = {da_mean:.2f}%, P25 = {da_p25:.2f}%, P75 = {da_p75:.2f}%")
print(f"Model B: Median Drift = {db_med:.2f}%, Mean Drift = {db_mean:.2f}%, P25 = {db_p25:.2f}%, P75 = {db_p75:.2f}%")

fa_med = df['fpe_a_m'].median()
fa_mean = df['fpe_a_m'].mean()
fb_med = df['fpe_b_m'].median()
fb_mean = df['fpe_b_m'].mean()

print(f"Model A: Median FPE = {fa_med:.2f} m, Mean FPE = {fa_mean:.2f} m")
print(f"Model B: Median FPE = {fb_med:.2f} m, Mean FPE = {fb_mean:.2f} m")

aa_med = df['along_a_m'].median()
aa_mean = df['along_a_m'].mean()
ab_med = df['along_b_m'].median()
ab_mean = df['along_b_m'].mean()

print(f"Model A: Median Along-Track = {aa_med:.2f} m, Mean Along-Track = {aa_mean:.2f} m")
print(f"Model B: Median Along-Track = {ab_med:.2f} m, Mean Along-Track = {ab_mean:.2f} m")

ca_med = df['cross_a_m'].median()
cb_med = df['cross_b_m'].median()
print(f"Model A: Median Cross-Track = {ca_med:.2f} m, Model B: Median Cross-Track = {cb_med:.2f} m")

ha_med = df['hdg_a_deg'].median()
hb_med = df['hdg_b_deg'].median()
print(f"Model A: Median Heading Err = {ha_med:.2f} deg, Model B: Median Heading Err = {hb_med:.2f} deg")

diff_fpe = df['fpe_b_m'] - df['fpe_a_m']
improved = int((diff_fpe < -0.5).sum())
worsened = int((diff_fpe > 0.5).sum())
unchanged = int(len(df) - improved - worsened)
print(f"Delta Breakdown (B vs A): Improved = {improved}, Worsened = {worsened}, Unchanged = {unchanged}")

below_10_a = int((df['drift_a_pct'] < 10.0).sum())
below_10_b = int((df['drift_b_pct'] < 10.0).sum())
print(f"Outages < 10% Drift: Model A = {below_10_a}, Model B = {below_10_b}")

print("\n--- Per-Duration Results ---")
for dur in sorted(df['duration_s'].unique()):
    sub = df[df['duration_s'] == dur]
    da = sub['drift_a_pct'].median()
    db = sub['drift_b_pct'].median()
    fa = sub['fpe_a_m'].median()
    fb = sub['fpe_b_m'].median()
    aa = sub['along_a_m'].median()
    ab = sub['along_b_m'].median()
    print(f"Duration {dur:3d}s (N={len(sub):2d}): Drift A={da:.2f}%, Drift B={db:.2f}%, FPE A={fa:.2f}m, FPE B={fb:.2f}m, Along A={aa:.2f}m, Along B={ab:.2f}m")
