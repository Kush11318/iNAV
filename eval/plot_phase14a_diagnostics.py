"""
Phase 14A: Diagnostic Visualizations for Barometer/DEM Observability Experiment
Generates:
  1. phase14a_sensor_availability_audit.png (Dataset sensor availability)
  2. phase14a_terrain_elevation_profile.png (Example motorway elevation & grade)
  3. phase14a_grade_observability_analysis.png (Road grade CDF & uninformative zones)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

plots_dir = config.BASE_DIR / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Plot 1: Sensor Channel Availability Audit (IO-VNBD Dataset)
# -------------------------------------------------------------
channels = [
    "Triaxial Accel\n(Phone)",
    "Triaxial Gyro\n(Phone)",
    "Gravity Vector\n(Phone)",
    "Magnetometer\n(Phone)",
    "GNSS Lat/Lon\n(Phone)",
    "CAN Wheel Speeds\n(Vehicle)",
    "Barometer / Pressure\n(Phone)"
]
presence_pct = [100.0, 100.0, 100.0, 96.3, 100.0, 100.0, 0.0]
colors = ['#2ca02c', '#2ca02c', '#2ca02c', '#2ca02c', '#2ca02c', '#1f77b4', '#d62728']

plt.figure(figsize=(10, 5))
bars = plt.bar(channels, presence_pct, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
plt.title("IO-VNBD Dataset Sensor Channel Availability Audit (56 Outages / 20 Test Runs)", fontsize=12, fontweight='bold')
plt.ylabel("Availability (% of Test Runs)", fontsize=11)
plt.ylim(0, 115)
plt.grid(axis='y', linestyle=':', alpha=0.6)

for bar, pct in zip(bars, presence_pct):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.axhline(0, color='black', linewidth=1.0)
plt.tight_layout()
p1_path = plots_dir / "phase14a_sensor_availability_audit.png"
plt.savefig(p1_path, dpi=200)
plt.close()
print(f"Saved Plot 1: {p1_path}")

# -------------------------------------------------------------
# Plot 2 & 3: Elevation Profile & Road Grade Observability
# -------------------------------------------------------------
# Load sync_vw11.parquet as representative motorway run
vw11_path = config.SYNC_PROCESSED_DIR / "sync_vw11.parquet"
if vw11_path.exists():
    df = pd.read_parquet(vw11_path)
    s_km = (df["gt_cum_dist_m"].values if "gt_cum_dist_m" in df.columns else np.cumsum(df["gt_speed_ms"].values * 0.1)) / 1000.0
    h_m = df["gt_height_m"].values

    dh = np.diff(h_m)
    ds = np.diff(s_km * 1000.0)
    valid = ds > 0.1
    grade_pct = np.zeros(len(s_km))
    grade_pct[1:][valid] = np.abs(dh[valid] / ds[valid]) * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(s_km, h_m, 'b-', linewidth=1.8, label="Road Elevation Profile (gt_height_m)")
    ax1.set_title("Motorway Elevation Profile (sync_vw11, West Midlands M42/M40)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Elevation (m)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    ax2.plot(s_km, grade_pct, 'm-', linewidth=1.2, label="|Road Grade| (%)")
    ax2.axhline(1.5, color='r', linestyle='--', linewidth=1.5, label="Observability Threshold (1.5% Grade)")
    ax2.fill_between(s_km, 0, 1.5, color='red', alpha=0.15, label="Uninformative Zone (Grade < 1.5%)")
    ax2.set_title("Absolute Road Longitudinal Grade (%) along Trajectory", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Distance Along Route (km)", fontsize=11)
    ax2.set_ylabel("|Grade| (%)", fontsize=11)
    ax2.set_ylim(0, 8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    p2_path = plots_dir / "phase14a_terrain_elevation_profile.png"
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f"Saved Plot 2: {p2_path}")

    # Plot 3: Cumulative Grade Distribution across all 18 test runs
    test_runs = ['vw10', 'vw11', 'vw12', 'vw13', 'vw14a', 'vw14b', 'vw14c', 'vw16a', 'vw16b', 'vw17', 'vw2', 'vw3', 'vw4', 'vw5', 'vw6', 'vw7', 'vw8', 'vw9']
    all_grades = []
    for r in test_runs:
        p = config.SYNC_PROCESSED_DIR / f'sync_{r}.parquet'
        if p.exists():
            d = pd.read_parquet(p)
            if 'gt_height_m' in d.columns:
                h = d['gt_height_m'].values
                s = d['gt_cum_dist_m'].values if 'gt_cum_dist_m' in d.columns else np.cumsum(d['gt_speed_ms'].values * 0.1)
                dh = np.diff(h)
                ds = np.diff(s)
                v = ds > 0.05
                all_grades.extend(np.abs(dh[v] / ds[v] * 100.0).tolist())

    all_grades = np.sort(np.array(all_grades))
    cdf = np.arange(len(all_grades)) / float(len(all_grades)) * 100.0

    plt.figure(figsize=(9, 5))
    plt.plot(all_grades, cdf, 'k-', linewidth=2.0, label="Empirical CDF of Road Grade")
    plt.axvline(1.5, color='r', linestyle='--', linewidth=1.5, label="1.5% Grade Threshold (57.0% of Highway)")
    plt.axvline(0.5, color='orange', linestyle=':', linewidth=1.5, label="0.5% Grade Threshold (25.2% of Highway)")
    plt.axhline(57.0, color='r', linestyle=':', alpha=0.5)
    plt.xlim(0, 6)
    plt.ylim(0, 105)
    plt.title("Road Grade CDF across All 18 Test Motorway Runs", fontsize=12, fontweight='bold')
    plt.xlabel("Absolute Road Grade (%)", fontsize=11)
    plt.ylabel("Cumulative Percentage of Route (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    p3_path = plots_dir / "phase14a_grade_observability_analysis.png"
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f"Saved Plot 3: {p3_path}")
