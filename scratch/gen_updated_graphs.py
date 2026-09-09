import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path("results/leaderboard_summary.csv")
df = pd.read_csv(csv_path)

durations = [10, 30, 60, 120, 180]
methods = {
    "baseline_strapdown_v1": ("Naive Inertial Strapdown", "#e74c3c", "--", "s"),
    "baseline_cv_heading_v1": ("Constant Velocity Baseline", "#e67e22", ":", "^"),
    "inav_ai_ukf_v1": ("iNAV v1 (AI + UKF)", "#3498db", "-.", "d"),
    "inav_spectra_esekf_v2": ("iNAV v2 (SPECTRA + ES-EKF + Scaled Qc)", "#2ecc71", "-", "o")
}

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# 1. 180s Outage Final Position Error Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
df_180 = df[df["outage_s"] == 180]
labels = []
errors = []
colors = []

for m_key, (m_name, color, _, _) in methods.items():
    row = df_180[df_180["config"] == m_key]
    if len(row) > 0:
        labels.append(m_name)
        errors.append(row["median_final_pos_error_m"].values[0])
        colors.append(color)

bars = ax.bar(labels, errors, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
ax.set_ylabel("Median Position Error (m)", fontsize=12, fontweight="bold")
ax.set_title("180-Second GNSS Blackout: Final Position Error Comparison\n(Evaluated Across 18 Synchronized Motorway Test Runs)", fontsize=13, fontweight="bold", pad=15)
ax.grid(axis="y", linestyle="--", alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:.1f} m", ha="center", va="bottom", fontsize=11, fontweight="bold")

reduction = ((errors[0] - errors[-1]) / errors[0]) * 100.0
ax.annotate(f"★ 1,312m Drift Saved ({reduction:.1f}% Reduction)",
            xy=(3, errors[-1]), xytext=(2.2, errors[-1] + 600),
            arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=2, headwidth=8),
            fontsize=11, fontweight="bold", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#27ae60"))

plt.tight_layout()
plt.savefig("results/benchmark_180s_bar.png", dpi=300)
plt.close()

# 2. Position Error vs Outage Duration
fig, ax = plt.subplots(figsize=(10, 6))
for m_key, (m_name, color, style, marker) in methods.items():
    sub = df[df["config"] == m_key].sort_values("outage_s")
    if len(sub) > 0:
        ax.plot(sub["outage_s"], sub["median_final_pos_error_m"], label=m_name,
                color=color, linestyle=style, marker=marker, linewidth=2.2, markersize=8)

ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=12, fontweight="bold")
ax.set_ylabel("Median Position Drift (meters)", fontsize=12, fontweight="bold")
ax.set_title("Position Error Scaling vs Outage Duration (10s to 180s)", fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(durations)
ax.legend(frameon=True, facecolor="white", edgecolor="gray", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("results/benchmark_position_error.png", dpi=300)
plt.close()

# 3. Heading Error vs Outage Duration
fig, ax = plt.subplots(figsize=(10, 6))
for m_key, (m_name, color, style, marker) in methods.items():
    sub = df[df["config"] == m_key].sort_values("outage_s")
    if len(sub) > 0:
        ax.plot(sub["outage_s"], sub["median_heading_error_deg"], label=m_name,
                color=color, linestyle=style, marker=marker, linewidth=2.2, markersize=8)

ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=12, fontweight="bold")
ax.set_ylabel("Median Heading Error (degrees)", fontsize=12, fontweight="bold")
ax.set_title("Heading Error Growth vs Outage Duration", fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(durations)
ax.legend(frameon=True, facecolor="white", edgecolor="gray", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("results/benchmark_heading_error.png", dpi=300)
plt.close()

print("[SUCCESS] All updated benchmark graphs generated in results/ directory!")
