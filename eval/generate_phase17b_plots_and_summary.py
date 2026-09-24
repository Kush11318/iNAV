"""
Post-processing, verification, plotting, and forensic analysis for Phase 17B.
Loads the complete 56-outage results from eval/phase17b_heading_ablation_results.csv,
computes all statistical metrics, generates figures, and prints the audit tables.
"""

import math
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

csv_path = BASE_DIR / "eval" / "phase17b_heading_ablation_results.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} scenarios from {csv_path}")

# =============================================================================
# 1. VERIFY NUMERICAL REPRODUCTION OF PHASE 17A BASELINE (SYSTEM B vs 17A REF)
# =============================================================================
p17a_csv = BASE_DIR / "eval" / "phase17a_map_fusion_results.csv"
if p17a_csv.exists():
    df_17a = pd.read_csv(p17a_csv)
    merged = pd.merge(df, df_17a, on="scenario", suffixes=("_17b", "_17a"))
    fpe_diff = np.abs(merged["fpe_b_m_17b"] - merged["fpe_c_m_17a"]).max()
    print(f"\n[VERIFICATION] Max FPE diff between Phase 17B System B and Phase 17A System C: {fpe_diff:.6f} m")
    if fpe_diff < 1e-2:
        print("[VERIFICATION PASS] System B reproduces Phase 17A reference within numerical precision (<0.01m)!")
    else:
        print(f"[VERIFICATION WARNING] Max discrepancy: {fpe_diff:.4f} m")

# =============================================================================
# 2. OVERALL AGGREGATE METRICS (A vs B vs C)
# =============================================================================
med_fpe_a, med_fpe_b, med_fpe_c = df["fpe_a_m"].median(), df["fpe_b_m"].median(), df["fpe_c_m"].median()
mean_fpe_a, mean_fpe_b, mean_fpe_c = df["fpe_a_m"].mean(), df["fpe_b_m"].mean(), df["fpe_c_m"].mean()

med_drf_a, med_drf_b, med_drf_c = df["drift_a_pct"].median(), df["drift_b_pct"].median(), df["drift_c_pct"].median()
mean_drf_a, mean_drf_b, mean_drf_c = df["drift_a_pct"].mean(), df["drift_b_pct"].mean(), df["drift_c_pct"].mean()

med_along_a, med_along_b, med_along_c = df["along_a_m"].abs().median(), df["along_b_m"].abs().median(), df["along_c_m"].abs().median()
mean_along_a, mean_along_b, mean_along_c = df["along_a_m"].abs().mean(), df["along_b_m"].abs().mean(), df["along_c_m"].abs().mean()

med_cross_a, med_cross_b, med_cross_c = df["cross_a_m"].abs().median(), df["cross_b_m"].abs().median(), df["cross_c_m"].abs().median()
mean_cross_a, mean_cross_b, mean_cross_c = df["cross_a_m"].abs().mean(), df["cross_b_m"].abs().mean(), df["cross_c_m"].abs().mean()

med_hdg_a, med_hdg_b, med_hdg_c = df["hdg_a_deg"].dropna().median(), df["hdg_b_deg"].dropna().median(), df["hdg_c_deg"].dropna().median()
mean_hdg_a, mean_hdg_b, mean_hdg_c = df["hdg_a_deg"].dropna().mean(), df["hdg_b_deg"].dropna().mean(), df["hdg_c_deg"].dropna().mean()

n_below_10_a = (df["drift_a_pct"] < 10.0).sum()
n_below_10_b = (df["drift_b_pct"] < 10.0).sum()
n_below_10_c = (df["drift_c_pct"] < 10.0).sum()

n_imp = (df["b_to_c_status"] == "IMPROVED").sum()
n_wor = (df["b_to_c_status"] == "WORSENED").sum()
n_unc = (df["b_to_c_status"] == "UNCHANGED").sum()

tot_pos_acc = df["pos_map_accepted"].sum()
tot_pos_tot = df["pos_map_accepted"].sum() + df["pos_map_rejected"].sum()
tot_hdg_acc = df["hdg_map_accepted"].sum()
tot_hdg_tot = df["hdg_map_accepted"].sum() + df["hdg_map_rejected"].sum()

mean_pos_nis = df["mean_pos_nis"].mean()
mean_hdg_nis = df["mean_hdg_nis"].mean()

print("\n" + "=" * 135)
print("PHASE 17B ABLATION: OVERALL AGGREGATE SUMMARY (56 HELD-OUT SCENARIOS)")
print("=" * 135)

def fmt(label, va, vb, vc, unit=""):
    dbc = vc - vb
    pbc = (dbc / vb * 100.0) if abs(vb) > 1e-6 else 0.0
    s = "+" if dbc > 0 else ""
    return f"{label:<32} | {va:8.2f} {unit:<3} | {vb:8.2f} {unit:<3} | {vc:8.2f} {unit:<3} | {s}{dbc:6.2f} {unit:<3} ({s}{pbc:5.1f}%)"

print(f"{'Metric':<32} | {'System A':<12} | {'System B':<12} | {'System C':<12} | {'B -> C Delta':<18}")
print("-" * 135)
print(fmt("Median FPE", med_fpe_a, med_fpe_b, med_fpe_c, "m"))
print(fmt("Mean FPE", mean_fpe_a, mean_fpe_b, mean_fpe_c, "m"))
print(fmt("Median Drift %", med_drf_a, med_drf_b, med_drf_c, "%"))
print(fmt("Mean Drift %", mean_drf_a, mean_drf_b, mean_drf_c, "%"))
print(fmt("Median Abs Along-Track Error", med_along_a, med_along_b, med_along_c, "m"))
print(fmt("Mean Abs Along-Track Error", mean_along_a, mean_along_b, mean_along_c, "m"))
print(fmt("Median Abs Cross-Track Error", med_cross_a, med_cross_b, med_cross_c, "m"))
print(fmt("Mean Abs Cross-Track Error", mean_cross_a, mean_cross_b, mean_cross_c, "m"))
print(fmt("Median Heading Error", med_hdg_a, med_hdg_b, med_hdg_c, "°"))
print(fmt("Mean Heading Error", mean_hdg_a, mean_hdg_b, mean_hdg_c, "°"))
print("-" * 135)
print(f"Scenarios Below 10% Drift: A = {n_below_10_a}/56 ({n_below_10_a/56*100:.1f}%), B = {n_below_10_b}/56 ({n_below_10_b/56*100:.1f}%), C = {n_below_10_c}/56 ({n_below_10_c/56*100:.1f}%)")
print(f"B -> C Comparison: {n_imp} Improved, {n_wor} Worsened, {n_unc} Unchanged (Net: {n_imp - n_wor:+d})")
print(f"Map Position Updates: {tot_pos_acc}/{tot_pos_tot} ({tot_pos_acc/max(tot_pos_tot, 1)*100:.1f}%) accepted | Mean NIS: {mean_pos_nis:.2f}")
print(f"Map Heading Updates:  {tot_hdg_acc}/{tot_hdg_tot} ({tot_hdg_acc/max(tot_hdg_tot, 1)*100:.1f}%) accepted | Mean NIS: {mean_hdg_nis:.2f}")

# =============================================================================
# 3. DURATION BREAKDOWN
# =============================================================================
print("\n" + "=" * 135)
print("DURATION BREAKDOWN (Median FPE / Drift % / Heading Error)")
print("=" * 135)
print(f"{'Duration':<10} | {'Count':<6} | {'A FPE (m)':<10} | {'B FPE (m)':<10} | {'C FPE (m)':<10} | {'B->C Δ FPE':<12} | {'B Drift%':<10} | {'C Drift%':<10} | {'B Hdg(°)':<10} | {'C Hdg(°)':<10} | {'B->C Imp/Wor'}")
print("-" * 135)

durs = [10, 30, 60, 120, 180]
for dur_val in durs:
    sub = df[df["duration_s"] == dur_val]
    cnt = len(sub)
    if cnt == 0:
        continue
    fpe_a = sub["fpe_a_m"].median()
    fpe_b = sub["fpe_b_m"].median()
    fpe_c = sub["fpe_c_m"].median()
    dfpe = fpe_c - fpe_b
    drf_b = sub["drift_b_pct"].median()
    drf_c = sub["drift_c_pct"].median()
    hdg_b = sub["hdg_b_deg"].dropna().median()
    hdg_c = sub["hdg_c_deg"].dropna().median()
    imp = (sub["b_to_c_status"] == "IMPROVED").sum()
    wor = (sub["b_to_c_status"] == "WORSENED").sum()
    print(f"{dur_val:3d}s        | {cnt:<6} | {fpe_a:10.2f} | {fpe_b:10.2f} | {fpe_c:10.2f} | {dfpe:+10.2f}m  | {drf_b:9.2f}% | {drf_c:9.2f}% | {hdg_b:9.2f}° | {hdg_c:9.2f}° | {imp} / {wor}")

# =============================================================================
# 4. SENSITIVITY BREAKDOWN
# =============================================================================
print("\n" + "=" * 135)
print("PARAMETER SENSITIVITY: sigma_heading = {1.0°, 2.0°, 5.0°}")
print("=" * 135)
print(f"System B (No Heading):       Median FPE = {med_fpe_b:6.2f} m | Mean FPE = {mean_fpe_b:6.2f} m")
print(f"System C (sigma = 1.0 deg):  Median FPE = {df['fpe_c_1deg_m'].median():6.2f} m | Mean FPE = {df['fpe_c_1deg_m'].mean():6.2f} m")
print(f"System C (sigma = 2.0 deg):  Median FPE = {med_fpe_c:6.2f} m | Mean FPE = {mean_fpe_c:6.2f} m")
print(f"System C (sigma = 5.0 deg):  Median FPE = {df['fpe_c_5deg_m'].median():6.2f} m | Mean FPE = {df['fpe_c_5deg_m'].mean():6.2f} m")

# =============================================================================
# 5. FORENSIC AUDIT OF 120s & 180s OUTAGES
# =============================================================================
print("\n" + "=" * 135)
print("FORENSIC AUDIT: 120s & 180s SCENARIOS (Phase 17A vs Phase 17B)")
print("=" * 135)
long_df = df[df["duration_s"] >= 120].sort_values("duration_s")
for _, r in long_df.iterrows():
    sc = r["scenario"]
    dur = r["duration_s"]
    fa, fb, fc = r["fpe_a_m"], r["fpe_b_m"], r["fpe_c_m"]
    hb, hc = r["hdg_b_deg"], r["hdg_c_deg"]
    p_acc = r["pos_map_accepted"]
    p_rej = r["pos_map_rejected"]
    h_acc = r["hdg_map_accepted"]
    h_rej = r["hdg_map_rejected"]
    dfpe = fc - fb
    stat = r["b_to_c_status"]

    # Diagnose mechanism
    if p_acc + p_rej == 0:
        diag = "No OSM graph coverage (off-network / abstained safely)"
    elif h_acc == 0 and h_rej > 0:
        diag = "Heading gated out (angular discrepancy > 15° or NIS > 6.635)"
    elif h_acc > 0 and dfpe < -10.0:
        diag = "Heading update corrected gyro drift and improved position"
    elif h_acc > 0 and dfpe > 10.0:
        diag = "Heading update pulled filter toward misaligned segment / road-bend"
    elif abs(dfpe) <= 10.0:
        diag = "Negligible effect / filter already constrained by road-normal"
    else:
        diag = "Mixed interaction"

    print(f"[{sc:<30}] Dur={dur:3d}s | FPE: B={fb:6.1f}m, C={fc:6.1f}m (Δ={dfpe:+6.1f}m, {stat:<9}) | Hdg: B={hb:4.1f}°, C={hc:4.1f}° | PosAcc={p_acc}/{p_acc+p_rej} | HdgAcc={h_acc}/{h_acc+h_rej} | Diag: {diag}")

# =============================================================================
# 6. GENERATE PUBLICATION-GRADE PLOTS
# =============================================================================
print("\nGenerating publication-grade plots...")

# Plot 1: Summary Comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Phase 17B: Map Heading Ablation Summary (56 Held-Out Outages)", fontsize=15, fontweight="bold")

metrics_to_plot = [
    ("Median FPE (m)", [med_fpe_a, med_fpe_b, med_fpe_c], axes[0, 0]),
    ("Median Drift (%)", [med_drf_a, med_drf_b, med_drf_c], axes[0, 1]),
    ("Median Heading Error (°)", [med_hdg_a, med_hdg_b, med_hdg_c], axes[0, 2]),
    ("Median Abs Along-Track Error (m)", [med_along_a, med_along_b, med_along_c], axes[1, 0]),
    ("Median Abs Cross-Track Error (m)", [med_cross_a, med_cross_b, med_cross_c], axes[1, 1]),
    ("Scenarios < 10% Drift", [n_below_10_a, n_below_10_b, n_below_10_c], axes[1, 2])
]

colors = ["#4A90E2", "#E2844A", "#2ECC71"]
systems = ["Sys A\n(CAN Only)", "Sys B\n(CAN+Normal)", "Sys C\n(CAN+Normal+Hdg)"]

for title, vals, ax in metrics_to_plot:
    bars = ax.bar(systems, vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, yval * 1.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
fig1_path = PLOTS_DIR / "phase17b_fig1_ablation_summary.png"
fig1_art = ARTIFACTS_DIR / "phase17b_fig1_ablation_summary.png"
plt.savefig(fig1_path, dpi=300)
plt.savefig(fig1_art, dpi=300)
plt.close()

# Plot 2: Duration Breakdown
fpe_a_durs = [df[df["duration_s"] == d]["fpe_a_m"].median() for d in durs]
fpe_b_durs = [df[df["duration_s"] == d]["fpe_b_m"].median() for d in durs]
fpe_c_durs = [df[df["duration_s"] == d]["fpe_c_m"].median() for d in durs]

hdg_a_durs = [df[df["duration_s"] == d]["hdg_a_deg"].dropna().median() for d in durs]
hdg_b_durs = [df[df["duration_s"] == d]["hdg_b_deg"].dropna().median() for d in durs]
hdg_c_durs = [df[df["duration_s"] == d]["hdg_c_deg"].dropna().median() for d in durs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Phase 17B: Performance by Outage Duration (10s to 180s)", fontsize=15, fontweight="bold")

x = np.arange(len(durs))
width = 0.25

ax1.bar(x - width, fpe_a_durs, width, label="System A (CAN Only)", color="#4A90E2", edgecolor="black")
ax1.bar(x, fpe_b_durs, width, label="System B (CAN + Road-Normal)", color="#E2844A", edgecolor="black")
ax1.bar(x + width, fpe_c_durs, width, label="System C (CAN + Normal + Heading)", color="#2ECC71", edgecolor="black")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{d}s" for d in durs])
ax1.set_ylabel("Median FPE (m)", fontsize=12)
ax1.set_title("Median FPE vs Outage Duration", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.bar(x - width, hdg_a_durs, width, label="System A (CAN Only)", color="#4A90E2", edgecolor="black")
ax2.bar(x, hdg_b_durs, width, label="System B (CAN + Road-Normal)", color="#E2844A", edgecolor="black")
ax2.bar(x + width, hdg_c_durs, width, label="System C (CAN + Normal + Heading)", color="#2ECC71", edgecolor="black")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{d}s" for d in durs])
ax2.set_ylabel("Median Heading Error (°)", fontsize=12)
ax2.set_title("Median Heading Error vs Outage Duration", fontsize=13, fontweight="bold")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig2_path = PLOTS_DIR / "phase17b_fig2_duration_breakdown.png"
fig2_art = ARTIFACTS_DIR / "phase17b_fig2_duration_breakdown.png"
plt.savefig(fig2_path, dpi=300)
plt.savefig(fig2_art, dpi=300)
plt.close()

# Plot 3: Sensitivity Comparison
fig, ax = plt.subplots(figsize=(10, 6))
sig_labels = ["Sys B\n(No Heading)", "Sys C\n(σ=1.0°)", "Sys C\n(σ=2.0°)", "Sys C\n(σ=5.0°)"]
sig_med_fpe = [med_fpe_b, df['fpe_c_1deg_m'].median(), med_fpe_c, df['fpe_c_5deg_m'].median()]
sig_mean_fpe = [mean_fpe_b, df['fpe_c_1deg_m'].mean(), mean_fpe_c, df['fpe_c_5deg_m'].mean()]

x_s = np.arange(len(sig_labels))
w = 0.35
b1 = ax.bar(x_s - w/2, sig_med_fpe, w, label="Median FPE (m)", color="#2ECC71", edgecolor="black")
b2 = ax.bar(x_s + w/2, sig_mean_fpe, w, label="Mean FPE (m)", color="#3498DB", edgecolor="black")
ax.set_xticks(x_s)
ax.set_xticklabels(sig_labels, fontsize=11, fontweight="bold")
ax.set_ylabel("FPE (m)", fontsize=12)
ax.set_title("Phase 17B: Parameter Sensitivity on Heading Uncertainty (σ_heading)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

for b in b1:
    yval = b.get_height()
    ax.text(b.get_x() + b.get_width()/2.0, yval * 1.02, f"{yval:.1f}m", ha="center", va="bottom", fontsize=10)
for b in b2:
    yval = b.get_height()
    ax.text(b.get_x() + b.get_width()/2.0, yval * 1.02, f"{yval:.1f}m", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
fig3_path = PLOTS_DIR / "phase17b_fig3_sensitivity_comparison.png"
fig3_art = ARTIFACTS_DIR / "phase17b_fig3_sensitivity_comparison.png"
plt.savefig(fig3_path, dpi=300)
plt.savefig(fig3_art, dpi=300)
plt.close()

print("All plots generated and saved successfully!")
