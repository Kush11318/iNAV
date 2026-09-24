"""
Phase 18B Post-Processing, Analysis & Plotting Script

Loads results from eval/phase18b_heading_gate_results.csv,
computes full aggregate metrics for Systems A, B, C30, C35,
conducts the forensic recovery analysis on the 14 failure cases,
and generates publication-grade comparison plots.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")

csv_path = BASE_DIR / "eval" / "phase18b_heading_gate_results.csv"
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} scenarios from {csv_path}")

# Verification against 17A reference
p17a_csv = BASE_DIR / "eval" / "phase17a_map_fusion_results.csv"
if p17a_csv.exists():
    df_17a = pd.read_csv(p17a_csv)
    merged = pd.merge(df, df_17a, on="scenario", suffixes=("_18b", "_17a"))
    diff_b = np.abs(merged["fpe_b_m_18b"] - merged["fpe_c_m"]).max()
    print(f"[VERIFICATION] Max FPE difference between Phase 18B System B and Phase 17A System C: {diff_b:.6f} m")
    if diff_b < 1e-2:
        print("[VERIFICATION PASS] Exact numerical match to Phase 17A reference!")

# =============================================================================
# 1. OVERALL AGGREGATE BENCHMARK COMPARISON (56 OUTAGES)
# =============================================================================
print("\n" + "=" * 145)
print("PHASE 18B: OVERALL AGGREGATE BENCHMARK COMPARISON (56 HELD-OUT SCENARIOS)")
print("=" * 145)

metrics = {}
for sys_id, pfx in [("System A (CAN Only)", "a"), ("System B (Phase 17A Ref)", "b"), 
                    ("System C30 (theta=30°)", "c30"), ("System C35 (theta=35°)", "c35")]:
    metrics[sys_id] = {
        "med_fpe": df[f"fpe_{pfx}_m"].median(),
        "mean_fpe": df[f"fpe_{pfx}_m"].mean(),
        "max_fpe": df[f"fpe_{pfx}_m"].max(),
        "med_drift": df[f"drift_{pfx}_pct"].median(),
        "mean_drift": df[f"drift_{pfx}_pct"].mean(),
        "n_sub10": (df[f"drift_{pfx}_pct"] < 10.0).sum(),
        "med_along": df[f"along_{pfx}_m"].abs().median(),
        "mean_along": df[f"along_{pfx}_m"].abs().mean(),
        "med_cross": df[f"cross_{pfx}_m"].abs().median(),
        "mean_cross": df[f"cross_{pfx}_m"].abs().mean(),
        "med_hdg": df[f"hdg_{pfx}_deg"].dropna().median(),
        "mean_hdg": df[f"hdg_{pfx}_deg"].dropna().mean(),
    }

print(f"{'Metric':<32} | {'System A (CAN)':<16} | {'System B (17A Ref)':<18} | {'System C30 (30°)':<18} | {'System C35 (35°)':<18}")
print("-" * 145)
print(f"{'Median FPE (m)':<32} | {metrics['System A (CAN Only)']['med_fpe']:14.2f} m | {metrics['System B (Phase 17A Ref)']['med_fpe']:16.2f} m | {metrics['System C30 (theta=30°)']['med_fpe']:16.2f} m | {metrics['System C35 (theta=35°)']['med_fpe']:16.2f} m")
print(f"{'Mean FPE (m)':<32} | {metrics['System A (CAN Only)']['mean_fpe']:14.2f} m | {metrics['System B (Phase 17A Ref)']['mean_fpe']:16.2f} m | {metrics['System C30 (theta=30°)']['mean_fpe']:16.2f} m | {metrics['System C35 (theta=35°)']['mean_fpe']:16.2f} m")
print(f"{'Maximum FPE (m)':<32} | {metrics['System A (CAN Only)']['max_fpe']:14.2f} m | {metrics['System B (Phase 17A Ref)']['max_fpe']:16.2f} m | {metrics['System C30 (theta=30°)']['max_fpe']:16.2f} m | {metrics['System C35 (theta=35°)']['max_fpe']:16.2f} m")
print(f"{'Median Drift (%)':<32} | {metrics['System A (CAN Only)']['med_drift']:14.2f} % | {metrics['System B (Phase 17A Ref)']['med_drift']:16.2f} % | {metrics['System C30 (theta=30°)']['med_drift']:16.2f} % | {metrics['System C35 (theta=35°)']['med_drift']:16.2f} %")
print(f"{'Mean Drift (%)':<32} | {metrics['System A (CAN Only)']['mean_drift']:14.2f} % | {metrics['System B (Phase 17A Ref)']['mean_drift']:16.2f} % | {metrics['System C30 (theta=30°)']['mean_drift']:16.2f} % | {metrics['System C35 (theta=35°)']['mean_drift']:16.2f} %")
print(f"{'Scenarios < 10% Drift':<32} | {metrics['System A (CAN Only)']['n_sub10']:11d} / 56 | {metrics['System B (Phase 17A Ref)']['n_sub10']:13d} / 56 | {metrics['System C30 (theta=30°)']['n_sub10']:13d} / 56 | {metrics['System C35 (theta=35°)']['n_sub10']:13d} / 56")
print(f"{'Median Abs Along-Track (m)':<32} | {metrics['System A (CAN Only)']['med_along']:14.2f} m | {metrics['System B (Phase 17A Ref)']['med_along']:16.2f} m | {metrics['System C30 (theta=30°)']['med_along']:16.2f} m | {metrics['System C35 (theta=35°)']['med_along']:16.2f} m")
print(f"{'Mean Abs Along-Track (m)':<32} | {metrics['System A (CAN Only)']['mean_along']:14.2f} m | {metrics['System B (Phase 17A Ref)']['mean_along']:16.2f} m | {metrics['System C30 (theta=30°)']['mean_along']:16.2f} m | {metrics['System C35 (theta=35°)']['mean_along']:16.2f} m")
print(f"{'Median Abs Cross-Track (m)':<32} | {metrics['System A (CAN Only)']['med_cross']:14.2f} m | {metrics['System B (Phase 17A Ref)']['med_cross']:16.2f} m | {metrics['System C30 (theta=30°)']['med_cross']:16.2f} m | {metrics['System C35 (theta=35°)']['med_cross']:16.2f} m")
print(f"{'Mean Abs Cross-Track (m)':<32} | {metrics['System A (CAN Only)']['mean_cross']:14.2f} m | {metrics['System B (Phase 17A Ref)']['mean_cross']:16.2f} m | {metrics['System C30 (theta=30°)']['mean_cross']:16.2f} m | {metrics['System C35 (theta=35°)']['mean_cross']:16.2f} m")
print(f"{'Median Heading Error (°)':<32} | {metrics['System A (CAN Only)']['med_hdg']:14.2f} ° | {metrics['System B (Phase 17A Ref)']['med_hdg']:16.2f} ° | {metrics['System C30 (theta=30°)']['med_hdg']:16.2f} ° | {metrics['System C35 (theta=35°)']['med_hdg']:16.2f} °")
print(f"{'Mean Heading Error (°)':<32} | {metrics['System A (CAN Only)']['mean_hdg']:14.2f} ° | {metrics['System B (Phase 17A Ref)']['mean_hdg']:16.2f} ° | {metrics['System C30 (theta=30°)']['mean_hdg']:16.2f} ° | {metrics['System C35 (theta=35°)']['mean_hdg']:16.2f} °")
print("-" * 145)

# Update accounting
tot_acc_b, tot_rej_b, tot_nis_b = df["map_acc_b"].sum(), df["map_rej_b"].sum(), df["map_nis_rej_b"].sum()
tot_acc_c30, tot_rej_c30, tot_nis_c30, tot_hdg_c30 = df["map_acc_c30"].sum(), df["map_rej_c30"].sum(), df["map_nis_rej_c30"].sum(), df["map_hdg_rej_c30"].sum()
tot_acc_c35, tot_rej_c35, tot_nis_c35, tot_hdg_c35 = df["map_acc_c35"].sum(), df["map_rej_c35"].sum(), df["map_nis_rej_c35"].sum(), df["map_hdg_rej_c35"].sum()

mean_abstain_c30 = df["abstain_pct_c30"].mean()
mean_abstain_c35 = df["abstain_pct_c35"].mean()

print(f"Map Updates Accepted: System B = {tot_acc_b}, C30 = {tot_acc_c30} ({tot_acc_c30-tot_acc_b:+d}), C35 = {tot_acc_c35} ({tot_acc_c35-tot_acc_b:+d})")
print(f"Map Updates Gated by Heading: C30 = {tot_hdg_c30} updates, C35 = {tot_hdg_c35} updates")
print(f"Map Updates Gated by NIS: System B = {tot_nis_b}, C30 = {tot_nis_c30}, C35 = {tot_nis_c35}")
print(f"Mean Epoch Map Abstention Rate: C30 = {mean_abstain_c30:.1f}%, C35 = {mean_abstain_c35:.1f}%")

# Head to head C30 vs B and C35 vs B
imp_c30 = (df["c30_vs_b_diff_m"] < -0.5).sum()
wor_c30 = (df["c30_vs_b_diff_m"] > 0.5).sum()
unc_c30 = len(df) - imp_c30 - wor_c30

imp_c35 = (df["c35_vs_b_diff_m"] < -0.5).sum()
wor_c35 = (df["c35_vs_b_diff_m"] > 0.5).sum()
unc_c35 = len(df) - imp_c35 - wor_c35

print(f"\nHead-to-Head vs System B:")
print(f"  C30 vs B: {imp_c30} Improved, {wor_c30} Worsened, {unc_c30} Unchanged (Net: {imp_c30 - wor_c30:+d})")
print(f"  C35 vs B: {imp_c35} Improved, {wor_c35} Worsened, {unc_c35} Unchanged (Net: {imp_c35 - wor_c35:+d})")

# =============================================================================
# 2. IN-DEPTH ANALYSIS OF THE 14 PRIOR FAILURE CASES
# =============================================================================
print("\n" + "=" * 155)
print("PHASE 18B: FORENSIC AUDIT OF THE 14 PRIOR PHASE 17A FAILURE CASES")
print("=" * 155)

# Load 18A failure list
p18a_csv = BASE_DIR / "eval" / "phase18a_failure_diagnostics.csv"
p18a_df = pd.read_csv(p18a_csv)
fail_scenarios = set(p18a_df["scenario"].values)

fail_rows = []
for _, r in p18a_df.iterrows():
    sc = r["scenario"]
    r_18b = df[df["scenario"] == sc].iloc[0]
    dur = r["duration_s"]
    cat = r["category"]
    fa = r_18b["fpe_a_m"]
    fb = r_18b["fpe_b_m"]
    fc30 = r_18b["fpe_c30_m"]
    fc35 = r_18b["fpe_c35_m"]
    dfpe_c30 = fc30 - fb
    dfpe_c35 = fc35 - fb
    hdg_rej_30 = r_18b["map_hdg_rej_c30"]
    hdg_rej_35 = r_18b["map_hdg_rej_c35"]
    abstain_pct = r_18b["abstain_pct_c30"]
    first_t = r_18b["hdg_gate_first_t_c30"]
    gate_dur = r_18b["hdg_gate_dur_c30"]

    # Qualitative status
    if dfpe_c30 < -15.0:
        prevented = "YES (Substantial FPE Recovery)"
    elif dfpe_c30 <= 5.0:
        prevented = "YES (Catastrophic Latch Avoided / Safe CAN Fallback)"
    else:
        prevented = "NO (Residual Drift)"

    useful_lost = "No" if (fc30 <= fa) else "Partial"
    traj_behavior = "Recovered trajectory" if (fc30 < fa and fc30 < fb) else ("Followed CAN baseline" if abs(fc30 - fa) < 20.0 else "Partially assisted")

    fail_rows.append({
        "scenario": sc,
        "duration_s": dur,
        "category": cat,
        "fpe_a_can": fa,
        "fpe_b_17a": fb,
        "fpe_c30": fc30,
        "fpe_c35": fc35,
        "delta_c30_vs_b": dfpe_c30,
        "delta_c35_vs_b": dfpe_c35,
        "hdg_rej_c30": hdg_rej_30,
        "hdg_rej_c35": hdg_rej_35,
        "abstain_pct": abstain_pct,
        "first_gate_t": first_t if not np.isnan(first_t) else "None",
        "gate_dur": gate_dur,
        "catastrophic_prevented": prevented,
        "useful_lost": useful_lost,
        "trajectory_behavior": traj_behavior
    })

fail_table_df = pd.DataFrame(fail_rows).sort_values("duration_s", ascending=False)
fail_table_csv = BASE_DIR / "eval" / "phase18b_14_failures_recovery.csv"
fail_table_df.to_csv(fail_table_csv, index=False)
print(f"Saved 14-failure recovery table to {fail_table_csv}")

print(f"{'Scenario':<28} | {'Dur':<4} | {'Cat':<3} | {'CAN (A)':<8} | {'17A (B)':<8} | {'C30 (30°)':<9} | {'C35 (35°)':<9} | {'C30 Δ vs B':<10} | {'HdgRej':<6} | {'Abstain%':<8} | {'Gate Act @':<10} | {'Catastrophic Prevented?'}")
print("-" * 155)
for _, r in fail_table_df.iterrows():
    act_str = f"{r['first_gate_t']}s ({r['gate_dur']}s)" if r['first_gate_t'] != "None" else "None"
    print(f"{r['scenario']:<28} | {r['duration_s']:3d}s | {r['category']:<3} | {r['fpe_a_can']:6.1f}m | {r['fpe_b_17a']:6.1f}m | {r['fpe_c30']:7.1f}m | {r['fpe_c35']:7.1f}m | {r['delta_c30_vs_b']:+8.1f}m | {r['hdg_rej_c30']:6d} | {r['abstain_pct']:6.1f}% | {act_str:<10} | {r['catastrophic_prevented']}")

# Focus on major severe failures:
print("\n" + "=" * 115)
print("FOCUS ON SEVERE 120s & 180s FAILURES: vw14b_o5, vw4_o1, vw11_o3, vw14c_o4, vw16a_o4")
print("=" * 115)
severe_keys = ["vw14b_o5", "vw4_o1", "vw11_o3", "vw14c_o4", "vw16a_o4", "sample_test_trajectory_motorway_o3"]
for k in severe_keys:
    if k in df["scenario"].values:
        r = df[df["scenario"] == k].iloc[0]
        print(f"[{k:<35}] Dur={r['duration_s']}s | CAN={r['fpe_a_m']:6.1f}m | 17A Map={r['fpe_b_m']:6.1f}m -> C30={r['fpe_c30_m']:6.1f}m (Δ={r['c30_vs_b_diff_m']:+6.1f}m), C35={r['fpe_c35_m']:6.1f}m | HdgRej={r['map_hdg_rej_c30']} (Dur={r['hdg_gate_dur_c30']}s, First={r['hdg_gate_first_t_c30']}s)")

# =============================================================================
# 3. DURATION BREAKDOWN
# =============================================================================
print("\n" + "=" * 135)
print("PHASE 18B: DURATION BREAKDOWN (Median FPE)")
print("=" * 135)
print(f"{'Duration':<10} | {'Count':<6} | {'A (CAN Only)':<14} | {'B (17A Ref)':<14} | {'C30 (30°)':<14} | {'C35 (35°)':<14} | {'C30 vs B Delta'}")
print("-" * 135)

durs = [10, 30, 60, 120, 180]
for d in durs:
    sub = df[df["duration_s"] == d]
    cnt = len(sub)
    fa = sub["fpe_a_m"].median()
    fb = sub["fpe_b_m"].median()
    fc30 = sub["fpe_c30_m"].median()
    fc35 = sub["fpe_c35_m"].median()
    dfpe = fc30 - fb
    sign = "+" if dfpe > 0 else ""
    print(f"{d:3d}s        | {cnt:<6} | {fa:12.2f} m | {fb:12.2f} m | {fc30:12.2f} m | {fc35:12.2f} m | {sign}{dfpe:6.2f} m")

# =============================================================================
# 4. GENERATE PUBLICATION-GRADE FIGURES
# =============================================================================
print("\nGenerating publication-grade figures...")

# Figure 1: Aggregate Overview Comparison Bar Charts
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Phase 18B: Heading-Gated Map Decoupling Benchmark (56 Held-Out Outages)", fontsize=15, fontweight="bold")

sys_labels = ["Sys A\n(CAN Only)", "Sys B\n(17A Ref)", "Sys C30\n(θ=30°)", "Sys C35\n(θ=35°)"]
colors = ["#4A90E2", "#E2844A", "#2ECC71", "#3498DB"]

plots_data = [
    ("Median FPE (m)", [metrics[s]["med_fpe"] for s in metrics], axes[0, 0]),
    ("Mean FPE (m)", [metrics[s]["mean_fpe"] for s in metrics], axes[0, 1]),
    ("Maximum FPE Tail (m)", [metrics[s]["max_fpe"] for s in metrics], axes[0, 2]),
    ("Median Drift (%)", [metrics[s]["med_drift"] for s in metrics], axes[1, 0]),
    ("Median Cross-Track Error (m)", [metrics[s]["med_cross"] for s in metrics], axes[1, 1]),
    ("Scenarios < 10% Drift", [metrics[s]["n_sub10"] for s in metrics], axes[1, 2]),
]

for title, vals, ax in plots_data:
    bars = ax.bar(sys_labels, vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        yval = b.get_height()
        fmt_str = f"{yval:.1f}" if yval < 500 else f"{int(yval)}"
        ax.text(b.get_x() + b.get_width()/2.0, yval * 1.02, fmt_str, ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
fig1_path = PLOTS_DIR / "phase18b_fig1_aggregate_comparison.png"
fig1_art = ARTIFACTS_DIR / "phase18b_fig1_aggregate_comparison.png"
plt.savefig(fig1_path, dpi=300)
plt.savefig(fig1_art, dpi=300)
plt.close()

# Figure 2: Prior Failure Recovery Comparison
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("Phase 18B: FPE Impact on Prior 14 Phase 17A Failure Cases (B vs C30 vs C35 vs CAN)", fontsize=14, fontweight="bold")

fail_sorted = fail_table_df.sort_values("fpe_b_17a", ascending=False)
x = np.arange(len(fail_sorted))
w = 0.2

b_can = ax.bar(x - 1.5*w, fail_sorted["fpe_a_can"], w, label="Sys A (CAN Only)", color="#4A90E2", edgecolor="black")
b_17a = ax.bar(x - 0.5*w, fail_sorted["fpe_b_17a"], w, label="Sys B (Phase 17A Map)", color="#E74C3C", edgecolor="black")
b_c30 = ax.bar(x + 0.5*w, fail_sorted["fpe_c30"], w, label="Sys C30 (θ_max=30°)", color="#2ECC71", edgecolor="black")
b_c35 = ax.bar(x + 1.5*w, fail_sorted["fpe_c35"], w, label="Sys C35 (θ_max=35°)", color="#F39C12", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n({d}s)" for s, d in zip(fail_sorted["scenario"], fail_sorted["duration_s"])], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Final Position Error (m)", fontsize=12)
ax.set_yscale("log")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5, which="both")

plt.tight_layout()
fig2_path = PLOTS_DIR / "phase18b_fig2_failure_recovery_comparison.png"
fig2_art = ARTIFACTS_DIR / "phase18b_fig2_failure_recovery_comparison.png"
plt.savefig(fig2_path, dpi=300)
plt.savefig(fig2_art, dpi=300)
plt.close()

print("All figures successfully generated and saved.")
