"""
Plotting and Visualization for Phase 15A: Discrete Event Observability
Generates:
  1. Along-Track vs Cross-Track Error CDF comparison (Variant A vs B vs C)
  2. Event-level case study on accepted junction event (vw3, Outage 2)
  3. FPE distribution comparison by outage duration
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_CSV = BASE_DIR / "eval" / "phase15a_nav_results.csv"
JUNC_CSV = BASE_DIR / "eval" / "phase15a_junction_events.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_phase15a_plots():
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)

    # -------------------------------------------------------------
    # Plot 1: Along-Track vs Cross-Track Error Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    along_a = df["along_a"].abs()
    along_b = df["along_b"].abs()
    along_c = df["along_c"].abs()

    cross_a = df["cross_a"].abs()
    cross_b = df["cross_b"].abs()
    cross_c = df["cross_c"].abs()

    # Sorted CDFs
    n = len(df)
    p = np.linspace(0, 100, n)

    ax1.plot(np.sort(along_a), p, label=f"Variant A (Control): Med = {along_a.median():.1f}m", color="#555555", lw=2, ls="--")
    ax1.plot(np.sort(along_b), p, label=f"Variant B (Standstill): Med = {along_b.median():.1f}m", color="#2b5c8f", lw=2.5)
    ax1.plot(np.sort(along_c), p, label=f"Variant C (Topological): Med = {along_c.median():.1f}m", color="#d95f02", lw=2, ls=":")

    ax1.set_title("Along-Track (Longitudinal) Error CDF across 56 Outages", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Absolute Along-Track Error |e_||| (meters)", fontsize=11)
    ax1.set_ylabel("Cumulative Percentage (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.set_xlim(0, 300)

    ax2.plot(np.sort(cross_a), p, label=f"Variant A (Control): Med = {cross_a.median():.1f}m", color="#555555", lw=2, ls="--")
    ax2.plot(np.sort(cross_b), p, label=f"Variant B (Standstill): Med = {cross_b.median():.1f}m", color="#2b5c8f", lw=2.5)
    ax2.plot(np.sort(cross_c), p, label=f"Variant C (Topological): Med = {cross_c.median():.1f}m", color="#d95f02", lw=2, ls=":")

    ax2.set_title("Cross-Track (Lateral) Error CDF across 56 Outages", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Absolute Cross-Track Error |e_perp| (meters)", fontsize=11)
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.set_xlim(0, 300)

    plt.tight_layout()
    out_p1 = PLOTS_DIR / "phase15a_error_decomposition_cdf.png"
    plt.savefig(out_p1, dpi=300)
    plt.close()
    print(f"Saved: {out_p1}")

    # -------------------------------------------------------------
    # Plot 2: Event-Level Step Response (Junction Event #1 on vw3)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    events = [
        {"metric": "Along-Track Error |e_|||", "before": 61.2, "after": 66.2, "delta": +4.9, "color": "#d95f02"},
        {"metric": "Cross-Track Error |e_perp|", "before": 84.0, "after": 63.4, "delta": -20.5, "color": "#1b9e77"},
        {"metric": "Total 2D Error ||e||", "before": 103.9, "after": 91.7, "delta": -12.3, "color": "#7570b3"}
    ]

    labels = [e["metric"] for e in events]
    before_vals = [e["before"] for e in events]
    after_vals = [e["after"] for e in events]

    x = np.arange(len(labels))
    width = 0.32

    rects1 = ax.bar(x - width/2, before_vals, width, label="Before Node Update", color="#a6cee3", edgecolor="#1f78b4")
    rects2 = ax.bar(x + width/2, after_vals, width, label="After Node Update", color="#b2df8a", edgecolor="#33a02c")

    for i, e in enumerate(events):
        diff_str = f"{e['delta']:+.1f} m"
        y_pos = max(before_vals[i], after_vals[i]) + 3.0
        c = "red" if e['delta'] > 0 else "green"
        ax.text(x[i], y_pos, diff_str, ha="center", va="bottom", fontweight="bold", color=c, fontsize=11)

    ax.set_title("Junction Event #1 (vw3 Outage 2, t=60.5s): Impact on Error Decomposition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Magnitude (meters)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 130)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    # Add subtitle text box
    ax.text(
        0.5, 0.88,
        "NIS = 2.69 (Passed gate <= 9.21) | Node Distance = 23.5m | Turn = 30.5 deg\n"
        "Key Finding: Cross-track error drops by 20.5m (-24.4%), but Along-track error slightly increases (+4.9m)",
        transform=ax.transAxes, fontsize=9.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fdfbf7", ec="#cccccc", lw=1)
    )

    plt.tight_layout()
    out_p2 = PLOTS_DIR / "phase15a_junction_event_impact.png"
    plt.savefig(out_p2, dpi=300)
    plt.close()
    print(f"Saved: {out_p2}")

    # -------------------------------------------------------------
    # Plot 3: Median FPE by Outage Duration
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    durations = [10, 30, 60, 120, 180]
    med_a, med_b, med_c = [], [], []

    for d in durations:
        sub = df[df["duration_s"] == d]
        med_a.append(sub["fpe_a"].median())
        med_b.append(sub["fpe_b"].median())
        med_c.append(sub["fpe_c"].median())

    ax.plot(durations, med_a, marker="o", label="Variant A (Control)", color="#555555", lw=2, ls="--")
    ax.plot(durations, med_b, marker="s", label="Variant B (Standstill)", color="#2b5c8f", lw=2.5)
    ax.plot(durations, med_c, marker="^", label="Variant C (Topological)", color="#d95f02", lw=2, ls=":")

    ax.set_title("Median Final Position Error (FPE) by Outage Duration", fontsize=12, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=11)
    ax.set_ylabel("Median FPE (meters)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    out_p3 = PLOTS_DIR / "phase15a_fpe_by_duration.png"
    plt.savefig(out_p3, dpi=300)
    plt.close()
    print(f"Saved: {out_p3}")


if __name__ == "__main__":
    generate_phase15a_plots()
