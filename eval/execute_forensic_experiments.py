"""
Execution script for Experiments 1 through 7.
Collects real measured numbers, generates plots, and outputs forensic tables.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from eval.run_forensic_suite import evaluate_test_set

def run_all_experiments():
    print("=" * 80)
    print("          STARTING FORENSIC INVESTIGATION (EXPERIMENTS 1 - 7)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Baseline: Current iNAV
    # Notice: In C++, deadband=0.02 rad/s was active and bias was NOT injected.
    # In Python esekf.py, deadband was 0.0 and bias was NOT injected.
    # We will test both to be 100% rigorous!
    # -------------------------------------------------------------------------
    print("\n>>> [1/7] Evaluating Baseline (Current iNAV: No bias injected, deadband=0.02 rad/s as in C++ core)...")
    df_base_cpp, diag_base_cpp = evaluate_test_set(
        "current_inav_cpp_baseline",
        inject_bias=False,
        deadband_rads=0.02,
        return_first_diag=True
    )
    
    print(">>> Evaluating Baseline (Python esekf baseline: No bias injected, deadband=0.0)...")
    df_base_py, diag_base_py = evaluate_test_set(
        "current_inav_py_baseline",
        inject_bias=False,
        deadband_rads=0.0,
        return_first_diag=True
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: GYRO BIAS INJECTION ONLY
    # -------------------------------------------------------------------------
    print("\n>>> [2/7] Experiment 1: Gyro Bias Injection ONLY (deadband kept as in C++)...")
    df_exp1, diag_exp1 = evaluate_test_set(
        "exp1_bias_only",
        inject_bias=True,
        deadband_rads=0.02,
        return_first_diag=True
    )

    print(">>> Experiment 1 (Py version: Gyro Bias Injection ONLY with deadband=0.0)...")
    df_exp1_py, diag_exp1_py = evaluate_test_set(
        "exp1_bias_only_nodeadband",
        inject_bias=True,
        deadband_rads=0.0,
        return_first_diag=True
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: DEADBAND ONLY
    # -------------------------------------------------------------------------
    print("\n>>> [3/7] Experiment 2: Deadband Ablation (No bias injected)...")
    deadbands = [0.02, 0.005, 0.001, 0.0]
    db_results = {}
    db_diags = {}
    for db in deadbands:
        name = f"deadband_{db}"
        df_db, diag_db = evaluate_test_set(
            name,
            inject_bias=False,
            deadband_rads=db,
            return_first_diag=True
        )
        db_results[db] = df_db
        db_diags[db] = diag_db

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: BIAS + DEADBAND FIX
    # -------------------------------------------------------------------------
    print("\n>>> [4/7] Experiment 3: Bias Injection + Deadband Fix (deadband=0.0)...")
    df_exp3, diag_exp3 = evaluate_test_set(
        "exp3_bias_and_deadband_fix",
        inject_bias=True,
        deadband_rads=0.0,
        return_first_diag=True
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: TRACE HEADING ERROR (Diagnostic CSV)
    # -------------------------------------------------------------------------
    print("\n>>> [5/7] Experiment 4: Tracing Heading Error Diagnostic Time Series...")
    diag_csv_path = Path("results/forensic_heading_trace_180s.csv")
    diag_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if diag_base_cpp is not None:
        diag_base_cpp.to_csv(diag_csv_path, index=False)
        print(f"Saved diagnostic CSV to {diag_csv_path} ({len(diag_base_cpp)} timesteps)")

    # -------------------------------------------------------------------------
    # EXPERIMENT 6 & 7: NHC, COUPLED JACOBIAN, FEJ, ROAD HEADING
    # -------------------------------------------------------------------------
    print("\n>>> [6/7] Experiment 6 & 7: NHC, Velocity Jacobian, and FEJ...")
    df_nhc, _ = evaluate_test_set(
        "plus_nhc",
        inject_bias=True,
        deadband_rads=0.0,
        apply_nhc=True,
        return_first_diag=False
    )

    df_jac, _ = evaluate_test_set(
        "plus_velocity_jacobian",
        inject_bias=True,
        deadband_rads=0.0,
        apply_nhc=True,
        coupled_jacobian=True,
        return_first_diag=False
    )

    df_fej, _ = evaluate_test_set(
        "plus_fej",
        inject_bias=True,
        deadband_rads=0.0,
        apply_nhc=True,
        coupled_jacobian=True,
        use_fej=True,
        return_first_diag=False
    )

    print("\n" + "=" * 80)
    print("                    FORENSIC EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # Compile Summary Table
    configs = [
        ("Current iNAV (C++ Core: Deadband 0.02 rad/s, No Bias)", df_base_cpp),
        ("Current iNAV (Python: Deadband 0.0, No Bias)", df_base_py),
        ("+ Gyro Bias (Deadband 0.02 rad/s)", df_exp1),
        ("+ Deadband Fix Only (Deadband 0.0, No Bias)", df_base_py),
        ("+ Bias + Deadband Fix", df_exp3),
        ("+ NHC (Lateral / Vertical Constraints)", df_nhc),
        ("+ Coupled Velocity Jacobian", df_jac),
        ("+ FEJ (Observability Consistency)", df_fej),
    ]

    summary_rows = []
    for label, df in configs:
        med_pos = float(df["final_pos_error_m"].median())
        med_drift = float(df["pct_of_distance"].median())
        med_hdg = float(df["heading_error_deg"].median())
        med_xtrack = float(df["cross_track_m"].median()) if "cross_track_m" in df.columns else float('nan')
        med_atrack = float(df["along_track_m"].median()) if "along_track_m" in df.columns else float('nan')
        summary_rows.append({
            "Configuration": label,
            "Position Error (m)": med_pos,
            "Drift (%)": med_drift,
            "Heading Error (°)": med_hdg,
            "Cross-Track (m)": med_xtrack,
            "Along-Track (m)": med_atrack,
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv("results/forensic_error_budget.csv", index=False)
    print("\nSaved summary table to results/forensic_error_budget.csv")

    # -------------------------------------------------------------------------
    # Generate Plots
    # -------------------------------------------------------------------------
    print("\nGenerating Diagnostic Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Heading Error vs Time (Base vs Bias vs Both)
    ax1 = axes[0, 0]
    t = np.arange(len(diag_base_cpp)) * 0.1
    ax1.plot(t, diag_base_cpp["heading_error"], label="Baseline (No bias, Deadband 0.02)", color="crimson", lw=2)
    if diag_base_py is not None:
        ax1.plot(t, diag_base_py["heading_error"], label="Deadband Fix Only (No bias)", color="orange", lw=1.5, ls="--")
    if diag_exp3 is not None:
        ax1.plot(t, diag_exp3["heading_error"], label="Bias + Deadband Fix", color="forestgreen", lw=2.5)
    ax1.set_title("Experiment 1 & 3: Heading Error vs Time (180s Outage)")
    ax1.set_xlabel("Outage Time (s)")
    ax1.set_ylabel("Heading Error (°)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Deadband Comparison (Experiment 2)
    ax2 = axes[0, 1]
    for db in deadbands:
        d = db_diags[db]
        if d is not None:
            ax2.plot(t, d["heading_error"], label=f"Deadband = {db} rad/s", lw=1.8)
    ax2.set_title("Experiment 2: Heading Error across Deadbands")
    ax2.set_xlabel("Outage Time (s)")
    ax2.set_ylabel("Heading Error (°)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Gyro Rate vs Corrected Rate in Deadband
    ax3 = axes[1, 0]
    ax3.plot(t[:300], diag_base_cpp["angular_rate"].iloc[:300], label="Vehicle Gyro Yaw Rate", color="gray", lw=1.2)
    ax3.plot(t[:300], diag_base_cpp["gyro_z_corrected"].iloc[:300], label="Clamped Rate (0.02 rad/s deadband)", color="purple", lw=1.8)
    ax3.axhline(0.02, color="red", ls=":", label="+0.02 rad/s threshold")
    ax3.axhline(-0.02, color="red", ls=":", label="-0.02 rad/s threshold")
    ax3.set_title("Experiment 2: Gyro Rate Clamping Effect")
    ax3.set_xlabel("Outage Time (s)")
    ax3.set_ylabel("Angular Rate (rad/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Position Error vs Time
    ax4 = axes[1, 1]
    ax4.plot(t, diag_base_cpp["position_error"], label="Baseline (54.3% Drift)", color="crimson", lw=2)
    if diag_exp3 is not None:
        ax4.plot(t, diag_exp3["position_error"], label="Bias + Deadband Fix", color="forestgreen", lw=2.5)
    ax4.set_title("Experiment 3: Position Error vs Time")
    ax4.set_xlabel("Outage Time (s)")
    ax4.set_ylabel("Position Error (m)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plot_path = Path("results/forensic_experiments_diagnostic.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plots to {plot_path}")

    return df_summary

if __name__ == "__main__":
    run_all_experiments()
