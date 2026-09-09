"""
iNAV Diagnostic & Comparative Evaluation Visualizer
Generates publication-ready figures:
1. results/plots/error_vs_outage_duration.png: Log-scale final position error vs outage duration
2. results/plots/error_ratio_bar.png: Drift percentage / divergence ratio comparison
3. results/plots/trajectory_showcase.png: Motorway outage multi-trajectory comparison in local ENU frame
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.baseline import run_strapdown_baseline, run_constant_velocity_baseline
from eval.replay import run_inav_ukf_pipeline
from modules.velocity_net import VelocityNetPredictor
from modules.alignment import AlignmentEngine


EARTH_RADIUS_M = 6371000.0


def geodetic_to_enu(lats: np.ndarray, lons: np.ndarray, ref_lat: float, ref_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert geodetic latitude/longitude to local tangent plane ENU meters."""
    cos_ref = np.cos(np.deg2rad(ref_lat))
    d2r = np.pi / 180.0
    east = (lons - ref_lon) * d2r * EARTH_RADIUS_M * cos_ref
    north = (lats - ref_lat) * d2r * EARTH_RADIUS_M
    return east, north


def plot_error_vs_outage_duration(
    summary_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot median and p95 position error vs outage duration on a log scale."""
    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=300)

    # Sort and filter configs
    cv_df = summary_df[summary_df["config"].isin(["baseline_cv_heading_v1", "constant_velocity"])].sort_values("outage_s")
    sd_df = summary_df[summary_df["config"].isin(["baseline_strapdown_v1", "strapdown"])].sort_values("outage_s")
    inav_df = summary_df[summary_df["config"].isin(["inav_ai_ukf_v1", "inav_ukf"])].sort_values("outage_s")

    durations = sorted(list(set(cv_df["outage_s"].unique()).union(inav_df["outage_s"].unique())))

    # Colors & styling
    c_cv = "#1f77b4"      # Blue
    c_sd = "#d62728"      # Crimson
    c_inav = "#2ca02c"    # Emerald green

    # 1. Constant-Velocity
    if not cv_df.empty:
        x_cv = cv_df["outage_s"].values
        ax.plot(x_cv, cv_df["median_final_pos_error_m"], color=c_cv, marker="o", linewidth=2.4, markersize=7, label="Constant-Velocity (Median)")
        ax.plot(x_cv, cv_df["p95_final_pos_error_m"], color=c_cv, linestyle="--", marker="o", markerfacecolor="none", linewidth=1.5, markersize=5, label="Constant-Velocity (95th %tile)")
        ax.fill_between(x_cv, cv_df["median_final_pos_error_m"], cv_df["p95_final_pos_error_m"], color=c_cv, alpha=0.08)

    # 2. Strapdown INS
    if not sd_df.empty:
        x_sd = sd_df["outage_s"].values
        ax.plot(x_sd, sd_df["median_final_pos_error_m"], color=c_sd, marker="s", linewidth=2.4, markersize=7, label="Strapdown INS (Median)")
        ax.plot(x_sd, sd_df["p95_final_pos_error_m"], color=c_sd, linestyle="--", marker="s", markerfacecolor="none", linewidth=1.5, markersize=5, label="Strapdown INS (95th %tile)")
        ax.fill_between(x_sd, sd_df["median_final_pos_error_m"], sd_df["p95_final_pos_error_m"], color=c_sd, alpha=0.08)

    # 3. iNAV AI-UKF
    if not inav_df.empty:
        x_inav = inav_df["outage_s"].values
        ax.plot(x_inav, inav_df["median_final_pos_error_m"], color=c_inav, marker="^", linewidth=2.8, markersize=8, label="iNAV AI-UKF (Median)")
        ax.plot(x_inav, inav_df["p95_final_pos_error_m"], color=c_inav, linestyle="--", marker="^", markerfacecolor="none", linewidth=1.8, markersize=6, label="iNAV AI-UKF (95th %tile)")
        ax.fill_between(x_inav, inav_df["median_final_pos_error_m"], inav_df["p95_final_pos_error_m"], color=c_inav, alpha=0.12)

    ax.set_yscale("log")
    ax.set_xticks(durations)
    ax.set_xticklabels([f"{int(d)}s" for d in durations], fontsize=11, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Position Error (meters, log scale)", fontsize=12, fontweight="bold")

    n_total = int(summary_df["n_paired"].sum())
    ax.set_title(
        f"Dead-Reckoning Position Error vs. Outage Duration\n"
        f"Motorway Test Cohort (Driver E Vw) [Logarithmic Y-Scale]",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.5, color="#888888")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9.5, loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_error_ratio_bar(
    summary_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot bar chart of drift percentage / error comparison across durations."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)

    cv_df = summary_df[summary_df["config"].isin(["baseline_cv_heading_v1", "constant_velocity"])].set_index("outage_s")
    sd_df = summary_df[summary_df["config"].isin(["baseline_strapdown_v1", "strapdown"])].set_index("outage_s")
    inav_df = summary_df[summary_df["config"].isin(["inav_ai_ukf_v1", "inav_ukf"])].set_index("outage_s")

    durations = [d for d in [10, 30, 60, 120, 180] if d in inav_df.index]
    x = np.arange(len(durations))
    width = 0.25

    # Plot grouped bars
    bars_sd = ax.bar(x - width, [sd_df.loc[d, "median_pct_of_distance"] if d in sd_df.index else 0 for d in durations], width, label="Strapdown INS", color="#d62728", edgecolor="#333", alpha=0.9)
    bars_cv = ax.bar(x, [cv_df.loc[d, "median_pct_of_distance"] if d in cv_df.index else 0 for d in durations], width, label="Constant-Velocity", color="#1f77b4", edgecolor="#333", alpha=0.9)
    bars_inav = ax.bar(x + width, [inav_df.loc[d, "median_pct_of_distance"] if d in inav_df.index else 0 for d in durations], width, label="iNAV AI-UKF", color="#2ca02c", edgecolor="#333", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations], fontsize=11, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Median Drift (% of Trajectory Distance)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Trajectory Drift Ratio: iNAV AI-UKF vs. Conventional Baselines\n"
        "(Motorway Benchmark Test Suite)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    # Callout values on iNAV bars
    for bar in bars_inav:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b5e20")

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_trajectory_showcase(
    test_parquet_path: Path,
    output_path: Path
) -> None:
    """Generate high-resolution local ENU comparison on a representative motorway run."""
    df = pd.read_parquet(test_parquet_path)
    total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]

    # Generate 60s outage
    schedule = generate_outage_schedule(total_dur, durations=[60], run_id=test_parquet_path.stem)
    if not schedule:
        schedule = [(30.0, 90.0, 60)]

    df_sim, df_outages = inject_outages(df, schedule)
    row = df_outages.iloc[0]
    oid = int(row["outage_id"])
    mask = (df_sim["outage_id"] == oid)
    df_sub = df.loc[mask]

    outage_start_idx = mask.idxmax()
    pre_idx_start = max(0, outage_start_idx - 10)
    pre_df = df.iloc[pre_idx_start:outage_start_idx]

    init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
    init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
    init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
    h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
    init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)

    # Warmup calibration for iNAV
    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(str(best_pt)) if best_pt.exists() else None
    warmup = df[df[config.COL_TIME] < 30.0]

    run_align = AlignmentEngine()
    k_calib = 1.0
    if len(warmup) >= 20 and predictor is not None:
        cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
        wins = [np.ascontiguousarray(warmup[cols].iloc[i:i+20].values, dtype=np.float32) for i in range(0, len(warmup)-20, 5)]
        preds = [predictor.predict(w)[0] / 2.0 for w in wins]
        trues = [warmup[config.COL_TRUE_SPEED_MS].iloc[i:i+20].mean() for i in range(0, len(warmup)-20, 5)]
        ratios = [t / max(pr, 1.0) for t, pr in zip(trues, preds) if t > 3.0]
        if ratios:
            k_calib = float(np.clip(np.median(ratios), 0.5, 4.0))

        acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
        gyro_w = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
        mean_a = np.mean(acc_w, axis=0)
        run_align.u_z_body = -mean_a / np.linalg.norm(mean_a)
        z = run_align.u_z_body
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        y = np.cross(z, ref)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        run_align.R_b_to_v = np.vstack([x, y, z])
        run_align.static_calibrated = True
        run_align.calibrate_dynamic(acc_w, gyro_w)

    # Run predictions
    p_lat_cv, p_lon_cv, _ = run_constant_velocity_baseline(df_sub, init_lat, init_lon, init_spd, init_hdg)
    p_lat_sd, p_lon_sd, _ = run_strapdown_baseline(df_sub, init_lat, init_lon, init_spd, init_hdg)
    p_lat_inav, p_lon_inav, _ = run_inav_ukf_pipeline(df_sub, init_lat, init_lon, init_spd, init_hdg, predictor=predictor, alignment=run_align, k_scale=k_calib)

    gt_lats = df_sub[config.COL_TRUE_LAT].values
    gt_lons = df_sub[config.COL_TRUE_LON].values

    # Project to local ENU meters
    e_gt, n_gt = geodetic_to_enu(gt_lats, gt_lons, init_lat, init_lon)
    e_cv, n_cv = geodetic_to_enu(p_lat_cv, p_lon_cv, init_lat, init_lon)
    e_sd, n_sd = geodetic_to_enu(p_lat_sd, p_lon_sd, init_lat, init_lon)
    e_inav, n_inav = geodetic_to_enu(p_lat_inav, p_lon_inav, init_lat, init_lon)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252526")

    # Plot Ground Truth
    ax.plot(e_gt, n_gt, color="#00e676", linewidth=3.0, label="CAN Ground Truth", zorder=4)
    ax.scatter([e_gt[0]], [n_gt[0]], color="#00e676", s=110, marker="o", edgecolors="white", label="Outage Inset", zorder=6)
    ax.scatter([e_gt[-1]], [n_gt[-1]], color="#ff1744", s=110, marker="s", edgecolors="white", label="Ground Truth Endpoint", zorder=6)

    # Plot Baselines
    ax.plot(e_sd, n_sd, color="#ff5252", linestyle="--", linewidth=2.0, label="Strapdown INS", zorder=2)
    ax.plot(e_cv, n_cv, color="#40c4ff", linestyle=":", linewidth=2.2, label="Constant-Velocity", zorder=3)
    ax.plot(e_inav, n_inav, color="#ffd700", linewidth=2.8, label="iNAV AI-UKF", zorder=5)

    ax.set_xlabel("East (meters)", fontsize=11, color="white", fontweight="bold")
    ax.set_ylabel("North (meters)", fontsize=11, color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.35, color="#aaa")

    ax.set_title(
        f"60s GNSS Blackout Trajectory Comparison (Local ENU Tangent Plane)\n"
        f"Run: {test_parquet_path.stem} | Motorway High-Speed Segment",
        fontsize=12,
        fontweight="bold",
        color="white",
        pad=12
    )

    leg = ax.legend(frameon=True, facecolor="#2d2d30", edgecolor="#555", fontsize=9.5, loc="best")
    for text in leg.get_texts():
        text.set_color("white")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_diagnostic_plots():
    plots_dir = config.RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_p = config.LEADERBOARD_SUMMARY_PATH
    if summary_p.exists():
        df_sum = pd.read_csv(summary_p)
        plot_error_vs_outage_duration(df_sum, plots_dir / "error_vs_outage_duration.png")
        plot_error_ratio_bar(df_sum, plots_dir / "error_ratio_bar.png")
    else:
        print(f"Summary table not found at {summary_p}")

    # Find sample Vw test file
    vw_files = sorted(list((config.SYNC_PROCESSED_DIR).glob("sync_vw*.parquet")))
    if vw_files:
        sample_file = vw_files[0]
        plot_trajectory_showcase(sample_file, plots_dir / "sample_trajectories_motorway.png")


if __name__ == "__main__":
    generate_all_diagnostic_plots()
