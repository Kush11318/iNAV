"""
iNAV Trajectory Visualization Module
Plots GPS ground truth against estimated trajectories, highlighting
GNSS blackout zones, endpoint drift vectors, and baseline comparisons.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.baseline import run_strapdown_baseline, run_constant_velocity_baseline
from eval.replay import run_inav_ukf_pipeline
from modules.velocity_net import VelocityNetPredictor
from modules.alignment import AlignmentEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.viz.trajectory")


def plot_trajectory_comparison(
    sync_parquet_path: Path,
    out_png: Optional[Path] = None,
    outage_durations: List[int] = [30, 60]
) -> Path:
    """
    Generate static Matplotlib comparison plot for a test run showing
    Ground Truth vs Constant Velocity vs Strapdown baselines during GNSS outages.
    """
    df = pd.read_parquet(sync_parquet_path)
    run_key = sync_parquet_path.stem.replace("sync_", "")

    total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]
    schedule = generate_outage_schedule(total_dur, durations=outage_durations, warmup_sec=20.0, recovery_sec=40.0)

    if not schedule:
        schedule = [(20.0, min(total_dur - 10.0, 80.0), int(min(total_dur - 30.0, 60.0)))]

    df_sim, df_outages = inject_outages(df, schedule)

    # Set up figure
    fig, (ax_map, ax_speed) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1.0]})
    fig.patch.set_facecolor("#18191a")
    ax_map.set_facecolor("#242526")
    ax_speed.set_facecolor("#242526")

    # Plot Full Ground Truth
    gt_lat = df[config.COL_TRUE_LAT].values
    gt_lon = df[config.COL_TRUE_LON].values
    ax_map.plot(gt_lon, gt_lat, color="#00e676", linewidth=2.5, label="GNSS Ground Truth (CAN)", zorder=3)

    # Start and End markers
    ax_map.scatter([gt_lon[0]], [gt_lat[0]], color="#00e676", s=90, marker="o", edgecolors="white", label="Start", zorder=5)
    ax_map.scatter([gt_lon[-1]], [gt_lat[-1]], color="#ff1744", s=90, marker="s", edgecolors="white", label="End", zorder=5)

    # Colors
    colors_cv = ["#29b6f6", "#0288d1", "#01579b"]
    colors_sd = ["#ff9100", "#ff6d00", "#e65100"]
    colors_inav = ["#00e5ff", "#00b0ff", "#0091ea"]

    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(str(best_pt)) if best_pt.exists() else None

    # Pre-calibrate vehicle scale factor k and alignment during warmup
    k_calib = 1.0
    run_align = None
    warmup = df[df[config.COL_TIME] < 30.0]
    if len(warmup) >= 20 and predictor is not None:
        cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
        wins = [np.ascontiguousarray(warmup[cols].iloc[i:i+20].values, dtype=np.float32) for i in range(0, len(warmup)-20, 5)]
        preds = [predictor.predict(w)[0] / 2.0 for w in wins]
        trues = [warmup[config.COL_TRUE_SPEED_MS].iloc[i:i+20].mean() for i in range(0, len(warmup)-20, 5)]
        ratios = [t / max(pr, 1.0) for t, pr in zip(trues, preds) if t > 3.0]
        if ratios:
            k_calib = float(np.clip(np.median(ratios), 0.5, 4.0))

        run_align = AlignmentEngine()
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

    for idx, row in df_outages.iterrows():
        oid = int(row["outage_id"])
        dur = int(row["duration_s"])
        mask = (df_sim["outage_id"] == oid)
        df_sub = df.loc[mask]

        if len(df_sub) < 5:
            continue

        # Extract pre-outage state
        outage_start_idx = mask.idxmax()
        pre_idx_start = max(0, outage_start_idx - 10)
        pre_df = df.iloc[pre_idx_start:outage_start_idx]

        if len(pre_df) > 0:
            init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
            init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
            init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
            h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
            init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
        else:
            init_idx = max(0, outage_start_idx - 1)
            init_lat = float(df.loc[init_idx, config.COL_TRUE_LAT])
            init_lon = float(df.loc[init_idx, config.COL_TRUE_LON])
            init_spd = float(df.loc[init_idx, config.COL_TRUE_SPEED_MS])
            init_hdg = float(df.loc[init_idx, config.COL_TRUE_HEADING])

        # Run Baselines & iNAV
        cv_lat, cv_lon, cv_spd = run_constant_velocity_baseline(df_sub, init_lat, init_lon, init_spd, init_hdg)
        sd_lat, sd_lon, sd_spd = run_strapdown_baseline(df_sub, init_lat, init_lon, init_spd, init_hdg)
        inav_lat, inav_lon, inav_spd = run_inav_ukf_pipeline(
            df_sub, init_lat, init_lon, init_spd, init_hdg,
            predictor=predictor, alignment=run_align, k_scale=k_calib
        )

        # Highlight ground truth blackout portion
        ax_map.plot(df_sub[config.COL_TRUE_LON], df_sub[config.COL_TRUE_LAT], color="#ff1744", linewidth=3.5, linestyle="--",
                    label="Blackout Ground Truth" if idx == 0 else "", zorder=4)

        # Plot estimated tracks
        ax_map.plot(inav_lon, inav_lat, color=colors_inav[idx % len(colors_inav)], linewidth=2.5, linestyle="-",
                    label=f"iNAV AI-UKF ({dur}s outage)" if idx == 0 else "", zorder=6)
        ax_map.plot(cv_lon, cv_lat, color=colors_cv[idx % len(colors_cv)], linewidth=1.8, linestyle="-.",
                    label=f"Constant Vel ({dur}s)" if idx == 0 else "", zorder=4)
        ax_map.plot(sd_lon, sd_lat, color=colors_sd[idx % len(colors_sd)], linewidth=1.5, linestyle=":",
                    label=f"Strapdown IMU ({dur}s)" if idx == 0 else "", zorder=4)

        # Endpoint marker
        ax_map.scatter([inav_lon[-1]], [inav_lat[-1]], color="#00e5ff", s=60, marker="^", edgecolors="white", zorder=7)

    ax_map.set_title(f"Trajectory Comparison (Run: {run_key})", color="white", fontsize=14, fontweight="bold", pad=12)
    ax_map.set_xlabel("Longitude (deg)", color="#b0b3b8")
    ax_map.set_ylabel("Latitude (deg)", color="#b0b3b8")
    ax_map.tick_params(colors="#b0b3b8")
    ax_map.grid(True, linestyle="--", alpha=0.3, color="#484f58")
    ax_map.legend(loc="best", facecolor="#18191a", edgecolor="#3a3b3c", labelcolor="white", fontsize=9)

    # Plot Speed Profile
    time_s = df[config.COL_TIME].values
    gt_speed = df[config.COL_TRUE_SPEED_MS].values * config.MS_TO_KMH
    phone_speed = df[config.COL_GPS_SPEED_MS].values * config.MS_TO_KMH

    ax_speed.plot(time_s, gt_speed, color="#00e676", linewidth=1.8, label="CAN True Speed (km/h)")
    ax_speed.plot(time_s, phone_speed, color="#81d4fa", linewidth=1.2, alpha=0.7, label="Phone GPS Speed (km/h)")

    # Shade outages
    for idx, row in df_outages.iterrows():
        st, et, dur = row["start_time_s"], row["end_time_s"], row["duration_s"]
        ax_speed.axvspan(st, et, color="#ff5252", alpha=0.25, label="GNSS Blackout" if idx == 0 else "")
        ax_speed.text((st + et) / 2.0, max(gt_speed) * 0.85, f"{dur}s", color="#ff8a80", ha="center", fontsize=9, fontweight="bold")

    ax_speed.set_title("Speed Profile & Blackout Intervals", color="white", fontsize=14, fontweight="bold", pad=12)
    ax_speed.set_xlabel("Time (seconds)", color="#b0b3b8")
    ax_speed.set_ylabel("Speed (km/h)", color="#b0b3b8")
    ax_speed.tick_params(colors="#b0b3b8")
    ax_speed.grid(True, linestyle="--", alpha=0.3, color="#484f58")
    ax_speed.legend(loc="upper right", facecolor="#18191a", edgecolor="#3a3b3c", labelcolor="white", fontsize=9)

    plt.tight_layout()

    if out_png is None:
        out_png = config.RESULTS_DIR / f"trajectory_{run_key}.png"
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    logger.info(f"Trajectory plot saved to {out_png}")
    return out_png


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV Trajectory Visualizer")
    parser.add_argument("--run", type=str, default="vta10", help="Run key to plot (e.g. 'vta10', 's1')")
    args = parser.parse_args()

    p = config.SYNC_PROCESSED_DIR / f"sync_{args.run.lower()}.parquet"
    if not p.exists():
        logger.error(f"File not found: {p}")
    else:
        plot_trajectory_comparison(p)
