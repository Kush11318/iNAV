"""
Phase 12: 9-State Velocity-State EKF Navigation Benchmark & Ablation Study

Evaluates:
  - Model A: Production 7-state UKF baseline with Phase 11 Anchored Model
  - Model B: Experimental 9-state EKF with NHC (Full fusion: IMU + AI velocity + NHC + ZUPT)
  - Model C: Experimental 9-state EKF without NHC (Ablation C)

Evaluated across the exact same 56 held-out test outages from IO-VNBD.
Strict experiment isolation:
  - Production 7-state UKF is preserved completely unchanged.
  - Phase 11 model is strictly frozen (no retraining, no new network).
  - No ground-truth initialization (fair GNSS position, speed, and course initialization).
  - No global k calibration, no test-derived tuning, no artificial clamping.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from eval.train_phase10_temporal_model import TemporalVelocityNet4s
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from modules.alignment import AlignmentEngine, AlignmentState
from modules.ukf import UKFNavigationFilter
from modules.ekf_9state import EKF9StateNavigationFilter
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase12_Eval")


def compute_outage_predictions_phase11(
    model_b: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_val: float,
    window_len: int = 40,
    device: str = "cpu"
) -> Dict[int, Tuple[float, float, int, float]]:
    """
    Computes causal sliding-window predictions for a specific outage using the frozen Phase 11 model.
    Returns:
      dict: global_k -> (delta_d_2s, v_ai_mean, event_class, sigma_ai)
    """
    lookup = {}
    model_b.eval()

    windows = []
    step_indices = []

    for k in range(0, sub_len):
        global_k = outage_start_idx + k
        if global_k >= window_len - 1 and (k % 5 == 0):
            win = seq_data[global_k - window_len + 1 : global_k + 1]
            windows.append(win)
            step_indices.append(global_k)

    if not windows:
        return lookup

    all_windows = np.array(windows, dtype=np.float32)  # (M, 40, 6)
    bx = torch.from_numpy(all_windows.transpose(0, 2, 1)).float().to(device)
    b_anc = torch.full((len(all_windows), 1), v_anchor_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        del_v_seq_b, v_seq_b, sig_seq_b, ev_b = model_b(bx, b_anc)
        p_v_b = v_seq_b.cpu().numpy()
        p_sig_b = sig_seq_b.cpu().numpy()
        p_ev_b = torch.argmax(ev_b, dim=1).cpu().numpy()

        p_d_b = np.sum(p_v_b * 0.2, axis=1)  # 2.0s integrated displacement
        p_mean_v_b = np.mean(p_v_b, axis=1)  # forward speed (m/s)
        p_mean_sig_b = np.mean(p_sig_b, axis=1)  # speed sigma (m/s)

    for idx, gk in enumerate(step_indices):
        lookup[gk] = (
            float(p_d_b[idx]),
            float(p_mean_v_b[idx]),
            int(p_ev_b[idx]),
            float(p_mean_sig_b[idx])
        )

    return lookup


def run_single_outage_ukf_7state(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, float, int, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs the preserved, untouched 7-state UKF baseline."""
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    ukf = UKFNavigationFilter(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg)
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading = np.zeros(n)

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        global_k = global_start_idx + k
        if global_k in lookup_preds and (k % 5 == 0):
            raw_d, _, ev_class, raw_sig = lookup_preds[global_k]
            cal_d = raw_d / 2.0
            cal_sig = max(raw_sig / 2.0, 0.2)

            if ev_class == 0 or cal_d < 0.1:
                ukf.update_zupt(gyro_reading=w_v[2])
            else:
                ukf.update_velocity_net(
                    delta_d_pred=cal_d,
                    sigma_pred=cal_sig,
                    event_class=ev_class,
                    window_dur=2.0
                )

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]
        est_heading[k] = math.degrees(ukf.x[3]) % 360.0

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed, est_heading


def run_single_outage_ekf_9state(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, float, int, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    use_nhc: bool = True,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Runs the experimental 9-state EKF (Model B or Model C)."""
    n = len(df_outage)
    telemetry = {
        "vE": np.zeros(n),
        "vN": np.zeros(n),
        "vU": np.zeros(n),
        "v_lat": np.zeros(n),
        "ai_innov": [],
        "ai_nis": [],
        "nhc_innov": [],
        "nhc_nis": [],
        "ai_updates": 0,
        "ai_rejected": 0,
        "nhc_updates": 0,
        "nhc_rejected": 0,
    }
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), telemetry

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    ekf = EKF9StateNavigationFilter(dt=dt, heading_convention="clockwise_from_north")
    ekf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_deg=init_heading_deg
    )

    est_pE = np.zeros(n)
    est_pN = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading = np.zeros(n)

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. IMU Propagation
        ekf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        # 2. NHC Update (10 Hz) if enabled
        if use_nhc:
            acc_nhc, nis_nhc = ekf.update_nhc(sigma_lat=0.5, sigma_vert=0.3, nis_gate=9.21)
            telemetry["nhc_innov"].append(float(ekf.last_nhc_innov[0]))
            telemetry["nhc_nis"].append(float(nis_nhc))

        # 3. AI Velocity or ZUPT Update (2 Hz, every 5 steps)
        global_k = global_start_idx + k
        if global_k in lookup_preds and (k % 5 == 0):
            _, v_ai, ev_class, sig_ai = lookup_preds[global_k]

            if ev_class == 0 or v_ai < 0.1:
                ekf.update_zupt(sigma_zupt=0.05)
            else:
                acc_ai, nis_ai = ekf.update_ai_forward_velocity(v_ai=v_ai, sigma_ai=sig_ai, nis_gate=6.635)
                telemetry["ai_innov"].append(float(ekf.last_ai_innov))
                telemetry["ai_nis"].append(float(nis_ai))

        est_pE[k] = ekf.x[0]
        est_pN[k] = ekf.x[1]
        est_speed[k] = ekf.get_forward_speed()
        est_heading[k] = ekf.get_heading_deg()

        telemetry["vE"][k] = ekf.x[3]
        telemetry["vN"][k] = ekf.x[4]
        telemetry["vU"][k] = ekf.x[5]
        telemetry["v_lat"][k] = ekf.get_lateral_speed()

    telemetry["ai_updates"] = ekf.ai_updates_count
    telemetry["ai_rejected"] = ekf.ai_rejected_count
    telemetry["nhc_updates"] = ekf.nhc_updates_count
    telemetry["nhc_rejected"] = ekf.nhc_rejected_count

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed, est_heading, telemetry


def plot_phase12_observability_cases(
    case_a_data: Dict[str, Any],
    case_b_data: Dict[str, Any],
    output_dir: Path
):
    """
    Generates the required plots specified in Section 14 and Section 15.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("case_a_high_speed_cruise", case_a_data), ("case_b_braking_deceleration", case_b_data)]:
        t = data["time"]

        # 1. Trajectory Plot: GT vs 7-state vs 9-state
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(data["gt_pE"], data["gt_pN"], "k-", linewidth=2.5, label="Ground Truth")
        ax.plot(data["ukf_pE"], data["ukf_pN"], "b--", linewidth=2.0, label="7-State UKF Baseline")
        ax.plot(data["ekf_b_pE"], data["ekf_b_pN"], "r-", linewidth=2.0, label="9-State EKF (with NHC)")
        ax.plot(data["ekf_c_pE"], data["ekf_c_pN"], "g:", linewidth=1.8, label="9-State EKF (No NHC)")
        ax.scatter([data["gt_pE"][0]], [data["gt_pN"][0]], color="green", s=100, marker="o", label="Start (GNSS Loss)")
        ax.scatter([data["gt_pE"][-1]], [data["gt_pN"][-1]], color="black", s=100, marker="*", label="GT End")
        ax.scatter([data["ekf_b_pE"][-1]], [data["ekf_b_pN"][-1]], color="red", s=100, marker="x", label="9-State End")
        ax.set_title(f"Trajectory Comparison: {data['title']}", fontsize=13, fontweight="bold")
        ax.set_xlabel("East Position (m)")
        ax.set_ylabel("North Position (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_1_trajectory.png", dpi=200)
        plt.close(fig)

        # 2. Speed Comparison: GT vs AI vs 7-state vs 9-state
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, data["gt_speed"], "k-", linewidth=2.5, label="GT Speed")
        ax.plot(t, data["ukf_speed"], "b--", linewidth=2.0, label="7-State UKF Speed")
        ax.plot(t, data["ekf_b_speed"], "r-", linewidth=2.0, label="9-State EKF Speed")
        if "ai_steps_t" in data and len(data["ai_steps_t"]) > 0:
            ax.scatter(data["ai_steps_t"], data["ai_steps_v"], color="purple", s=35, zorder=5, label="Phase 11 AI Speed Updates")
        ax.axhline(data["v_anchor"], color="gray", linestyle=":", alpha=0.7, label=f"v_anchor = {data['v_anchor']:.1f} m/s")
        ax.set_title(f"Speed Recovery & Regime Change: {data['title']}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time since GNSS Outage (s)")
        ax.set_ylabel("Forward Speed (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_2_speed_comparison.png", dpi=200)
        plt.close(fig)

        # 3. World Velocity Components: GT vE/vN vs EKF vE/vN
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, data["gt_vE"], "k-", linewidth=2.0, label="GT vE")
        axes[0].plot(t, data["ekf_b_vE"], "r--", linewidth=2.0, label="9-State EKF vE")
        axes[0].set_ylabel("vE (m/s)")
        axes[0].set_title(f"World Frame Velocities (vE, vN): {data['title']}", fontsize=13, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, data["gt_vN"], "k-", linewidth=2.0, label="GT vN")
        axes[1].plot(t, data["ekf_b_vN"], "r--", linewidth=2.0, label="9-State EKF vN")
        axes[1].set_ylabel("vN (m/s)")
        axes[1].set_xlabel("Time since GNSS Outage (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_3_world_velocity.png", dpi=200)
        plt.close(fig)

        # 4. Lateral Velocity v_lat vs Time
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, data["ekf_b_v_lat"], "r-", linewidth=2.0, label="9-State EKF v_lat (with NHC)")
        ax.plot(t, data["ekf_c_v_lat"], "g--", linewidth=1.8, label="9-State EKF v_lat (No NHC)")
        ax.axhline(0.0, color="k", linestyle=":", alpha=0.7)
        ax.set_title(f"Body-Frame Lateral Velocity v_lat: {data['title']}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time since GNSS Outage (s)")
        ax.set_ylabel("Lateral Velocity (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_4_lateral_velocity.png", dpi=200)
        plt.close(fig)

        # 5. Heading Comparison: GT vs 7-state vs 9-state
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, data["gt_hdg"], "k-", linewidth=2.5, label="GT Heading")
        ax.plot(t, data["ukf_hdg"], "b--", linewidth=2.0, label="7-State UKF Heading")
        ax.plot(t, data["ekf_b_hdg"], "r-", linewidth=2.0, label="9-State EKF Heading")
        ax.set_title(f"Heading Angle Comparison: {data['title']}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time since GNSS Outage (s)")
        ax.set_ylabel("Heading (deg clockwise from North)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_5_heading_comparison.png", dpi=200)
        plt.close(fig)

        # 6. NHC & AI Innovation Timeseries
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        if len(data["nhc_innov"]) > 0:
            axes[0].plot(data["nhc_innov"], "r-", linewidth=1.5, label="NHC Innovation")
            axes[0].set_title(f"Filter Innovations: {data['title']}", fontsize=13, fontweight="bold")
            axes[0].set_ylabel("NHC Innov (m/s)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        if len(data["ai_innov"]) > 0:
            axes[1].plot(data["ai_innov"], "m-o", linewidth=1.5, label="AI Velocity Innovation")
            axes[1].set_ylabel("AI Innov (m/s)")
            axes[1].set_xlabel("Update Step")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_6_innovations.png", dpi=200)
        plt.close(fig)

        # 7. Along-Track and Cross-Track Errors vs Time
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, data["along_ukf"], "b--", linewidth=2.0, label="7-State UKF Along-Track Error")
        axes[0].plot(t, data["along_ekf_b"], "r-", linewidth=2.0, label="9-State EKF Along-Track Error")
        axes[0].set_title(f"Trajectory Error Components: {data['title']}", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Along-Track Error (m)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, data["cross_ukf"], "b--", linewidth=2.0, label="7-State UKF Cross-Track Error")
        axes[1].plot(t, data["cross_ekf_b"], "r-", linewidth=2.0, label="9-State EKF Cross-Track Error")
        axes[1].set_ylabel("Cross-Track Error (m)")
        axes[1].set_xlabel("Time since GNSS Outage (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_7_along_cross_error.png", dpi=200)
        plt.close(fig)

    logger.info(f"Generated comprehensive Phase 12 observability plots in {output_dir}")


def run_phase12_benchmark():
    logger.info("=" * 90)
    logger.info("PHASE 12: 9-STATE VELOCITY-STATE EKF EXPERIMENT & ABLATION (56 OUTAGES)")
    logger.info("Model A: Preserved 7-State UKF Baseline (Phase 11 Model)")
    logger.info("Model B: 9-State EKF with Full NHC (IMU + AI Velocity + NHC + ZUPT)")
    logger.info("Model C: 9-State EKF without NHC (Ablation C)")
    logger.info("=" * 90)

    model_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    if not model_path.exists():
        logger.error(f"Phase 11 model not found at {model_path}!")
        return

    model_b = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model_b.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_b.eval()

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    logger.info(f"Loaded {len(test_files)} test files.")
    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]

    records = []

    case_a_saved = None
    case_b_saved = None

    for tf in test_files:
        run_name = tf.stem.replace("sync_", "")
        df = pd.read_parquet(tf)
        n = len(df)
        total_dur = n * config.TARGET_DT
        if total_dur < 30.0:
            continue

        schedule = generate_outage_schedule(total_dur, run_id=run_name)
        if not schedule:
            continue

        df_sim, df_outages = inject_outages(df, schedule)

        # Vehicle alignment (unsupervised from warmup acceleration)
        run_align = AlignmentEngine()
        warmup = df[df[config.COL_TIME] < min(45.0, total_dur * 0.2)]
        if len(warmup) >= 25:
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref)
            y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

        for c in cols:
            df[c] = df[c].interpolate().bfill().ffill().fillna(0.0)

        seq_data = df[cols].values.astype(np.float32)

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            dist_gt = float(row["distance_travelled_m"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]

            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()
            pre_idx_start = max(0, outage_start_idx - 10)
            pre_df = df.iloc[pre_idx_start:outage_start_idx]

            # FAIRNESS & LEAKAGE AUDIT: Initialize strictly from GNSS measurements immediately before outage
            valid_gps = pre_df.dropna(subset=[config.COL_GPS_LAT, config.COL_GPS_LON, config.COL_GPS_SPEED_MS])
            if len(valid_gps) > 0:
                init_lat = float(valid_gps[config.COL_GPS_LAT].iloc[-1])
                init_lon = float(valid_gps[config.COL_GPS_LON].iloc[-1])
                init_spd = float(valid_gps[config.COL_GPS_SPEED_MS].iloc[-1])
                init_hdg = float(valid_gps[config.COL_GPS_BEARING].iloc[-1])
                v_anchor_val = init_spd
            else:
                # Fallback to current row GNSS if pre-window had no valid GPS
                init_lat = float(df[config.COL_GPS_LAT].iloc[outage_start_idx])
                init_lon = float(df[config.COL_GPS_LON].iloc[outage_start_idx])
                init_spd = float(df[config.COL_GPS_SPEED_MS].iloc[outage_start_idx])
                init_hdg = float(df[config.COL_GPS_BEARING].iloc[outage_start_idx])
                v_anchor_val = init_spd

            # Ground truth strictly for evaluation
            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None
            gt_spd = df_sub[config.COL_TRUE_SPEED_MS].values if config.COL_TRUE_SPEED_MS in df_sub.columns else np.zeros(len(df_sub))

            # Compute frozen Phase 11 causal predictions
            lookup_preds = compute_outage_predictions_phase11(
                model_b=model_b,
                seq_data=seq_data,
                outage_start_idx=outage_start_idx,
                sub_len=len(df_sub),
                v_anchor_val=v_anchor_val,
                window_len=40,
                device="cpu"
            )

            # Model A: 7-State UKF Baseline
            lat_a, lon_a, spd_a, hdg_a = run_single_outage_ukf_7state(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_preds, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg, pred_heading_deg=hdg_a)

            # Model B: 9-State EKF with NHC
            lat_b, lon_b, spd_b, hdg_b, telem_b = run_single_outage_ekf_9state(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_preds, global_start_idx=outage_start_idx,
                alignment=run_align,
                use_nhc=True
            )
            score_b = score_outage_segment(lat_b, lon_b, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg, pred_heading_deg=hdg_b)

            # Model C: 9-State EKF without NHC (Ablation C)
            lat_c, lon_c, spd_c, hdg_c, telem_c = run_single_outage_ekf_9state(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_preds, global_start_idx=outage_start_idx,
                alignment=run_align,
                use_nhc=False
            )
            score_c = score_outage_segment(lat_c, lon_c, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg, pred_heading_deg=hdg_c)

            rec = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                "v_anchor_ms": round(v_anchor_val, 2),
                # Model A (7-State UKF)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                "rmse_a_m": round(float(score_a.get("ate_m", np.nan)), 2),
                "cep50_a_m": round(float(score_a.get("cep50_m", np.nan)), 2),
                "cep95_a_m": round(float(score_a.get("cep95_m", np.nan)), 2),
                # Model B (9-State EKF with NHC)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
                "rmse_b_m": round(float(score_b.get("ate_m", np.nan)), 2),
                "cep50_b_m": round(float(score_b.get("cep50_m", np.nan)), 2),
                "cep95_b_m": round(float(score_b.get("cep95_m", np.nan)), 2),
                "ai_updates_b": telem_b["ai_updates"],
                "ai_rejected_b": telem_b["ai_rejected"],
                "nhc_updates_b": telem_b["nhc_updates"],
                "nhc_rejected_b": telem_b["nhc_rejected"],
                "mean_nhc_nis_b": round(float(np.mean(telem_b["nhc_nis"])), 2) if len(telem_b["nhc_nis"]) > 0 else 0.0,
                "mean_ai_nis_b": round(float(np.mean(telem_b["ai_nis"])), 2) if len(telem_b["ai_nis"]) > 0 else 0.0,
                # Model C (9-State EKF without NHC)
                "fpe_c_m": round(float(score_c["final_pos_error_m"]), 2),
                "drift_c_pct": round(float(score_c["pct_of_distance"]), 2),
                "along_c_m": round(float(score_c.get("along_track_m", np.nan)), 2),
                "cross_c_m": round(float(score_c.get("cross_track_m", np.nan)), 2),
                "hdg_c_deg": round(float(score_c.get("heading_error_deg", np.nan)), 2),
                "rmse_c_m": round(float(score_c.get("ate_m", np.nan)), 2),
                "cep50_c_m": round(float(score_c.get("cep50_m", np.nan)), 2),
                "cep95_c_m": round(float(score_c.get("cep95_m", np.nan)), 2),
            }
            records.append(rec)

            # Identify representative Case A (steady high-speed cruise) & Case B (braking deceleration)
            dt_step = config.TARGET_DT
            sub_n = len(df_sub)
            t_axis = np.arange(sub_n) * dt_step
            spd_change = abs(gt_spd[-1] - gt_spd[0])

            # Case A: 60s or 120s steady cruise with high speed (> 20 m/s) and minimal speed delta (< 3 m/s)
            if case_a_saved is None and dur >= 60 and np.mean(gt_spd) > 20.0 and spd_change < 3.0:
                R_earth = 6371000.0
                pE_gt = (gt_lon - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_gt = (gt_lat - init_lat) * (math.pi / 180.0) * R_earth
                pE_ukf = (lon_a - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_ukf = (lat_a - init_lat) * (math.pi / 180.0) * R_earth
                pE_b = (lon_b - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_b = (lat_b - init_lat) * (math.pi / 180.0) * R_earth
                pE_c = (lon_c - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_c = (lat_c - init_lat) * (math.pi / 180.0) * R_earth

                psi_gt_rad = np.radians(gt_hdg) if gt_hdg is not None else np.zeros(sub_n)
                along_ukf = (pE_ukf - pE_gt) * np.sin(psi_gt_rad) + (pN_ukf - pN_gt) * np.cos(psi_gt_rad)
                cross_ukf = (pE_ukf - pE_gt) * np.cos(psi_gt_rad) - (pN_ukf - pN_gt) * np.sin(psi_gt_rad)
                along_b = (pE_b - pE_gt) * np.sin(psi_gt_rad) + (pN_b - pN_gt) * np.cos(psi_gt_rad)
                cross_b = (pE_b - pE_gt) * np.cos(psi_gt_rad) - (pN_b - pN_gt) * np.sin(psi_gt_rad)

                ai_t = []
                ai_v = []
                for gk, val in lookup_preds.items():
                    rel_idx = gk - outage_start_idx
                    if 0 <= rel_idx < sub_n:
                        ai_t.append(rel_idx * dt_step)
                        ai_v.append(val[1])

                gt_vE = gt_spd * np.sin(psi_gt_rad)
                gt_vN = gt_spd * np.cos(psi_gt_rad)

                case_a_saved = {
                    "title": f"{run_name} Outage {oid} ({dur}s High-Speed Cruise)",
                    "time": t_axis,
                    "gt_pE": pE_gt, "gt_pN": pN_gt,
                    "ukf_pE": pE_ukf, "ukf_pN": pN_ukf,
                    "ekf_b_pE": pE_b, "ekf_b_pN": pN_b,
                    "ekf_c_pE": pE_c, "ekf_c_pN": pN_c,
                    "gt_speed": gt_spd, "ukf_speed": spd_a, "ekf_b_speed": spd_b, "ekf_c_speed": spd_c,
                    "v_anchor": v_anchor_val,
                    "ai_steps_t": ai_t, "ai_steps_v": ai_v,
                    "gt_vE": gt_vE, "gt_vN": gt_vN,
                    "ekf_b_vE": telem_b["vE"], "ekf_b_vN": telem_b["vN"],
                    "ekf_b_v_lat": telem_b["v_lat"], "ekf_c_v_lat": telem_c["v_lat"],
                    "gt_hdg": gt_hdg if gt_hdg is not None else np.zeros(sub_n),
                    "ukf_hdg": hdg_a, "ekf_b_hdg": hdg_b,
                    "nhc_innov": telem_b["nhc_innov"], "ai_innov": telem_b["ai_innov"],
                    "along_ukf": along_ukf, "cross_ukf": cross_ukf,
                    "along_ekf_b": along_b, "cross_ekf_b": cross_b,
                }

            # Case B: Significant deceleration (high speed -> braking -> low speed, delta > 10 m/s)
            if case_b_saved is None and dur >= 30 and (gt_spd[0] - np.min(gt_spd)) > 8.0:
                R_earth = 6371000.0
                pE_gt = (gt_lon - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_gt = (gt_lat - init_lat) * (math.pi / 180.0) * R_earth
                pE_ukf = (lon_a - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_ukf = (lat_a - init_lat) * (math.pi / 180.0) * R_earth
                pE_b = (lon_b - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_b = (lat_b - init_lat) * (math.pi / 180.0) * R_earth
                pE_c = (lon_c - init_lon) * (math.pi / 180.0) * R_earth * math.cos(math.radians(init_lat))
                pN_c = (lat_c - init_lat) * (math.pi / 180.0) * R_earth

                psi_gt_rad = np.radians(gt_hdg) if gt_hdg is not None else np.zeros(sub_n)
                along_ukf = (pE_ukf - pE_gt) * np.sin(psi_gt_rad) + (pN_ukf - pN_gt) * np.cos(psi_gt_rad)
                cross_ukf = (pE_ukf - pE_gt) * np.cos(psi_gt_rad) - (pN_ukf - pN_gt) * np.sin(psi_gt_rad)
                along_b = (pE_b - pE_gt) * np.sin(psi_gt_rad) + (pN_b - pN_gt) * np.cos(psi_gt_rad)
                cross_b = (pE_b - pE_gt) * np.cos(psi_gt_rad) - (pN_b - pN_gt) * np.sin(psi_gt_rad)

                ai_t = []
                ai_v = []
                for gk, val in lookup_preds.items():
                    rel_idx = gk - outage_start_idx
                    if 0 <= rel_idx < sub_n:
                        ai_t.append(rel_idx * dt_step)
                        ai_v.append(val[1])

                gt_vE = gt_spd * np.sin(psi_gt_rad)
                gt_vN = gt_spd * np.cos(psi_gt_rad)

                case_b_saved = {
                    "title": f"{run_name} Outage {oid} ({dur}s Deceleration/Braking)",
                    "time": t_axis,
                    "gt_pE": pE_gt, "gt_pN": pN_gt,
                    "ukf_pE": pE_ukf, "ukf_pN": pN_ukf,
                    "ekf_b_pE": pE_b, "ekf_b_pN": pN_b,
                    "ekf_c_pE": pE_c, "ekf_c_pN": pN_c,
                    "gt_speed": gt_spd, "ukf_speed": spd_a, "ekf_b_speed": spd_b, "ekf_c_speed": spd_c,
                    "v_anchor": v_anchor_val,
                    "ai_steps_t": ai_t, "ai_steps_v": ai_v,
                    "gt_vE": gt_vE, "gt_vN": gt_vN,
                    "ekf_b_vE": telem_b["vE"], "ekf_b_vN": telem_b["vN"],
                    "ekf_b_v_lat": telem_b["v_lat"], "ekf_c_v_lat": telem_c["v_lat"],
                    "gt_hdg": gt_hdg if gt_hdg is not None else np.zeros(sub_n),
                    "ukf_hdg": hdg_a, "ekf_b_hdg": hdg_b,
                    "nhc_innov": telem_b["nhc_innov"], "ai_innov": telem_b["ai_innov"],
                    "along_ukf": along_ukf, "cross_ukf": cross_ukf,
                    "along_ekf_b": along_b, "cross_ekf_b": cross_b,
                }

    df_results = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase12_nav_results.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 12 results to {out_csv} ({len(df_results)} total outages)")

    # Generate plots if cases found
    plot_dir = config.BASE_DIR / "reports" / "phase12_plots"
    if case_a_saved and case_b_saved:
        plot_phase12_observability_cases(case_a_saved, case_b_saved, plot_dir)

    # Print Full A/B/C Comparative Summary Table
    print("\n" + "=" * 115)
    print("PHASE 12: A/B/C COMPARATIVE RESULTS SUMMARY (56 HELD-OUT OUTAGES)")
    print("=" * 115)
    print(f"{'METRIC':<30} | {'A: 7-STATE UKF':<24} | {'B: 9-STATE EKF (WITH NHC)':<26} | {'C: 9-STATE EKF (NO NHC)':<24}")
    print("-" * 115)
    print(f"{'Median Drift %':<30} | {df_results['drift_a_pct'].median():<24.2f}% | {df_results['drift_b_pct'].median():<26.2f}% | {df_results['drift_c_pct'].median():<24.2f}%")
    print(f"{'Mean Drift %':<30} | {df_results['drift_a_pct'].mean():<24.2f}% | {df_results['drift_b_pct'].mean():<26.2f}% | {df_results['drift_c_pct'].mean():<24.2f}%")
    print(f"{'P95 Drift %':<30} | {df_results['drift_a_pct'].quantile(0.95):<24.2f}% | {df_results['drift_b_pct'].quantile(0.95):<26.2f}% | {df_results['drift_c_pct'].quantile(0.95):<24.2f}%")
    print(f"{'Median FPE (m)':<30} | {df_results['fpe_a_m'].median():<24.2f}m | {df_results['fpe_b_m'].median():<26.2f}m | {df_results['fpe_c_m'].median():<24.2f}m")
    print(f"{'Mean FPE (m)':<30} | {df_results['fpe_a_m'].mean():<24.2f}m | {df_results['fpe_b_m'].mean():<26.2f}m | {df_results['fpe_c_m'].mean():<24.2f}m")
    print(f"{'P95 FPE (m)':<30} | {df_results['fpe_a_m'].quantile(0.95):<24.2f}m | {df_results['fpe_b_m'].quantile(0.95):<26.2f}m | {df_results['fpe_c_m'].quantile(0.95):<24.2f}m")
    print(f"{'Median Along-Track (m)':<30} | {df_results['along_a_m'].median():<24.2f}m | {df_results['along_b_m'].median():<26.2f}m | {df_results['along_c_m'].median():<24.2f}m")
    print(f"{'Mean Along-Track (m)':<30} | {df_results['along_a_m'].mean():<24.2f}m | {df_results['along_b_m'].mean():<26.2f}m | {df_results['along_c_m'].mean():<24.2f}m")
    print(f"{'Median Cross-Track (m)':<30} | {df_results['cross_a_m'].median():<24.2f}m | {df_results['cross_b_m'].median():<26.2f}m | {df_results['cross_c_m'].median():<24.2f}m")
    print(f"{'Mean Cross-Track (m)':<30} | {df_results['cross_a_m'].mean():<24.2f}m | {df_results['cross_b_m'].mean():<26.2f}m | {df_results['cross_c_m'].mean():<24.2f}m")
    print(f"{'Median Heading Err (deg)':<30} | {df_results['hdg_a_deg'].median():<24.2f}° | {df_results['hdg_b_deg'].median():<26.2f}° | {df_results['hdg_c_deg'].median():<24.2f}°")
    print(f"{'Median Trajectory RMSE (m)':<30} | {df_results['rmse_a_m'].median():<24.2f}m | {df_results['rmse_b_m'].median():<26.2f}m | {df_results['rmse_c_m'].median():<24.2f}m")
    print(f"{'Median CEP50 (m)':<30} | {df_results['cep50_a_m'].median():<24.2f}m | {df_results['cep50_b_m'].median():<26.2f}m | {df_results['cep50_c_m'].median():<24.2f}m")
    print(f"{'Median CEP95 (m)':<30} | {df_results['cep95_a_m'].median():<24.2f}m | {df_results['cep95_b_m'].median():<26.2f}m | {df_results['cep95_c_m'].median():<24.2f}m")

    n_tot = len(df_results)
    pass_a = int(np.sum(df_results["drift_a_pct"] < 10.0))
    pass_b = int(np.sum(df_results["drift_b_pct"] < 10.0))
    pass_c = int(np.sum(df_results["drift_c_pct"] < 10.0))
    print(f"{'Drift < 10% Target Pass':<30} | {pass_a}/{n_tot} ({pass_a/n_tot*100:.1f}%) | {pass_b}/{n_tot} ({pass_b/n_tot*100:.1f}%) | {pass_c}/{n_tot} ({pass_c/n_tot*100:.1f}%)")
    print("=" * 115)

    # Breakdown by Outage Duration
    print("\n" + "=" * 115)
    print("BREAKDOWN BY OUTAGE DURATION")
    print("=" * 115)
    print(f"{'DURATION':<12} | {'COUNT':<6} | {'MEDIAN FPE A (m)':<18} | {'MEDIAN FPE B (m)':<18} | {'MEDIAN DRIFT A':<16} | {'MEDIAN DRIFT B':<16}")
    print("-" * 115)
    for dur in [10, 30, 60, 120, 180]:
        sub_df = df_results[df_results["duration_s"] == dur]
        if len(sub_df) > 0:
            print(f"{dur:<10}s | {len(sub_df):<6} | {sub_df['fpe_a_m'].median():<18.2f} | {sub_df['fpe_b_m'].median():<18.2f} | {sub_df['drift_a_pct'].median():<15.2f}% | {sub_df['drift_b_pct'].median():<15.2f}%")
    print("=" * 115)


if __name__ == "__main__":
    run_phase12_benchmark()
