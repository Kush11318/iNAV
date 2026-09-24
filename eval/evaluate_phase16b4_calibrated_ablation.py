"""
Phase 16B-4: Calibrated Differential Wheel-Yaw Final Benchmark

Evaluates:
  - System A (Control): 7-State UKF + Phase 11 Anchored Temporal VelocityNet + NHC + ZUPT
  - System B (CAN Speed Only): 7-State UKF + CAN Speed Replacement (10 Hz, R_can = 0.0325) + NHC + ZUPT
  - System C (CAN Speed + CALIBRATED Differential Wheel Yaw):
      7-State UKF + CAN Speed + Rear-Wheel Calibrated Differential Yaw
      (10 Hz, R_yaw = (1.56 deg/s)^2 = 0.0007413 rad^2/s^2, B_eff = 1.4976 m,
       k_diff estimated strictly from pre-outage straight-driving data).

Evaluates all 56 held-out test outages from the IO-VNBD synchronized benchmark.
Strict experiment isolation:
  - Production code remains completely untouched.
  - Zero ground-truth leakage during simulated outages.
  - Strictly healthy pre-outage data for calibration (t < outage_start).
  - Fixed B_eff = 1.4976 m, fixed R_yaw = 0.0007413 rad^2/s^2.
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
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from modules.alignment import AlignmentEngine, AlignmentState
from modules.can_fusion_ukf import CANFusionUKF, compute_pre_outage_can_calibration
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase16B4_Ablation")

ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")
B_EFF_VALIDATED = 1.4976  # Phase 16B-1 validated rear track width (m)
# Phase 16B-3 calibrated yaw noise variance: sigma_yaw = 1.56 deg/s -> 0.027227 rad/s
R_YAW_CALIBRATED = (1.56 * math.pi / 180.0) ** 2  # 0.0007413 rad^2/s^2
K_DIFF_PRIOR = 0.9946  # Established prior from Phase 16B-3 global analysis


def compute_pre_outage_diff_calibration(
    pre_wheel_rl: np.ndarray,
    pre_wheel_rr: np.ndarray,
    pre_gt_yaw_deg: np.ndarray,
    pre_gps_speed: np.ndarray,
    prior_k: float = K_DIFF_PRIOR,
    min_speed_ms: float = 2.5,
    min_samples: int = 15
) -> Tuple[float, str, int]:
    """
    Estimates k_diff = median(omega_RR / omega_RL) strictly from healthy
    pre-outage straight-driving data (|gt_yaw_rate| < 0.5 deg/s, speed > 2.5 m/s).
    """
    valid = (
        (pre_gps_speed > min_speed_ms) &
        (pre_wheel_rl > 5.0) &
        (pre_wheel_rr > 5.0) &
        (np.abs(pre_gt_yaw_deg) < 0.5) &
        (~np.isnan(pre_wheel_rl)) &
        (~np.isnan(pre_wheel_rr))
    )
    n_valid = int(np.sum(valid))
    if n_valid >= min_samples:
        k_diff = float(np.median(pre_wheel_rr[valid] / pre_wheel_rl[valid]))
        return k_diff, "pre_outage_calibrated", n_valid
    else:
        return prior_k, "prior_fallback", n_valid


def compute_baseline_a_predictions(
    model: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_frozen: float,
    window_len: int = 40,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float, float]]:
    lookup = {}
    model.eval()
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

    all_windows = np.array(windows, dtype=np.float32)
    bx = torch.from_numpy(all_windows.transpose(0, 2, 1)).float().to(device)
    b_anc = torch.full((len(all_windows), 1), v_anchor_frozen, dtype=torch.float32).to(device)

    with torch.no_grad():
        del_v_seq, v_seq, sig_seq, ev = model(bx, b_anc)
        p_v = v_seq.cpu().numpy()
        p_sig = sig_seq.cpu().numpy()
        p_ev = torch.argmax(ev, dim=1).cpu().numpy()
        p_v_mean = np.mean(p_v, axis=1)
        p_v_end = p_v[:, -1]
        p_sig_mean = np.mean(p_sig, axis=1)

    for idx, gk in enumerate(step_indices):
        lookup[gk] = (
            float(p_v_mean[idx]),
            int(p_ev[idx]),
            float(p_sig_mean[idx]),
            float(p_v_end[idx])
        )
    return lookup


def run_outage_simulation_ablation(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    mode: str = "A",  # 'A': Control, 'B': CAN Speed Only, 'C': CAN Speed + CALIBRATED Differential Wheel Yaw
    r_eff: float = 0.2776,
    b_eff: float = B_EFF_VALIDATED,
    k_diff: float = K_DIFF_PRIOR,
    r_can_var: float = 0.0325,
    r_yaw_var: float = R_YAW_CALIBRATED,
    nis_gate: float = 16.0,
    dt: float = config.TARGET_DT
) -> Dict[str, Any]:
    n = len(df_outage)
    if n == 0:
        return {}

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    omega_rl = df_outage[config.COL_TRUE_WHEEL_RL].values
    omega_rr = df_outage[config.COL_TRUE_WHEEL_RR].values

    # Forward velocity uses symmetrical average (Phase 16A)
    v_rl = omega_rl * r_eff
    v_rr = omega_rr * r_eff
    v_rear = 0.5 * (v_rl + v_rr)

    # Calibrated differential yaw rate: (omega_RR - k_diff * omega_RL) * R_eff / B_eff
    z_wheel_yaw_arr = (omega_rr - k_diff * omega_rl) * r_eff / b_eff

    allow_bias_learning = (mode == "C")
    ukf = CANFusionUKF(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg),
        allow_bias_learning=allow_bias_learning
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading_deg = np.zeros(n)
    est_bg = np.zeros(n)

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Prediction step
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)
        global_k = global_start_idx + k

        # 2. Measurement updates
        if mode == "A":
            if global_k in lookup_preds and (k % 5 == 0):
                v_meas, ev_class, sig_meas, _ = lookup_preds[global_k]
                cal_d = v_meas * 2.0
                cal_sig = max(sig_meas * 2.0, 0.2)
                if ev_class == 0 or v_meas < 0.15:
                    ukf.update_zupt(gyro_reading=w_v[2])
                else:
                    ukf.update_velocity_net(cal_d, cal_sig, ev_class, window_dur=2.0)

        elif mode == "B":
            v_val = float(v_rear[k])
            rear_diff = abs(omega_rl[k] - omega_rr[k])
            is_slipping = rear_diff > 25.0
            if v_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_val, r_var=r_can_var)

        elif mode == "C":
            # CAN forward speed update
            v_val = float(v_rear[k])
            rear_diff = abs(omega_rl[k] - omega_rr[k])
            is_slipping = rear_diff > 25.0
            if v_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_val, r_var=r_can_var)

            # Calibrated differential rear-wheel yaw rate update
            if not is_slipping:
                z_yaw = float(z_wheel_yaw_arr[k])
                ukf.update_wheel_yaw(z_wheel_yaw=z_yaw, gyro_reading=w_v[2], r_yaw=r_yaw_var, nis_gate=nis_gate)

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]
        est_heading_deg[k] = math.degrees(ukf.x[3]) % 360.0
        est_bg[k] = ukf.x[4]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)

    gt_lat = df_outage[config.COL_TRUE_LAT].values
    gt_lon = df_outage[config.COL_TRUE_LON].values
    R_earth = 6371000.0
    d_lat = np.radians(est_lat - gt_lat) * R_earth
    d_lon = np.radians(est_lon - gt_lon) * R_earth * np.cos(np.radians(gt_lat))
    pos_err_series = np.sqrt(d_lat**2 + d_lon**2)

    return {
        "lat": est_lat,
        "lon": est_lon,
        "speed": est_speed,
        "heading_deg": est_heading_deg,
        "pN": est_pN,
        "pE": est_pE,
        "bg": est_bg,
        "pos_err": pos_err_series,
        "yaw_accepted": ukf.accepted_yaw_updates,
        "yaw_rejected": ukf.rejected_yaw_updates,
        "yaw_total": ukf.total_yaw_updates,
        "yaw_nis_hist": ukf.yaw_nis_history,
        "rejection_reasons": ukf.rejection_reasons
    }


def run_phase16b4_ablation():
    logger.info("=" * 95)
    logger.info("PHASE 16B-4: CALIBRATED DIFFERENTIAL WHEEL-YAW FINAL BENCHMARK")
    logger.info("A = Existing Control (Phase 11 Anchored VelocityNet)")
    logger.info("B = CAN Speed Only (Phase 16A)")
    logger.info("C = CAN Speed + CALIBRATED Differential Wheel Yaw (Phase 16B-4)")
    logger.info(f"Fixed Parameters: B_eff = {B_EFF_VALIDATED} m, R_yaw = {R_YAW_CALIBRATED:.7f} rad^2/s^2 (sigma=1.56 deg/s)")
    logger.info("=" * 95)

    model_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    records = []
    traces_store = {}
    all_yaw_nis = []
    total_accepted_yaw = 0
    total_attempted_yaw = 0

    for tf in test_files:
        run_name = tf.stem.replace("sync_", "")
        df = pd.read_parquet(tf)
        total_dur = len(df) * config.TARGET_DT
        if total_dur < 30.0:
            continue

        schedule = generate_outage_schedule(total_dur, run_id=run_name)
        if not schedule:
            continue

        df_sim, df_outages = inject_outages(df, schedule)

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

            if len(pre_df) > 0:
                init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
                init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
                init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
                h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
                init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
                v_anchor_val = float(pre_df[config.COL_TRUE_SPEED_MS].iloc[-1])
            else:
                init_lat = float(df[config.COL_TRUE_LAT].iloc[outage_start_idx])
                init_lon = float(df[config.COL_TRUE_LON].iloc[outage_start_idx])
                init_spd = float(df[config.COL_TRUE_SPEED_MS].iloc[outage_start_idx])
                init_hdg = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx])
                v_anchor_val = init_spd

            # 1. Forward Speed R_eff calibration strictly from pre-outage GNSS
            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, cal_method_can, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values,
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

            # 2. Multiplicative Wheel-Yaw k_diff calibration strictly from pre-outage straight driving
            gt_yaw_col = cal_pre_df["gt_yaw_rate_degs"].values if "gt_yaw_rate_degs" in cal_pre_df.columns else np.zeros(len(cal_pre_df))
            k_diff_est, cal_method_diff, n_str_samples = compute_pre_outage_diff_calibration(
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values,
                pre_gt_yaw_deg=gt_yaw_col,
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values,
                prior_k=K_DIFF_PRIOR
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            lookup_a = compute_baseline_a_predictions(model, seq_data, outage_start_idx, len(df_sub), v_anchor_val)

            # 1. System A: Existing Control
            res_a = run_outage_simulation_ablation(
                df_sub, init_lat, init_lon, init_spd, init_hdg, lookup_a,
                outage_start_idx, run_align, mode="A"
            )
            score_a = score_outage_segment(res_a["lat"], res_a["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 2. System B: CAN Speed Only
            res_b = run_outage_simulation_ablation(
                df_sub, init_lat, init_lon, init_spd, init_hdg, lookup_a,
                outage_start_idx, run_align, mode="B", r_eff=r_eff_est
            )
            score_b = score_outage_segment(res_b["lat"], res_b["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 3. System C: CAN Speed + CALIBRATED Differential Wheel Yaw
            res_c = run_outage_simulation_ablation(
                df_sub, init_lat, init_lon, init_spd, init_hdg, lookup_a,
                outage_start_idx, run_align, mode="C", r_eff=r_eff_est, b_eff=B_EFF_VALIDATED,
                k_diff=k_diff_est, r_yaw_var=R_YAW_CALIBRATED, nis_gate=16.0
            )
            score_c = score_outage_segment(res_c["lat"], res_c["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            all_yaw_nis.extend(res_c["yaw_nis_hist"])
            total_accepted_yaw += res_c["yaw_accepted"]
            total_attempted_yaw += res_c["yaw_total"]

            sc_id = f"{run_name}_o{oid}"
            traces_store[sc_id] = {
                "t": np.arange(len(df_sub)) * config.TARGET_DT,
                "dur": dur, "run": run_name, "oid": oid,
                "gt_lat": gt_lat, "gt_lon": gt_lon, "gt_hdg": gt_hdg,
                "lat_a": res_a["lat"], "lon_a": res_a["lon"], "hdg_a": res_a["heading_deg"], "pos_err_a": res_a["pos_err"], "bg_a": res_a["bg"],
                "lat_b": res_b["lat"], "lon_b": res_b["lon"], "hdg_b": res_b["heading_deg"], "pos_err_b": res_b["pos_err"], "bg_b": res_b["bg"],
                "lat_c": res_c["lat"], "lon_c": res_c["lon"], "hdg_c": res_c["heading_deg"], "pos_err_c": res_c["pos_err"], "bg_c": res_c["bg"],
            }

            yaw_acc_rate = (res_c["yaw_accepted"] / max(res_c["yaw_total"], 1)) * 100.0
            mean_nis = float(np.mean(res_c["yaw_nis_hist"])) if res_c["yaw_nis_hist"] else 0.0

            # Initial vs final heading error
            if gt_hdg is not None and len(gt_hdg) > 0:
                init_hdg_err = abs((init_hdg - gt_hdg[0] + 180.0) % 360.0 - 180.0)
                final_hdg_err_a = abs((res_a["heading_deg"][-1] - gt_hdg[-1] + 180.0) % 360.0 - 180.0)
                final_hdg_err_b = abs((res_b["heading_deg"][-1] - gt_hdg[-1] + 180.0) % 360.0 - 180.0)
                final_hdg_err_c = abs((res_c["heading_deg"][-1] - gt_hdg[-1] + 180.0) % 360.0 - 180.0)
            else:
                init_hdg_err = np.nan
                final_hdg_err_a = np.nan
                final_hdg_err_b = np.nan
                final_hdg_err_c = np.nan

            rec = {
                "scenario": sc_id,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                "k_diff": round(k_diff_est, 6),
                "k_cal_method": cal_method_diff,
                "n_pre_str": n_str_samples,
                "r_eff": round(r_eff_est, 4),
                # System A (Control)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                "init_hdg_err_deg": round(float(init_hdg_err), 2),
                "final_hdg_a_deg": round(float(final_hdg_err_a), 2),
                "init_bg_a": round(float(res_a["bg"][0]), 6),
                "final_bg_a": round(float(res_a["bg"][-1]), 6),
                # System B (CAN Speed Only)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
                "final_hdg_b_deg": round(float(final_hdg_err_b), 2),
                "init_bg_b": round(float(res_b["bg"][0]), 6),
                "final_bg_b": round(float(res_b["bg"][-1]), 6),
                # System C (CAN Speed + CALIBRATED Differential Wheel Yaw)
                "fpe_c_m": round(float(score_c["final_pos_error_m"]), 2),
                "drift_c_pct": round(float(score_c["pct_of_distance"]), 2),
                "along_c_m": round(float(score_c.get("along_track_m", np.nan)), 2),
                "cross_c_m": round(float(score_c.get("cross_track_m", np.nan)), 2),
                "hdg_c_deg": round(float(score_c.get("heading_error_deg", np.nan)), 2),
                "final_hdg_c_deg": round(float(final_hdg_err_c), 2),
                "init_bg_c": round(float(res_c["bg"][0]), 6),
                "final_bg_c": round(float(res_c["bg"][-1]), 6),
                # Yaw Diagnostics
                "yaw_accepted": res_c["yaw_accepted"],
                "yaw_rejected": res_c["yaw_rejected"],
                "yaw_acc_pct": round(yaw_acc_rate, 1),
                "mean_yaw_nis": round(mean_nis, 2),
            }
            records.append(rec)
            logger.info(
                f"[{sc_id}] Dur={dur:3d}s | FPE: A={rec['fpe_a_m']:6.1f}m, B={rec['fpe_b_m']:6.1f}m, C={rec['fpe_c_m']:6.1f}m | "
                f"HdgErr: A={rec['hdg_a_deg']:5.1f}°, B={rec['hdg_b_deg']:5.1f}°, C={rec['hdg_c_deg']:5.1f}° | "
                f"k_diff={rec['k_diff']:.5f} ({cal_method_diff}) | YawAcc={yaw_acc_rate:4.1f}%"
            )

    df_results = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase16b4_calibrated_results.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved benchmark results to {out_csv}")

    # =========================================================================
    # AGGREGATE SUMMARY CALCULATIONS
    # =========================================================================
    print("\n" + "=" * 125)
    print("PHASE 16B-4: CALIBRATED DIFFERENTIAL WHEEL-YAW FINAL BENCHMARK RESULTS")
    print("=" * 125)

    def print_metric_row(metric_name, a_val, b_val, c_val, unit="", is_pct=False):
        c_vs_a = c_val - a_val
        c_vs_b = c_val - b_val
        diff_str_a = f"{c_vs_a:+7.2f}{unit}"
        diff_str_b = f"{c_vs_b:+7.2f}{unit}"
        fmt = "{:8.2f}" if is_pct else "{:8.2f}"
        print(f"| {metric_name:<26} | {fmt.format(a_val)}{unit} | {fmt.format(b_val)}{unit} | {fmt.format(c_val)}{unit} | {diff_str_a:<16} | {diff_str_b:<16} |")

    print(f"| {'Metric':<26} | {'System A (Ctrl)':<15} | {'System B (CAN)':<15} | {'System C (Calib)':<16} | {'C vs A':<16} | {'C vs B':<16} |")
    print("-" * 125)
    print_metric_row("Median Drift %", df_results["drift_a_pct"].median(), df_results["drift_b_pct"].median(), df_results["drift_c_pct"].median(), "%", True)
    print_metric_row("Mean Drift %", df_results["drift_a_pct"].mean(), df_results["drift_b_pct"].mean(), df_results["drift_c_pct"].mean(), "%", True)
    print_metric_row("P95 Drift %", np.percentile(df_results["drift_a_pct"], 95), np.percentile(df_results["drift_b_pct"], 95), np.percentile(df_results["drift_c_pct"], 95), "%", True)
    print_metric_row("Median FPE", df_results["fpe_a_m"].median(), df_results["fpe_b_m"].median(), df_results["fpe_c_m"].median(), " m")
    print_metric_row("Mean FPE", df_results["fpe_a_m"].mean(), df_results["fpe_b_m"].mean(), df_results["fpe_c_m"].mean(), " m")
    print_metric_row("Median Along-Track", df_results["along_a_m"].median(), df_results["along_b_m"].median(), df_results["along_c_m"].median(), " m")
    print_metric_row("Mean Along-Track", df_results["along_a_m"].mean(), df_results["along_b_m"].mean(), df_results["along_c_m"].mean(), " m")
    print_metric_row("Median Cross-Track", df_results["cross_a_m"].median(), df_results["cross_b_m"].median(), df_results["cross_c_m"].median(), " m")
    print_metric_row("Mean Cross-Track", df_results["cross_a_m"].mean(), df_results["cross_b_m"].mean(), df_results["cross_c_m"].mean(), " m")
    print_metric_row("Median Heading Error", df_results["hdg_a_deg"].median(), df_results["hdg_b_deg"].median(), df_results["hdg_c_deg"].median(), "°")
    print_metric_row("Mean Heading Error", df_results["hdg_a_deg"].mean(), df_results["hdg_b_deg"].mean(), df_results["hdg_c_deg"].mean(), "°")

    cnt_a10 = int(np.sum(df_results["drift_a_pct"] < 10.0))
    cnt_b10 = int(np.sum(df_results["drift_b_pct"] < 10.0))
    cnt_c10 = int(np.sum(df_results["drift_c_pct"] < 10.0))
    print(f"| {'Drift < 10% Target Count':<26} | {cnt_a10:>7d} / 56   | {cnt_b10:>7d} / 56   | {cnt_c10:>8d} / 56   | {cnt_c10 - cnt_a10:+16d} | {cnt_c10 - cnt_b10:+16d} |")
    print("=" * 125)

    # Acceptance statistics
    overall_acc_rate = (total_accepted_yaw / max(total_attempted_yaw, 1)) * 100.0
    print(f"\nWheel-Yaw NIS Gate Statistics:")
    print(f"  - Total updates attempted: {total_attempted_yaw}")
    print(f"  - Total updates accepted : {total_accepted_yaw} ({overall_acc_rate:.2f}%)")
    print(f"  - Total updates rejected : {total_attempted_yaw - total_accepted_yaw} ({100.0 - overall_acc_rate:.2f}%)")
    if all_yaw_nis:
        print(f"  - Median NIS: {np.median(all_yaw_nis):.3f} | Mean NIS: {np.mean(all_yaw_nis):.3f} | P95 NIS: {np.percentile(all_yaw_nis, 95):.3f}")

    # =========================================================================
    # PHASE 16A 19-FAILURE RECOVERY ANALYSIS
    # =========================================================================
    worsened_scenarios = df_results[df_results["fpe_b_m"] > df_results["fpe_a_m"]]["scenario"].tolist()
    df_19 = df_results[df_results["scenario"].isin(worsened_scenarios)].copy()
    print("\n" + "=" * 115)
    print(f"PHASE 16A FAILURE RECOVERY ANALYSIS ({len(df_19)} / 19 SCENARIOS WHERE CAN SPEED WORSENED FPE)")
    print("=" * 115)
    print(f"| {'Scenario':<26} | {'Dur':<4} | {'Ctrl FPE (A)':<13} | {'CAN FPE (B)':<13} | {'Calib FPE (C)':<14} | {'Hdg A':<8} | {'Hdg C':<8} | {'Status':<16} |")
    print("-" * 115)

    n_recovered = 0
    n_partial = 0
    n_still_failed = 0

    for _, r in df_19.iterrows():
        fpe_a = r["fpe_a_m"]
        fpe_b = r["fpe_b_m"]
        fpe_c = r["fpe_c_m"]
        hdg_a = r["hdg_a_deg"]
        hdg_c = r["hdg_c_deg"]

        if fpe_c < fpe_a:
            status = "RECOVERED (C<A)"
            n_recovered += 1
        elif fpe_c < fpe_b:
            status = "PARTIAL (C<B)"
            n_partial += 1
        else:
            status = "FAILED (C>=B)"
            n_still_failed += 1

        print(f"| {r['scenario']:<26} | {int(r['duration_s']):<4d} | {fpe_a:11.2f}m | {fpe_b:11.2f}m | {fpe_c:12.2f}m | {hdg_a:6.1f}° | {hdg_c:6.1f}° | {status:<16} |")

    print("=" * 115)
    print(f"Recovery Summary:")
    print(f"  - Fully Recovered (C < A): {n_recovered} / {len(df_19)} ({n_recovered / len(df_19) * 100.0:.1f}%)")
    print(f"  - Partially Recovered (C < B): {n_partial} / {len(df_19)} ({n_partial / len(df_19) * 100.0:.1f}%)")
    print(f"  - Still Failed (C >= B): {n_still_failed} / {len(df_19)} ({n_still_failed / len(df_19) * 100.0:.1f}%)")

    # =========================================================================
    # DURATION BREAKDOWN TABLE
    # =========================================================================
    print("\n" + "=" * 115)
    print("DURATION BREAKDOWN: SYSTEM A vs B vs C (MEDIANS)")
    print("=" * 115)
    durations = [10, 30, 60, 120, 180]
    print(f"| {'Dur':<5} | {'N':<4} | {'Drift% (A/B/C)':<22} | {'FPE (A/B/C)':<24} | {'Along-Track (A/B/C)':<24} | {'Heading (A/B/C)':<22} |")
    print("-" * 115)
    for d in durations:
        sub = df_results[df_results["duration_s"] == d]
        if len(sub) > 0:
            d_str = f"{sub['drift_a_pct'].median():.1f} / {sub['drift_b_pct'].median():.1f} / {sub['drift_c_pct'].median():.1f}%"
            f_str = f"{sub['fpe_a_m'].median():.1f} / {sub['fpe_b_m'].median():.1f} / {sub['fpe_c_m'].median():.1f}m"
            a_str = f"{sub['along_a_m'].median():.1f} / {sub['along_b_m'].median():.1f} / {sub['along_c_m'].median():.1f}m"
            h_str = f"{sub['hdg_a_deg'].median():.1f} / {sub['hdg_b_deg'].median():.1f} / {sub['hdg_c_deg'].median():.1f}°"
            print(f"| {d:<5d} | {len(sub):<4d} | {d_str:<22} | {f_str:<24} | {a_str:<24} | {h_str:<22} |")
    print("=" * 115)

    # =========================================================================
    # GENERATE DIAGNOSTIC FIGURES FOR ARTIFACTS
    # =========================================================================
    logger.info("Generating Phase 16B-4 visualization artifacts...")

    # Fig 1: Aggregate FPE, Heading, Along-Track, Cross-Track Comparison (A vs B vs C)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Panel 1: Median Metrics
    metrics_labels = ["Median FPE (m)", "Median Along (m)", "Median Cross (m)", "Median Hdg (°)"]
    vals_a = [df_results["fpe_a_m"].median(), df_results["along_a_m"].median(), df_results["cross_a_m"].median(), df_results["hdg_a_deg"].median()]
    vals_b = [df_results["fpe_b_m"].median(), df_results["along_b_m"].median(), df_results["cross_b_m"].median(), df_results["hdg_b_deg"].median()]
    vals_c = [df_results["fpe_c_m"].median(), df_results["along_c_m"].median(), df_results["cross_c_m"].median(), df_results["hdg_c_deg"].median()]

    x_pos = np.arange(len(metrics_labels))
    w = 0.25
    axs[0, 0].bar(x_pos - w, vals_a, width=w, label="Control (A)", color="#4A90E2")
    axs[0, 0].bar(x_pos, vals_b, width=w, label="CAN Speed (B)", color="#F5A623")
    axs[0, 0].bar(x_pos + w, vals_c, width=w, label="Calib Wheel Yaw (C)", color="#2ECC71")
    axs[0, 0].set_xticks(x_pos)
    axs[0, 0].set_xticklabels(metrics_labels, fontweight="bold")
    axs[0, 0].set_title("Figure 1A: Aggregate Median Metrics Comparison", fontweight="bold")
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    axs[0, 0].legend()

    # Panel 2: FPE by Outage Duration
    dur_fpe_a = [df_results[df_results["duration_s"] == d]["fpe_a_m"].median() for d in durations]
    dur_fpe_b = [df_results[df_results["duration_s"] == d]["fpe_b_m"].median() for d in durations]
    dur_fpe_c = [df_results[df_results["duration_s"] == d]["fpe_c_m"].median() for d in durations]
    axs[0, 1].plot(durations, dur_fpe_a, "o-", color="#4A90E2", linewidth=2, label="Control (A)")
    axs[0, 1].plot(durations, dur_fpe_b, "s-", color="#F5A623", linewidth=2, label="CAN Speed (B)")
    axs[0, 1].plot(durations, dur_fpe_c, "^-", color="#2ECC71", linewidth=2.5, label="Calib Wheel Yaw (C)")
    axs[0, 1].set_xlabel("Outage Duration (s)", fontweight="bold")
    axs[0, 1].set_ylabel("Median FPE (m)", fontweight="bold")
    axs[0, 1].set_title("Figure 1B: Median FPE vs Outage Duration", fontweight="bold")
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    axs[0, 1].legend()

    # Panel 3: Heading Error by Duration
    dur_hdg_a = [df_results[df_results["duration_s"] == d]["hdg_a_deg"].median() for d in durations]
    dur_hdg_b = [df_results[df_results["duration_s"] == d]["hdg_b_deg"].median() for d in durations]
    dur_hdg_c = [df_results[df_results["duration_s"] == d]["hdg_c_deg"].median() for d in durations]
    axs[1, 0].plot(durations, dur_hdg_a, "o-", color="#4A90E2", linewidth=2, label="Control (A)")
    axs[1, 0].plot(durations, dur_hdg_b, "s-", color="#F5A623", linewidth=2, label="CAN Speed (B)")
    axs[1, 0].plot(durations, dur_hdg_c, "^-", color="#2ECC71", linewidth=2.5, label="Calib Wheel Yaw (C)")
    axs[1, 0].set_xlabel("Outage Duration (s)", fontweight="bold")
    axs[1, 0].set_ylabel("Median Heading Error (°)", fontweight="bold")
    axs[1, 0].set_title("Figure 1C: Median Heading Error vs Outage Duration", fontweight="bold")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    axs[1, 0].legend()

    # Panel 4: 19 Failure Recovery Bar Chart
    rec_cats = ["Fully Recovered\n(C < A)", "Partially Recovered\n(C < B)", "Still Failed\n(C >= B)"]
    rec_counts = [n_recovered, n_partial, n_still_failed]
    rec_colors = ["#2ECC71", "#F39C12", "#E74C3C"]
    axs[1, 1].bar(rec_cats, rec_counts, color=rec_colors, width=0.5)
    for i, v in enumerate(rec_counts):
        axs[1, 1].text(i, v + 0.3, f"{v} ({v/len(df_19)*100:.1f}%)", ha="center", fontweight="bold")
    axs[1, 1].set_ylabel("Count of Outages", fontweight="bold")
    axs[1, 1].set_title(f"Figure 1D: Phase 16A Failure Recovery ({len(df_19)} Scenarios)", fontweight="bold")
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b4_fig1_benchmark_summary.png", dpi=180)
    plt.close()

    # Fig 2: Representative Cases
    # Identify 4 representative cases:
    # 1. Clean successful case: sample_test_trajectory_motorway_o1
    # 2. Typical case: median improvement
    # 3. Phase 16A failure recovered: e.g. vw14c_o2 or vw11_o2
    # 4. Difficult case: e.g. high slip
    rep_cases = [
        ("Clean Successful Case", "sample_test_trajectory_motorway_o1"),
        ("Phase 16A Failure Recovered", "vw14c_o2"),
        ("Typical Urban Outage", "vw10_o1"),
        ("Difficult Outage Case", "vw16a_o4")
    ]

    fig, axs = plt.subplots(4, 2, figsize=(16, 18))
    for row_idx, (case_title, sc_id) in enumerate(rep_cases):
        if sc_id not in traces_store:
            # Fallback to first available
            sc_id = list(traces_store.keys())[row_idx]

        tr = traces_store[sc_id]
        t = tr["t"]

        # Trajectory (Lat/Lon)
        axs[row_idx, 0].plot(tr["gt_lon"], tr["gt_lat"], "k-", linewidth=2.5, label="Ground Truth")
        axs[row_idx, 0].plot(tr["lon_a"], tr["lat_a"], "--", color="#4A90E2", linewidth=1.8, label="Control (A)")
        axs[row_idx, 0].plot(tr["lon_b"], tr["lat_b"], "-.", color="#F5A623", linewidth=1.8, label="CAN Speed (B)")
        axs[row_idx, 0].plot(tr["lon_c"], tr["lat_c"], "-", color="#2ECC71", linewidth=2.2, label="Calib Wheel Yaw (C)")
        axs[row_idx, 0].set_title(f"Case {row_idx+1}: {case_title} ({sc_id}, {tr['dur']}s)", fontweight="bold")
        axs[row_idx, 0].set_xlabel("Longitude")
        axs[row_idx, 0].set_ylabel("Latitude")
        axs[row_idx, 0].grid(True, linestyle=":", alpha=0.6)
        axs[row_idx, 0].legend(loc="best")

        # Heading & Position Error Traces
        ax_pos = axs[row_idx, 1]
        ax_pos.plot(t, tr["pos_err_a"], "--", color="#4A90E2", linewidth=1.8, label="Pos Err A (Control)")
        ax_pos.plot(t, tr["pos_err_b"], "-.", color="#F5A623", linewidth=1.8, label="Pos Err B (CAN Speed)")
        ax_pos.plot(t, tr["pos_err_c"], "-", color="#2ECC71", linewidth=2.2, label="Pos Err C (Calib Wheel Yaw)")
        ax_pos.set_title(f"Position Error Trace ({sc_id})", fontweight="bold")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position Error (m)")
        ax_pos.grid(True, linestyle=":", alpha=0.6)
        ax_pos.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b4_fig2_representative_cases.png", dpi=180)
    plt.close()

    logger.info("Saved all diagnostic figures to artifacts directory.")
    return df_results


if __name__ == "__main__":
    run_phase16b4_ablation()
