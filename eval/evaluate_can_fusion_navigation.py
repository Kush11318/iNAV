"""
Phase 16A: CAN / Wheel-Speed Fusion Navigation Benchmark

Compares:
  - Baseline A (Control): 7-State UKF + Phase 11 Anchored Temporal VelocityNet + NHC + ZUPT
  - Experiment B1: 7-State UKF + CAN Speed Replacement (10 Hz, R_can = 0.0325) + NHC + ZUPT
  - Experiment B2: 7-State UKF + Simultaneous CAN Speed + VelocityNet + NHC + ZUPT

Evaluates across all 56 held-out test outages from the IO-VNBD synchronized benchmark.
Strict experiment isolation:
  - Zero production code modified.
  - Zero ground-truth leakage during simulated outages.
  - Pre-outage calibration strictly uses healthy pre-outage phone GPS and CAN signals.
  - Decouples Along-Track error from Cross-Track error to assess longitudinal observability.
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
from modules.ukf import UKFNavigationFilter
from modules.can_fusion_ukf import CANFusionUKF, compute_pre_outage_can_calibration
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase16A_CAN")


def compute_baseline_a_predictions(
    model: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_frozen: float,
    window_len: int = 40,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float, float]]:
    """
    Baseline A: Frozen pre-outage anchor for the entire outage.
    Returns lookup: step_idx -> (v_mean, event_class, sigma, v_end)
    """
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


def run_outage_simulation(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    mode: str = "A",  # 'A': Control, 'B1': CAN replaces VNet, 'B2': CAN + VNet simultaneous
    r_eff: float = 0.2776,
    r_can_var: float = 0.0325,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Executes the 7-state UKF forward navigation across the outage window.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), []

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    # Calibrated rear wheel speeds
    omega_rl = df_outage[config.COL_TRUE_WHEEL_RL].values
    omega_rr = df_outage[config.COL_TRUE_WHEEL_RR].values
    omega_rear = 0.5 * (omega_rl + omega_rr)
    v_can_arr = omega_rear * r_eff

    ukf = CANFusionUKF(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg)
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Prediction step
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        global_k = global_start_idx + k

        # 2. Measurement updates
        if mode == "A":
            # Baseline A: VelocityNet + ZUPT
            if global_k in lookup_preds and (k % 5 == 0):
                v_meas, ev_class, sig_meas, _ = lookup_preds[global_k]
                cal_d = v_meas * 2.0
                cal_sig = max(sig_meas * 2.0, 0.2)

                if ev_class == 0 or v_meas < 0.15:
                    ukf.update_zupt(gyro_reading=w_v[2])
                else:
                    ukf.update_velocity_net(
                        delta_d_pred=cal_d,
                        sigma_pred=cal_sig,
                        event_class=ev_class,
                        window_dur=2.0
                    )
        elif mode == "B1":
            # Experiment B1: CAN speed replaces VelocityNet (at 10 Hz)
            v_can_val = float(v_can_arr[k])
            # Slip check: rear differential divergence
            rear_diff = abs(omega_rl[k] - omega_rr[k])
            is_slipping = rear_diff > 25.0

            if v_can_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_can_val, r_var=r_can_var)

        elif mode == "B2":
            # Experiment B2: Simultaneous CAN speed (10 Hz) + VelocityNet (0.5 Hz)
            v_can_val = float(v_can_arr[k])
            rear_diff = abs(omega_rl[k] - omega_rr[k])
            is_slipping = rear_diff > 25.0

            if v_can_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_can_val, r_var=r_can_var)

            # Also apply VelocityNet at 0.5 Hz with its uncertainty
            if global_k in lookup_preds and (k % 5 == 0):
                v_meas, ev_class, sig_meas, _ = lookup_preds[global_k]
                cal_d = v_meas * 2.0
                cal_sig = max(sig_meas * 2.0, 0.2)
                if ev_class != 0 and v_meas >= 0.15:
                    ukf.update_velocity_net(
                        delta_d_pred=cal_d,
                        sigma_pred=cal_sig,
                        event_class=ev_class,
                        window_dur=2.0
                    )

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed, list(est_speed)


def run_phase16a_benchmark():
    logger.info("=" * 90)
    logger.info("PHASE 16A: CAN / WHEEL-SPEED FUSION NAVIGATION BENCHMARK")
    logger.info("Baseline A: 7-State UKF + Phase 11 Anchored VelocityNet (Frozen v_anchor)")
    logger.info("Experiment B1: 7-State UKF + CAN Speed Replacement (10 Hz, R_can = 0.0325)")
    logger.info("Experiment B2: 7-State UKF + Simultaneous CAN Speed + VelocityNet")
    logger.info("=" * 90)

    model_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    if not model_path.exists():
        logger.error(f"Missing model checkpoint at {model_path}!")
        return None

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

    logger.info(f"Evaluating {len(test_files)} test files across 56 held-out outages...")

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    records = []
    detailed_profiles = {}

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

        # Standard warmup alignment (identical to previous benchmarks)
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

            # Pre-outage R_eff calibration (strictly pre-outage window, up to 300 epochs)
            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, cal_method, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values,
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None
            gt_speed = df_sub[config.COL_TRUE_SPEED_MS].values

            # Compute Baseline A predictions
            lookup_a = compute_baseline_a_predictions(
                model=model,
                seq_data=seq_data,
                outage_start_idx=outage_start_idx,
                sub_len=len(df_sub),
                v_anchor_frozen=v_anchor_val,
                window_len=40,
                device="cpu"
            )

            # 1. Baseline A
            lat_a, lon_a, spd_a, ukf_spd_a = run_outage_simulation(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_a, global_start_idx=outage_start_idx,
                alignment=run_align, mode="A", dt=config.TARGET_DT
            )
            score_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 2. Experiment B1 (CAN Replaces VelocityNet)
            lat_b1, lon_b1, spd_b1, ukf_spd_b1 = run_outage_simulation(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_a, global_start_idx=outage_start_idx,
                alignment=run_align, mode="B1", r_eff=r_eff_est, dt=config.TARGET_DT
            )
            score_b1 = score_outage_segment(lat_b1, lon_b1, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 3. Experiment B2 (Simultaneous CAN + VelocityNet)
            lat_b2, lon_b2, spd_b2, ukf_spd_b2 = run_outage_simulation(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_a, global_start_idx=outage_start_idx,
                alignment=run_align, mode="B2", r_eff=r_eff_est, dt=config.TARGET_DT
            )
            score_b2 = score_outage_segment(lat_b2, lon_b2, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            rec_id = f"{run_name}_outage_{oid}"
            if rec_id in ["vw11_outage_3", "vw14b_outage_1"] or len(detailed_profiles) < 3:
                detailed_profiles[rec_id] = {
                    "t": np.arange(len(df_sub)) * config.TARGET_DT,
                    "gt_speed": gt_speed,
                    "ukf_spd_a": ukf_spd_a,
                    "ukf_spd_b1": ukf_spd_b1,
                    "ukf_spd_b2": ukf_spd_b2,
                    "v_anchor": v_anchor_val,
                    "dist_gt": dist_gt,
                    "dur": dur
                }

            rec = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                "v_anchor_ms": round(v_anchor_val, 2),
                "r_eff_est": round(r_eff_est, 5),
                "cal_method": cal_method,
                "true_mean_spd_ms": round(float(np.mean(gt_speed)), 2),
                "true_final_spd_ms": round(float(gt_speed[-1]), 2),
                # Baseline A (Control)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                # Experiment B1 (CAN Replaces VelocityNet)
                "fpe_b1_m": round(float(score_b1["final_pos_error_m"]), 2),
                "drift_b1_pct": round(float(score_b1["pct_of_distance"]), 2),
                "along_b1_m": round(float(score_b1.get("along_track_m", np.nan)), 2),
                "cross_b1_m": round(float(score_b1.get("cross_track_m", np.nan)), 2),
                "hdg_b1_deg": round(float(score_b1.get("heading_error_deg", np.nan)), 2),
                # Experiment B2 (CAN + VelocityNet Simultaneous)
                "fpe_b2_m": round(float(score_b2["final_pos_error_m"]), 2),
                "drift_b2_pct": round(float(score_b2["pct_of_distance"]), 2),
                "along_b2_m": round(float(score_b2.get("along_track_m", np.nan)), 2),
                "cross_b2_m": round(float(score_b2.get("cross_track_m", np.nan)), 2),
                "hdg_b2_deg": round(float(score_b2.get("heading_error_deg", np.nan)), 2),
            }
            records.append(rec)

    df_res = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase16a_can_fusion_results.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 16A benchmark results to {out_csv} ({len(df_res)} outages)")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print(f"{'METRIC':<36} | {'BASELINE A (Control)':<25} | {'EXP B1 (CAN Speed)':<25} | {'EXP B2 (CAN + VNet)':<25}")
    print("=" * 125)
    print(f"{'Median Drift %':<36} | {df_res['drift_a_pct'].median():<25.2f}% | {df_res['drift_b1_pct'].median():<25.2f}% | {df_res['drift_b2_pct'].median():<25.2f}%")
    print(f"{'Mean Drift %':<36} | {df_res['drift_a_pct'].mean():<25.2f}% | {df_res['drift_b1_pct'].mean():<25.2f}% | {df_res['drift_b2_pct'].mean():<25.2f}%")
    print(f"{'P95 Drift %':<36} | {df_res['drift_a_pct'].quantile(0.95):<25.2f}% | {df_res['drift_b1_pct'].quantile(0.95):<25.2f}% | {df_res['drift_b2_pct'].quantile(0.95):<25.2f}%")
    print(f"{'Median FPE (m)':<36} | {df_res['fpe_a_m'].median():<25.2f}m | {df_res['fpe_b1_m'].median():<25.2f}m | {df_res['fpe_b2_m'].median():<25.2f}m")
    print(f"{'Mean FPE (m)':<36} | {df_res['fpe_a_m'].mean():<25.2f}m | {df_res['fpe_b1_m'].mean():<25.2f}m | {df_res['fpe_b2_m'].mean():<25.2f}m")
    print(f"{'Median Along-Track Error (m)':<36} | {df_res['along_a_m'].abs().median():<25.2f}m | {df_res['along_b1_m'].abs().median():<25.2f}m | {df_res['along_b2_m'].abs().median():<25.2f}m")
    print(f"{'Mean Along-Track Error (m)':<36} | {df_res['along_a_m'].abs().mean():<25.2f}m | {df_res['along_b1_m'].abs().mean():<25.2f}m | {df_res['along_b2_m'].abs().mean():<25.2f}m")
    print(f"{'Median Cross-Track Error (m)':<36} | {df_res['cross_a_m'].abs().median():<25.2f}m | {df_res['cross_b1_m'].abs().median():<25.2f}m | {df_res['cross_b2_m'].abs().median():<25.2f}m")
    print(f"{'Median Heading Error (deg)':<36} | {df_res['hdg_a_deg'].median():<25.2f}deg | {df_res['hdg_b1_deg'].median():<25.2f}deg | {df_res['hdg_b2_deg'].median():<25.2f}deg")
    print("-" * 125)
    improved_along_b1 = int(np.sum(df_res["along_b1_m"].abs() < df_res["along_a_m"].abs()))
    worsened_along_b1 = int(np.sum(df_res["along_b1_m"].abs() > df_res["along_a_m"].abs()))
    improved_fpe_b1 = int(np.sum(df_res["fpe_b1_m"] < df_res["fpe_a_m"]))
    worsened_fpe_b1 = int(np.sum(df_res["fpe_b1_m"] > df_res["fpe_a_m"]))
    target_a = int(np.sum(df_res["drift_a_pct"] < 10.0))
    target_b1 = int(np.sum(df_res["drift_b1_pct"] < 10.0))
    target_b2 = int(np.sum(df_res["drift_b2_pct"] < 10.0))
    
    med_pct_imp_along = float(np.median((df_res["along_a_m"].abs() - df_res["along_b1_m"].abs()) / np.maximum(df_res["along_a_m"].abs(), 1.0)) * 100.0)
    med_pct_imp_fpe = float(np.median((df_res["fpe_a_m"] - df_res["fpe_b1_m"]) / np.maximum(df_res["fpe_a_m"], 1.0)) * 100.0)

    print(f"{'Along-Track Improved (B1 vs A)':<36} | {'-':<25} | {improved_along_b1}/{len(df_res)} ({improved_along_b1/len(df_res)*100:.1f}%) | -")
    print(f"{'Along-Track Worsened (B1 vs A)':<36} | {'-':<25} | {worsened_along_b1}/{len(df_res)} ({worsened_along_b1/len(df_res)*100:.1f}%) | -")
    print(f"{'Median % Along-Track Improvement':<36} | {'-':<25} | {med_pct_imp_along:+.2f}% | -")
    print(f"{'FPE Improved (B1 vs A)':<36} | {'-':<25} | {improved_fpe_b1}/{len(df_res)} ({improved_fpe_b1/len(df_res)*100:.1f}%) | -")
    print(f"{'FPE Worsened (B1 vs A)':<36} | {'-':<25} | {worsened_fpe_b1}/{len(df_res)} ({worsened_fpe_b1/len(df_res)*100:.1f}%) | -")
    print(f"{'Median % FPE Improvement':<36} | {'-':<25} | {med_pct_imp_fpe:+.2f}% | -")
    print(f"{'Drift < 10% Target Count':<36} | {target_a}/{len(df_res)} ({target_a/len(df_res)*100:.1f}%) | {target_b1}/{len(df_res)} ({target_b1/len(df_res)*100:.1f}%) | {target_b2}/{len(df_res)} ({target_b2/len(df_res)*100:.1f}%)")
    print("=" * 125)

    # -------------------------------------------------------------
    # Breakdown by Outage Duration
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("BREAKDOWN BY OUTAGE DURATION (MEDIANS)")
    print("=" * 125)
    print(f"{'Dur (s)':<8} | {'N':<4} | {'Drift A %':<11} | {'Drift B1 %':<11} | {'FPE A (m)':<11} | {'FPE B1 (m)':<11} | {'Along A (m)':<13} | {'Along B1 (m)':<13} | {'Cross A (m)':<13} | {'Cross B1 (m)':<13}")
    print("-" * 125)
    for dur in sorted(df_res["duration_s"].unique()):
        sub = df_res[df_res["duration_s"] == dur]
        d_a = sub["drift_a_pct"].median()
        d_b1 = sub["drift_b1_pct"].median()
        f_a = sub["fpe_a_m"].median()
        f_b1 = sub["fpe_b1_m"].median()
        al_a = sub["along_a_m"].abs().median()
        al_b1 = sub["along_b1_m"].abs().median()
        cr_a = sub["cross_a_m"].abs().median()
        cr_b1 = sub["cross_b1_m"].abs().median()
        print(f"{dur:<8} | {len(sub):<4} | {d_a:<11.2f} | {d_b1:<11.2f} | {f_a:<11.2f} | {f_b1:<11.2f} | {al_a:<13.2f} | {al_b1:<13.2f} | {cr_a:<13.2f} | {cr_b1:<13.2f}")
    print("=" * 125)

    # -------------------------------------------------------------
    # 56 Scenario Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 135)
    print(f"{'Scenario':<22} | {'Dur':<4} | {'Control FPE':<12} | {'CAN FPE':<12} | {'Ctrl Along':<12} | {'CAN Along':<12} | {'Ctrl Drift':<12} | {'CAN Drift':<12} | {'Along Imp %':<12}")
    print("=" * 135)
    for _, r in df_res.iterrows():
        sc_name = f"{r['run']}_o{int(r['outage_id'])}"
        along_imp = ((abs(r['along_a_m']) - abs(r['along_b1_m'])) / max(abs(r['along_a_m']), 1.0)) * 100.0
        print(f"{sc_name:<22} | {int(r['duration_s']):<4} | {r['fpe_a_m']:<12.2f} | {r['fpe_b1_m']:<12.2f} | {abs(r['along_a_m']):<12.2f} | {abs(r['along_b1_m']):<12.2f} | {r['drift_a_pct']:<12.2f}% | {r['drift_b1_pct']:<12.2f}% | {along_imp:+11.1f}%")
    print("=" * 135)

    return df_res, detailed_profiles


if __name__ == "__main__":
    run_phase16a_benchmark()
