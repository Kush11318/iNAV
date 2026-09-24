"""
Phase 16B-2: Differential Wheel-Yaw UKF Ablation Benchmark

Compares:
  - System A (Control): 7-State UKF + Phase 11 Anchored Temporal VelocityNet + NHC + ZUPT
  - System B (CAN Speed Only): 7-State UKF + CAN Speed Replacement (10 Hz, R_can = 0.0325) + NHC + ZUPT
  - System C (CAN Speed + Differential Wheel Yaw): 7-State UKF + CAN Speed + Rear-Wheel Differential Yaw (10 Hz, R_yaw = 0.002213, B_eff = 1.4976m) + NHC + ZUPT

Evaluates across all 56 held-out test outages from the IO-VNBD synchronized benchmark.
Strict experiment isolation:
  - Zero production code modified.
  - Zero ground-truth leakage during simulated outages.
  - Pre-outage calibration strictly uses healthy pre-outage phone GPS and CAN signals.
  - B_eff = 1.4976 m strictly fixed (validated in Phase 16B-1).
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
logger = logging.getLogger("iNAV.Phase16B2_Ablation")

ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")
B_EFF_VALIDATED = 1.4976  # Phase 16B-1 validated rear track width
R_YAW_NOISE = 0.002213     # Phase 16B-1 straight-road noise variance (0.04704 rad/s)^2


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
    mode: str = "A",  # 'A': Control, 'B': CAN Speed Only, 'C': CAN Speed + Differential Wheel Yaw
    r_eff: float = 0.2776,
    b_eff: float = B_EFF_VALIDATED,
    r_can_var: float = 0.0325,
    r_yaw_var: float = R_YAW_NOISE,
    dt: float = config.TARGET_DT
) -> Dict[str, Any]:
    n = len(df_outage)
    if n == 0:
        return {}

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    omega_rl = df_outage[config.COL_TRUE_WHEEL_RL].values
    omega_rr = df_outage[config.COL_TRUE_WHEEL_RR].values

    v_rl = omega_rl * r_eff
    v_rr = omega_rr * r_eff
    v_rear = 0.5 * (v_rl + v_rr)
    z_wheel_yaw_arr = (v_rr - v_rl) / b_eff

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
            # CAN speed update
            v_val = float(v_rear[k])
            rear_diff = abs(omega_rl[k] - omega_rr[k])
            is_slipping = rear_diff > 25.0
            if v_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_val, r_var=r_can_var)

            # Differential rear-wheel yaw rate update
            if not is_slipping:
                z_yaw = float(z_wheel_yaw_arr[k])
                ukf.update_wheel_yaw(z_wheel_yaw=z_yaw, gyro_reading=w_v[2], r_yaw=r_yaw_var, nis_gate=16.0)

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


def run_phase16b2_ablation():
    logger.info("=" * 95)
    logger.info("PHASE 16B-2: DIFFERENTIAL WHEEL-YAW UKF ABLATION BENCHMARK")
    logger.info("A = Existing Control (Frozen Anchor VelocityNet)")
    logger.info("B = CAN Speed Only (Phase 16A)")
    logger.info("C = CAN Speed + Differential Wheel Yaw (Phase 16B-2)")
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

            # 3. System C: CAN Speed + Differential Wheel Yaw
            res_c = run_outage_simulation_ablation(
                df_sub, init_lat, init_lon, init_spd, init_hdg, lookup_a,
                outage_start_idx, run_align, mode="C", r_eff=r_eff_est, b_eff=B_EFF_VALIDATED
            )
            score_c = score_outage_segment(res_c["lat"], res_c["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            all_yaw_nis.extend(res_c["yaw_nis_hist"])

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

            rec = {
                "scenario": sc_id,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                # System A (Control)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                "mean_pos_a_m": round(float(np.mean(res_a["pos_err"])), 2),
                "max_pos_a_m": round(float(np.max(res_a["pos_err"])), 2),
                # System B (CAN Speed Only)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
                "mean_pos_b_m": round(float(np.mean(res_b["pos_err"])), 2),
                "max_pos_b_m": round(float(np.max(res_b["pos_err"])), 2),
                # System C (CAN Speed + Differential Wheel Yaw)
                "fpe_c_m": round(float(score_c["final_pos_error_m"]), 2),
                "drift_c_pct": round(float(score_c["pct_of_distance"]), 2),
                "along_c_m": round(float(score_c.get("along_track_m", np.nan)), 2),
                "cross_c_m": round(float(score_c.get("cross_track_m", np.nan)), 2),
                "hdg_c_deg": round(float(score_c.get("heading_error_deg", np.nan)), 2),
                "mean_pos_c_m": round(float(np.mean(res_c["pos_err"])), 2),
                "max_pos_c_m": round(float(np.max(res_c["pos_err"])), 2),
                # Wheel Yaw Diagnostics
                "yaw_accepted": res_c["yaw_accepted"],
                "yaw_rejected": res_c["yaw_rejected"],
                "yaw_acc_rate_pct": round(yaw_acc_rate, 2),
                "mean_yaw_nis": round(mean_nis, 3),
                "bg_init_rads": round(float(res_c["bg"][0]), 5),
                "bg_final_rads": round(float(res_c["bg"][-1]), 5),
                "delta_bg_rads": round(float(res_c["bg"][-1] - res_c["bg"][0]), 5),
            }
            records.append(rec)

    df_res = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase16b2_ablation_results.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 16B-2 benchmark results to {out_csv} ({len(df_res)} outages)")

    # -------------------------------------------------------------
    # Summary Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 135)
    print(f"{'METRIC':<34} | {'SYSTEM A (Control)':<25} | {'SYSTEM B (CAN Speed)':<25} | {'SYSTEM C (CAN + Wheel Yaw)':<25}")
    print("=" * 135)
    print(f"{'Median Drift %':<34} | {df_res['drift_a_pct'].median():<25.2f}% | {df_res['drift_b_pct'].median():<25.2f}% | {df_res['drift_c_pct'].median():<25.2f}%")
    print(f"{'Mean Drift %':<34} | {df_res['drift_a_pct'].mean():<25.2f}% | {df_res['drift_b_pct'].mean():<25.2f}% | {df_res['drift_c_pct'].mean():<25.2f}%")
    print(f"{'P95 Drift %':<34} | {df_res['drift_a_pct'].quantile(0.95):<25.2f}% | {df_res['drift_b_pct'].quantile(0.95):<25.2f}% | {df_res['drift_c_pct'].quantile(0.95):<25.2f}%")
    print(f"{'Median FPE (m)':<34} | {df_res['fpe_a_m'].median():<25.2f}m | {df_res['fpe_b_m'].median():<25.2f}m | {df_res['fpe_c_m'].median():<25.2f}m")
    print(f"{'Mean FPE (m)':<34} | {df_res['fpe_a_m'].mean():<25.2f}m | {df_res['fpe_b_m'].mean():<25.2f}m | {df_res['fpe_c_m'].mean():<25.2f}m")
    print(f"{'Median Along-Track (m)':<34} | {df_res['along_a_m'].abs().median():<25.2f}m | {df_res['along_b_m'].abs().median():<25.2f}m | {df_res['along_c_m'].abs().median():<25.2f}m")
    print(f"{'Mean Along-Track (m)':<34} | {df_res['along_a_m'].abs().mean():<25.2f}m | {df_res['along_b_m'].abs().mean():<25.2f}m | {df_res['along_c_m'].abs().mean():<25.2f}m")
    print(f"{'Median Cross-Track (m)':<34} | {df_res['cross_a_m'].abs().median():<25.2f}m | {df_res['cross_b_m'].abs().median():<25.2f}m | {df_res['cross_c_m'].abs().median():<25.2f}m")
    print(f"{'Mean Cross-Track (m)':<34} | {df_res['cross_a_m'].abs().mean():<25.2f}m | {df_res['cross_b_m'].abs().mean():<25.2f}m | {df_res['cross_c_m'].abs().mean():<25.2f}m")
    print(f"{'Median Heading Error (deg)':<34} | {df_res['hdg_a_deg'].median():<25.2f}deg | {df_res['hdg_b_deg'].median():<25.2f}deg | {df_res['hdg_c_deg'].median():<25.2f}deg")
    print(f"{'Mean Heading Error (deg)':<34} | {df_res['hdg_a_deg'].mean():<25.2f}deg | {df_res['hdg_b_deg'].mean():<25.2f}deg | {df_res['hdg_c_deg'].mean():<25.2f}deg")
    print("-" * 135)
    fpe_imp_c_vs_a = int(np.sum(df_res["fpe_c_m"] < df_res["fpe_a_m"]))
    fpe_wors_c_vs_a = int(np.sum(df_res["fpe_c_m"] > df_res["fpe_a_m"]))
    fpe_imp_c_vs_b = int(np.sum(df_res["fpe_c_m"] < df_res["fpe_b_m"]))
    fpe_wors_c_vs_b = int(np.sum(df_res["fpe_c_m"] > df_res["fpe_b_m"]))
    hdg_imp_c_vs_a = int(np.sum(df_res["hdg_c_deg"] < df_res["hdg_a_deg"]))
    along_imp_c_vs_a = int(np.sum(df_res["along_c_m"].abs() < df_res["along_a_m"].abs()))
    cross_imp_c_vs_a = int(np.sum(df_res["cross_c_m"].abs() < df_res["cross_a_m"].abs()))

    print(f"{'FPE Improved (C vs A)':<34} | {'-':<25} | {'-':<25} | {fpe_imp_c_vs_a}/{len(df_res)} ({fpe_imp_c_vs_a/len(df_res)*100:.1f}%)")
    print(f"{'FPE Worsened (C vs A)':<34} | {'-':<25} | {'-':<25} | {fpe_wors_c_vs_a}/{len(df_res)} ({fpe_wors_c_vs_a/len(df_res)*100:.1f}%)")
    print(f"{'FPE Improved (C vs B)':<34} | {'-':<25} | {'-':<25} | {fpe_imp_c_vs_b}/{len(df_res)} ({fpe_imp_c_vs_b/len(df_res)*100:.1f}%)")
    print(f"{'FPE Worsened (C vs B)':<34} | {'-':<25} | {'-':<25} | {fpe_wors_c_vs_b}/{len(df_res)} ({fpe_wors_c_vs_b/len(df_res)*100:.1f}%)")
    print(f"{'Heading Improved (C vs A)':<34} | {'-':<25} | {'-':<25} | {hdg_imp_c_vs_a}/{len(df_res)} ({hdg_imp_c_vs_a/len(df_res)*100:.1f}%)")
    print(f"{'Along-Track Improved (C vs A)':<34} | {'-':<25} | {'-':<25} | {along_imp_c_vs_a}/{len(df_res)} ({along_imp_c_vs_a/len(df_res)*100:.1f}%)")
    print(f"{'Cross-Track Improved (C vs A)':<34} | {'-':<25} | {'-':<25} | {cross_imp_c_vs_a}/{len(df_res)} ({cross_imp_c_vs_a/len(df_res)*100:.1f}%)")
    print(f"{'Drift < 10% Target Count':<34} | {(df_res['drift_a_pct'] < 10.0).sum()}/56 ({(df_res['drift_a_pct'] < 10.0).mean()*100:.1f}%) | {(df_res['drift_b_pct'] < 10.0).sum()}/56 ({(df_res['drift_b_pct'] < 10.0).mean()*100:.1f}%) | {(df_res['drift_c_pct'] < 10.0).sum()}/56 ({(df_res['drift_c_pct'] < 10.0).mean()*100:.1f}%)")
    print("=" * 135)

    # -------------------------------------------------------------
    # Duration Breakdown
    # -------------------------------------------------------------
    print("\n" + "=" * 135)
    print("DURATION BREAKDOWN (MEDIANS: SYSTEM A vs B vs C)")
    print("=" * 135)
    print(f"{'Dur':<4} | {'N':<3} | {'Drift A%':<9} | {'Drift B%':<9} | {'Drift C%':<9} | {'FPE A(m)':<9} | {'FPE B(m)':<9} | {'FPE C(m)':<9} | {'Hdg A°':<7} | {'Hdg B°':<7} | {'Hdg C°':<7}")
    print("-" * 135)
    for dur in sorted(df_res["duration_s"].unique()):
        sub = df_res[df_res["duration_s"] == dur]
        da = sub["drift_a_pct"].median()
        db = sub["drift_b_pct"].median()
        dc = sub["drift_c_pct"].median()
        fa = sub["fpe_a_m"].median()
        fb = sub["fpe_b_m"].median()
        fc = sub["fpe_c_m"].median()
        ha = sub["hdg_a_deg"].median()
        hb = sub["hdg_b_deg"].median()
        hc = sub["hdg_c_deg"].median()
        print(f"{dur:<4} | {len(sub):<3} | {da:<9.2f} | {db:<9.2f} | {dc:<9.2f} | {fa:<9.2f} | {fb:<9.2f} | {fc:<9.2f} | {ha:<7.1f} | {hb:<7.1f} | {hc:<7.1f}")
    print("=" * 135)

    # -------------------------------------------------------------
    # 19 Failure Case Analysis
    # -------------------------------------------------------------
    # Phase 16A failures were cases where fpe_b > fpe_a
    p16a_failures = df_res[df_res["fpe_b_m"] > df_res["fpe_a_m"]].copy()
    print("\n" + "=" * 135)
    print(f"RECOVERY OF THE 19 PHASE 16A FAILURES (WHERE CAN SPEED WORSENED FPE) — TOTAL {len(p16a_failures)} CASES")
    print("=" * 135)
    print(f"{'Scenario':<18} | {'Dur':<4} | {'Ctrl FPE (A)':<13} | {'CAN FPE (B)':<13} | {'CAN+Yaw FPE(C)':<15} | {'Hdg A°':<8} | {'Hdg C°':<8} | {'Recovery Status':<16}")
    print("-" * 135)
    recovered_count = 0
    for _, r in p16a_failures.iterrows():
        is_recovered = r["fpe_c_m"] <= r["fpe_a_m"]
        if is_recovered:
            recovered_count += 1
            status = "RECOVERED"
        else:
            status = "STILL FAILED" if r["fpe_c_m"] > r["fpe_b_m"] else "PARTIAL RECOV"
        print(f"{r['scenario']:<18} | {int(r['duration_s']):<4} | {r['fpe_a_m']:<13.2f} | {r['fpe_b_m']:<13.2f} | {r['fpe_c_m']:<15.2f} | {r['hdg_a_deg']:<8.1f} | {r['hdg_c_deg']:<8.1f} | {status:<16}")
    print("=" * 135)
    print(f"19 Phase 16A failures -> RECOVERED BY DIFFERENTIAL YAW: {recovered_count} / {len(p16a_failures)} ({recovered_count/len(p16a_failures)*100:.1f}%)")
    print(f"19 Phase 16A failures -> STILL FAILED: {len(p16a_failures) - recovered_count} / {len(p16a_failures)}")
    print("=" * 135)

    # -------------------------------------------------------------
    # Generating 11 Required Visualizations
    # -------------------------------------------------------------
    logger.info("Generating all 11 required figures...")

    durs = sorted(df_res["duration_s"].unique())
    x = np.arange(len(durs))
    width = 0.25

    # Fig 1: A/B/C Aggregate Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ["Median Drift %", "Median Along (m)", "Median Cross (m)", "Median Heading (°)"]
    vals_a = [df_res["drift_a_pct"].median(), df_res["along_a_m"].abs().median(), df_res["cross_a_m"].abs().median(), df_res["hdg_a_deg"].median()]
    vals_b = [df_res["drift_b_pct"].median(), df_res["along_b_m"].abs().median(), df_res["cross_b_m"].abs().median(), df_res["hdg_b_deg"].median()]
    vals_c = [df_res["drift_c_pct"].median(), df_res["along_c_m"].abs().median(), df_res["cross_c_m"].abs().median(), df_res["hdg_c_deg"].median()]

    xm = np.arange(len(metrics_names))
    ax.bar(xm - width, vals_a, width, label="System A (Control)", color="#e74c3c")
    ax.bar(xm, vals_b, width, label="System B (CAN Speed Only)", color="#e67e22")
    ax.bar(xm + width, vals_c, width, label="System C (CAN + Wheel Yaw)", color="#2ecc71")
    ax.set_xticks(xm)
    ax.set_xticklabels(metrics_names, fontweight="bold")
    ax.set_title("Figure 1: Aggregate System Comparison (A vs B vs C)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig1_aggregate_comparison.png", dpi=180)
    plt.close()

    # Fig 2: Heading Error by Duration
    fig, ax = plt.subplots(figsize=(10, 5))
    h_a = [df_res[df_res["duration_s"] == d]["hdg_a_deg"].median() for d in durs]
    h_b = [df_res[df_res["duration_s"] == d]["hdg_b_deg"].median() for d in durs]
    h_c = [df_res[df_res["duration_s"] == d]["hdg_c_deg"].median() for d in durs]
    ax.bar(x - width, h_a, width, label="System A (Control)", color="#e74c3c")
    ax.bar(x, h_b, width, label="System B (CAN Speed)", color="#e67e22")
    ax.bar(x + width, h_c, width, label="System C (CAN + Wheel Yaw)", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durs], fontweight="bold")
    ax.set_ylabel("Median Heading Error (°)", fontweight="bold")
    ax.set_title("Figure 2: Heading Error by Outage Duration (A vs B vs C)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig2_heading_by_duration.png", dpi=180)
    plt.close()

    # Fig 3: FPE by Duration
    fig, ax = plt.subplots(figsize=(10, 5))
    f_a = [df_res[df_res["duration_s"] == d]["fpe_a_m"].median() for d in durs]
    f_b = [df_res[df_res["duration_s"] == d]["fpe_b_m"].median() for d in durs]
    f_c = [df_res[df_res["duration_s"] == d]["fpe_c_m"].median() for d in durs]
    ax.bar(x - width, f_a, width, label="System A (Control)", color="#e74c3c")
    ax.bar(x, f_b, width, label="System B (CAN Speed)", color="#e67e22")
    ax.bar(x + width, f_c, width, label="System C (CAN + Wheel Yaw)", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durs], fontweight="bold")
    ax.set_ylabel("Median FPE (m)", fontweight="bold")
    ax.set_title("Figure 3: Final Position Error (FPE) by Outage Duration", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig3_fpe_by_duration.png", dpi=180)
    plt.close()

    # Fig 4: Along-Track Error by Duration
    fig, ax = plt.subplots(figsize=(10, 5))
    al_a = [df_res[df_res["duration_s"] == d]["along_a_m"].abs().median() for d in durs]
    al_b = [df_res[df_res["duration_s"] == d]["along_b_m"].abs().median() for d in durs]
    al_c = [df_res[df_res["duration_s"] == d]["along_c_m"].abs().median() for d in durs]
    ax.bar(x - width, al_a, width, label="System A", color="#e74c3c")
    ax.bar(x, al_b, width, label="System B", color="#e67e22")
    ax.bar(x + width, al_c, width, label="System C", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durs], fontweight="bold")
    ax.set_ylabel("Median Along-Track Error (m)", fontweight="bold")
    ax.set_title("Figure 4: Along-Track Error by Outage Duration", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig4_along_track_by_duration.png", dpi=180)
    plt.close()

    # Fig 5: Cross-Track Error by Duration
    fig, ax = plt.subplots(figsize=(10, 5))
    cr_a = [df_res[df_res["duration_s"] == d]["cross_a_m"].abs().median() for d in durs]
    cr_b = [df_res[df_res["duration_s"] == d]["cross_b_m"].abs().median() for d in durs]
    cr_c = [df_res[df_res["duration_s"] == d]["cross_c_m"].abs().median() for d in durs]
    ax.bar(x - width, cr_a, width, label="System A", color="#e74c3c")
    ax.bar(x, cr_b, width, label="System B", color="#e67e22")
    ax.bar(x + width, cr_c, width, label="System C", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durs], fontweight="bold")
    ax.set_ylabel("Median Cross-Track Error (m)", fontweight="bold")
    ax.set_title("Figure 5: Cross-Track Error by Outage Duration", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig5_cross_track_by_duration.png", dpi=180)
    plt.close()

    # Fig 6: The 19 Phase 16A-worsened cases comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x19 = np.arange(len(p16a_failures))
    ax.bar(x19 - width, p16a_failures["fpe_a_m"], width, label="System A (Control)", color="#e74c3c")
    ax.bar(x19, p16a_failures["fpe_b_m"], width, label="System B (CAN Speed Worsened)", color="#e67e22")
    ax.bar(x19 + width, p16a_failures["fpe_c_m"], width, label="System C (CAN + Wheel Yaw)", color="#2ecc71")
    ax.set_xticks(x19)
    ax.set_xticklabels([f"{r['run']}_o{int(r['outage_id'])}" for _, r in p16a_failures.iterrows()], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("FPE (m)", fontweight="bold")
    ax.set_title("Figure 6: Recovery of the 19 Phase 16A Failure Scenarios", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig6_19_failures_recovery.png", dpi=180)
    plt.close()

    # Representative cases for Figures 7, 8, 9
    rep_cases = ["vw11_o2", "vw14c_o2", "vw3_o4"]
    fig7, axs7 = plt.subplots(1, 3, figsize=(18, 6))
    fig8, axs8 = plt.subplots(1, 3, figsize=(18, 5))
    fig9, axs9 = plt.subplots(1, 3, figsize=(18, 5))

    for idx, sc_id in enumerate(rep_cases):
        tdata = traces_store[sc_id]
        t = tdata["t"]

        # Trajectory (Fig 7)
        lat0, lon0 = tdata["gt_lat"][0], tdata["gt_lon"][0]
        R_e = 6371000.0
        gt_E = (tdata["gt_lon"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        gt_N = (tdata["gt_lat"] - lat0) * np.radians(1.0) * R_e
        a_E = (tdata["lon_a"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        a_N = (tdata["lat_a"] - lat0) * np.radians(1.0) * R_e
        b_E = (tdata["lon_b"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        b_N = (tdata["lat_b"] - lat0) * np.radians(1.0) * R_e
        c_E = (tdata["lon_c"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        c_N = (tdata["lat_c"] - lat0) * np.radians(1.0) * R_e

        axs7[idx].plot(gt_E, gt_N, "k-", linewidth=2.5, label="Ground Truth")
        axs7[idx].plot(a_E, a_N, "r--", linewidth=1.8, label="A (Control)")
        axs7[idx].plot(b_E, b_N, "m-.", linewidth=1.8, label="B (CAN Speed)")
        axs7[idx].plot(c_E, c_N, "g-", linewidth=2.2, label="C (CAN+Yaw)")
        axs7[idx].scatter([0], [0], color="black", s=60, marker="o")
        axs7[idx].set_title(f"Trajectory: {sc_id} ({tdata['dur']}s)", fontweight="bold")
        axs7[idx].set_xlabel("East (m)", fontweight="bold")
        axs7[idx].set_ylabel("North (m)", fontweight="bold")
        axs7[idx].grid(True, linestyle=":", alpha=0.6)
        axs7[idx].axis("equal")
        if idx == 0:
            axs7[idx].legend()

        # Heading (Fig 8)
        if tdata["gt_hdg"] is not None:
            axs8[idx].plot(t, tdata["gt_hdg"], "k-", linewidth=2.5, label="Ground Truth")
        axs8[idx].plot(t, tdata["hdg_a"], "r--", linewidth=1.8, label="A (Control)")
        axs8[idx].plot(t, tdata["hdg_b"], "m-.", linewidth=1.8, label="B (CAN Speed)")
        axs8[idx].plot(t, tdata["hdg_c"], "g-", linewidth=2.2, label="C (CAN+Yaw)")
        axs8[idx].set_title(f"Heading: {sc_id} ({tdata['dur']}s)", fontweight="bold")
        axs8[idx].set_xlabel("Time (s)", fontweight="bold")
        axs8[idx].set_ylabel("Heading (°)", fontweight="bold")
        axs8[idx].grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            axs8[idx].legend()

        # Gyro Bias Trace (Fig 9)
        axs9[idx].plot(t, np.degrees(tdata["bg_c"]), "g-", linewidth=2.2, label="Estimated b_g (System C)")
        axs9[idx].plot(t, np.degrees(tdata["bg_a"]), "r--", linewidth=1.8, label="Estimated b_g (System A)")
        axs9[idx].set_title(f"Gyro Bias: {sc_id} ({tdata['dur']}s)", fontweight="bold")
        axs9[idx].set_xlabel("Time (s)", fontweight="bold")
        axs9[idx].set_ylabel("Gyro Bias (°/s)", fontweight="bold")
        axs9[idx].grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            axs9[idx].legend()

    fig7.suptitle("Figure 7: Representative 2D Trajectories (GT vs A vs B vs C)", fontsize=14, fontweight="bold")
    fig7.tight_layout()
    fig7.savefig(ARTIFACTS_DIR / "phase16b2_fig7_representative_trajectories.png", dpi=180)
    plt.close(fig7)

    fig8.suptitle("Figure 8: Representative Heading Tracking (GT vs A vs B vs C)", fontsize=14, fontweight="bold")
    fig8.tight_layout()
    fig8.savefig(ARTIFACTS_DIR / "phase16b2_fig8_representative_headings.png", dpi=180)
    plt.close(fig8)

    fig9.suptitle("Figure 9: Representative Gyro Bias Adaptation (System C)", fontsize=14, fontweight="bold")
    fig9.tight_layout()
    fig9.savefig(ARTIFACTS_DIR / "phase16b2_fig9_representative_gyro_bias.png", dpi=180)
    plt.close(fig9)

    # Fig 10: Wheel-Yaw NIS Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_yaw_nis, bins=50, range=(0, 20), color="teal", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Theoretical Mean NIS = 1.0 (1-DOF)")
    ax.axvline(16.0, color="orange", linestyle=":", linewidth=2, label="Chi-Square Gate = 16.0 (4σ)")
    ax.set_xlabel("Normalized Innovation Squared (NIS)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title(f"Figure 10: Wheel-Yaw NIS Distribution (Mean NIS = {np.mean(all_yaw_nis):.2f})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig10_wheel_yaw_nis_dist.png", dpi=180)
    plt.close()

    # Fig 11: Yaw Update Acceptance/Rejection Statistics
    fig, ax = plt.subplots(figsize=(8, 5))
    total_acc = int(df_res["yaw_accepted"].sum())
    total_rej = int(df_res["yaw_rejected"].sum())
    bars = ax.bar(["Accepted Updates", "Rejected Updates"], [total_acc, total_rej], color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set_ylabel("Total Number of Updates", fontweight="bold")
    ax.set_title(f"Figure 11: Wheel-Yaw Acceptance Statistics (Acceptance Rate: {total_acc/(total_acc+total_rej)*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 300, f"{b.get_height():,}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase16b2_fig11_acceptance_statistics.png", dpi=180)
    plt.close()

    return df_res, p16a_failures


if __name__ == "__main__":
    run_phase16b2_ablation()
