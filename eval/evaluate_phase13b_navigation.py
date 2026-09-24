"""
Phase 13B: Event-Gated Neural Velocity Correction Navigation Benchmark
Compares:
  - Baseline A: Phase 11 Anchored Temporal Velocity Net with FROZEN v_anchor
  - Experiment B: Phase 13B Event-Gated Temporary Neural Velocity Correction Tracker

Across all 56 held-out test outages from the IO-VNBD synchronized benchmark.
Strict experiment isolation:
  - Fixed v_anchor is NEVER updated recursively.
  - 7-state UKF parameters, Q/R, alignment, and test outages are strictly identical.
  - Zero ground-truth leakage into runtime estimation.
  - Decouples Along-Track error from Cross-Track error to evaluate the longitudinal drift hypothesis.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
from modules.phase13b_event_gated_velocity import Phase13BEventGatedVelocityTracker
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from modules.alignment import AlignmentEngine, AlignmentState
from modules.ukf import UKFNavigationFilter
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase13B_Nav")


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
        lookup[gk] = (float(p_v_mean[idx]), int(p_ev[idx]), float(p_sig_mean[idx]), float(p_v_end[idx]))

    return lookup


def run_experiment_b_event_gated(
    model: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_fixed: float,
    alignment: AlignmentEngine,
    window_len: int = 40,
    device: str = "cpu"
) -> Tuple[Dict[int, Tuple[float, int, float, float]], List[Dict[str, Any]]]:
    """
    Experiment B: Event-Gated Temporary Neural Velocity Correction Tracker.
    v_anchor remains constant throughout the outage.
    """
    lookup = {}
    event_timeline = []
    model.eval()

    tracker = Phase13BEventGatedVelocityTracker(tau_decay_s=2.5, tau_attack_s=1.0)
    tracker.initialize(v_anchor_fixed, initial_sigma=0.2)

    for k in range(0, sub_len):
        global_k = outage_start_idx + k
        if global_k >= window_len - 1 and (k % 5 == 0):
            win = seq_data[global_k - window_len + 1 : global_k + 1]  # (40, 6)

            # Transform IMU window to vehicle frame to compute aligned forward acceleration & yaw rate
            acc_raw = win[:, :3]
            gyro_raw = win[:, 3:]
            a_v_list = []
            w_v_list = []
            for i in range(len(win)):
                av, wv = alignment.transform_imu(acc_raw[i], gyro_raw[i])
                a_v_list.append(av)
                w_v_list.append(wv)
            a_v_arr = np.array(a_v_list)
            w_v_arr = np.array(w_v_list)

            a_fwd_mean = float(np.mean(a_v_arr[-10:, 0]))  # recent 1.0s forward acceleration
            yaw_rate_mean = float(np.mean(w_v_arr[-10:, 2]))  # recent 1.0s yaw rate

            v_meas, sig_meas, ev_class, gate_open, gate_type, corr_val = tracker.update_step(
                model=model,
                imu_window_40x6=win,
                a_fwd_mean=a_fwd_mean,
                yaw_rate_mean=yaw_rate_mean,
                dt_elapsed=0.5,
                device=device
            )

            lookup[global_k] = (float(v_meas), int(ev_class), float(sig_meas), float(v_meas))
            event_timeline.append({
                "time_s": k * config.TARGET_DT,
                "v_meas": v_meas,
                "corr_val": corr_val,
                "gate_open": gate_open,
                "gate_type": gate_type,
                "sigma": sig_meas
            })

    return lookup, event_timeline


def run_single_outage_ukf(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float]]:
    """
    Executes the 7-state UKF forward navigation across the outage window.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), [], []

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
    speed_preds_profile = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # Filter prediction
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        global_k = global_start_idx + k
        if global_k in lookup_preds and (k % 5 == 0):
            v_meas, ev_class, sig_meas, _ = lookup_preds[global_k]
            speed_preds_profile.append(v_meas)

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

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed, speed_preds_profile, list(est_speed)


def run_phase13b_navigation_benchmark():
    logger.info("=" * 80)
    logger.info("PHASE 13B: EVENT-GATED NEURAL VELOCITY CORRECTION BENCHMARK")
    logger.info("Baseline A: Phase 11 Anchored Temporal Velocity Net (Frozen v_anchor)")
    logger.info("Experiment B: Phase 13B Event-Gated Temporary Neural Velocity Correction")
    logger.info("=" * 80)

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
    
    # Store detailed data for diagnostic plots
    detailed_profiles = {}
    gate_timelines = {}

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

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None
            gt_speed = df_sub[config.COL_TRUE_SPEED_MS].values

            # 1. Baseline A: Frozen anchor predictions
            lookup_a = compute_baseline_a_predictions(
                model=model,
                seq_data=seq_data,
                outage_start_idx=outage_start_idx,
                sub_len=len(df_sub),
                v_anchor_frozen=v_anchor_val,
                window_len=40,
                device="cpu"
            )

            # 2. Experiment B: Event-Gated temporary neural velocity correction
            lookup_b, timeline_b = run_experiment_b_event_gated(
                model=model,
                seq_data=seq_data,
                outage_start_idx=outage_start_idx,
                sub_len=len(df_sub),
                v_anchor_fixed=v_anchor_val,
                alignment=run_align,
                window_len=40,
                device="cpu"
            )

            # 3. Filter runs
            lat_a, lon_a, spd_a, preds_a, ukf_spd_a = run_single_outage_ukf(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_a, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            lat_b, lon_b, spd_b, preds_b, ukf_spd_b = run_single_outage_ukf(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_b, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_b = score_outage_segment(lat_b, lon_b, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            rec_id = f"{run_name}_outage_{oid}"
            if rec_id == "vw11_outage_3" or len(detailed_profiles) < 3:
                detailed_profiles[rec_id] = {
                    "t": np.arange(len(df_sub)) * config.TARGET_DT,
                    "gt_speed": gt_speed,
                    "ukf_spd_a": ukf_spd_a,
                    "ukf_spd_b": ukf_spd_b,
                    "v_anchor": v_anchor_val,
                    "dist_gt": dist_gt,
                }
                gate_timelines[rec_id] = timeline_b

            # Categorize outage regime for critical analysis
            spd_start = v_anchor_val
            spd_end = float(gt_speed[-1])
            spd_min = float(np.min(gt_speed))
            spd_max = float(np.max(gt_speed))
            spd_std = float(np.std(gt_speed))

            if spd_min < 0.15:
                regime = "Stationary Transition"
            elif (spd_start - spd_end) > 6.0:
                regime = "Strong Deceleration"
            elif (spd_end - spd_start) > 6.0:
                regime = "Strong Acceleration"
            elif (spd_max - spd_min) > 8.0:
                regime = "Cruise -> Braking/Accel"
            elif spd_std < 1.5:
                regime = "Stable Cruise"
            else:
                regime = "Mixed Dynamic"

            rec = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                "v_anchor_ms": round(v_anchor_val, 2),
                "true_mean_spd_ms": round(float(np.mean(gt_speed)), 2),
                "true_final_spd_ms": round(float(gt_speed[-1]), 2),
                "regime": regime,
                # Baseline A (Phase 11 Frozen Anchor)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                # Experiment B (Phase 13B Event-Gated)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
            }
            records.append(rec)

    df_res = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase13b_nav_results.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 13B benchmark results to {out_csv} ({len(df_res)} outages)")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 110)
    print(f"{'METRIC':<36} | {'BASELINE A (Phase 11 Frozen)':<32} | {'EXPERIMENT B (Phase 13B Event-Gated)':<32}")
    print("=" * 110)
    print(f"{'Median Drift %':<36} | {df_res['drift_a_pct'].median():<32.2f}% | {df_res['drift_b_pct'].median():<32.2f}%")
    print(f"{'Mean Drift %':<36} | {df_res['drift_a_pct'].mean():<32.2f}% | {df_res['drift_b_pct'].mean():<32.2f}%")
    print(f"{'P95 Drift %':<36} | {df_res['drift_a_pct'].quantile(0.95):<32.2f}% | {df_res['drift_b_pct'].quantile(0.95):<32.2f}%")
    print(f"{'Median FPE (m)':<36} | {df_res['fpe_a_m'].median():<32.2f}m | {df_res['fpe_b_m'].median():<32.2f}m")
    print(f"{'Mean FPE (m)':<36} | {df_res['fpe_a_m'].mean():<32.2f}m | {df_res['fpe_b_m'].mean():<32.2f}m")
    print(f"{'Median Along-Track Error (m)':<36} | {df_res['along_a_m'].abs().median():<32.2f}m | {df_res['along_b_m'].abs().median():<32.2f}m")
    print(f"{'Mean Along-Track Error (m)':<36} | {df_res['along_a_m'].abs().mean():<32.2f}m | {df_res['along_b_m'].abs().mean():<32.2f}m")
    print(f"{'Median Cross-Track Error (m)':<36} | {df_res['cross_a_m'].abs().median():<32.2f}m | {df_res['cross_b_m'].abs().median():<32.2f}m")
    print(f"{'Median Heading Error (deg)':<36} | {df_res['hdg_a_deg'].median():<32.2f}deg | {df_res['hdg_b_deg'].median():<32.2f}deg")
    print("-" * 110)
    improved_drift = int(np.sum(df_res["drift_b_pct"] < df_res["drift_a_pct"]))
    worsened_drift = int(np.sum(df_res["drift_b_pct"] > df_res["drift_a_pct"]))
    improved_along = int(np.sum(df_res["along_b_m"].abs() < df_res["along_a_m"].abs()))
    worsened_along = int(np.sum(df_res["along_b_m"].abs() > df_res["along_a_m"].abs()))
    target_a = int(np.sum(df_res["drift_a_pct"] < 10.0))
    target_b = int(np.sum(df_res["drift_b_pct"] < 10.0))
    print(f"{'Scenarios Drift Improved':<36} | {'-':<32} | {improved_drift}/{len(df_res)} ({improved_drift/len(df_res)*100:.1f}%)")
    print(f"{'Scenarios Drift Worsened':<36} | {'-':<32} | {worsened_drift}/{len(df_res)} ({worsened_drift/len(df_res)*100:.1f}%)")
    print(f"{'Along-Track Improved':<36} | {'-':<32} | {improved_along}/{len(df_res)} ({improved_along/len(df_res)*100:.1f}%)")
    print(f"{'Along-Track Worsened':<36} | {'-':<32} | {worsened_along}/{len(df_res)} ({worsened_along/len(df_res)*100:.1f}%)")
    print(f"{'Drift < 10% Target Count':<36} | {target_a}/{len(df_res)} ({target_a/len(df_res)*100:.1f}%) | {target_b}/{len(df_res)} ({target_b/len(df_res)*100:.1f}%)")
    print("=" * 110)

    # -------------------------------------------------------------
    # Breakdown by Outage Duration
    # -------------------------------------------------------------
    print("\n" + "=" * 120)
    print("BREAKDOWN BY OUTAGE DURATION (MEDIANS)")
    print("=" * 120)
    print(f"{'Dur (s)':<8} | {'N':<4} | {'Drift A %':<11} | {'Drift B %':<11} | {'FPE A (m)':<11} | {'FPE B (m)':<11} | {'Along A (m)':<13} | {'Along B (m)':<13} | {'Cross A (m)':<13} | {'Cross B (m)':<13}")
    print("-" * 120)
    for dur in sorted(df_res["duration_s"].unique()):
        sub = df_res[df_res["duration_s"] == dur]
        d_a = sub["drift_a_pct"].median()
        d_b = sub["drift_b_pct"].median()
        f_a = sub["fpe_a_m"].median()
        f_b = sub["fpe_b_m"].median()
        al_a = sub["along_a_m"].abs().median()
        al_b = sub["along_b_m"].abs().median()
        cr_a = sub["cross_a_m"].abs().median()
        cr_b = sub["cross_b_m"].abs().median()
        print(f"{dur:<8} | {len(sub):<4} | {d_a:<11.2f} | {d_b:<11.2f} | {f_a:<11.2f} | {f_b:<11.2f} | {al_a:<13.2f} | {al_b:<13.2f} | {cr_a:<13.2f} | {cr_b:<13.2f}")
    print("=" * 120)

    # -------------------------------------------------------------
    # Critical Analysis: Breakdown by Driving Regime
    # -------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CRITICAL ANALYSIS: BREAKDOWN BY DRIVING REGIME (MEDIANS)")
    print("=" * 120)
    print(f"{'Regime':<26} | {'N':<4} | {'Drift A %':<11} | {'Drift B %':<11} | {'Along A (m)':<13} | {'Along B (m)':<13} | {'Along Imp %':<12}")
    print("-" * 120)
    for reg, grp in df_res.groupby("regime"):
        d_a = grp["drift_a_pct"].median()
        d_b = grp["drift_b_pct"].median()
        al_a = grp["along_a_m"].abs().median()
        al_b = grp["along_b_m"].abs().median()
        n_imp = int(np.sum(grp["along_b_m"].abs() < grp["along_a_m"].abs()))
        pct_imp = (n_imp / len(grp)) * 100.0
        print(f"{reg:<26} | {len(grp):<4} | {d_a:<11.2f} | {d_b:<11.2f} | {al_a:<13.2f} | {al_b:<13.2f} | {pct_imp:<12.1f}%")
    print("=" * 120)

    # -------------------------------------------------------------
    # Generate the 5 Required Diagnostic Plots
    # -------------------------------------------------------------
    plots_dir = config.BASE_DIR / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Select representative deceleration outage (vw11 Outage 3)
    p_key = "vw11_outage_3" if "vw11_outage_3" in detailed_profiles else list(detailed_profiles.keys())[0]
    p_data = detailed_profiles[p_key]
    tl_data = gate_timelines[p_key]

    # Plot 1: Speed Trace (GT vs Phase 11 vs Phase 13B)
    plt.figure(figsize=(10, 5))
    plt.plot(p_data["t"], p_data["gt_speed"], 'k-', label="Ground Truth Speed", linewidth=2.2)
    plt.plot(p_data["t"], p_data["ukf_spd_a"], 'r--', label="Baseline A (Phase 11 Frozen Anchor)", linewidth=1.8)
    plt.plot(p_data["t"], p_data["ukf_spd_b"], 'b-', label="Experiment B (Phase 13B Event-Gated)", linewidth=2.0)
    plt.axhline(p_data["v_anchor"], color='gray', linestyle=':', label=f"Fixed Anchor ({p_data['v_anchor']:.1f} m/s)")
    plt.title(f"Plot 1: Speed Tracking on Deceleration Outage ({p_key})", fontsize=12, fontweight="bold")
    plt.xlabel("Outage Time (s)", fontsize=11)
    plt.ylabel("Forward Speed (m/s)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    p1_path = plots_dir / "phase13b_speed_trace.png"
    plt.savefig(p1_path, dpi=200)
    plt.close()
    logger.info(f"Saved Plot 1: {p1_path}")

    # Plot 2: Cumulative Distance Profile
    cum_gt = np.cumsum(p_data["gt_speed"]) * config.TARGET_DT
    cum_a = np.cumsum(p_data["ukf_spd_a"]) * config.TARGET_DT
    cum_b = np.cumsum(p_data["ukf_spd_b"]) * config.TARGET_DT

    plt.figure(figsize=(10, 5))
    plt.plot(p_data["t"], cum_gt, 'k-', label=f"True Ground Truth ({cum_gt[-1]:.1f}m)", linewidth=2.2)
    plt.plot(p_data["t"], cum_a, 'r--', label=f"Phase 11 Frozen ({cum_a[-1]:.1f}m, err: {cum_a[-1]-cum_gt[-1]:+.1f}m)", linewidth=1.8)
    plt.plot(p_data["t"], cum_b, 'b-', label=f"Phase 13B Event-Gated ({cum_b[-1]:.1f}m, err: {cum_b[-1]-cum_gt[-1]:+.1f}m)", linewidth=2.0)
    plt.title(f"Plot 2: Cumulative Distance Profile ({p_key})", fontsize=12, fontweight="bold")
    plt.xlabel("Outage Time (s)", fontsize=11)
    plt.ylabel("Cumulative Distance (m)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    p2_path = plots_dir / "phase13b_distance_trace.png"
    plt.savefig(p2_path, dpi=200)
    plt.close()
    logger.info(f"Saved Plot 2: {p2_path}")

    # Plot 3: Along-Track Error vs Outage Duration across all 56 scenarios
    durations = sorted(df_res["duration_s"].unique())
    along_a_meds = [df_res[df_res["duration_s"] == d]["along_a_m"].abs().median() for d in durations]
    along_b_meds = [df_res[df_res["duration_s"] == d]["along_b_m"].abs().median() for d in durations]

    plt.figure(figsize=(10, 5))
    plt.plot(durations, along_a_meds, 'ro-', label="Baseline A (Phase 11 Frozen Anchor)", linewidth=2.0, markersize=7)
    plt.plot(durations, along_b_meds, 'bs-', label="Experiment B (Phase 13B Event-Gated)", linewidth=2.0, markersize=7)
    plt.title("Plot 3: Median Along-Track Error vs Outage Duration (56 Outages)", fontsize=12, fontweight="bold")
    plt.xlabel("Outage Duration (s)", fontsize=11)
    plt.ylabel("Median Along-Track Error (m)", fontsize=11)
    plt.xticks(durations)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    p3_path = plots_dir / "phase13b_along_track_error.png"
    plt.savefig(p3_path, dpi=200)
    plt.close()
    logger.info(f"Saved Plot 3: {p3_path}")

    # Plot 4: Event Gate Timeline
    tl_times = [item["time_s"] for item in tl_data]
    tl_gates = [1 if item["gate_open"] else 0 for item in tl_data]
    tl_types = [item["gate_type"] for item in tl_data]

    plt.figure(figsize=(10, 4))
    plt.step(tl_times, tl_gates, where='post', color='darkorange', linewidth=2.0, label="Event Gate State (1=Open, 0=Closed)")
    for idx in range(0, len(tl_times), 4):
        if tl_gates[idx] == 1:
            plt.annotate(tl_types[idx], (tl_times[idx], 1.05), rotation=45, fontsize=8, color='maroon')
    plt.ylim(-0.1, 1.4)
    plt.yticks([0, 1], ["Gate Closed (Cruise)", "Gate Open (Dynamic)"])
    plt.title(f"Plot 4: Event Gate State Timeline ({p_key})", fontsize=12, fontweight="bold")
    plt.xlabel("Outage Time (s)", fontsize=11)
    plt.ylabel("Gate Status", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    p4_path = plots_dir / "phase13b_event_gate_timeline.png"
    plt.savefig(p4_path, dpi=200)
    plt.close()
    logger.info(f"Saved Plot 4: {p4_path}")

    # Plot 5: Correction Magnitude (Neural Correction vs Fixed Anchor)
    tl_corrs = [item["corr_val"] for item in tl_data]
    plt.figure(figsize=(10, 4))
    plt.plot(tl_times, tl_corrs, 'm-', linewidth=2.0, label="Gated Correction c(t) (m/s)")
    plt.axhline(0.0, color='gray', linestyle='--', label="Zero Baseline (Fixed Anchor)")
    plt.title(f"Plot 5: Gated Neural Correction Magnitude vs Anchor ({p_key})", fontsize=12, fontweight="bold")
    plt.xlabel("Outage Time (s)", fontsize=11)
    plt.ylabel("Velocity Correction (m/s)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    p5_path = plots_dir / "phase13b_correction_magnitude.png"
    plt.savefig(p5_path, dpi=200)
    plt.close()
    logger.info(f"Saved Plot 5: {p5_path}")

    return df_res


if __name__ == "__main__":
    run_phase13b_navigation_benchmark()
