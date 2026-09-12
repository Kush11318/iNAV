"""
Phase 11: Navigation A/B Benchmark Script
Compares:
  - Baseline Model A: Phase 10 4s Temporal Velocity Net (Absolute Velocity Prediction)
  - Experiment Model B: Phase 11 4s Anchored Temporal Velocity Net (GNSS-Anchored Velocity Residual)
    where v_anchor = last healthy GNSS forward speed immediately before outage (frozen for entire outage).

Evaluates across the exact same 56 held-out test outages used in Phases 8B, 8C, 8D, 9, and 10.
Strict experiment isolation: UKF 7-state equations, Q/R, alignment, and parameters are strictly frozen.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase10_temporal_model import TemporalVelocityNet4s
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
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
logger = logging.getLogger("iNAV.Phase11_Nav")


def compute_outage_predictions_phase11(
    model_a: TemporalVelocityNet4s,
    model_b: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_val: float,
    window_len: int = 40,
    device: str = "cpu"
) -> Tuple[Dict[int, Tuple[float, int, float]], Dict[int, Tuple[float, int, float]]]:
    """
    Computes causal sliding-window predictions for a specific outage.
    v_anchor_val is strictly the pre-outage speed, frozen for the entire outage.
    """
    lookup_a = {}
    lookup_b = {}

    model_a.eval()
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
        return lookup_a, lookup_b

    all_windows = np.array(windows, dtype=np.float32)  # (M, 40, 6)
    bx = torch.from_numpy(all_windows.transpose(0, 2, 1)).float().to(device)
    b_anc = torch.full((len(all_windows), 1), v_anchor_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Model A: Phase 10 absolute velocity
        v_seq_a, sig_seq_a, ev_a = model_a(bx)
        p_v_a = v_seq_a.cpu().numpy()
        p_sig_a = sig_seq_a.cpu().numpy()
        p_ev_a = torch.argmax(ev_a, dim=1).cpu().numpy()

        p_d_a = np.sum(p_v_a * 0.2, axis=1)
        p_int_sig_a = 0.2 * np.sqrt(np.sum(p_sig_a ** 2, axis=1))

        # Model B: Phase 11 anchored velocity
        del_v_seq_b, v_seq_b, sig_seq_b, ev_b = model_b(bx, b_anc)
        p_v_b = v_seq_b.cpu().numpy()
        p_sig_b = sig_seq_b.cpu().numpy()
        p_ev_b = torch.argmax(ev_b, dim=1).cpu().numpy()

        p_d_b = np.sum(p_v_b * 0.2, axis=1)
        p_int_sig_b = 0.2 * np.sqrt(np.sum(p_sig_b ** 2, axis=1))

    for idx, gk in enumerate(step_indices):
        lookup_a[gk] = (float(p_d_a[idx]), int(p_ev_a[idx]), float(p_int_sig_a[idx]))
        lookup_b[gk] = (float(p_d_b[idx]), int(p_ev_b[idx]), float(p_int_sig_b[idx]))

    return lookup_a, lookup_b


def run_single_outage_ukf(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), []

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
    nis_list = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # Filter prediction
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        global_k = global_start_idx + k
        if global_k in lookup_preds and (k % 5 == 0):
            raw_d, ev_class, raw_sig = lookup_preds[global_k]
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

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed, nis_list


def run_phase11_navigation_benchmark():
    logger.info("=" * 80)
    logger.info("PHASE 11: NAVIGATION A/B BENCHMARK (56 HELD-OUT OUTAGES)")
    logger.info("Baseline A: Model A (Phase 10 Absolute Velocity Net)")
    logger.info("Experiment B: Model B (Phase 11 Anchored Velocity Net, Frozen v_anchor)")
    logger.info("=" * 80)

    model_a_path = config.MODELS_DIR / "phase10_temporal_velocity_net.pt"
    model_b_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"

    if not model_a_path.exists() or not model_b_path.exists():
        logger.error("Missing model checkpoint(s)!")
        return None

    model_a = TemporalVelocityNet4s(in_channels=6, num_events=5)
    model_a.load_state_dict(torch.load(model_a_path, map_location="cpu"))
    model_a.eval()

    model_b = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model_b.load_state_dict(torch.load(model_b_path, map_location="cpu"))
    model_b.eval()

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    logger.info(f"Evaluating {len(test_files)} test files...")

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    records = []

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

            # Compute predictions for this outage
            lookup_a, lookup_b = compute_outage_predictions_phase11(
                model_a=model_a,
                model_b=model_b,
                seq_data=seq_data,
                outage_start_idx=outage_start_idx,
                sub_len=len(df_sub),
                v_anchor_val=v_anchor_val,
                window_len=40,
                device="cpu"
            )

            # Model A UKF
            lat_a, lon_a, spd_a, _ = run_single_outage_ukf(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_a, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # Model B UKF
            lat_b, lon_b, spd_b, _ = run_single_outage_ukf(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup_b, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_b = score_outage_segment(lat_b, lon_b, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            rec = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                "v_anchor_ms": round(v_anchor_val, 2),
                # Model A (Phase 10 Absolute)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                # Model B (Phase 11 Anchored)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
            }
            records.append(rec)

    df_nav = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase11_nav_results.csv"
    df_nav.to_csv(out_csv, index=False)
    logger.info(f"Saved navigation benchmark results to {out_csv} ({len(df_nav)} outages)")

    # Print Summary Table
    print("\n" + "=" * 100)
    print(f"{'METRIC':<32} | {'MODEL A (Phase 10 Absolute)':<30} | {'MODEL B (Phase 11 Anchored)':<30}")
    print("=" * 100)
    print(f"{'Median Drift %':<32} | {df_nav['drift_a_pct'].median():<30.2f}% | {df_nav['drift_b_pct'].median():<30.2f}%")
    print(f"{'Mean Drift %':<32} | {df_nav['drift_a_pct'].mean():<30.2f}% | {df_nav['drift_b_pct'].mean():<30.2f}%")
    print(f"{'Median FPE (m)':<32} | {df_nav['fpe_a_m'].median():<30.2f}m | {df_nav['fpe_b_m'].median():<30.2f}m")
    print(f"{'Mean FPE (m)':<32} | {df_nav['fpe_a_m'].mean():<30.2f}m | {df_nav['fpe_b_m'].mean():<30.2f}m")
    print(f"{'Median Along-Track Error (m)':<32} | {df_nav['along_a_m'].median():<30.2f}m | {df_nav['along_b_m'].median():<30.2f}m")
    print(f"{'Mean Along-Track Error (m)':<32} | {df_nav['along_a_m'].mean():<30.2f}m | {df_nav['along_b_m'].mean():<30.2f}m")
    print(f"{'Median Cross-Track Error (m)':<32} | {df_nav['cross_a_m'].median():<30.2f}m | {df_nav['cross_b_m'].median():<30.2f}m")
    print(f"{'Median Heading Error (deg)':<32} | {df_nav['hdg_a_deg'].median():<30.2f}deg | {df_nav['hdg_b_deg'].median():<30.2f}deg")
    print("-" * 100)
    improved = int(np.sum(df_nav["drift_b_pct"] < df_nav["drift_a_pct"]))
    worsened = int(np.sum(df_nav["drift_b_pct"] > df_nav["drift_a_pct"]))
    unchanged = int(np.sum(df_nav["drift_b_pct"] == df_nav["drift_a_pct"]))
    target_count_a = int(np.sum(df_nav["drift_a_pct"] < 10.0))
    target_count_b = int(np.sum(df_nav["drift_b_pct"] < 10.0))
    print(f"{'Drift Improved Outages':<32} | {'-':<30} | {improved}/{len(df_nav)} ({improved/len(df_nav)*100:.1f}%)")
    print(f"{'Drift Worsened Outages':<32} | {'-':<30} | {worsened}/{len(df_nav)} ({worsened/len(df_nav)*100:.1f}%)")
    print(f"{'Drift < 10% Target Count':<32} | {target_count_a}/{len(df_nav)} ({target_count_a/len(df_nav)*100:.1f}%) | {target_count_b}/{len(df_nav)} ({target_count_b/len(df_nav)*100:.1f}%)")
    print("=" * 100)

    # Breakdown by outage duration
    print("\n" + "=" * 105)
    print("BREAKDOWN BY OUTAGE DURATION (MEDIANS)")
    print("=" * 105)
    print(f"{'Duration':<10} | {'N':<5} | {'Drift A %':<12} | {'Drift B %':<12} | {'FPE A (m)':<12} | {'FPE B (m)':<12} | {'Along A (m)':<14} | {'Along B (m)':<14}")
    print("-" * 105)
    for dur in sorted(df_nav["duration_s"].unique()):
        sub = df_nav[df_nav["duration_s"] == dur]
        d_a = sub["drift_a_pct"].median()
        d_b = sub["drift_b_pct"].median()
        f_a = sub["fpe_a_m"].median()
        f_b = sub["fpe_b_m"].median()
        al_a = sub["along_a_m"].median()
        al_b = sub["along_b_m"].median()
        print(f"{dur:<10} | {len(sub):<5} | {d_a:<12.2f} | {d_b:<12.2f} | {f_a:<12.2f} | {f_b:<12.2f} | {al_a:<14.2f} | {al_b:<14.2f}")
    print("=" * 105)

    return df_nav


if __name__ == "__main__":
    run_phase11_navigation_benchmark()
