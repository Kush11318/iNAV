"""
Phase 8B: Controlled Online RLS Calibration Experiment (Vectorized & ASCII Safe)
Strict Audit / Isolation Mode.

Baseline: Phase 8A 4-Second VelocityNet (phase8a_4s_velocity_net.pt)
Experiment: Phase 8B Online Pre-Outage RLS Calibration

All UKF parameters, Q/R matrices, map matcher, and model weights are strictly FROZEN.
"""

import sys
import glob
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase8a_4s_model import VelocityNet4s
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
logger = logging.getLogger("iNAV.Phase8B_Exp")


class OnlineScalarRLS:
    """
    Causal Online Recursive Least Squares (RLS) scale factor estimator.
    Learns k online from healthy GNSS prior to outage:
        v_gnss = k * v_ai
    Uses fixed engineering hyperparameters:
        - k0 = 1.0 (unit scale initialization)
        - P0 = 0.1
        - lambda = 0.985 (standard forgetting factor for vehicle scale adaptation)
        - k_clamp in [0.60, 1.80] (physical chassis/tire variance bound)
    """

    def __init__(
        self,
        k0: float = 1.0,
        P0: float = 0.1,
        lam: float = 0.985,
        k_min: float = 0.60,
        k_max: float = 1.80,
        min_v_gnss: float = 2.0,
        min_v_ai: float = 1.0
    ):
        self.k = float(k0)
        self.P = float(P0)
        self.lam = float(lam)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.min_v_gnss = float(min_v_gnss)
        self.min_v_ai = float(min_v_ai)
        self.update_count = 0
        self.history = []

    def update(
        self,
        time_s: float,
        v_gnss: float,
        v_ai: float,
        is_gnss_healthy: bool,
        gnss_accuracy_m: float = 3.0
    ) -> bool:
        self.history.append((time_s, self.k, self.P))

        if not is_gnss_healthy:
            return False
        if not (math.isfinite(v_gnss) and math.isfinite(v_ai)):
            return False
        if v_gnss < self.min_v_gnss or v_ai < self.min_v_ai:
            return False
        if gnss_accuracy_m > 5.0:
            return False

        G = (self.P * v_ai) / (self.lam + (v_ai**2) * self.P)
        error = v_gnss - self.k * v_ai
        self.k = float(np.clip(self.k + G * error, self.k_min, self.k_max))
        self.P = float(np.clip((self.P - G * v_ai * self.P) / self.lam, 1e-5, 1.0))
        self.update_count += 1
        return True

    def get_k_at_time(self, query_time_s: float) -> float:
        if not self.history:
            return self.k
        for t_s, k_val, _ in reversed(self.history):
            if t_s <= query_time_s:
                return k_val
        return self.history[0][1]


def compute_batched_velocitynet_predictions(
    model: VelocityNet4s,
    seq_data: np.ndarray,
    needed_step_indices: List[int],
    window_len: int = 40,
    batch_size: int = 2048,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float]]:
    """
    Extracts all sliding windows for specified step indices and runs batch inference in high-speed chunks.
    Returns a lookup dictionary mapping step_idx -> (pred_d_4s, ev_class, sig).
    """
    n_samples = len(seq_data)
    valid_indices = sorted([k for k in needed_step_indices if window_len - 1 <= k < n_samples])

    if not valid_indices:
        return {}

    windows = []
    for k in valid_indices:
        win = seq_data[k - window_len + 1 : k + 1]
        windows.append(win)

    all_windows = np.array(windows, dtype=np.float32)  # (M, 40, 6)

    preds_d = []
    preds_ev = []
    preds_sig = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]  # (B, 40, 6)
            bx = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)  # (B, 6, 40)
            d_d, ev_logits, sig = model(bx)
            p_d = d_d.squeeze(-1).cpu().numpy()
            p_ev = torch.argmax(ev_logits, dim=1).cpu().numpy()
            p_sig = sig.squeeze(-1).cpu().numpy()

            if p_d.ndim == 0:
                p_d = np.array([p_d])
                p_ev = np.array([p_ev])
                p_sig = np.array([p_sig])

            preds_d.append(p_d)
            preds_ev.append(p_ev)
            preds_sig.append(p_sig)

    cat_d = np.concatenate(preds_d)
    cat_ev = np.concatenate(preds_ev)
    cat_sig = np.concatenate(preds_sig)

    lookup = {}
    for idx, k in enumerate(valid_indices):
        lookup[k] = (float(cat_d[idx]), int(cat_ev[idx]), float(cat_sig[idx]))

    return lookup


def run_single_outage_ukf_fast(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    k_scale: float,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Executes UKF dead reckoning during a single GPS blackout using precomputed AI predictions.
    UKF equations and noise matrices remain strictly frozen.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    ukf = UKFNavigationFilter()
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg)
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    raw_ai_displacements = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Filter prediction step
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        # 2. Check precomputed prediction for step
        global_k = global_start_idx + k
        if global_k in lookup_preds and (k % 5 == 0):
            raw_d, ev_class, raw_sig = lookup_preds[global_k]
            raw_ai_displacements.append(raw_d)

            # Interface adaptation: delta_d_interface = (raw_d * k_scale) / 2.0
            # So in UKF: v_net = delta_d_interface / 2.0 = (raw_d * k_scale) / 4.0 = k_scale * v_ai
            cal_d = (raw_d * k_scale) / 2.0
            cal_sig = max((raw_sig * k_scale) / 2.0, 0.2)

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

    dur_s = n * dt
    ai_mean_spd = (float(np.mean(raw_ai_displacements)) / 4.0) if raw_ai_displacements else init_speed_ms
    integrated_ai_dist = ai_mean_spd * dur_s
    integrated_gt_dist = float(df_outage[config.COL_TRUE_SPEED_MS].sum() * dt) if config.COL_TRUE_SPEED_MS in df_outage.columns else 0.0

    return est_lat, est_lon, est_speed, integrated_ai_dist, integrated_gt_dist


def evaluate_phase8b_causal():
    print("="*80, flush=True)
    print("PHASE 8B: CONTROLLED ONLINE RLS CAUSAL NAVIGATION BENCHMARK", flush=True)
    print("Baseline: Phase 8A 4s (Uncalibrated, k=1.0)", flush=True)
    print("Experiment: Phase 8B 4s + Causal Pre-Outage RLS Calibration", flush=True)
    print("="*80, flush=True)

    model_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    if not model_path.exists():
        print(f"Error: {model_path} not found!", flush=True)
        return

    model = VelocityNet4s(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Test file list (held-out test runs)
    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    outage_results = []

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]

    print(f"Total test files to evaluate: {len(test_files)}", flush=True)

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

        # Pre-calibrate run alignment using warmup (first 45s)
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

        # 1. High-speed batched prediction for all RLS steps and all outage steps
        needed_indices = set(k for k in range(0, n, 5) if k >= 39)
        for _, row in df_outages.iterrows():
            mask = (df_sim["outage_id"] == int(row["outage_id"]))
            s_idx = mask.idxmax()
            sub_len = len(df.loc[mask])
            for k in range(0, sub_len, 5):
                if s_idx + k >= 39:
                    needed_indices.add(s_idx + k)

        lookup_preds = compute_batched_velocitynet_predictions(
            model=model,
            seq_data=seq_data,
            needed_step_indices=list(needed_indices),
            window_len=40,
            batch_size=2048,
            device="cpu"
        )

        # 2. Run CAUSAL RLS pass across the entire trajectory
        # Updates strictly when GNSS is available outside outages
        rls = OnlineScalarRLS(
            k0=1.0, P0=0.1, lam=0.985, k_min=0.60, k_max=1.80,
            min_v_gnss=2.0, min_v_ai=1.0
        )

        for step in range(n):
            t_curr = float(df[config.COL_TIME].iloc[step]) if config.COL_TIME in df.columns else step * 0.1

            is_out = bool(df_sim["is_outage"].iloc[step]) if "is_outage" in df_sim.columns else False
            g_lat = float(df_sim[config.COL_GPS_LAT].iloc[step]) if config.COL_GPS_LAT in df_sim.columns else 0.0
            g_spd = float(df_sim[config.COL_GPS_SPEED_MS].iloc[step]) if config.COL_GPS_SPEED_MS in df_sim.columns else 0.0
            g_acc = float(df_sim[config.COL_GPS_ACCURACY].iloc[step]) if config.COL_GPS_ACCURACY in df_sim.columns else 3.0

            gnss_healthy = (not is_out) and math.isfinite(g_lat) and (g_lat != 0.0)

            if step in lookup_preds and (step % 5 == 0):
                raw_d, _, _ = lookup_preds[step]
                v_ai = raw_d / 4.0

                rls.update(
                    time_s=t_curr,
                    v_gnss=g_spd,
                    v_ai=v_ai,
                    is_gnss_healthy=gnss_healthy,
                    gnss_accuracy_m=g_acc
                )

        # 3. Evaluate each outage strictly using causally frozen k
        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]

            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()
            outage_onset_time = float(df[config.COL_TIME].iloc[outage_start_idx]) if config.COL_TIME in df.columns else outage_start_idx * 0.1

            k_onset = rls.get_k_at_time(outage_onset_time)
            k_2s_before = rls.get_k_at_time(outage_onset_time - 2.0)
            k_5s_before = rls.get_k_at_time(outage_onset_time - 5.0)

            pre_idx_start = max(0, outage_start_idx - 10)
            pre_df = df.iloc[pre_idx_start:outage_start_idx]
            if len(pre_df) > 0:
                init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
                init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
                init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
                h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
                init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
            else:
                init_lat = float(df[config.COL_TRUE_LAT].iloc[outage_start_idx])
                init_lon = float(df[config.COL_TRUE_LON].iloc[outage_start_idx])
                init_spd = float(df[config.COL_TRUE_SPEED_MS].iloc[outage_start_idx])
                init_hdg = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx])

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            # ----------------------------------------------------
            # A. Phase 8A: Uncalibrated (k = 1.0)
            # ----------------------------------------------------
            lat_8a, lon_8a, spd_8a, ai_d_8a, gt_d = run_single_outage_ukf_fast(
                df_outage=df_sub,
                init_lat=init_lat,
                init_lon=init_lon,
                init_speed_ms=init_spd,
                init_heading_deg=init_hdg,
                lookup_preds=lookup_preds,
                global_start_idx=outage_start_idx,
                alignment=run_align,
                k_scale=1.0
            )
            score_8a = score_outage_segment(
                lat_8a, lon_8a, gt_lat, gt_lon,
                distance_travelled_m=row["distance_travelled_m"],
                duration_s=dur,
                gt_heading_deg=gt_hdg
            )

            # ----------------------------------------------------
            # B. Phase 8B: Online Pre-Outage RLS (k = k_onset)
            # ----------------------------------------------------
            lat_8b, lon_8b, spd_8b, ai_d_8b, _ = run_single_outage_ukf_fast(
                df_outage=df_sub,
                init_lat=init_lat,
                init_lon=init_lon,
                init_speed_ms=init_spd,
                init_heading_deg=init_hdg,
                lookup_preds=lookup_preds,
                global_start_idx=outage_start_idx,
                alignment=run_align,
                k_scale=k_onset
            )
            score_8b = score_outage_segment(
                lat_8b, lon_8b, gt_lat, gt_lon,
                distance_travelled_m=row["distance_travelled_m"],
                duration_s=dur,
                gt_heading_deg=gt_hdg
            )

            calib_ai_dist = ai_d_8a * k_onset

            record = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "gt_dist_m": round(float(row["distance_travelled_m"]), 2),
                "ai_dist_8a_m": round(float(ai_d_8a), 2),
                "calib_dist_8b_m": round(float(calib_ai_dist), 2),
                "k_minus_5s": round(float(k_5s_before), 4),
                "k_minus_2s": round(float(k_2s_before), 4),
                "k_onset": round(float(k_onset), 4),
                # 8A Scores
                "fpe_8a_m": round(float(score_8a["final_pos_error_m"]), 2),
                "drift_8a_pct": round(float(score_8a["pct_of_distance"]), 2),
                "along_8a_m": round(float(score_8a["along_track_m"]), 2),
                "cross_8a_m": round(float(score_8a["cross_track_m"]), 2),
                "hdg_8a_deg": round(float(score_8a["heading_error_deg"]), 2),
                # 8B Scores
                "fpe_8b_m": round(float(score_8b["final_pos_error_m"]), 2),
                "drift_8b_pct": round(float(score_8b["pct_of_distance"]), 2),
                "along_8b_m": round(float(score_8b["along_track_m"]), 2),
                "cross_8b_m": round(float(score_8b["cross_track_m"]), 2),
                "hdg_8b_deg": round(float(score_8b["heading_error_deg"]), 2),
                # Improvement delta
                "fpe_change_m": round(float(score_8b["final_pos_error_m"] - score_8a["final_pos_error_m"]), 2),
                "drift_change_pct": round(float(score_8b["pct_of_distance"] - score_8a["pct_of_distance"]), 2),
                "target_pass_8a": score_8a["pct_of_distance"] < 10.0,
                "target_pass_8b": score_8b["pct_of_distance"] < 10.0
            }
            outage_results.append(record)

        print(f"Finished {run_name} (Total outages logged: {len(outage_results)})", flush=True)

    df_outages = pd.DataFrame(outage_results)

    print("\n" + "="*80, flush=True)
    print("1. OUTAGE-BY-OUTAGE A/B COMPARISON TABLE", flush=True)
    print("="*80, flush=True)
    display_cols = [
        "run", "duration_s", "gt_dist_m", "k_onset",
        "fpe_8a_m", "fpe_8b_m", "drift_8a_pct", "drift_8b_pct",
        "along_8a_m", "along_8b_m", "target_pass_8a", "target_pass_8b"
    ]
    print(df_outages[display_cols].to_string(index=False), flush=True)

    print("\n" + "="*80, flush=True)
    print("2. AGGREGATE SUMMARY BY OUTAGE DURATION", flush=True)
    print("="*80, flush=True)
    agg_df = df_outages.groupby("duration_s").agg({
        "k_onset": ["count", "mean", "median", "std"],
        "fpe_8a_m": "median",
        "fpe_8b_m": "median",
        "drift_8a_pct": "median",
        "drift_8b_pct": "median",
        "along_8a_m": "median",
        "along_8b_m": "median",
        "cross_8a_m": "median",
        "cross_8b_m": "median",
        "hdg_8a_deg": "median",
        "hdg_8b_deg": "median",
    })
    print(agg_df.to_string(), flush=True)

    print("\n" + "="*80, flush=True)
    print("3. OVERALL AGGREGATE DRIFT & FPE METRICS", flush=True)
    print("="*80, flush=True)
    n_total = len(df_outages)
    n_improved = (df_outages["drift_change_pct"] < 0).sum()
    n_worsened = (df_outages["drift_change_pct"] > 0).sum()
    n_cross_under = ((~df_outages["target_pass_8a"]) & df_outages["target_pass_8b"]).sum()
    n_cross_over = (df_outages["target_pass_8a"] & (~df_outages["target_pass_8b"])).sum()
    n_pass_8a = df_outages["target_pass_8a"].sum()
    n_pass_8b = df_outages["target_pass_8b"].sum()

    print(f"Total Outages Evaluated         : {n_total}", flush=True)
    print(f"Overall Median Drift (8A Raw)   : {df_outages['drift_8a_pct'].median():.2f}%", flush=True)
    print(f"Overall Median Drift (8B RLS)   : {df_outages['drift_8b_pct'].median():.2f}%", flush=True)
    print(f"Overall Median FPE (8A Raw)     : {df_outages['fpe_8a_m'].median():.2f} m", flush=True)
    print(f"Overall Median FPE (8B RLS)     : {df_outages['fpe_8b_m'].median():.2f} m", flush=True)
    print(f"Outages Improved with RLS       : {n_improved} / {n_total} ({n_improved/n_total*100:.1f}%)", flush=True)
    print(f"Outages Worsened with RLS       : {n_worsened} / {n_total} ({n_worsened/n_total*100:.1f}%)", flush=True)
    print(f"Achieving <10% Target in 8A Raw : {n_pass_8a} / {n_total} ({n_pass_8a/n_total*100:.1f}%)", flush=True)
    print(f"Achieving <10% Target in 8B RLS : {n_pass_8b} / {n_total} ({n_pass_8b/n_total*100:.1f}%)", flush=True)
    print(f"Crossed Below 10% Target (Gain) : +{n_cross_under}", flush=True)
    print(f"Crossed Above 10% Target (Loss) : -{n_cross_over}", flush=True)

    print("\n" + "="*80, flush=True)
    print("4. CONVERGENCE & STABILITY DIAGNOSTICS", flush=True)
    print("="*80, flush=True)
    dk_5_to_2 = np.abs(df_outages["k_minus_2s"] - df_outages["k_minus_5s"])
    dk_2_to_0 = np.abs(df_outages["k_onset"] - df_outages["k_minus_2s"])
    print(f"Mean |k(-2s) - k(-5s)| (3s pre-outage stability): {dk_5_to_2.mean():.4f}", flush=True)
    print(f"Mean |k(onset) - k(-2s)| (2s pre-outage stability): {dk_2_to_0.mean():.4f}", flush=True)
    print(f"Max  |k(onset) - k(-2s)|: {dk_2_to_0.max():.4f}", flush=True)
    print(f"Min k_onset observed: {df_outages['k_onset'].min():.4f}", flush=True)
    print(f"Max k_onset observed: {df_outages['k_onset'].max():.4f}", flush=True)
    print(f"Clamped to boundary (0.60 or 1.80): {((df_outages['k_onset'] <= 0.601) | (df_outages['k_onset'] >= 1.799)).sum()} times", flush=True)

    # Save artifact results
    results_path = config.BASE_DIR / "eval" / "phase8b_results.csv"
    df_outages.to_csv(results_path, index=False)
    print(f"\nSaved detailed outage results to {results_path}", flush=True)


if __name__ == "__main__":
    evaluate_phase8b_causal()
