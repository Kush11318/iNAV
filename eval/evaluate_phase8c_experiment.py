"""
Phase 8C: Controlled Speed-Conditioned Calibration Experiment
Strict Audit / Isolation Mode.

Compares across identical 56 held-out test outages:
  Condition A: Phase 8A raw 4s VelocityNet (Uncalibrated, k=1.0)
  Condition B: Phase 8B scalar causal RLS (Frozen k_onset)
  Condition C: Phase 8C speed-conditioned calibration: k_eff = k(v_hat_previous)
  Condition D: Phase 8C speed-conditioned calibration + speed-conditioned measurement covariance

All bin boundaries and variance parameters derived exclusively from training/validation sets.
All UKF parameters, Q/R baselines, map matcher, and models remain strictly FROZEN.
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
logger = logging.getLogger("iNAV.Phase8C_Exp")

# ==============================================================================
# SPEED-CONDITIONED BIN SPECIFICATION (DERIVED FROM TRAIN/VAL DISTRIBUTIONS)
# ==============================================================================
# Bins (m/s):
# 0: [0, 5)   - Urban crawl (0-18 km/h)
# 1: [5, 12)  - City arterial (18-43 km/h)
# 2: [12, 20) - Fast suburban / expressway (43-72 km/h)
# 3: [20, 28) - Motorway cruising (72-100 km/h)
# 4: [28, 100)- High-speed motorway (>100 km/h)
BIN_EDGES = [0.0, 5.0, 12.0, 20.0, 28.0, 100.0]
NUM_BINS = len(BIN_EDGES) - 1

# Speed-conditioned residual standard deviations (m/s) strictly from train/val audit:
# Res std: [4.04, 4.39, 5.40, 3.90, 2.81]
TRAIN_VAL_RESIDUAL_STD_MS = np.array([4.04, 4.39, 5.40, 3.90, 2.81], dtype=np.float32)


def get_speed_bin_idx(v_ms: float) -> int:
    v = max(0.0, float(v_ms))
    for b in range(NUM_BINS):
        if BIN_EDGES[b] <= v < BIN_EDGES[b + 1]:
            return b
    return NUM_BINS - 1


class ScalarRLSEstimator:
    """Standard Phase 8B Scalar Causal RLS."""
    def __init__(self, k0: float = 1.0, P0: float = 0.1, lam: float = 0.985):
        self.k = float(k0)
        self.P = float(P0)
        self.lam = float(lam)
        self.history = []

    def update(self, time_s: float, v_gnss: float, v_ai: float, is_healthy: bool, acc_m: float = 3.0):
        self.history.append((time_s, self.k))
        if not is_healthy or v_gnss < 2.0 or v_ai < 1.0 or acc_m > 5.0:
            return
        G = (self.P * v_ai) / (self.lam + (v_ai**2) * self.P)
        err = v_gnss - self.k * v_ai
        self.k = float(np.clip(self.k + G * err, 0.60, 1.80))
        self.P = float(np.clip((self.P - G * v_ai * self.P) / self.lam, 1e-5, 1.0))

    def get_k_at_time(self, t_s: float) -> float:
        if not self.history:
            return self.k
        for ts, kval in reversed(self.history):
            if ts <= t_s:
                return kval
        return self.history[0][1]


class SpeedConditionedRLSEstimator:
    """
    Phase 8C Speed-Conditioned Online Calibration Estimator.
    Maintains independent RLS scale parameters and residual statistics across speed bins.
    """
    def __init__(self, k0: float = 1.0, P0: float = 0.1, lam: float = 0.985):
        self.lam = float(lam)
        self.k_bins = np.full(NUM_BINS, k0, dtype=np.float32)
        self.P_bins = np.full(NUM_BINS, P0, dtype=np.float32)
        self.counts = np.zeros(NUM_BINS, dtype=np.int32)
        self.res_sum = np.zeros(NUM_BINS, dtype=np.float32)
        self.res_sq_sum = np.zeros(NUM_BINS, dtype=np.float32)
        self.history = []  # (time_s, copy of k_bins, copy of counts)

    def update(self, time_s: float, v_gnss: float, v_ai: float, is_healthy: bool, acc_m: float = 3.0):
        self.history.append((time_s, self.k_bins.copy(), self.counts.copy()))
        if not is_healthy or v_gnss < 2.0 or v_ai < 1.0 or acc_m > 5.0:
            return

        # Determine bin based on observed AI speed
        b = get_speed_bin_idx(v_ai)

        G = (self.P_bins[b] * v_ai) / (self.lam + (v_ai**2) * self.P_bins[b])
        err = v_gnss - self.k_bins[b] * v_ai
        self.k_bins[b] = float(np.clip(self.k_bins[b] + G * err, 0.60, 1.80))
        self.P_bins[b] = float(np.clip((self.P_bins[b] - G * v_ai * self.P_bins[b]) / self.lam, 1e-5, 1.0))
        self.counts[b] += 1
        self.res_sum[b] += err
        self.res_sq_sum[b] += err**2

    def get_snapshot_at_time(self, t_s: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.history:
            return self.k_bins.copy(), self.counts.copy()
        for ts, k_arr, c_arr in reversed(self.history):
            if ts <= t_s:
                return k_arr.copy(), c_arr.copy()
        return self.history[0][1].copy(), self.history[0][2].copy()


class SpeedConditionedLookupTable:
    """
    Frozen snapshot of speed-conditioned calibration queried causally during outage
    using ONLY previous UKF speed estimate: k_eff = k(v_hat_previous).
    """
    def __init__(self, k_bins: np.ndarray, counts: np.ndarray):
        self.k_bins = k_bins.copy()
        self.counts = counts.copy()
        # Confidence weighting: if bin has fewer than 10 samples, blend towards nearest populated bin or 1.0
        self.confidence = np.clip(self.counts / 10.0, 0.0, 1.0)

    def query_k(self, v_hat_ms: float) -> float:
        b = get_speed_bin_idx(v_hat_ms)
        conf = self.confidence[b]
        k_val = self.k_bins[b]
        if conf >= 0.9:
            return float(k_val)

        # Fallback to nearest populated bin
        best_diff = 999
        best_b = b
        for ob in range(NUM_BINS):
            if self.counts[ob] >= 5 and abs(ob - b) < best_diff:
                best_diff = abs(ob - b)
                best_b = ob

        if best_diff < 999:
            k_fallback = self.k_bins[best_b]
            return float(conf * k_val + (1.0 - conf) * k_fallback)
        else:
            return float(conf * k_val + (1.0 - conf) * 1.0)

    def get_bin_residual_std(self, v_hat_ms: float) -> float:
        b = get_speed_bin_idx(v_hat_ms)
        return float(TRAIN_VAL_RESIDUAL_STD_MS[b])


def compute_batched_velocitynet_predictions(
    model: VelocityNet4s,
    seq_data: np.ndarray,
    needed_step_indices: List[int],
    window_len: int = 40,
    batch_size: int = 2048,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float]]:
    n_samples = len(seq_data)
    valid_indices = sorted([k for k in needed_step_indices if window_len - 1 <= k < n_samples])
    if not valid_indices:
        return {}

    windows = [seq_data[k - window_len + 1 : k + 1] for k in valid_indices]
    all_windows = np.array(windows, dtype=np.float32)

    preds_d, preds_ev, preds_sig = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            bx = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)
            d_d, ev_logits, sig = model(bx)
            p_d = d_d.squeeze(-1).cpu().numpy()
            p_ev = torch.argmax(ev_logits, dim=1).cpu().numpy()
            p_sig = sig.squeeze(-1).cpu().numpy()

            if p_d.ndim == 0:
                p_d, p_ev, p_sig = np.array([p_d]), np.array([p_ev]), np.array([p_sig])

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


def run_outage_simulation_mode(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    mode: str,  # '8A', '8B', '8C', '8D'
    k_scalar: float = 1.0,
    table_8c: Optional[SpeedConditionedLookupTable] = None,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Runs single outage UKF under Condition A, B, C, or D.
    Returns:
        (lat, lon, spd, ai_dist, gt_dist, mean_nis)
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0

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
    nis_values = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Filter prediction step
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        global_k = global_start_idx + k
        if global_k in lookup_preds and (k % 5 == 0):
            raw_d, ev_class, raw_sig = lookup_preds[global_k]
            raw_ai_displacements.append(raw_d)

            # Query scale factor causally based on previous speed estimate
            v_prev = float(ukf.x[2])

            if mode == "8A":
                k_eff = 1.0
                sigma_in = max(raw_sig / 2.0, 0.2)
            elif mode == "8B":
                k_eff = float(k_scalar)
                sigma_in = max((raw_sig * k_eff) / 2.0, 0.2)
            elif mode == "8C":
                k_eff = table_8c.query_k(v_prev) if table_8c else 1.0
                sigma_in = max((raw_sig * k_eff) / 2.0, 0.2)
            elif mode == "8D":
                k_eff = table_8c.query_k(v_prev) if table_8c else 1.0
                # Speed-conditioned residual covariance from train/val:
                res_std = table_8c.get_bin_residual_std(v_prev) if table_8c else 4.0
                # sigma_in scaled to 2.0s interface: res_std * 2.0
                sigma_base = max((raw_sig * k_eff) / 2.0, 0.2)
                sigma_in = max(sigma_base, res_std * 2.0)
            else:
                k_eff = 1.0
                sigma_in = max(raw_sig / 2.0, 0.2)

            cal_d = (raw_d * k_eff) / 2.0

            if ev_class == 0 or cal_d < 0.1:
                ukf.update_zupt(gyro_reading=w_v[2])
            else:
                v_meas = cal_d / 2.0
                v_filter = float(ukf.x[2])
                r_val = (sigma_in / 2.0)**2
                s_innov = float(ukf.P[2, 2] + r_val)
                nis_step = float((v_meas - v_filter)**2 / max(s_innov, 1e-6))
                nis_values.append(nis_step)

                ukf.update_velocity_net(
                    delta_d_pred=cal_d,
                    sigma_pred=sigma_in,
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
    mean_nis = float(np.mean(nis_values)) if nis_values else 1.0

    return est_lat, est_lon, est_speed, integrated_ai_dist, integrated_gt_dist, mean_nis


def run_phase8c_experiment():
    print("="*80, flush=True)
    print("PHASE 8C: CONTROLLED SPEED-CONDITIONED CALIBRATION EXPERIMENT", flush=True)
    print("Comparing 8A (Raw 4s) -> 8B (Scalar RLS) -> 8C (Speed-Conditioned) -> 8D (Speed-Cond + Covariance)", flush=True)
    print("="*80, flush=True)

    model_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    if not model_path.exists():
        print(f"Error: {model_path} not found!", flush=True)
        return

    model = VelocityNet4s(in_channels=6, num_events=5)
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

    outage_records = []

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

        # Online Causal Estimators running on healthy GNSS:
        rls_scalar = ScalarRLSEstimator(k0=1.0, P0=0.1, lam=0.985)
        rls_binned = SpeedConditionedRLSEstimator(k0=1.0, P0=0.1, lam=0.985)

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

                rls_scalar.update(t_curr, g_spd, v_ai, gnss_healthy, g_acc)
                rls_binned.update(t_curr, g_spd, v_ai, gnss_healthy, g_acc)

        # Replay each outage under all 4 conditions
        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()
            outage_onset_time = float(df[config.COL_TIME].iloc[outage_start_idx]) if config.COL_TIME in df.columns else outage_start_idx * 0.1

            # Freeze parameters causally
            k_scalar_onset = rls_scalar.get_k_at_time(outage_onset_time)
            k_bins_onset, counts_onset = rls_binned.get_snapshot_at_time(outage_onset_time)
            table_8c = SpeedConditionedLookupTable(k_bins_onset, counts_onset)

            # Pre-outage filter state
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

            v_start = float(df_sub[config.COL_TRUE_SPEED_MS].iloc[0]) if config.COL_TRUE_SPEED_MS in df_sub.columns else init_spd
            v_end = float(df_sub[config.COL_TRUE_SPEED_MS].iloc[-1]) if config.COL_TRUE_SPEED_MS in df_sub.columns else init_spd
            delta_v = v_end - v_start

            if v_start > 15.0 and delta_v < -5.0:
                dynamics_regime = "Decel_HighToLow"
            elif v_start < 10.0 and delta_v > 5.0:
                dynamics_regime = "Accel_LowToHigh"
            else:
                dynamics_regime = "Cruising_Steady"

            # Condition A: Raw 8A
            lat_a, lon_a, _, _, _, nis_a = run_outage_simulation_mode(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                lookup_preds, outage_start_idx, run_align, mode="8A"
            )
            sc_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, distance_travelled_m=row["distance_travelled_m"], duration_s=dur, gt_heading_deg=gt_hdg)

            # Condition B: Scalar RLS 8B
            lat_b, lon_b, _, _, _, nis_b = run_outage_simulation_mode(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                lookup_preds, outage_start_idx, run_align, mode="8B", k_scalar=k_scalar_onset
            )
            sc_b = score_outage_segment(lat_b, lon_b, gt_lat, gt_lon, distance_travelled_m=row["distance_travelled_m"], duration_s=dur, gt_heading_deg=gt_hdg)

            # Condition C: Speed-Conditioned 8C
            lat_c, lon_c, _, _, _, nis_c = run_outage_simulation_mode(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                lookup_preds, outage_start_idx, run_align, mode="8C", table_8c=table_8c
            )
            sc_c = score_outage_segment(lat_c, lon_c, gt_lat, gt_lon, distance_travelled_m=row["distance_travelled_m"], duration_s=dur, gt_heading_deg=gt_hdg)

            # Condition D: Speed-Conditioned 8C + Uncertainty 8D
            lat_d, lon_d, _, _, _, nis_d = run_outage_simulation_mode(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                lookup_preds, outage_start_idx, run_align, mode="8D", table_8c=table_8c
            )
            sc_d = score_outage_segment(lat_d, lon_d, gt_lat, gt_lon, distance_travelled_m=row["distance_travelled_m"], duration_s=dur, gt_heading_deg=gt_hdg)

            outage_records.append({
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(float(row["distance_travelled_m"]), 2),
                "v_start_ms": round(v_start, 2),
                "v_end_ms": round(v_end, 2),
                "delta_v_ms": round(delta_v, 2),
                "dynamics": dynamics_regime,
                "k_scalar": round(k_scalar_onset, 4),
                # Condition A (Raw 8A)
                "fpe_8a_m": round(float(sc_a["final_pos_error_m"]), 2),
                "drift_8a_pct": round(float(sc_a["pct_of_distance"]), 2),
                "along_8a_m": round(float(sc_a["along_track_m"]), 2),
                "cross_8a_m": round(float(sc_a["cross_track_m"]), 2),
                "nis_8a": round(nis_a, 2),
                # Condition B (Scalar 8B)
                "fpe_8b_m": round(float(sc_b["final_pos_error_m"]), 2),
                "drift_8b_pct": round(float(sc_b["pct_of_distance"]), 2),
                "along_8b_m": round(float(sc_b["along_track_m"]), 2),
                "cross_8b_m": round(float(sc_b["cross_track_m"]), 2),
                "nis_8b": round(nis_b, 2),
                # Condition C (Speed-Conditioned 8C)
                "fpe_8c_m": round(float(sc_c["final_pos_error_m"]), 2),
                "drift_8c_pct": round(float(sc_c["pct_of_distance"]), 2),
                "along_8c_m": round(float(sc_c["along_track_m"]), 2),
                "cross_8c_m": round(float(sc_c["cross_track_m"]), 2),
                "nis_8c": round(nis_c, 2),
                # Condition D (Speed-Conditioned + Covariance 8D)
                "fpe_8d_m": round(float(sc_d["final_pos_error_m"]), 2),
                "drift_8d_pct": round(float(sc_d["pct_of_distance"]), 2),
                "along_8d_m": round(float(sc_d["along_track_m"]), 2),
                "cross_8d_m": round(float(sc_d["cross_track_m"]), 2),
                "nis_8d": round(nis_d, 2),
                # Target flags
                "target_pass_8a": sc_a["pct_of_distance"] < 10.0,
                "target_pass_8b": sc_b["pct_of_distance"] < 10.0,
                "target_pass_8c": sc_c["pct_of_distance"] < 10.0,
                "target_pass_8d": sc_d["pct_of_distance"] < 10.0,
            })

        print(f"Finished {run_name} (Total outages logged: {len(outage_records)})", flush=True)

    df_out = pd.DataFrame(outage_records)
    results_path = config.BASE_DIR / "eval" / "phase8c_results.csv"
    df_out.to_csv(results_path, index=False)
    print(f"\nSaved Phase 8C results to {results_path}", flush=True)

    print("\n" + "="*80, flush=True)
    print("1. SUMMARY BY OUTAGE DURATION (MEDIAN DRIFT & FPE)", flush=True)
    print("="*80, flush=True)
    dur_summary = df_out.groupby("duration_s").agg({
        "drift_8a_pct": "median",
        "drift_8b_pct": "median",
        "drift_8c_pct": "median",
        "drift_8d_pct": "median",
        "fpe_8a_m": "median",
        "fpe_8b_m": "median",
        "fpe_8c_m": "median",
        "fpe_8d_m": "median",
        "along_8a_m": "median",
        "along_8b_m": "median",
        "along_8c_m": "median",
        "along_8d_m": "median",
        "nis_8a": "median",
        "nis_8b": "median",
        "nis_8c": "median",
        "nis_8d": "median",
    })
    print(dur_summary.to_string(), flush=True)

    print("\n" + "="*80, flush=True)
    print("2. OVERALL AGGREGATE PERFORMANCE ACROSS ALL 56 OUTAGES", flush=True)
    print("="*80, flush=True)
    n_total = len(df_out)
    print(f"Total Outages Evaluated                 : {n_total}", flush=True)
    print(f"Median Drift %: 8A={df_out['drift_8a_pct'].median():.2f}% | 8B={df_out['drift_8b_pct'].median():.2f}% | 8C={df_out['drift_8c_pct'].median():.2f}% | 8D={df_out['drift_8d_pct'].median():.2f}%", flush=True)
    print(f"Median FPE (m): 8A={df_out['fpe_8a_m'].median():.2f}m | 8B={df_out['fpe_8b_m'].median():.2f}m | 8C={df_out['fpe_8c_m'].median():.2f}m | 8D={df_out['fpe_8d_m'].median():.2f}m", flush=True)
    print(f"Median Along  : 8A={df_out['along_8a_m'].median():.2f}m | 8B={df_out['along_8b_m'].median():.2f}m | 8C={df_out['along_8c_m'].median():.2f}m | 8D={df_out['along_8d_m'].median():.2f}m", flush=True)
    print(f"Median NIS    : 8A={df_out['nis_8a'].median():.2f} | 8B={df_out['nis_8b'].median():.2f} | 8C={df_out['nis_8c'].median():.2f} | 8D={df_out['nis_8d'].median():.2f}", flush=True)
    print(f"<10% Target Pass Count: 8A={df_out['target_pass_8a'].sum()} | 8B={df_out['target_pass_8b'].sum()} | 8C={df_out['target_pass_8c'].sum()} | 8D={df_out['target_pass_8d'].sum()}", flush=True)

    # 3. TRANSITION / FAILURE CASE ANALYSIS
    print("\n" + "="*80, flush=True)
    print("3. DYNAMICS REGIME BREAKDOWN (CRITICAL FAILURE CASE AUDIT)", flush=True)
    print("="*80, flush=True)
    reg_summary = df_out.groupby("dynamics").agg({
        "outage_id": "count",
        "drift_8a_pct": "median",
        "drift_8b_pct": "median",
        "drift_8c_pct": "median",
        "drift_8d_pct": "median",
        "fpe_8a_m": "median",
        "fpe_8b_m": "median",
        "fpe_8c_m": "median",
        "fpe_8d_m": "median",
        "along_8a_m": "median",
        "along_8b_m": "median",
        "along_8c_m": "median",
        "along_8d_m": "median",
    })
    print(reg_summary.to_string(), flush=True)

    # Improvements relative to Phase 8B
    improved_c_vs_b = (df_out["drift_8c_pct"] < df_out["drift_8b_pct"]).sum()
    worsened_c_vs_b = (df_out["drift_8c_pct"] > df_out["drift_8b_pct"]).sum()
    improved_d_vs_b = (df_out["drift_8d_pct"] < df_out["drift_8b_pct"]).sum()
    worsened_d_vs_b = (df_out["drift_8d_pct"] > df_out["drift_8b_pct"]).sum()

    print("\n" + "="*80, flush=True)
    print("4. HEAD-TO-HEAD TRANSITIONS (8C vs 8B and 8D vs 8B)", flush=True)
    print("="*80, flush=True)
    print(f"8C vs 8B: Improved={improved_c_vs_b} / {n_total} ({improved_c_vs_b/n_total*100:.1f}%), Worsened={worsened_c_vs_b} / {n_total} ({worsened_c_vs_b/n_total*100:.1f}%)", flush=True)
    print(f"8D vs 8B: Improved={improved_d_vs_b} / {n_total} ({improved_d_vs_b/n_total*100:.1f}%), Worsened={worsened_d_vs_b} / {n_total} ({worsened_d_vs_b/n_total*100:.1f}%)", flush=True)


if __name__ == "__main__":
    run_phase8c_experiment()
