"""
Phase 9: AI-Adaptive NHC Experiment Evaluation Script
Benchmarks:
  - Baseline A: 7-State UKF + Fixed NHC Covariance (diag(0.10, 0.05) m^2/s^2)
  - Experiment B: 7-State UKF + NoiseNet AI-Adaptive NHC Covariance (NoiseNet(IMU))
  - Ablation C: 7-State UKF + Fixed Mean Train Covariance (diag(0.05745, 0.23538) m^2/s^2)
  - Secondary Diagnostic: 15-State ES-EKF with Active NHC (A vs B vs C)

Evaluates all 56 held-out test outages across all durations (10s, 30s, 60s, 120s, 180s)
and driving regimes (Straight Cruising, Acceleration, Braking, Turning, Rough Road).
Strict experiment isolation: No test set tuning, no production code changes.
"""

import sys
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
from eval.train_phase9_noisenet import NoiseNet
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from modules.alignment import AlignmentEngine, AlignmentState
from modules.ukf import UKFNavigationFilter
from modules.esekf import ESEKFNavigationFilter, quat_to_rot_matrix
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase9_Eval")

FIXED_VAR_LAT_A = 0.10
FIXED_VAR_VERT_A = 0.05
MEAN_TRAIN_VAR_LAT_C = 0.05745
MEAN_TRAIN_VAR_VERT_C = 0.23538


class UKFWithNHC(UKFNavigationFilter):
    """
    Subclass of UKFNavigationFilter that adds explicit NHC measurement machinery
    without touching production code.
    """

    def __init__(self, dt: float = 0.1):
        super().__init__(dt=dt)
        self.nhc_nis_history = []

    def update_nhc(self, sigma2_lat: float, sigma2_vert: float) -> Tuple[bool, float]:
        """
        Non-Holonomic Constraint (NHC) measurement update:
        z = [0, 0] (lateral and vertical zero-velocity constraints)
        In the 7-state unicycle planar model:
            x = [p_N, p_E, v_fwd, psi, b_g, b_a, k]
            v_lat(x) = -v*sin(psi)*cos(psi) + v*cos(psi)*sin(psi) = 0.0
            v_vert(x) = 0.0
        """
        def h_nhc(s):
            return np.array([0.0, 0.0])

        z = np.array([0.0, 0.0])
        R_nhc = np.diag([max(sigma2_lat, 1e-4), max(sigma2_vert, 1e-4)])

        accepted, nis = self.update_measurement(z, h_nhc, R_nhc)
        self.nhc_nis_history.append(nis)
        return accepted, nis


def compute_batched_predictions(
    vnet_model: VelocityNet4s,
    noisenet_model: NoiseNet,
    seq_data: np.ndarray,
    needed_step_indices: List[int],
    window_len: int = 40,
    batch_size: int = 2048,
    device: str = "cpu"
) -> Tuple[Dict[int, Tuple[float, int, float]], Dict[int, Tuple[float, float]]]:
    """
    Runs batch inference for VelocityNet and NoiseNet in vectorized chunks.
    """
    n_samples = len(seq_data)
    valid_indices = sorted([k for k in needed_step_indices if window_len - 1 <= k < n_samples])

    if not valid_indices:
        return {}, {}

    windows = []
    for k in valid_indices:
        win = seq_data[k - window_len + 1 : k + 1]
        windows.append(win)

    all_windows = np.array(windows, dtype=np.float32)  # (M, 40, 6)

    preds_vnet = []
    preds_noise = []

    vnet_model.eval()
    noisenet_model.eval()

    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]  # (B, 40, 6)
            bx = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)  # (B, 6, 40)

            # 1. VelocityNet
            d_d, ev_logits, sig = vnet_model(bx)
            p_d = d_d.squeeze(-1).cpu().numpy()
            p_ev = torch.argmax(ev_logits, dim=1).cpu().numpy()
            p_sig = sig.squeeze(-1).cpu().numpy()

            if p_d.ndim == 0:
                p_d = np.array([p_d])
                p_ev = np.array([p_ev])
                p_sig = np.array([p_sig])

            for idx in range(len(p_d)):
                preds_vnet.append((float(p_d[idx]), int(p_ev[idx]), float(p_sig[idx])))

            # 2. NoiseNet
            cov = noisenet_model(bx).cpu().numpy()  # (B, 2)
            if cov.ndim == 1:
                cov = np.expand_dims(cov, axis=0)

            for idx in range(len(cov)):
                preds_noise.append((float(cov[idx, 0]), float(cov[idx, 1])))

    vnet_lookup = {}
    noise_lookup = {}
    for idx, k in enumerate(valid_indices):
        vnet_lookup[k] = preds_vnet[idx]
        noise_lookup[k] = preds_noise[idx]

    return vnet_lookup, noise_lookup


def run_single_outage_phase9(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    vnet_lookup: Dict[int, Tuple[float, int, float]],
    noise_lookup: Dict[int, Tuple[float, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    nhc_mode: str = "fixed_A",  # 'fixed_A', 'adaptive_B', 'mean_C'
    filter_type: str = "ukf",    # 'ukf' or 'esekf'
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float], List[float]]:
    """
    Executes dead reckoning on a single outage with specified NHC covariance configuration.
    Returns: est_lat, est_lon, est_speed, var_lat_history, var_vert_history, nis_history
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), [], [], []

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    var_lat_hist = []
    var_vert_hist = []
    nis_hist = []

    if filter_type == "ukf":
        ukf = UKFWithNHC(dt=dt)
        ukf.initialize(
            init_lat=init_lat,
            init_lon=init_lon,
            init_speed_ms=init_speed_ms,
            init_heading_rad=math.radians(init_heading_deg)
        )

        for k in range(n):
            a_b = acc_raw[k]
            w_b = gyro_raw[k]
            a_v, w_v = alignment.transform_imu(a_b, w_b)

            # 1. Prediction step
            ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

            # Determine NHC covariance for this step
            global_k = global_start_idx + k
            if nhc_mode == "fixed_A":
                v_lat, v_vert = FIXED_VAR_LAT_A, FIXED_VAR_VERT_A
            elif nhc_mode == "mean_C":
                v_lat, v_vert = MEAN_TRAIN_VAR_LAT_C, MEAN_TRAIN_VAR_VERT_C
            elif nhc_mode == "adaptive_B":
                if global_k in noise_lookup:
                    v_lat, v_vert = noise_lookup[global_k]
                else:
                    v_lat, v_vert = MEAN_TRAIN_VAR_LAT_C, MEAN_TRAIN_VAR_VERT_C
            else:
                v_lat, v_vert = FIXED_VAR_LAT_A, FIXED_VAR_VERT_A

            var_lat_hist.append(v_lat)
            var_vert_hist.append(v_vert)

            # 2. NHC update step at 10 Hz
            _, nis_val = ukf.update_nhc(sigma2_lat=v_lat, sigma2_vert=v_vert)
            nis_hist.append(nis_val)

            # 3. VelocityNet measurement every 5 steps (0.5s)
            if global_k in vnet_lookup and (k % 5 == 0):
                raw_d, ev_class, raw_sig = vnet_lookup[global_k]
                cal_d = raw_d / 2.0  # standard interface
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
        return est_lat, est_lon, est_speed, var_lat_hist, var_vert_hist, nis_hist

    else:
        # ES-EKF Filter with Active 3D Velocity States
        esekf = ESEKFNavigationFilter(dt=dt)
        esekf.initialize(
            init_lat_ned=0.0,
            init_lon_ned=0.0,
            init_speed_ms=init_speed_ms,
            init_heading_rad=math.radians(init_heading_deg)
        )

        for k in range(n):
            a_b = acc_raw[k]
            w_b = gyro_raw[k]
            a_v, w_v = alignment.transform_imu(a_b, w_b)

            # ES-EKF strapdown propagation
            esekf.predict(a_v, w_v, dt=dt)

            global_k = global_start_idx + k
            if nhc_mode == "fixed_A":
                v_lat, v_vert = FIXED_VAR_LAT_A, FIXED_VAR_VERT_A
            elif nhc_mode == "mean_C":
                v_lat, v_vert = MEAN_TRAIN_VAR_LAT_C, MEAN_TRAIN_VAR_VERT_C
            elif nhc_mode == "adaptive_B":
                if global_k in noise_lookup:
                    v_lat, v_vert = noise_lookup[global_k]
                else:
                    v_lat, v_vert = MEAN_TRAIN_VAR_LAT_C, MEAN_TRAIN_VAR_VERT_C
            else:
                v_lat, v_vert = FIXED_VAR_LAT_A, FIXED_VAR_VERT_A

            var_lat_hist.append(v_lat)
            var_vert_hist.append(v_vert)

            # Active NHC update
            esekf.update_nhc(var_lat=v_lat, var_vert=v_vert)

            # VelocityNet update
            if global_k in vnet_lookup and (k % 5 == 0):
                raw_d, ev_class, raw_sig = vnet_lookup[global_k]
                scaled_d = raw_d / 2.0
                v_fwd = scaled_d / 2.0
                var = max((raw_sig / 2.0)**2, 0.04)

                if ev_class == 0 or raw_d < 0.1:
                    esekf.update_zupt()
                else:
                    R_mat = quat_to_rot_matrix(esekf.q)
                    fwd_vec = R_mat[:, 0]
                    v_meas = fwd_vec * v_fwd
                    R_cov = np.eye(3) * var
                    H = np.zeros((3, esekf.dim_error))
                    H[:, 3:6] = np.eye(3)
                    y = v_meas - esekf.v
                    esekf.update_error_state(y, H, R_cov, chi2_gate=False)

            est_pN[k] = esekf.p[0]
            est_pE[k] = esekf.p[1]
            est_speed[k] = np.linalg.norm(esekf.v[:2])

        est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
        return est_lat, est_lon, est_speed, var_lat_hist, var_vert_hist, nis_hist


def classify_outage_regime(
    df_sub: pd.DataFrame,
    dt: float = config.TARGET_DT
) -> str:
    """
    Classifies the dominant driving condition during the outage segment:
      - STRAIGHT: low yaw rate (< 1.5 deg/s) and steady speed
      - TURNING: significant curve or turns (mean |yaw_rate| > 3.0 deg/s)
      - ACCELERATION: positive longitudinal accel (> 0.8 m/s^2)
      - BRAKING: negative longitudinal accel (< -0.8 m/s^2)
      - ROUGH_ROAD: high vertical acceleration variance (std(acc_z) > 1.2 m/s^2)
    """
    if config.COL_TRUE_YAW_RATE in df_sub.columns:
        mean_yaw_rate = float(np.mean(np.abs(df_sub[config.COL_TRUE_YAW_RATE])))
    elif config.COL_GYRO_Z in df_sub.columns:
        mean_yaw_rate = float(np.mean(np.abs(df_sub[config.COL_GYRO_Z]))) * config.RAD_TO_DEG
    else:
        mean_yaw_rate = 0.0

    if config.COL_ACC_Z in df_sub.columns:
        std_acc_z = float(np.std(df_sub[config.COL_ACC_Z]))
    else:
        std_acc_z = 0.0

    if config.COL_TRUE_SPEED_MS in df_sub.columns:
        spd = df_sub[config.COL_TRUE_SPEED_MS].values
        acc = (spd[-1] - spd[0]) / max(len(spd) * dt, 1.0)
    else:
        acc = 0.0

    if std_acc_z > 1.2:
        return "ROUGH_ROAD"
    if mean_yaw_rate > 3.0:
        return "TURNING"
    if acc > 0.8:
        return "ACCELERATION"
    if acc < -0.8:
        return "BRAKING"
    return "STRAIGHT"


def run_phase9_evaluation():
    logger.info("=" * 80)
    logger.info("PHASE 9: AI-ADAPTIVE NHC BENCHMARK ON 56 HELD-OUT OUTAGES")
    logger.info("=" * 80)

    # 1. Load frozen Phase 8A VelocityNet
    vnet_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    if not vnet_path.exists():
        logger.error(f"VelocityNet model {vnet_path} not found!")
        return

    vnet_model = VelocityNet4s(in_channels=6, num_events=5)
    vnet_model.load_state_dict(torch.load(vnet_path, map_location="cpu"))
    vnet_model.eval()

    # 2. Load trained Phase 9 NoiseNet
    noisenet_path = config.MODELS_DIR / "phase9_noisenet.pt"
    if not noisenet_path.exists():
        logger.error(f"NoiseNet model {noisenet_path} not found!")
        return

    noisenet_model = NoiseNet(in_channels=6)
    noisenet_model.load_state_dict(torch.load(noisenet_path, map_location="cpu"))
    noisenet_model.eval()

    # 3. Identify all held-out test files
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

        # Pre-calibrate run alignment using warmup
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

        # Batch compute all windows
        needed_indices = set(k for k in range(0, n, 5) if k >= 39)
        for _, row in df_outages.iterrows():
            mask = (df_sim["outage_id"] == int(row["outage_id"]))
            s_idx = mask.idxmax()
            sub_len = len(df.loc[mask])
            for k in range(0, sub_len):
                if s_idx + k >= 39:
                    needed_indices.add(s_idx + k)

        vnet_lookup, noise_lookup = compute_batched_predictions(
            vnet_model=vnet_model,
            noisenet_model=noisenet_model,
            seq_data=seq_data,
            needed_step_indices=list(needed_indices),
            window_len=40,
            batch_size=2048,
            device="cpu"
        )

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
            else:
                init_lat = float(df[config.COL_TRUE_LAT].iloc[outage_start_idx])
                init_lon = float(df[config.COL_TRUE_LON].iloc[outage_start_idx])
                init_spd = float(df[config.COL_TRUE_SPEED_MS].iloc[outage_start_idx])
                init_hdg = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx])

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            regime = classify_outage_regime(df_sub)

            # =========================================================================
            # 1. 7-STATE UKF: Baseline A (Fixed NHC)
            # =========================================================================
            lat_a, lon_a, spd_a, vlat_a, vvert_a, nis_a = run_single_outage_phase9(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                vnet_lookup=vnet_lookup, noise_lookup=noise_lookup,
                global_start_idx=outage_start_idx, alignment=run_align,
                nhc_mode="fixed_A", filter_type="ukf"
            )
            score_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # =========================================================================
            # 2. 7-STATE UKF: Experiment B (AI-Adaptive NoiseNet NHC)
            # =========================================================================
            lat_b, lon_b, spd_b, vlat_b, vvert_b, nis_b = run_single_outage_phase9(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                vnet_lookup=vnet_lookup, noise_lookup=noise_lookup,
                global_start_idx=outage_start_idx, alignment=run_align,
                nhc_mode="adaptive_B", filter_type="ukf"
            )
            score_b = score_outage_segment(lat_b, lon_b, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # =========================================================================
            # 3. 7-STATE UKF: Ablation C (Mean Train Covariance NHC)
            # =========================================================================
            lat_c, lon_c, spd_c, vlat_c, vvert_c, nis_c = run_single_outage_phase9(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                vnet_lookup=vnet_lookup, noise_lookup=noise_lookup,
                global_start_idx=outage_start_idx, alignment=run_align,
                nhc_mode="mean_C", filter_type="ukf"
            )
            score_c = score_outage_segment(lat_c, lon_c, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # =========================================================================
            # 4. 15-STATE ES-EKF: Diagnostic Control (Active 3D Velocity States)
            # =========================================================================
            lat_es_a, lon_es_a, _, _, _, _ = run_single_outage_phase9(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                vnet_lookup=vnet_lookup, noise_lookup=noise_lookup,
                global_start_idx=outage_start_idx, alignment=run_align,
                nhc_mode="fixed_A", filter_type="esekf"
            )
            score_es_a = score_outage_segment(lat_es_a, lon_es_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            lat_es_b, lon_es_b, _, _, _, _ = run_single_outage_phase9(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                vnet_lookup=vnet_lookup, noise_lookup=noise_lookup,
                global_start_idx=outage_start_idx, alignment=run_align,
                nhc_mode="adaptive_B", filter_type="esekf"
            )
            score_es_b = score_outage_segment(lat_es_b, lon_es_b, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            lat_es_c, lon_es_c, _, _, _, _ = run_single_outage_phase9(
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                vnet_lookup=vnet_lookup, noise_lookup=noise_lookup,
                global_start_idx=outage_start_idx, alignment=run_align,
                nhc_mode="mean_C", filter_type="esekf"
            )
            score_es_c = score_outage_segment(lat_es_c, lon_es_c, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            rec = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "distance_m": round(dist_gt, 2),
                "regime": regime,
                # Covariance Stats
                "min_sigma2_lat": round(float(np.min(vlat_b)), 5),
                "median_sigma2_lat": round(float(np.median(vlat_b)), 5),
                "max_sigma2_lat": round(float(np.max(vlat_b)), 5),
                "min_sigma2_vert": round(float(np.min(vvert_b)), 5),
                "median_sigma2_vert": round(float(np.median(vvert_b)), 5),
                "max_sigma2_vert": round(float(np.max(vvert_b)), 5),
                # UKF Baseline A
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                "nis_a_mean": round(float(np.mean(nis_a)), 5),
                # UKF Experiment B (NoiseNet)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
                "nis_b_mean": round(float(np.mean(nis_b)), 5),
                # UKF Ablation C (Mean Train Covariance)
                "fpe_c_m": round(float(score_c["final_pos_error_m"]), 2),
                "drift_c_pct": round(float(score_c["pct_of_distance"]), 2),
                # ES-EKF Active Comparison
                "es_fpe_a_m": round(float(score_es_a["final_pos_error_m"]), 2),
                "es_drift_a_pct": round(float(score_es_a["pct_of_distance"]), 2),
                "es_fpe_b_m": round(float(score_es_b["final_pos_error_m"]), 2),
                "es_drift_b_pct": round(float(score_es_b["pct_of_distance"]), 2),
                "es_fpe_c_m": round(float(score_es_c["final_pos_error_m"]), 2),
                "es_drift_c_pct": round(float(score_es_c["pct_of_distance"]), 2),
            }
            records.append(rec)

    df_res = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase9_results.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 9 results to: {out_csv} ({len(df_res)} total outages)")

    # Compute Summary Tables
    print("\n" + "=" * 90)
    print("                    PHASE 9 AI-ADAPTIVE NHC BENCHMARK SUMMARY")
    print("=" * 90)

    print(f"\n--- Primary Architecture: 7-State UKF (N = {len(df_res)}) ---")
    print(f"{'Metric':<25} | {'Baseline A (Fixed)':<20} | {'Experiment B (NoiseNet)':<22} | {'Ablation C (Mean Train)':<22}")
    print("-" * 95)
    print(f"{'Median Drift %':<25} | {df_res['drift_a_pct'].median():<20.2f} | {df_res['drift_b_pct'].median():<22.2f} | {df_res['drift_c_pct'].median():<22.2f}")
    print(f"{'Mean Drift %':<25} | {df_res['drift_a_pct'].mean():<20.2f} | {df_res['drift_b_pct'].mean():<22.2f} | {df_res['drift_c_pct'].mean():<22.2f}")
    print(f"{'P25 Drift %':<25} | {df_res['drift_a_pct'].quantile(0.25):<20.2f} | {df_res['drift_b_pct'].quantile(0.25):<22.2f} | {df_res['drift_c_pct'].quantile(0.25):<22.2f}")
    print(f"{'P75 Drift %':<25} | {df_res['drift_a_pct'].quantile(0.75):<20.2f} | {df_res['drift_b_pct'].quantile(0.75):<22.2f} | {df_res['drift_c_pct'].quantile(0.75):<22.2f}")
    print(f"{'Median FPE (m)':<25} | {df_res['fpe_a_m'].median():<20.2f} | {df_res['fpe_b_m'].median():<22.2f} | {df_res['fpe_c_m'].median():<22.2f}")
    print(f"{'Mean FPE (m)':<25} | {df_res['fpe_a_m'].mean():<20.2f} | {df_res['fpe_b_m'].mean():<22.2f} | {df_res['fpe_c_m'].mean():<22.2f}")
    print(f"{'Median Along-Track (m)':<25} | {df_res['along_a_m'].median():<20.2f} | {df_res['along_b_m'].median():<22.2f} | -")
    print(f"{'Median Cross-Track (m)':<25} | {df_res['cross_a_m'].median():<20.2f} | {df_res['cross_b_m'].median():<22.2f} | -")
    print(f"{'Median Heading Err (deg)':<25} | {df_res['hdg_a_deg'].median():<20.2f} | {df_res['hdg_b_deg'].median():<22.2f} | -")
    print(f"{'Outages < 10% Drift':<25} | {(df_res['drift_a_pct'] < 10.0).sum():<20} | {(df_res['drift_b_pct'] < 10.0).sum():<22} | {(df_res['drift_c_pct'] < 10.0).sum():<22}")

    # Difference between B and A
    diff_fpe = df_res["fpe_b_m"] - df_res["fpe_a_m"]
    improved = (diff_fpe < -0.05).sum()
    worsened = (diff_fpe > 0.05).sum()
    unchanged = ((diff_fpe >= -0.05) & (diff_fpe <= 0.05)).sum()

    print(f"\nUKF Delta Breakdown (B vs A): Improved: {improved}, Worsened: {worsened}, Unchanged: {unchanged}")

    print(f"\n--- Secondary Diagnostic: 15-State ES-EKF with Active Velocity States (N = {len(df_res)}) ---")
    print(f"{'Metric':<25} | {'ES-EKF Baseline A':<20} | {'ES-EKF NoiseNet B':<22} | {'ES-EKF Mean C':<22}")
    print("-" * 95)
    print(f"{'Median Drift %':<25} | {df_res['es_drift_a_pct'].median():<20.2f} | {df_res['es_drift_b_pct'].median():<22.2f} | {df_res['es_drift_c_pct'].median():<22.2f}")
    print(f"{'Mean Drift %':<25} | {df_res['es_drift_a_pct'].mean():<20.2f} | {df_res['es_drift_b_pct'].mean():<22.2f} | {df_res['es_drift_c_pct'].mean():<22.2f}")
    print(f"{'Median FPE (m)':<25} | {df_res['es_fpe_a_m'].median():<20.2f} | {df_res['es_fpe_b_m'].median():<22.2f} | {df_res['es_fpe_c_m'].median():<22.2f}")
    es_improved = (df_res['es_fpe_b_m'] < df_res['es_fpe_a_m'] - 0.5).sum()
    es_worsened = (df_res['es_fpe_b_m'] > df_res['es_fpe_a_m'] + 0.5).sum()
    es_unchanged = len(df_res) - es_improved - es_worsened
    print(f"ES-EKF Delta Breakdown (B vs A): Improved: {es_improved}, Worsened: {es_worsened}, Unchanged: {es_unchanged}")

    # Per duration
    print("\n--- Per-Duration Results (7-State UKF) ---")
    for dur in sorted(df_res["duration_s"].unique()):
        sub = df_res[df_res["duration_s"] == dur]
        print(f"Duration {dur:3d}s (N={len(sub):2d}): Baseline A Drift={sub['drift_a_pct'].median():.2f}%, NoiseNet B Drift={sub['drift_b_pct'].median():.2f}%, FPE A={sub['fpe_a_m'].median():.2f}m, FPE B={sub['fpe_b_m'].median():.2f}m")

    # Per regime
    print("\n--- Per-Regime Results (7-State UKF) ---")
    for reg in sorted(df_res["regime"].unique()):
        sub = df_res[df_res["regime"] == reg]
        print(f"Regime {reg:<15} (N={len(sub):2d}): Baseline A Drift={sub['drift_a_pct'].median():.2f}%, NoiseNet B Drift={sub['drift_b_pct'].median():.2f}%, FPE A={sub['fpe_a_m'].median():.2f}m, FPE B={sub['fpe_b_m'].median():.2f}m")

    # Covariance statistics
    print("\n--- NoiseNet Predicted Covariance Statistics on Test Set (m^2/s^2) ---")
    print(f"sigma^2_lat  : Min={df_res['min_sigma2_lat'].min():.5f}, Median={df_res['median_sigma2_lat'].median():.5f}, Max={df_res['max_sigma2_lat'].max():.5f}")
    print(f"sigma^2_vert : Min={df_res['min_sigma2_vert'].min():.5f}, Median={df_res['median_sigma2_vert'].median():.5f}, Max={df_res['max_sigma2_vert'].max():.5f}")


if __name__ == "__main__":
    run_phase9_evaluation()
