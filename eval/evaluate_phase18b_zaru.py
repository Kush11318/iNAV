"""
Phase 18B: ZARU / Gyro-Bias Observability Experiment.

Evaluates:
  1. Causal standstill detection and empirical smartphone IMU noise floor.
  2. Test 1: Gyro bias observability and covariance reduction during standstills.
  3. Test 2: Post-stop heading drift at 10s, 30s, 60s, 120s, 180s horizons.
  4. Test 3: 56-outage navigation benchmark comparing:
       - System A: Existing baseline (CAN speed + road-normal HMM/OSM map constraint)
       - System B: Baseline + ZARU (empirically derived R_ZARU = 3.2e-4)
       - System C: Baseline + Sustained ZUPT + ZARU
     Also analyzes pre-outage stop coverage (26 with stop vs 30 without stop).
  5. Test 4: Forensic audit of the 7 severe Phase 18A failure cases.
  6. Test 5: Critical failure test for false stationary detections.

Strict isolation:
  - Production code remains completely untouched.
  - Zero ground-truth leakage during simulated outages.
  - No map heading fusion, no differential wheel yaw, no neural model changes.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import time
import json
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
from modules.map_matcher import load_road_graph_from_osm_json, FixedLagHMMMapMatcher, RoadGraph
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase18B_ZARU")

ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")
RESULTS_DIR = config.BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. Causal Standstill Detector
# ==============================================================================
class CausalStandstillDetector:
    """
    Strictly causal stationary detector.
    Does NOT use future samples or ground truth.
    Requires sustained multi-sensor stability for >= min_duration_samples.
    """
    def __init__(
        self,
        speed_thresh_mps: float = 0.3,
        gyro_std_thresh_rads: float = 0.04,
        gyro_rate_thresh_rads: float = 0.05,  # 2.86 deg/s
        acc_std_thresh_ms2: float = 0.25,
        window_samples: int = 10,             # 1.0s at 10 Hz
        min_duration_samples: int = 20,       # 2.0s at 10 Hz
        dt: float = 0.1
    ):
        self.v_thresh = speed_thresh_mps
        self.gyro_thresh = gyro_std_thresh_rads
        self.rate_thresh = gyro_rate_thresh_rads
        self.acc_thresh = acc_std_thresh_ms2
        self.window = window_samples
        self.min_dur = min_duration_samples
        self.dt = dt

        self.buf_v: List[float] = []
        self.buf_gx: List[float] = []
        self.buf_gy: List[float] = []
        self.buf_gz: List[float] = []
        self.buf_amag: List[float] = []

        self.consecutive_stationary = 0
        self.is_stationary = False

    def reset(self):
        self.buf_v.clear()
        self.buf_gx.clear()
        self.buf_gy.clear()
        self.buf_gz.clear()
        self.buf_amag.clear()
        self.consecutive_stationary = 0
        self.is_stationary = False

    def update(
        self,
        v_can: float,
        gx: float,
        gy: float,
        gz: float,
        ax: float,
        ay: float,
        az: float
    ) -> Tuple[bool, bool]:
        """
        Returns (is_stationary_now, is_transition_to_stationary)
        """
        amag = math.sqrt(ax**2 + ay**2 + az**2)
        self.buf_v.append(v_can)
        self.buf_gx.append(gx)
        self.buf_gy.append(gy)
        self.buf_gz.append(gz)
        self.buf_amag.append(amag)

        if len(self.buf_v) > self.window:
            self.buf_v.pop(0)
            self.buf_gx.pop(0)
            self.buf_gy.pop(0)
            self.buf_gz.pop(0)
            self.buf_amag.pop(0)

        if len(self.buf_v) < self.window:
            return False, False

        v_now = abs(v_can)
        rate_ok = (abs(gx) < self.rate_thresh and
                   abs(gy) < self.rate_thresh and
                   abs(gz) < self.rate_thresh)
        std_gx = float(np.std(self.buf_gx))
        std_gy = float(np.std(self.buf_gy))
        std_gz = float(np.std(self.buf_gz))
        std_a = float(np.std(self.buf_amag))

        instant_ok = (
            v_now < self.v_thresh and
            rate_ok and
            std_gx < self.gyro_thresh and
            std_gy < self.gyro_thresh and
            std_gz < self.gyro_thresh and
            std_a < self.acc_thresh
        )

        prev_state = self.is_stationary

        if instant_ok:
            self.consecutive_stationary += 1
            if self.consecutive_stationary >= self.min_dur:
                self.is_stationary = True
        else:
            self.consecutive_stationary = 0
            self.is_stationary = False

        transition_to_stop = (not prev_state and self.is_stationary)
        return self.is_stationary, transition_to_stop


# ==============================================================================
# 2. Experimental UKF with ZARU Measurement Update
# ==============================================================================
class ZARU_CANFusionUKF(CANFusionUKF):
    """
    Experimental 7-State UKF with Zero Angular Rate Updates (ZARU).
    State: [pN, pE, v_fwd, psi, b_g, b_a, k]
    Preserves exact 7-state structure and Joseph-form covariance updates.
    """
    def __init__(self, dt: float = 0.1, r_zaru: float = 3.2e-4):
        super().__init__(dt=dt)
        self.r_zaru = r_zaru
        self.total_zaru_updates = 0
        self.accepted_zaru_updates = 0
        self.rejected_zaru_updates = 0
        self.zaru_nis_history: List[float] = []

    def update_zaru(
        self,
        omega_z_meas: float,
        nis_gate: float = 16.0
    ) -> Tuple[bool, float]:
        """
        ZARU measurement update:
        During verified standstill:
          true angular rate ≈ 0
          z_ZARU = 0
          Measurement model: h_ZARU(s) = omega_z_meas - s[4]
          Innovation: y = 0 - (omega_z_meas - x[4]) = x[4] - omega_z_meas
        """
        self.total_zaru_updates += 1

        def h_zaru(s: np.ndarray) -> np.ndarray:
            return np.array([omega_z_meas - s[4]])

        z = np.array([0.0])
        R = np.array([[self.r_zaru]])

        success, nis = self.update_measurement(z, h_zaru, R, nis_gate=nis_gate)
        if success:
            self.accepted_zaru_updates += 1
            self.zaru_nis_history.append(nis)
        else:
            self.rejected_zaru_updates += 1
        return success, nis


# ==============================================================================
# 3. Trajectory Preloading & Alignment
# ==============================================================================
def get_benchmark_trajectories():
    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)
    return test_files


def run_full_experiment():
    logger.info("=" * 90)
    logger.info("PHASE 18B: ZARU / GYRO-BIAS OBSERVABILITY EXPERIMENT")
    logger.info("=" * 90)

    test_files = get_benchmark_trajectories()
    logger.info(f"Loaded {len(test_files)} trajectories for evaluation.")

    # 1. Load Canonical OSM Road Graph
    osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    logger.info(f"Loading canonical RoadGraph from {osm_path}...")
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)
    logger.info(f"RoadGraph loaded ({len(graph.nodes)} nodes, {len(graph.edges)} edges).")

    # ==========================================================================
    # STEP 1 & 2: STANDSTILL DISCOVERY & ACTUAL GYRO NOISE AUDIT
    # ==========================================================================
    logger.info("\n--- STEP 1 & 2: DATA DISCOVERY & STANDSTILL GYRO NOISE AUDIT ---")
    all_stops = []
    false_positives = []

    candidate_count = 0
    accepted_count = 0

    runs_data = {}

    for tf in test_files:
        run_name = tf.stem.replace("sync_", "")
        df = pd.read_parquet(tf)
        total_dur = len(df) * config.TARGET_DT
        if total_dur < 30.0:
            continue

        # Alignment
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

        # Transform all IMU to vehicle frame
        acc_raw = df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
        gyro_raw = df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
        R_mat = run_align.result.R_b_to_v
        acc_v = acc_raw @ R_mat.T
        gyro_v = gyro_raw @ R_mat.T

        w_rl = df[config.COL_TRUE_WHEEL_RL].values
        w_rr = df[config.COL_TRUE_WHEEL_RR].values
        v_can = 0.5 * (w_rl + w_rr) * 0.2776

        gt_spd = df[config.COL_TRUE_SPEED_MS].values if config.COL_TRUE_SPEED_MS in df.columns else v_can
        gt_yaw = df["gyro_z"].values if "gyro_z" in df.columns else np.zeros(len(df))

        detector = CausalStandstillDetector()

        stop_active = False
        stop_start = 0
        stop_indices = []

        for k in range(len(df)):
            is_stop, is_trans = detector.update(
                v_can[k],
                gyro_v[k, 0], gyro_v[k, 1], gyro_v[k, 2],
                acc_v[k, 0], acc_v[k, 1], acc_v[k, 2]
            )

            # False positive check (vehicle moving > 1.0 m/s or rotating > 5 deg/s)
            if is_stop and (gt_spd[k] > 1.0 or abs(gyro_v[k, 2]) > math.radians(5.0)):
                false_positives.append({
                    "run": run_name,
                    "t": k * config.TARGET_DT,
                    "v_can": float(v_can[k]),
                    "gt_spd": float(gt_spd[k]),
                    "wv_z_deg": float(np.degrees(gyro_v[k, 2]))
                })

            if is_trans:
                stop_active = True
                stop_start = k - detector.min_dur
                stop_indices = list(range(stop_start, k + 1))
            elif is_stop:
                stop_indices.append(k)
            elif stop_active and not is_stop:
                stop_active = False
                dur = len(stop_indices) * config.TARGET_DT
                sub_wz = gyro_v[stop_indices, 2]
                sub_wx = gyro_v[stop_indices, 0]
                sub_wy = gyro_v[stop_indices, 1]
                sub_a = np.linalg.norm(acc_v[stop_indices], axis=1)
                sub_v = v_can[stop_indices]

                all_stops.append({
                    "stop_id": len(all_stops) + 1,
                    "run": run_name,
                    "start_idx": stop_indices[0],
                    "end_idx": stop_indices[-1],
                    "start_t": stop_indices[0] * config.TARGET_DT,
                    "end_t": stop_indices[-1] * config.TARGET_DT,
                    "dur_s": dur,
                    "n_samples": len(stop_indices),
                    "mean_v_can": float(np.mean(sub_v)),
                    "max_v_can": float(np.max(sub_v)),
                    "mean_wz_rad": float(np.mean(sub_wz)),
                    "std_wz_rad": float(np.std(sub_wz)),
                    "mean_wz_deg": float(np.degrees(np.mean(sub_wz))),
                    "std_wz_deg": float(np.degrees(np.std(sub_wz))),
                    "mean_wx_rad": float(np.mean(sub_wx)),
                    "std_wx_rad": float(np.std(sub_wx)),
                    "mean_wy_rad": float(np.mean(sub_wy)),
                    "std_wy_rad": float(np.std(sub_wy)),
                    "mean_amag": float(np.mean(sub_a)),
                    "std_amag": float(np.std(sub_a))
                })
                stop_indices = []

        runs_data[run_name] = {
            "df": df,
            "acc_v": acc_v,
            "gyro_v": gyro_v,
            "v_can": v_can,
            "alignment": run_align,
            "omega_rl": w_rl,
            "omega_rr": w_rr
        }

    df_stops = pd.DataFrame(all_stops)
    logger.info(f"Total accepted standstill events: {len(df_stops)}")
    logger.info(f"Total false positive standstill detections: {len(false_positives)}")

    # Calculate gyro noise floor statistics
    median_std_wz = df_stops['std_wz_rad'].median()
    mean_std_wz = df_stops['std_wz_rad'].mean()
    stat_r_zaru = float(mean_std_wz**2)

    logger.info(f"Median stationary gyro noise sigma_gz: {median_std_wz:.6f} rad/s ({np.degrees(median_std_wz):.4f} deg/s)")
    logger.info(f"Mean stationary gyro noise sigma_gz:   {mean_std_wz:.6f} rad/s ({np.degrees(mean_std_wz):.4f} deg/s)")
    logger.info(f"Statistically Justified R_ZARU = {stat_r_zaru:.8e} (rad/s)^2 (vs naive 1e-4)")

    # Save stops table
    df_stops.to_csv(BASE_DIR / "eval" / "phase18b_standstill_events.csv", index=False)

    # ==========================================================================
    # STEP 5: TEST 1 — DOES ZARU ACTUALLY ESTIMATE BIAS? (CALIBRATION ANALYSIS)
    # ==========================================================================
    logger.info("\n--- TEST 1: STANDSTILL GYRO BIAS OBSERVABILITY & COVARIANCE REDUCTION ---")
    calibration_records = []

    # Check if Test 1 and Test 2 results already exist
    calib_csv = BASE_DIR / "eval" / "phase18b_calibration_results.csv"
    drift_csv = BASE_DIR / "eval" / "phase18b_post_stop_heading_drift.csv"

    if calib_csv.exists():
        logger.info(f"Loading existing Test 1 calibration results from {calib_csv.name}...")
        df_calib = pd.read_csv(calib_csv)
    else:
        for idx, srow in df_stops.iterrows():
            run_name = srow["run"]
            rdata = runs_data[run_name]
            start_k = int(srow["start_idx"])
            end_k = int(srow["end_idx"])
            dur = srow["dur_s"]

            # Run UKF for 10 seconds before stop to establish prior filter state
            pre_k = max(0, start_k - 100)
            df_run = rdata["df"]
            init_lat = float(df_run[config.COL_TRUE_LAT].iloc[pre_k])
            init_lon = float(df_run[config.COL_TRUE_LON].iloc[pre_k])
            init_spd = float(rdata["v_can"][pre_k])
            init_hdg = math.radians(float(df_run[config.COL_TRUE_HEADING].iloc[pre_k])) if config.COL_TRUE_HEADING in df_run.columns else 0.0

            # Initialize UKF
            ukf = ZARU_CANFusionUKF(dt=0.1, r_zaru=stat_r_zaru)
            ukf.initialize(
                init_lat=init_lat,
                init_lon=init_lon,
                init_speed_ms=init_spd,
                init_heading_rad=init_hdg,
                allow_bias_learning=True
            )

            # Propagate through pre-stop period
            for k in range(pre_k, start_k):
                ukf.predict(acc_fwd=rdata["acc_v"][k, 0], gyro_yaw=rdata["gyro_v"][k, 2], dt=0.1)
                ukf.update_can_speed(v_can=rdata["v_can"][k])

            bg_before = float(ukf.x[4])
            P_bg_before = float(ukf.P[4, 4])

            # Apply ZARU during standstill
            for k in range(start_k, end_k + 1):
                ukf.predict(acc_fwd=rdata["acc_v"][k, 0], gyro_yaw=rdata["gyro_v"][k, 2], dt=0.1)
                ukf.update_can_speed(v_can=rdata["v_can"][k])
                ukf.update_zaru(omega_z_meas=rdata["gyro_v"][k, 2])

            bg_after = float(ukf.x[4])
            P_bg_after = float(ukf.P[4, 4])
            cov_reduction = P_bg_before / max(P_bg_after, 1e-12)
            corr_mag = abs(bg_after - bg_before)

            calibration_records.append({
                "stop_id": srow["stop_id"],
                "run": run_name,
                "start_t": srow["start_t"],
                "dur_s": dur,
                "mean_wz_deg": srow["mean_wz_deg"],
                "bg_before_rad": bg_before,
                "bg_after_rad": bg_after,
                "bg_before_deg": math.degrees(bg_before),
                "bg_after_deg": math.degrees(bg_after),
                "P_bg_before": P_bg_before,
                "P_bg_after": P_bg_after,
                "sigma_bg_before_deg": math.degrees(math.sqrt(P_bg_before)),
                "sigma_bg_after_deg": math.degrees(math.sqrt(P_bg_after)),
                "bias_corr_mag_deg": math.degrees(corr_mag),
                "cov_reduction_factor": cov_reduction
            })

        df_calib = pd.DataFrame(calibration_records)
        df_calib.to_csv(calib_csv, index=False)

    logger.info(f"Mean bias correction magnitude: {df_calib['bias_corr_mag_deg'].mean():.4f} deg/s")
    logger.info(f"Median covariance reduction factor: {df_calib['cov_reduction_factor'].median():.1f}x")
    logger.info(f"Mean sigma_bg before stop: {df_calib['sigma_bg_before_deg'].mean():.4f} deg/s -> after stop: {df_calib['sigma_bg_after_deg'].mean():.4f} deg/s")

    # ==========================================================================
    # STEP 6: TEST 2 — POST-STOP HEADING DRIFT HORIZON ANALYSIS
    # ==========================================================================
    logger.info("\n--- TEST 2: POST-STOP HEADING DRIFT HORIZON ANALYSIS ---")
    horizons = [10, 30, 60, 120, 180]

    if drift_csv.exists():
        logger.info(f"Loading existing Test 2 post-stop heading drift from {drift_csv.name}...")
        df_drift = pd.read_csv(drift_csv)
    else:
        drift_records = []
        for idx, srow in df_stops.iterrows():
            run_name = srow["run"]
            rdata = runs_data[run_name]
            df_run = rdata["df"]
            start_k = int(srow["start_idx"])
            end_k = int(srow["end_idx"])
            n_total = len(df_run)

            if end_k + 300 >= n_total:
                continue

            pre_k = max(0, start_k - 100)
            init_lat = float(df_run[config.COL_TRUE_LAT].iloc[pre_k])
            init_lon = float(df_run[config.COL_TRUE_LON].iloc[pre_k])
            init_spd = float(rdata["v_can"][pre_k])
            init_hdg = math.radians(float(df_run[config.COL_TRUE_HEADING].iloc[pre_k])) if config.COL_TRUE_HEADING in df_run.columns else 0.0

            max_post_k = min(n_total - 1, end_k + 1800)
            sim_systems = ["A", "B", "C"]
            sim_ukfs = {}
            for s in sim_systems:
                u = ZARU_CANFusionUKF(dt=0.1, r_zaru=stat_r_zaru)
                u.initialize(init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_spd, init_heading_rad=init_hdg, allow_bias_learning=True)
                sim_ukfs[s] = u

            for k in range(pre_k, start_k):
                for s in sim_systems:
                    sim_ukfs[s].predict(acc_fwd=rdata["acc_v"][k, 0], gyro_yaw=rdata["gyro_v"][k, 2], dt=0.1)
                    sim_ukfs[s].update_can_speed(v_can=rdata["v_can"][k])

            for k in range(start_k, end_k + 1):
                for s in sim_systems:
                    sim_ukfs[s].predict(acc_fwd=rdata["acc_v"][k, 0], gyro_yaw=rdata["gyro_v"][k, 2], dt=0.1)
                    if s == "A":
                        sim_ukfs[s].update_can_speed(v_can=rdata["v_can"][k])
                    elif s == "B":
                        sim_ukfs[s].update_can_speed(v_can=rdata["v_can"][k])
                        sim_ukfs[s].update_zaru(omega_z_meas=rdata["gyro_v"][k, 2])
                    elif s == "C":
                        sim_ukfs[s].update_zupt(gyro_reading=rdata["gyro_v"][k, 2])
                        sim_ukfs[s].update_zaru(omega_z_meas=rdata["gyro_v"][k, 2])

            hdg_errors = {s: {} for s in sim_systems}
            gt_headings = df_run[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_run.columns else None

            psi_stops = {s: float(sim_ukfs[s].x[3]) for s in sim_systems}
            bg_stops = {s: float(sim_ukfs[s].x[4]) for s in sim_systems}
            wz_post = rdata["gyro_v"][end_k + 1 : max_post_k + 1, 2]

            for h in horizons:
                h_samples = int(round(h / 0.1))
                if h_samples <= len(wz_post):
                    k_h = end_k + h_samples
                    cum_wz = float(np.sum(wz_post[:h_samples])) * 0.1
                    gt_h = gt_headings[k_h] if gt_headings is not None else 0.0

                    for s in sim_systems:
                        est_psi = (psi_stops[s] + cum_wz - bg_stops[s] * h) % (2.0 * np.pi)
                        est_hdg_deg = math.degrees(est_psi) % 360.0
                        diff_h = abs((est_hdg_deg - gt_h + 180.0) % 360.0 - 180.0)
                        hdg_errors[s][h] = diff_h

            rec = {
                "stop_id": srow["stop_id"],
                "run": run_name,
                "stop_dur_s": srow["dur_s"],
                "max_eval_horizon_s": (max_post_k - end_k) * 0.1,
                "bg_A_deg": math.degrees(sim_ukfs["A"].x[4]),
                "bg_B_deg": math.degrees(sim_ukfs["B"].x[4]),
                "bg_C_deg": math.degrees(sim_ukfs["C"].x[4]),
            }
            for h in horizons:
                rec[f"err_A_{h}s"] = hdg_errors["A"].get(h, np.nan)
                rec[f"err_B_{h}s"] = hdg_errors["B"].get(h, np.nan)
                rec[f"err_C_{h}s"] = hdg_errors["C"].get(h, np.nan)
            drift_records.append(rec)

        df_drift = pd.DataFrame(drift_records)
        df_drift.to_csv(drift_csv, index=False)

    logger.info(f"Evaluated post-stop heading drift across {len(df_drift)} driving intervals.")
    for h in horizons:
        val_A = df_drift[f"err_A_{h}s"].median()
        val_B = df_drift[f"err_B_{h}s"].median()
        val_C = df_drift[f"err_C_{h}s"].median()
        logger.info(f"Median Heading Error at {h:3d}s: System A = {val_A:.2f}° | System B (ZARU) = {val_B:.2f}° | System C (ZUPT+ZARU) = {val_C:.2f}°")

    # ==========================================================================
    # STEP 7: TEST 3 — 56-OUTAGE NAVIGATION BENCHMARK
    # ==========================================================================
    logger.info("\n--- TEST 3: 56-OUTAGE NAVIGATION BENCHMARK ---")
    benchmark_records = []

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
        rdata = runs_data[run_name]
        run_stops = df_stops[df_stops["run"] == run_name]

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            dist_gt = float(row["distance_travelled_m"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
            sub_len = len(df_sub)
            if sub_len == 0:
                continue

            global_start_idx = df_sub.index[0]
            outage_start_t = global_start_idx * 0.1

            # Check pre-outage standstill events
            pre_stops = run_stops[run_stops["end_t"] <= outage_start_t]
            has_pre_stop = len(pre_stops) > 0
            last_stop_age = (outage_start_t - pre_stops["end_t"].max()) if has_pre_stop else None

            # Check in-outage standstill events
            outage_end_t = outage_start_t + dur
            in_stops = run_stops[(run_stops["start_t"] < outage_end_t) & (run_stops["end_t"] > outage_start_t)]
            has_in_stop = len(in_stops) > 0

            # Ground truth initial states
            init_lat = float(df.loc[global_start_idx, config.COL_TRUE_LAT])
            init_lon = float(df.loc[global_start_idx, config.COL_TRUE_LON])
            init_speed_ms = float(df.loc[global_start_idx, config.COL_TRUE_SPEED_MS]) if config.COL_TRUE_SPEED_MS in df.columns else float(rdata["v_can"][global_start_idx])
            init_heading_deg = float(df.loc[global_start_idx, config.COL_TRUE_HEADING]) if config.COL_TRUE_HEADING in df.columns else 0.0

            # Evaluate Systems A, B, C
            # A: Phase 17A Baseline (CAN Speed + Road-Normal Map Constraint)
            # B: Baseline + ZARU
            # C: Baseline + Sustained ZUPT + ZARU
            modes = ["A", "B", "C"]
            mode_results = {}

            # Canonical CAN forward speed measurement variance
            r_can_var = 0.0325

            # Also determine if ZARU was applied pre-outage to calibrate gyro bias!
            pre_bg_zaru = 0.0
            if has_pre_stop:
                last_stop_row = pre_stops.iloc[-1]
                pre_bg_zaru = float(last_stop_row["mean_wz_rad"])

            for m in modes:
                ukf = ZARU_CANFusionUKF(dt=0.1, r_zaru=stat_r_zaru)
                init_bg = pre_bg_zaru if (m in ["B", "C"] and has_pre_stop) else 0.0
                ukf.initialize(
                    init_lat=init_lat,
                    init_lon=init_lon,
                    init_speed_ms=init_speed_ms,
                    init_heading_rad=math.radians(init_heading_deg),
                    init_gyro_bias=init_bg,
                    allow_bias_learning=(m in ["B", "C"])
                )

                matcher = FixedLagHMMMapMatcher(graph=graph)

                est_pN = np.zeros(sub_len)
                est_pE = np.zeros(sub_len)
                est_speed = np.zeros(sub_len)
                est_heading = np.zeros(sub_len)

                in_detector = CausalStandstillDetector()

                for k in range(sub_len):
                    gk = global_start_idx + k
                    a_v = rdata["acc_v"][gk]
                    w_v = rdata["gyro_v"][gk]
                    v_val = float(rdata["v_can"][gk])

                    # Standstill detection during outage
                    is_stop_outage, _ = in_detector.update(
                        v_val, w_v[0], w_v[1], w_v[2], a_v[0], a_v[1], a_v[2]
                    )

                    # 1. Prediction
                    ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=0.1)

                    # 2. Measurement updates
                    if m == "A":
                        # Standard Phase 17A baseline
                        if v_val < 0.15:
                            ukf.update_zupt(gyro_reading=w_v[2])
                        else:
                            ukf.update_can_speed(v_can=v_val, r_var=r_can_var)
                    elif m == "B":
                        # ZARU only during verified stop
                        if is_stop_outage:
                            ukf.update_zaru(omega_z_meas=w_v[2])
                        ukf.update_can_speed(v_can=v_val, r_var=r_can_var)
                    elif m == "C":
                        # Sustained ZUPT + ZARU during verified stop
                        if is_stop_outage:
                            ukf.update_zupt(gyro_reading=w_v[2])
                            ukf.update_zaru(omega_z_meas=w_v[2])
                        elif v_val < 0.15:
                            ukf.update_zupt(gyro_reading=w_v[2])
                        else:
                            ukf.update_can_speed(v_can=v_val, r_var=r_can_var)

                    # 3. Map update (2 Hz)
                    if (k % 5 == 0):
                        p_N_curr = float(ukf.x[0])
                        p_E_curr = float(ukf.x[1])
                        curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
                        p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
                        hdg_deg = math.degrees(ukf.x[3]) % 360.0
                        spd = float(ukf.x[2])
                        pos_sigma = float(math.sqrt(max(0.01, ukf.P[0, 0] + ukf.P[1, 1])))

                        mres = matcher.match(
                            point_xy=p_graph,
                            heading_deg=hdg_deg,
                            speed_ms=spd,
                            travel_dist_m=max(spd * 0.5, 0.05),
                            sigma_pos_m=pos_sigma,
                            dt=0.5
                        )

                        if not mres.is_off_road and mres.confidence >= 0.25:
                            snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                            d_lat = math.radians(snap_lat - init_lat)
                            d_lon = math.radians(snap_lon - init_lon)
                            p_N_match = float(6371000.0 * d_lat)
                            p_E_match = float(6371000.0 * d_lon * math.cos(math.radians(init_lat)))

                            ukf.update_map_match(
                                p_N_match=p_N_match,
                                p_E_match=p_E_match,
                                psi_road=float(mres.road_heading_rad),
                                confidence=float(mres.confidence),
                                is_heading_valid=False
                            )

                    est_pN[k] = ukf.x[0]
                    est_pE[k] = ukf.x[1]
                    est_speed[k] = ukf.x[2]
                    est_heading[k] = math.degrees(ukf.x[3]) % 360.0

                est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
                gt_lat = df_sub[config.COL_TRUE_LAT].values
                gt_lon = df_sub[config.COL_TRUE_LON].values
                gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

                scores = score_outage_segment(
                    pred_lat=est_lat,
                    pred_lon=est_lon,
                    gt_lat=gt_lat,
                    gt_lon=gt_lon,
                    distance_travelled_m=dist_gt,
                    duration_s=dur,
                    gt_heading_deg=gt_hdg,
                    pred_heading_deg=est_heading
                )
                mode_results[m] = scores

            benchmark_records.append({
                "scenario": f"{run_name}_o{oid}",
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "distance_m": dist_gt,
                "has_pre_stop": has_pre_stop,
                "n_pre_stops": len(pre_stops),
                "last_stop_age_s": last_stop_age if last_stop_age is not None else 999.0,
                "has_in_stop": has_in_stop,
                "fpe_A": mode_results["A"]["fpe_m"],
                "drift_pct_A": mode_results["A"]["drift_percent"],
                "along_track_A": mode_results["A"]["along_track_m"],
                "cross_track_A": mode_results["A"]["cross_track_m"],
                "heading_err_A": mode_results["A"]["heading_error_deg"],
                "fpe_B": mode_results["B"]["fpe_m"],
                "drift_pct_B": mode_results["B"]["drift_percent"],
                "along_track_B": mode_results["B"]["along_track_m"],
                "cross_track_B": mode_results["B"]["cross_track_m"],
                "heading_err_B": mode_results["B"]["heading_error_deg"],
                "fpe_C": mode_results["C"]["fpe_m"],
                "drift_pct_C": mode_results["C"]["drift_percent"],
                "along_track_C": mode_results["C"]["along_track_m"],
                "cross_track_C": mode_results["C"]["cross_track_m"],
                "heading_err_C": mode_results["C"]["heading_error_deg"],
                "fpe_delta_A_to_B": mode_results["B"]["fpe_m"] - mode_results["A"]["fpe_m"],
                "fpe_delta_A_to_C": mode_results["C"]["fpe_m"] - mode_results["A"]["fpe_m"]
            })

    df_bench = pd.DataFrame(benchmark_records)
    df_bench.to_csv(BASE_DIR / "eval" / "phase18b_zaru_benchmark_results.csv", index=False)

    logger.info(f"\n--- 56-OUTAGE BENCHMARK AGGREGATE RESULTS ---")
    logger.info(f"System A (Baseline): Median FPE = {df_bench['fpe_A'].median():.2f}m | Mean = {df_bench['fpe_A'].mean():.2f}m | Max = {df_bench['fpe_A'].max():.2f}m | Median Drift = {df_bench['drift_pct_A'].median():.2f}%")
    logger.info(f"System B (+ ZARU):   Median FPE = {df_bench['fpe_B'].median():.2f}m | Mean = {df_bench['fpe_B'].mean():.2f}m | Max = {df_bench['fpe_B'].max():.2f}m | Median Drift = {df_bench['drift_pct_B'].median():.2f}%")
    logger.info(f"System C (+ ZUPT+ZARU): Median FPE = {df_bench['fpe_C'].median():.2f}m | Mean = {df_bench['fpe_C'].mean():.2f}m | Max = {df_bench['fpe_C'].max():.2f}m | Median Drift = {df_bench['drift_pct_C'].median():.2f}%")

    # Sub-analysis by pre-outage stop availability
    with_stop = df_bench[df_bench["has_pre_stop"]]
    without_stop = df_bench[~df_bench["has_pre_stop"]]
    logger.info(f"\nSUBSET WITH PRE-OUTAGE STOP (N={len(with_stop)} / 56, {len(with_stop)/56*100:.1f}%):")
    logger.info(f"  System A: Median FPE = {with_stop['fpe_A'].median():.2f}m | Mean = {with_stop['fpe_A'].mean():.2f}m")
    logger.info(f"  System B: Median FPE = {with_stop['fpe_B'].median():.2f}m | Mean = {with_stop['fpe_B'].mean():.2f}m")
    logger.info(f"  System C: Median FPE = {with_stop['fpe_C'].median():.2f}m | Mean = {with_stop['fpe_C'].mean():.2f}m")

    logger.info(f"\nSUBSET WITHOUT PRE-OUTAGE STOP (N={len(without_stop)} / 56, {len(without_stop)/56*100:.1f}%):")
    logger.info(f"  System A: Median FPE = {without_stop['fpe_A'].median():.2f}m | Mean = {without_stop['fpe_A'].mean():.2f}m")
    logger.info(f"  System B: Median FPE = {without_stop['fpe_B'].median():.2f}m | Mean = {without_stop['fpe_B'].mean():.2f}m")
    logger.info(f"  System C: Median FPE = {without_stop['fpe_C'].median():.2f}m | Mean = {without_stop['fpe_C'].mean():.2f}m")

    # ==========================================================================
    # STEP 8: TEST 4 — PHASE 18A SEVERE FAILURE CASES AUDIT
    # ==========================================================================
    logger.info("\n--- TEST 4: PHASE 18A SEVERE FAILURE CASES AUDIT ---")
    fail_keys = ["vw14b_o5", "vw4_o1", "vw11_o3", "vw8_o1", "vw6_o2", "vw8_o3", "vw7_o1"]
    failure_audit = []

    for fk in fail_keys:
        match = df_bench[df_bench["scenario"] == fk]
        if len(match) > 0:
            r = match.iloc[0]
            has_stop = bool(r["has_pre_stop"])
            fpe_A = float(r["fpe_A"])
            fpe_B = float(r["fpe_B"])
            fpe_C = float(r["fpe_C"])
            delta_B = float(r["fpe_delta_A_to_B"])
            hdg_A = float(r["heading_err_A"])
            hdg_B = float(r["heading_err_B"])

            status_note = "ZARU applied from pre-outage standstill" if has_stop else "ZARU cannot address this case (zero pre-outage stops)"
            failure_audit.append({
                "scenario": fk,
                "duration_s": int(r["duration_s"]),
                "has_pre_stop": has_stop,
                "last_stop_age_s": float(r["last_stop_age_s"]) if has_stop else np.nan,
                "fpe_A": fpe_A,
                "fpe_B": fpe_B,
                "fpe_C": fpe_C,
                "fpe_delta": delta_B,
                "hdg_err_A": hdg_A,
                "hdg_err_B": hdg_B,
                "status_note": status_note
            })
            logger.info(f"[{fk:<12}] Dur={int(r['duration_s']):3d}s | Pre-Stop: {str(has_stop):<5} | FPE A={fpe_A:6.2f}m -> B={fpe_B:6.2f}m (Δ={delta_B:+6.2f}m) | Note: {status_note}")

    df_fails = pd.DataFrame(failure_audit)
    df_fails.to_csv(BASE_DIR / "eval" / "phase18b_failure_cases_audit.csv", index=False)

    # ==========================================================================
    # STEP 9: GENERATE DIAGNOSTIC FIGURES
    # ==========================================================================
    logger.info("\n--- GENERATING PLOTS ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Fig 1: Stationary Duration Distribution
    ax = axes[0, 0]
    ax.hist(df_stops["dur_s"], bins=25, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax.set_title("Standstill Duration Distribution (107 Verified Events)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Standstill Duration (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Fig 2: Gyro Noise Floor (std_wz) vs Standstill Duration
    ax = axes[0, 1]
    ax.scatter(df_stops["dur_s"], df_stops["std_wz_deg"], c="#2ca02c", alpha=0.7, edgecolors="none", s=50)
    ax.axhline(np.degrees(mean_std_wz), color="red", linestyle="--", label=f"Mean Noise = {np.degrees(mean_std_wz):.3f}°/s")
    ax.set_title("Observed Gyroscope Noise Floor (σ_wz)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Standstill Duration (s)")
    ax.set_ylabel("Standard Deviation (°/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fig 3: Benchmark FPE CDF (Subset with Pre-Outage Stops)
    ax = axes[1, 0]
    if len(with_stop) > 0:
        s_A = np.sort(with_stop["fpe_A"].values)
        s_B = np.sort(with_stop["fpe_B"].values)
        s_C = np.sort(with_stop["fpe_C"].values)
        p = np.linspace(0, 1, len(s_A))
        ax.plot(s_A, p, label=f"System A (Baseline) [Med={np.median(s_A):.1f}m]", color="#d62728", lw=2)
        ax.plot(s_B, p, label=f"System B (+ ZARU) [Med={np.median(s_B):.1f}m]", color="#1f77b4", lw=2, linestyle="--")
        ax.plot(s_C, p, label=f"System C (+ ZUPT+ZARU) [Med={np.median(s_C):.1f}m]", color="#2ca02c", lw=2, linestyle=":")
    ax.set_title("FPE CDF: Outages with Pre-Outage Stops (N=26)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Final Position Error (m)")
    ax.set_ylabel("Cumulative Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fig 4: Post-Stop Heading Error by Horizon
    ax = axes[1, 1]
    med_A = [df_drift[f"err_A_{h}s"].median() for h in horizons]
    med_B = [df_drift[f"err_B_{h}s"].median() for h in horizons]
    med_C = [df_drift[f"err_C_{h}s"].median() for h in horizons]
    x_pos = np.arange(len(horizons))
    width = 0.25
    ax.bar(x_pos - width, med_A, width, label="System A (Baseline)", color="#d62728", alpha=0.8)
    ax.bar(x_pos, med_B, width, label="System B (ZARU)", color="#1f77b4", alpha=0.8)
    ax.bar(x_pos + width, med_C, width, label="System C (ZUPT+ZARU)", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{h}s" for h in horizons])
    ax.set_title("Median Post-Stop Heading Error by Horizon", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Driving Time After Stop")
    ax.set_ylabel("Heading Error (°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "phase18b_fig1_zaru_comprehensive_benchmark.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved diagnostic figure to {plot_path}")

    # Copy to artifacts dir
    import shutil
    art_path = ARTIFACTS_DIR / "phase18b_fig1_zaru_comprehensive_benchmark.png"
    shutil.copy(plot_path, art_path)
    logger.info(f"Copied figure to {art_path}")

    logger.info("\nPHASE 18B EXPERIMENT COMPLETE!")


if __name__ == "__main__":
    run_full_experiment()
