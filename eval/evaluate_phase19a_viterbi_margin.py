"""
Phase 19A: Pre-Outage Course Anchoring + Viterbi Margin-Gated Map Aiding Benchmark

Hypothesis:
  Pre-outage GNSS Doppler course anchoring (eliminating initial heading bias)
  combined with a Viterbi Path Margin Gate (M >= 3.0, suppressing map updates
  on ambiguous parallel links) will eliminate the 14 Phase 18A map-degradation
  failures and prevent 120s–180s outage divergence without introducing false heading fixes.

Systems:
  - Arm A (Control): Phase 17A Production System (CAN speed + 1D road-normal map constraint + ZUPT/ZARU).
  - Arm B (Experimental): Arm A + Pre-Outage 5s GNSS Doppler Course & b_gz Lock + Viterbi Margin Gate (M >= 3.0).

Strict isolation:
  - Production code remains completely untouched.
  - Zero ground-truth leakage during outages.
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
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from modules.alignment import AlignmentEngine, AlignmentState
from modules.can_fusion_ukf import CANFusionUKF, compute_pre_outage_can_calibration
from modules.map_matcher import (
    load_road_graph_from_osm_json,
    FixedLagHMMMapMatcher,
    RoadGraph,
    RoadSegment,
    MapMatchResult,
    ROAD_EARTH_RADIUS
)
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase19A")

# 14 Phase 18A degradation scenarios
PHASE18A_FAILURE_CASES = [
    "vw14b_o5",
    "vw4_o1",
    "vw14c_o4",
    "vw11_o3",
    "vw16a_o4",
    "vw3_o3",
    "vw8_o1",
    "vw7_o1",
    "vw4_o3",
    "vw6_o2",
    "vw8_o3",
    "motorway_o2",
    "vw2_o2",
    "vw5_o2",
]


# ==============================================================================
# 1. Viterbi Margin-Gated HMM Map Matcher
# ==============================================================================
class ViterbiMarginHMMMapMatcher(FixedLagHMMMapMatcher):
    """
    Extends FixedLagHMMMapMatcher with a Viterbi Path Margin Gate:
      M = P_winner / P_runner-up
    where P_runner-up is the maximum probability among competing candidates
    belonging to a DISTINCT road edge (edge_id != winner_edge_id).
    
    If M < margin_thresh (default 3.0), the candidate is marked as ambiguous,
    enabling caller to suppress the map update to the UKF.
    """
    def __init__(self, graph: RoadGraph, margin_thresh: float = 3.0):
        super().__init__(graph=graph)
        self.margin_thresh = margin_thresh
        self.total_queries = 0
        self.ambiguous_queries = 0
        self.unambiguous_queries = 0

    def match_with_margin(
        self,
        point_xy: Tuple[float, float],
        heading_deg: float = 0.0,
        speed_ms: float = 0.0,
        travel_dist_m: float = 0.0,
        sigma_pos_m: float = 3.0,
        dt: float = 0.1
    ) -> Tuple[MapMatchResult, float, bool]:
        """
        Returns:
          (match_result, margin, is_ambiguous)
        """
        self.total_queries += 1
        mres = self.match(
            point_xy=point_xy,
            heading_deg=heading_deg,
            speed_ms=speed_ms,
            travel_dist_m=travel_dist_m,
            sigma_pos_m=sigma_pos_m,
            dt=dt
        )

        if mres.is_off_road or not self.prev_candidates:
            return mres, float("inf"), False

        # Compute winner and runner-up probabilities across DISTINCT road edges
        winner_edge_id = mres.edge_id
        winner_conf = mres.confidence

        # Find best candidate on a different road edge
        runner_up_conf = 0.0
        for c in self.prev_candidates:
            if c["seg"].edge_id != winner_edge_id:
                if c["confidence"] > runner_up_conf:
                    runner_up_conf = c["confidence"]

        if runner_up_conf <= 1e-9:
            margin = float("inf")
        else:
            margin = float(winner_conf / max(runner_up_conf, 1e-9))

        is_ambiguous = (margin < self.margin_thresh)
        if is_ambiguous:
            self.ambiguous_queries += 1
        else:
            self.unambiguous_queries += 1

        return mres, margin, is_ambiguous


# ==============================================================================
# 2. Pre-Outage 5s GNSS Doppler Course & b_gz Estimator
# ==============================================================================
def compute_pre_outage_doppler_anchor(
    df: pd.DataFrame,
    outage_start_idx: int,
    gyro_v: np.ndarray,
    window_s: float = 5.0,
    dt: float = 0.1
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Estimates anchored initial course and b_gz from the 5s pre-outage window.
    
    Returns:
      (anchored_heading_deg, locked_bgz_rad, info_dict)
    """
    n_win = int(round(window_s / dt))
    pre_start = max(0, outage_start_idx - n_win)
    pre_df = df.iloc[pre_start:outage_start_idx]

    if len(pre_df) < 5:
        # Fallback to standard initialization
        init_hdg = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx]) if config.COL_TRUE_HEADING in df.columns else 0.0
        return init_hdg, 0.0, {"mode": "fallback_insufficient_samples", "v_mean": 0.0}

    gps_spd = pre_df["gps_speed_ms"].values if "gps_speed_ms" in pre_df.columns else np.zeros(len(pre_df))
    gps_brg = pre_df["gps_bearing_deg"].values if "gps_bearing_deg" in pre_df.columns else np.zeros(len(pre_df))
    wz_pre = gyro_v[pre_start:outage_start_idx, 2]

    v_mean = float(np.mean(gps_spd))
    dur_actual = len(pre_df) * dt

    if v_mean > 1.5:
        # 1. Moving regime: Doppler course is valid
        # Use circular mean of the last 1.0s (10 samples) of Doppler bearing for initial heading
        last_n = min(10, len(gps_brg))
        rad_last = np.radians(gps_brg[-last_n:])
        anchored_hdg_deg = float(np.degrees(np.arctan2(np.mean(np.sin(rad_last)), np.mean(np.cos(rad_last)))) % 360.0)

        # 2. Estimate b_gz from gyro integral vs Doppler course change over the 5s window
        # Gyro integrated turn: delta_psi_gyro = sum(wz * dt)
        gyro_int_rad = float(np.sum(wz_pre) * dt)

        # GNSS Doppler delta heading (wrapped to [-pi, pi])
        brg_start_rad = math.radians(gps_brg[0])
        brg_end_rad = math.radians(gps_brg[-1])
        d_brg_rad = math.atan2(math.sin(brg_end_rad - brg_start_rad), math.cos(brg_end_rad - brg_start_rad))

        # Under kinematics: delta_psi_gyro = delta_psi_true + b_gz * T
        # and delta_psi_Doppler = delta_psi_true
        # Therefore: b_gz = (delta_psi_gyro - delta_psi_Doppler) / T
        raw_bgz = (gyro_int_rad - d_brg_rad) / max(dur_actual, 1.0)

        # Physical safety clamp: MEMS gyro bias is bounded within +/- 0.1 rad/s (~5.7 deg/s)
        locked_bgz_rad = float(np.clip(raw_bgz, -0.10, 0.10))

        mode_str = "moving_doppler_lock"
    else:
        # Stationary / low speed regime
        # Anchored heading from last valid moving course or true heading
        anchored_hdg_deg = float(pre_df[config.COL_TRUE_HEADING].iloc[-1]) if config.COL_TRUE_HEADING in pre_df.columns else float(gps_brg[-1])
        # Under standstill: true turn rate = 0, so b_gz = mean(wz)
        raw_bgz = float(np.mean(wz_pre))
        locked_bgz_rad = float(np.clip(raw_bgz, -0.05, 0.05))
        mode_str = "standstill_bias_lock"

    info = {
        "mode": mode_str,
        "v_mean": v_mean,
        "raw_bgz_deg": math.degrees(locked_bgz_rad),
        "anchored_hdg_deg": anchored_hdg_deg,
        "dur_s": dur_actual
    }
    return anchored_hdg_deg, locked_bgz_rad, info


# ==============================================================================
# 3. Single Outage Simulation Runner
# ==============================================================================
def run_single_outage_simulation(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    init_gyro_bias_rad: float,
    alignment: AlignmentEngine,
    graph: Optional[RoadGraph] = None,
    mode: str = "A",  # 'A': Control (Phase 17A Baseline), 'B': Experimental (Phase 19A Margin Gated)
    margin_thresh: float = 3.0,
    r_eff: float = 0.2776,
    r_can_var: float = 0.0325,
    dt: float = config.TARGET_DT
) -> Dict[str, Any]:
    """
    Executes a single outage dead reckoning simulation for Arm A or Arm B.
    """
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

    ukf = CANFusionUKF(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg),
        allow_bias_learning=False
    )
    # Set the locked gyro bias
    ukf.x[4] = init_gyro_bias_rad

    matcher = ViterbiMarginHMMMapMatcher(graph=graph, margin_thresh=margin_thresh) if graph is not None else None

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading_deg = np.zeros(n)
    est_bg = np.zeros(n)

    map_attempts = 0
    map_accepted = 0
    map_margin_gated = 0
    map_rejected = 0

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Prediction step
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        # 2. CAN forward speed update (10 Hz)
        v_val = float(v_rear[k])
        rear_diff = abs(omega_rl[k] - omega_rr[k])
        is_slipping = rear_diff > 25.0
        if v_val < 0.15 and not is_slipping:
            ukf.update_zupt(gyro_reading=w_v[2])
        elif not is_slipping:
            ukf.update_can_speed(v_can=v_val, r_var=r_can_var)

        # 3. Map matching update (every 5 epochs = 2 Hz / 0.5s)
        if matcher is not None and (k % 5 == 0):
            map_attempts += 1
            p_N_curr = float(ukf.x[0])
            p_E_curr = float(ukf.x[1])
            curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
            p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
            hdg_deg = math.degrees(ukf.x[3]) % 360.0
            spd = float(ukf.x[2])
            pos_sigma = float(math.sqrt(max(0.01, ukf.P[0, 0] + ukf.P[1, 1])))

            mres, margin, is_ambiguous = matcher.match_with_margin(
                point_xy=p_graph,
                heading_deg=hdg_deg,
                speed_ms=spd,
                travel_dist_m=max(spd * 0.5, 0.05),
                sigma_pos_m=pos_sigma,
                dt=0.5
            )

            # Check gating logic:
            # Arm A: Apply whenever confidence >= 0.25 and not off-road
            # Arm B: In addition, suppress if is_ambiguous (M < 3.0)
            should_apply_map = (not mres.is_off_road and mres.confidence >= 0.25)
            if mode == "B" and is_ambiguous:
                should_apply_map = False
                map_margin_gated += 1

            if should_apply_map:
                snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                d_lat = math.radians(snap_lat - init_lat)
                d_lon = math.radians(snap_lon - init_lon)
                p_N_match = float(ROAD_EARTH_RADIUS * d_lat)
                p_E_match = float(ROAD_EARTH_RADIUS * d_lon * math.cos(math.radians(init_lat)))

                applied, pos_nis, _ = ukf.update_map_match(
                    p_N_match=p_N_match,
                    p_E_match=p_E_match,
                    psi_road=float(mres.road_heading_rad),
                    confidence=float(mres.confidence),
                    is_heading_valid=False
                )
                if applied:
                    map_accepted += 1
                else:
                    map_rejected += 1
            else:
                if mode == "A" and not should_apply_map:
                    map_rejected += 1

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]
        est_heading_deg[k] = math.degrees(ukf.x[3]) % 360.0
        est_bg[k] = ukf.x[4]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    gt_lat = df_outage[config.COL_TRUE_LAT].values
    gt_lon = df_outage[config.COL_TRUE_LON].values
    d_lat = np.radians(est_lat - gt_lat) * ROAD_EARTH_RADIUS
    d_lon = np.radians(est_lon - gt_lon) * ROAD_EARTH_RADIUS * np.cos(np.radians(gt_lat))
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
        "map_attempts": map_attempts,
        "map_accepted": map_accepted,
        "map_margin_gated": map_margin_gated,
        "map_rejected": map_rejected
    }


# ==============================================================================
# 4. Master Benchmark Execution
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


def run_phase19a_benchmark(smoke_test: bool = False):
    logger.info("=" * 95)
    logger.info("PHASE 19A: PRE-OUTAGE COURSE ANCHORING + VITERBI MARGIN-GATED MAP AIDING")
    logger.info("Control Arm A: Phase 17A Production System (CAN Speed + 1D Map Normal + ZUPT/ZARU)")
    logger.info("Experimental Arm B: Arm A + Pre-Outage 5s GNSS Doppler Course & b_gz Lock + Viterbi Margin Gate (M >= 3.0)")
    logger.info("=" * 95)

    # 1. Load Road Graph from OSM
    osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    logger.info(f"Loading canonical RoadGraph from {osm_path}...")
    t0 = time.time()
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)
    logger.info(f"Loaded RoadGraph ({len(graph.nodes)} nodes, {len(graph.edges)} edges) in {time.time()-t0:.2f}s.")

    test_files = get_benchmark_trajectories()
    logger.info(f"Evaluating across {len(test_files)} trajectories.")

    benchmark_rows = []
    failure_audit_rows = []

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

        # Precompute body-to-vehicle alignment
        run_align = AlignmentEngine()
        warmup = df[df[config.COL_TIME] < min(45.0, total_dur * 0.2)]
        if len(warmup) >= 25:
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref); y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

        # Transform all gyros to vehicle frame for Doppler lock
        gyro_raw = df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
        gyro_v = gyro_raw @ run_align.result.R_b_to_v.T

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            dist_gt = float(row["distance_travelled_m"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()

            # Pre-outage CAN speed calibration
            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, _, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values if "gps_speed_ms" in cal_pre_df.columns else np.zeros(len(cal_pre_df)),
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

            # --- ARM A INITIALIZATION (Phase 17A Production Control) ---
            pre_idx_start = max(0, outage_start_idx - 10)
            pre_df = df.iloc[pre_idx_start:outage_start_idx]
            if len(pre_df) > 0:
                init_lat_A = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
                init_lon_A = float(pre_df[config.COL_TRUE_LON].iloc[-1])
                init_spd_A = float(pre_df[config.COL_TRUE_SPEED_MS].mean()) if config.COL_TRUE_SPEED_MS in pre_df.columns else 0.0
                h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values) if config.COL_TRUE_HEADING in pre_df.columns else np.zeros(len(pre_df))
                init_hdg_A = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
            else:
                init_lat_A = float(df[config.COL_TRUE_LAT].iloc[outage_start_idx])
                init_lon_A = float(df[config.COL_TRUE_LON].iloc[outage_start_idx])
                init_spd_A = float(df[config.COL_TRUE_SPEED_MS].iloc[outage_start_idx]) if config.COL_TRUE_SPEED_MS in df.columns else 0.0
                init_hdg_A = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx]) if config.COL_TRUE_HEADING in df.columns else 0.0
            init_bg_A = 0.0

            # --- ARM B INITIALIZATION (Pre-Outage 5s Doppler Course & bgz Lock) ---
            init_hdg_B, init_bg_B, lock_info = compute_pre_outage_doppler_anchor(
                df=df,
                outage_start_idx=outage_start_idx,
                gyro_v=gyro_v,
                window_s=5.0,
                dt=config.TARGET_DT
            )
            init_lat_B = init_lat_A
            init_lon_B = init_lon_A
            init_spd_B = init_spd_A

            # Execute Arm A Simulation
            res_A = run_single_outage_simulation(
                df_outage=df_sub,
                init_lat=init_lat_A,
                init_lon=init_lon_A,
                init_speed_ms=init_spd_A,
                init_heading_deg=init_hdg_A,
                init_gyro_bias_rad=init_bg_A,
                alignment=run_align,
                graph=graph,
                mode="A",
                r_eff=r_eff_est
            )

            # Execute Arm B Simulation
            res_B = run_single_outage_simulation(
                df_outage=df_sub,
                init_lat=init_lat_B,
                init_lon=init_lon_B,
                init_speed_ms=init_spd_B,
                init_heading_deg=init_hdg_B,
                init_gyro_bias_rad=init_bg_B,
                alignment=run_align,
                graph=graph,
                mode="B",
                margin_thresh=3.0,
                r_eff=r_eff_est
            )

            # Ground truth metrics
            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            # Score Arm A
            metrics_A = score_outage_segment(
                pred_lat=res_A["lat"],
                pred_lon=res_A["lon"],
                gt_lat=gt_lat,
                gt_lon=gt_lon,
                duration_s=dur,
                distance_travelled_m=dist_gt,
                pred_heading_deg=res_A["heading_deg"],
                gt_heading_deg=gt_hdg
            )

            # Score Arm B
            metrics_B = score_outage_segment(
                pred_lat=res_B["lat"],
                pred_lon=res_B["lon"],
                gt_lat=gt_lat,
                gt_lon=gt_lon,
                duration_s=dur,
                distance_travelled_m=dist_gt,
                pred_heading_deg=res_B["heading_deg"],
                gt_heading_deg=gt_hdg
            )

            scenario_name = f"{run_name}_o{oid}"
            fpe_A = float(metrics_A["fpe_m"])
            fpe_B = float(metrics_B["fpe_m"])
            delta_fpe = fpe_B - fpe_A

            rec = {
                "scenario": scenario_name,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "distance_m": dist_gt,
                # Arm A
                "fpe_A": fpe_A,
                "drift_pct_A": float(metrics_A["drift_percent"]),
                "along_track_A": float(metrics_A["along_track_m"]) if not np.isnan(metrics_A["along_track_m"]) else 0.0,
                "cross_track_A": float(metrics_A["cross_track_m"]) if not np.isnan(metrics_A["cross_track_m"]) else 0.0,
                "heading_err_A": float(metrics_A["heading_error_deg"]) if not np.isnan(metrics_A.get("heading_error_deg", np.nan)) else 0.0,
                "map_accepted_A": res_A["map_accepted"],
                "map_rejected_A": res_A["map_rejected"],
                # Arm B
                "fpe_B": fpe_B,
                "drift_pct_B": float(metrics_B["drift_percent"]),
                "along_track_B": float(metrics_B["along_track_m"]) if not np.isnan(metrics_B["along_track_m"]) else 0.0,
                "cross_track_B": float(metrics_B["cross_track_m"]) if not np.isnan(metrics_B["cross_track_m"]) else 0.0,
                "heading_err_B": float(metrics_B["heading_error_deg"]) if not np.isnan(metrics_B.get("heading_error_deg", np.nan)) else 0.0,
                "map_accepted_B": res_B["map_accepted"],
                "map_margin_gated_B": res_B["map_margin_gated"],
                "map_rejected_B": res_B["map_rejected"],
                # Delta & Initial state info
                "fpe_delta_A_to_B": delta_fpe,
                "init_hdg_A": init_hdg_A,
                "init_hdg_B": init_hdg_B,
                "locked_bgz_deg": math.degrees(init_bg_B),
                "lock_mode": lock_info["mode"]
            }
            benchmark_rows.append(rec)

            # Check if this scenario is one of the 14 Phase 18A failure cases
            if scenario_name in PHASE18A_FAILURE_CASES:
                f_rec = {
                    "scenario": scenario_name,
                    "duration_s": dur,
                    "fpe_A": fpe_A,
                    "fpe_B": fpe_B,
                    "delta_fpe": delta_fpe,
                    "improved": (fpe_B < fpe_A),
                    "hdg_err_A": rec["heading_err_A"],
                    "hdg_err_B": rec["heading_err_B"],
                    "map_gated_B": res_B["map_margin_gated"],
                    "lock_mode": lock_info["mode"],
                    "locked_bgz_deg": math.degrees(init_bg_B)
                }
                failure_audit_rows.append(f_rec)
                logger.info(f"[{scenario_name:12s}] Dur={dur:3d}s | FPE A={fpe_A:6.1f}m -> B={fpe_B:6.1f}m (Δ={delta_fpe:+6.1f}m) | Improved={f_rec['improved']} | Gated={res_B['map_margin_gated']}")

            if smoke_test and len(benchmark_rows) >= 3:
                break
        if smoke_test and len(benchmark_rows) >= 3:
            break

    df_results = pd.DataFrame(benchmark_rows)
    df_failure_audit = pd.DataFrame(failure_audit_rows)

    csv_path = config.BASE_DIR / "eval" / "phase19a_benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark results ({len(df_results)} rows) to {csv_path}")

    f_csv_path = config.BASE_DIR / "eval" / "phase19a_failure_cases_audit.csv"
    df_failure_audit.to_csv(f_csv_path, index=False)
    logger.info(f"Saved failure cases audit ({len(df_failure_audit)} rows) to {f_csv_path}")

    # ==========================================================================
    # 5. Aggregate Metrics & Success / Kill Verification
    # ==========================================================================
    med_fpe_A = float(df_results["fpe_A"].median())
    mean_fpe_A = float(df_results["fpe_A"].mean())
    max_fpe_A = float(df_results["fpe_A"].max())
    med_drift_A = float(df_results["drift_pct_A"].median())
    under10_A = int((df_results["drift_pct_A"] < 10.0).sum())

    med_fpe_B = float(df_results["fpe_B"].median())
    mean_fpe_B = float(df_results["fpe_B"].mean())
    max_fpe_B = float(df_results["fpe_B"].max())
    med_drift_B = float(df_results["drift_pct_B"].median())
    under10_B = int((df_results["drift_pct_B"] < 10.0).sum())

    # 180s Outages
    df_180 = df_results[df_results["duration_s"] == 180]
    med_180_A = float(df_180["fpe_A"].median()) if len(df_180) > 0 else 0.0
    med_180_B = float(df_180["fpe_B"].median()) if len(df_180) > 0 else 0.0

    # Failure recovery count
    n_failures_total = len(df_failure_audit)
    n_failures_improved = int(df_failure_audit["improved"].sum()) if n_failures_total > 0 else 0
    n_failures_worsened = n_failures_total - n_failures_improved

    logger.info("\n" + "=" * 95)
    logger.info("PHASE 19A BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 95)
    logger.info(f"Arm A (Control Baseline): Median FPE = {med_fpe_A:6.2f}m | Mean = {mean_fpe_A:6.2f}m | Drift = {med_drift_A:5.2f}% | <10% Drift = {under10_A}/{len(df_results)}")
    logger.info(f"Arm B (Experimental):     Median FPE = {med_fpe_B:6.2f}m | Mean = {mean_fpe_B:6.2f}m | Drift = {med_drift_B:5.2f}% | <10% Drift = {under10_B}/{len(df_results)}")
    logger.info(f"180s Outages Median FPE:  Arm A = {med_180_A:6.2f}m -> Arm B = {med_180_B:6.2f}m (Δ={med_180_B - med_180_A:+6.2f}m)")
    logger.info(f"Phase 18A Failures Audit: {n_failures_improved}/{n_failures_total} Improved | {n_failures_worsened}/{n_failures_total} Worsened")

    logger.info("\n--- DURATION BREAKDOWN ---")
    for dur in [30, 60, 120, 180]:
        sub = df_results[df_results["duration_s"] == dur]
        if len(sub) > 0:
            logger.info(f"Duration {dur:3d}s (N={len(sub):2d}): Arm A Med FPE = {sub['fpe_A'].median():6.2f}m | Arm B Med FPE = {sub['fpe_B'].median():6.2f}m (Δ={sub['fpe_B'].median() - sub['fpe_A'].median():+6.2f}m)")

    # Criteria Checks
    logger.info("\n--- SUCCESS / KILL CRITERIA EVALUATION ---")
    succ_1 = (med_fpe_B < 85.0)
    succ_2 = (n_failures_improved >= 10)
    succ_3 = (med_180_B < 350.0)
    succ_4 = (under10_B >= 20)

    kill_1 = (med_fpe_B > 108.69)
    kill_2 = (n_failures_worsened > 3)
    kill_3 = (med_fpe_B > 150.0)

    logger.info(f"Success 1 (Median FPE < 85m):       {med_fpe_B:.2f}m  -> {'PASS' if succ_1 else 'FAIL'}")
    logger.info(f"Success 2 (Recover >=10/14 cases):  {n_failures_improved}/{n_failures_total} -> {'PASS' if succ_2 else 'FAIL'}")
    logger.info(f"Success 3 (180s Outage FPE < 350m): {med_180_B:.2f}m -> {'PASS' if succ_3 else 'FAIL'}")
    logger.info(f"Success 4 (<10% Drift >= 20/56):    {under10_B}/{len(df_results)} -> {'PASS' if succ_4 else 'FAIL'}")

    logger.info(f"Kill 1 (Median FPE > 108.69m):      {med_fpe_B:.2f}m  -> {'TRIGGERED' if kill_1 else 'SAFE'}")
    logger.info(f"Kill 2 (Worsen >3/14 cases):        {n_failures_worsened}/{n_failures_total} -> {'TRIGGERED' if kill_2 else 'SAFE'}")
    logger.info(f"Kill 3 (Median FPE > 150m):         {med_fpe_B:.2f}m  -> {'TRIGGERED' if kill_3 else 'SAFE'}")

    # ==========================================================================
    # 6. Generate Diagnostic Plots
    # ==========================================================================
    logger.info("\n--- GENERATING PLOTS ---")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Cumulative Distribution of FPE
    ax = axes[0, 0]
    sorted_A = np.sort(df_results["fpe_A"])
    sorted_B = np.sort(df_results["fpe_B"])
    cdf = np.linspace(0, 1, len(df_results))
    ax.plot(sorted_A, cdf, label=f"Arm A (Phase 17A Baseline, Med={med_fpe_A:.1f}m)", color="red", lw=2)
    ax.plot(sorted_B, cdf, label=f"Arm B (Phase 19A Doppler+Margin, Med={med_fpe_B:.1f}m)", color="blue", lw=2.5)
    ax.set_xscale("log")
    ax.set_xlabel("Final Position Error (m) [Log Scale]", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    ax.set_title("CDF of Outage Final Position Error", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Plot 2: Median FPE by Outage Duration
    ax = axes[0, 1]
    durs = [30, 60, 120, 180]
    fpe_durs_A = [df_results[df_results["duration_s"] == d]["fpe_A"].median() for d in durs]
    fpe_durs_B = [df_results[df_results["duration_s"] == d]["fpe_B"].median() for d in durs]
    x_pos = np.arange(len(durs))
    width = 0.35
    ax.bar(x_pos - width/2, fpe_durs_A, width, label="Arm A (Baseline)", color="salmon")
    ax.bar(x_pos + width/2, fpe_durs_B, width, label="Arm B (Phase 19A)", color="royalblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{d}s" for d in durs], fontsize=10)
    ax.set_xlabel("Outage Duration", fontsize=11)
    ax.set_ylabel("Median FPE (m)", fontsize=11)
    ax.set_title("Median FPE Across Outage Durations", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Plot 3: 14 Failure Cases Comparison
    ax = axes[1, 0]
    if len(df_failure_audit) > 0:
        y_pos = np.arange(len(df_failure_audit))
        ax.barh(y_pos - width/2, df_failure_audit["fpe_A"], width, label="Arm A FPE", color="salmon")
        ax.barh(y_pos + width/2, df_failure_audit["fpe_B"], width, label="Arm B FPE", color="royalblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_failure_audit["scenario"], fontsize=9)
        ax.set_xlabel("FPE (m)", fontsize=11)
        ax.set_title("Audit of 14 Phase 18A Degradation Cases", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", ls="--", alpha=0.5)
        ax.legend(fontsize=10)

    # Plot 4: Viterbi Margin Gating & Heading Error Reduction
    ax = axes[1, 1]
    ax.scatter(df_results["heading_err_A"], df_results["heading_err_B"], color="purple", alpha=0.7, s=40)
    ax.plot([0, 180], [0, 180], "k--", alpha=0.5, label="1:1 Parity")
    ax.set_xlabel("Arm A Final Heading Error (deg)", fontsize=11)
    ax.set_ylabel("Arm B Final Heading Error (deg)", fontsize=11)
    ax.set_title("Heading Error Correlation (Arm A vs Arm B)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = config.BASE_DIR / "results" / "plots" / "phase19a_fig1_viterbi_margin_benchmark.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved diagnostic figure to {plot_path}")

    # Copy to brain artifact directory
    brain_plot_path = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16/phase19a_fig1_viterbi_margin_benchmark.png")
    if brain_plot_path.parent.exists():
        import shutil
        shutil.copy(plot_path, brain_plot_path)
        logger.info(f"Copied figure to {brain_plot_path}")

    logger.info("\nPHASE 19A EXPERIMENT COMPLETE!")


if __name__ == "__main__":
    smoke = ("--smoke-test" in sys.argv)
    run_phase19a_benchmark(smoke_test=smoke)
