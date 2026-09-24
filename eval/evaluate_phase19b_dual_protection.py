"""
Phase 19B: Dual Map-Protection Benchmark
(Viterbi Margin Gate M >= 3.0 + 30-deg Heading Consistency Gate)

Hypothesis:
  Combining topological protection (Viterbi Path Margin Gate M >= 3.0) with
  geometric protection (30-deg Heading Consistency Gate) will eliminate Phase 18A
  catastrophic map-latching failures without degrading the 42 already-good scenarios.

Strict Constraints:
  - NO Doppler bgz lock
  - NO explicit road-heading fusion
  - NO new neural models
  - NO global heading corrections
  - NO dynamic bias overwrite
  - Production code remains 100% untouched
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
logger = logging.getLogger("Phase19B")

# The 14 Phase 18A degradation scenarios
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
    "sample_test_trajectory_motorway_o2",
    "vw2_o2",
    "vw5_o2",
]


# ==============================================================================
# 1. Dual-Protected HMM Map Matcher
# ==============================================================================
class DualProtectedHMMMapMatcher(FixedLagHMMMapMatcher):
    """
    Extends FixedLagHMMMapMatcher to evaluate:
      1. Viterbi Path Margin: M = P_winner / P_runner-up
         (where runner-up belongs to a distinct road edge).
      2. Heading Discrepancy: d_psi = |wrapToPi(psi_filter - psi_road)|
    """
    def __init__(self, graph: RoadGraph, margin_thresh: float = 3.0, theta_max_deg: float = 30.0):
        super().__init__(graph=graph)
        self.margin_thresh = margin_thresh
        self.theta_max_deg = theta_max_deg
        self.theta_max_rad = math.radians(theta_max_deg)

    def match_dual_protected(
        self,
        point_xy: Tuple[float, float],
        heading_deg: float = 0.0,
        speed_ms: float = 0.0,
        travel_dist_m: float = 0.0,
        sigma_pos_m: float = 3.0,
        dt: float = 0.1
    ) -> Tuple[MapMatchResult, float, float, bool, bool]:
        """
        Returns:
          (mres, margin, d_psi_deg, is_margin_ambiguous, is_heading_inconsistent)
        """
        mres = self.match(
            point_xy=point_xy,
            heading_deg=heading_deg,
            speed_ms=speed_ms,
            travel_dist_m=travel_dist_m,
            sigma_pos_m=sigma_pos_m,
            dt=dt
        )

        if mres.is_off_road or not self.prev_candidates:
            return mres, float("inf"), 0.0, False, False

        # 1. Viterbi Margin Calculation across distinct road edges
        winner_edge_id = mres.edge_id
        winner_conf = mres.confidence

        runner_up_conf = 0.0
        for c in self.prev_candidates:
            if c["seg"].edge_id != winner_edge_id:
                if c["confidence"] > runner_up_conf:
                    runner_up_conf = c["confidence"]

        if runner_up_conf <= 1e-9:
            margin = float("inf")
        else:
            margin = float(winner_conf / max(runner_up_conf, 1e-9))

        is_margin_ambiguous = (margin < self.margin_thresh)

        # 2. Heading Consistency Calculation
        hdg_filter_rad = math.radians(heading_deg)
        d_psi_rad = abs(self.wrap_to_pi(hdg_filter_rad - mres.road_heading_rad))
        d_psi_deg = math.degrees(d_psi_rad)
        is_heading_inconsistent = (d_psi_deg > self.theta_max_deg)

        return mres, margin, d_psi_deg, is_margin_ambiguous, is_heading_inconsistent


# ==============================================================================
# 2. Outage Simulation Runner for Three Arms
# ==============================================================================
def run_three_arm_outage_simulation(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    alignment: AlignmentEngine,
    graph: RoadGraph,
    r_eff: float = 0.2776,
    r_can_var: float = 0.0325,
    dt: float = config.TARGET_DT
) -> Dict[str, Dict[str, Any]]:
    """
    Executes Arm A, Arm B, and Arm C side-by-side on the exact same data.
      - Arm A: Phase 17A Baseline (no gates)
      - Arm B: Dual-Protected (Viterbi Margin M>=3.0 + Heading Gate theta<=30 deg)
      - Arm C: Reference Diagnostic (Heading Gate theta<=30 deg alone)
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

    # Initialize UKFs for all three arms
    arms = ["A", "B", "C"]
    ukfs = {}
    matchers = {}
    for arm in arms:
        ukf = CANFusionUKF(dt=dt)
        ukf.initialize(
            init_lat=init_lat,
            init_lon=init_lon,
            init_speed_ms=init_speed_ms,
            init_heading_rad=math.radians(init_heading_deg),
            allow_bias_learning=False
        )
        ukfs[arm] = ukf
        matchers[arm] = DualProtectedHMMMapMatcher(graph=graph, margin_thresh=3.0, theta_max_deg=30.0)

    est_pN = {arm: np.zeros(n) for arm in arms}
    est_pE = {arm: np.zeros(n) for arm in arms}
    est_speed = {arm: np.zeros(n) for arm in arms}
    est_heading_deg = {arm: np.zeros(n) for arm in arms}

    map_stats = {
        arm: {
            "attempts": 0,
            "accepted": 0,
            "rejected_gate": 0,
            "margin_gated": 0,
            "heading_gated": 0
        } for arm in arms
    }

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        v_val = float(v_rear[k])
        rear_diff = abs(omega_rl[k] - omega_rr[k])
        is_slipping = rear_diff > 25.0

        for arm in arms:
            ukf = ukfs[arm]

            # 1. Prediction step
            ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

            # 2. CAN forward speed / ZUPT update
            if v_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_val, r_var=r_can_var)

            # 3. Map update (2 Hz)
            if k % 5 == 0:
                map_stats[arm]["attempts"] += 1
                p_N_curr = float(ukf.x[0])
                p_E_curr = float(ukf.x[1])
                curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
                p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
                hdg_deg = math.degrees(ukf.x[3]) % 360.0
                spd = float(ukf.x[2])
                pos_sigma = float(math.sqrt(max(0.01, ukf.P[0, 0] + ukf.P[1, 1])))

                mres, margin, d_psi_deg, is_margin_ambiguous, is_hdg_inconsistent = matchers[arm].match_dual_protected(
                    point_xy=p_graph,
                    heading_deg=hdg_deg,
                    speed_ms=spd,
                    travel_dist_m=max(spd * 0.5, 0.05),
                    sigma_pos_m=pos_sigma,
                    dt=0.5
                )

                # Gate check depending on arm:
                # Arm A: Apply whenever confidence >= 0.25 and not off-road
                # Arm B: In addition, suppress if is_margin_ambiguous (M < 3.0) OR is_hdg_inconsistent (d_psi > 30 deg)
                # Arm C: In addition, suppress if is_hdg_inconsistent (d_psi > 30 deg)
                should_apply = (not mres.is_off_road and mres.confidence >= 0.25)

                if arm == "B":
                    if is_margin_ambiguous:
                        map_stats[arm]["margin_gated"] += 1
                    if is_hdg_inconsistent:
                        map_stats[arm]["heading_gated"] += 1
                    if is_margin_ambiguous or is_hdg_inconsistent:
                        should_apply = False
                        map_stats[arm]["rejected_gate"] += 1

                elif arm == "C":
                    if is_hdg_inconsistent:
                        map_stats[arm]["heading_gated"] += 1
                        should_apply = False
                        map_stats[arm]["rejected_gate"] += 1

                if should_apply:
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
                        map_stats[arm]["accepted"] += 1

            est_pN[arm][k] = ukf.x[0]
            est_pE[arm][k] = ukf.x[1]
            est_speed[arm][k] = ukf.x[2]
            est_heading_deg[arm][k] = math.degrees(ukf.x[3]) % 360.0

    # Format outputs
    out = {}
    for arm in arms:
        lat_arr, lon_arr = local_xy_to_latlon(est_pE[arm], est_pN[arm], init_lat, init_lon)
        out[arm] = {
            "lat": lat_arr,
            "lon": lon_arr,
            "speed": est_speed[arm],
            "heading_deg": est_heading_deg[arm],
            "pN": est_pN[arm],
            "pE": est_pE[arm],
            "stats": map_stats[arm]
        }
    return out


# ==============================================================================
# 3. Master Evaluation Execution
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


def run_phase19b_benchmark():
    logger.info("=" * 95)
    logger.info("PHASE 19B: DUAL MAP-PROTECTION BENCHMARK")
    logger.info("Arm A (Control):   Phase 17A Baseline (CAN Speed + 1D Map Normal, no gates)")
    logger.info("Arm B (Dual Gate): Arm A + Viterbi Margin Gate (M >= 3.0) + Heading Consistency Gate (theta <= 30 deg)")
    logger.info("Arm C (Ref Diag):  Arm A + Heading Consistency Gate (theta <= 30 deg alone)")
    logger.info("=" * 95)

    # 1. Load Road Graph
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

            # Standard initialization for all arms (NO Doppler lock, untouched baseline protocol)
            pre_idx_start = max(0, outage_start_idx - 10)
            pre_df = df.iloc[pre_idx_start:outage_start_idx]
            if len(pre_df) > 0:
                init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
                init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
                init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean()) if config.COL_TRUE_SPEED_MS in pre_df.columns else 0.0
                h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values) if config.COL_TRUE_HEADING in pre_df.columns else np.zeros(len(pre_df))
                init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
            else:
                init_lat = float(df[config.COL_TRUE_LAT].iloc[outage_start_idx])
                init_lon = float(df[config.COL_TRUE_LON].iloc[outage_start_idx])
                init_spd = float(df[config.COL_TRUE_SPEED_MS].iloc[outage_start_idx]) if config.COL_TRUE_SPEED_MS in df.columns else 0.0
                init_hdg = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx]) if config.COL_TRUE_HEADING in df.columns else 0.0

            # Execute simulation for Arm A, Arm B, Arm C
            sim_res = run_three_arm_outage_simulation(
                df_outage=df_sub,
                init_lat=init_lat,
                init_lon=init_lon,
                init_speed_ms=init_spd,
                init_heading_deg=init_hdg,
                alignment=run_align,
                graph=graph,
                r_eff=r_eff_est
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            # Score each arm
            scores = {}
            for arm in ["A", "B", "C"]:
                scores[arm] = score_outage_segment(
                    pred_lat=sim_res[arm]["lat"],
                    pred_lon=sim_res[arm]["lon"],
                    gt_lat=gt_lat,
                    gt_lon=gt_lon,
                    distance_travelled_m=dist_gt,
                    duration_s=dur,
                    gt_heading_deg=gt_hdg,
                    pred_heading_deg=sim_res[arm]["heading_deg"]
                )

            scenario_name = f"{run_name}_o{oid}"
            fpe_A = float(scores["A"]["fpe_m"])
            fpe_B = float(scores["B"]["fpe_m"])
            fpe_C = float(scores["C"]["fpe_m"])

            rec = {
                "scenario": scenario_name,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "distance_m": dist_gt,
                # Arm A (Baseline)
                "fpe_A": fpe_A,
                "drift_pct_A": float(scores["A"]["drift_percent"]),
                "along_track_A": float(scores["A"]["along_track_m"]) if not np.isnan(scores["A"]["along_track_m"]) else 0.0,
                "cross_track_A": float(scores["A"]["cross_track_m"]) if not np.isnan(scores["A"]["cross_track_m"]) else 0.0,
                "heading_err_A": float(scores["A"]["heading_error_deg"]) if not np.isnan(scores["A"].get("heading_error_deg", np.nan)) else 0.0,
                "map_accepted_A": sim_res["A"]["stats"]["accepted"],
                # Arm B (Dual-Protected)
                "fpe_B": fpe_B,
                "drift_pct_B": float(scores["B"]["drift_percent"]),
                "along_track_B": float(scores["B"]["along_track_m"]) if not np.isnan(scores["B"]["along_track_m"]) else 0.0,
                "cross_track_B": float(scores["B"]["cross_track_m"]) if not np.isnan(scores["B"]["cross_track_m"]) else 0.0,
                "heading_err_B": float(scores["B"]["heading_error_deg"]) if not np.isnan(scores["B"].get("heading_error_deg", np.nan)) else 0.0,
                "map_accepted_B": sim_res["B"]["stats"]["accepted"],
                "margin_gated_B": sim_res["B"]["stats"]["margin_gated"],
                "heading_gated_B": sim_res["B"]["stats"]["heading_gated"],
                # Arm C (Heading Gate Alone)
                "fpe_C": fpe_C,
                "drift_pct_C": float(scores["C"]["drift_percent"]),
                "along_track_C": float(scores["C"]["along_track_m"]) if not np.isnan(scores["C"]["along_track_m"]) else 0.0,
                "cross_track_C": float(scores["C"]["cross_track_m"]) if not np.isnan(scores["C"]["cross_track_m"]) else 0.0,
                "heading_err_C": float(scores["C"]["heading_error_deg"]) if not np.isnan(scores["C"].get("heading_error_deg", np.nan)) else 0.0,
                "map_accepted_C": sim_res["C"]["stats"]["accepted"],
                "heading_gated_C": sim_res["C"]["stats"]["heading_gated"],
                # Deltas
                "delta_fpe_A_to_B": fpe_B - fpe_A,
                "delta_fpe_C_to_B": fpe_B - fpe_C,
            }
            benchmark_rows.append(rec)

            # Check if failure case
            is_fail_case = (scenario_name in PHASE18A_FAILURE_CASES) or (f"{run_name}_o{oid}" in PHASE18A_FAILURE_CASES)
            if is_fail_case:
                f_rec = {
                    "scenario": scenario_name,
                    "duration_s": dur,
                    "fpe_A": fpe_A,
                    "fpe_B": fpe_B,
                    "fpe_C": fpe_C,
                    "delta_fpe_A_to_B": fpe_B - fpe_A,
                    "improved_vs_A": (fpe_B < fpe_A),
                    "improved_vs_C": (fpe_B < fpe_C),
                    "margin_gated_B": sim_res["B"]["stats"]["margin_gated"],
                    "heading_gated_B": sim_res["B"]["stats"]["heading_gated"]
                }
                failure_audit_rows.append(f_rec)
                logger.info(f"[{scenario_name:35s}] Dur={dur:3d}s | A={fpe_A:6.1f}m -> B={fpe_B:6.1f}m (C={fpe_C:6.1f}m) | Δ(B-A)={fpe_B-fpe_A:+6.1f}m | M_Gated={f_rec['margin_gated_B']} | H_Gated={f_rec['heading_gated_B']}")

    df_results = pd.DataFrame(benchmark_rows)
    df_failure_audit = pd.DataFrame(failure_audit_rows)

    csv_path = config.BASE_DIR / "eval" / "phase19b_benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark results ({len(df_results)} rows) to {csv_path}")

    f_csv_path = config.BASE_DIR / "eval" / "phase19b_failure_cases_audit.csv"
    df_failure_audit.to_csv(f_csv_path, index=False)
    logger.info(f"Saved failure cases audit ({len(df_failure_audit)} rows) to {f_csv_path}")

    # ==========================================================================
    # 4. Summary Statistics & Regression Analysis
    # ==========================================================================
    med_fpe_A = float(df_results["fpe_A"].median())
    mean_fpe_A = float(df_results["fpe_A"].mean())
    med_drift_A = float(df_results["drift_pct_A"].median())
    under10_A = int((df_results["drift_pct_A"] < 10.0).sum())

    med_fpe_B = float(df_results["fpe_B"].median())
    mean_fpe_B = float(df_results["fpe_B"].mean())
    med_drift_B = float(df_results["drift_pct_B"].median())
    under10_B = int((df_results["drift_pct_B"] < 10.0).sum())

    med_fpe_C = float(df_results["fpe_C"].median())
    mean_fpe_C = float(df_results["fpe_C"].mean())
    med_drift_C = float(df_results["drift_pct_C"].median())
    under10_C = int((df_results["drift_pct_C"] < 10.0).sum())

    # Check 42 already-good cases (where Phase 17A improved or had acceptable error)
    df_good = df_results[~df_results["scenario"].isin(df_failure_audit["scenario"])]
    med_good_A = float(df_good["fpe_A"].median())
    med_good_B = float(df_good["fpe_B"].median())
    med_good_C = float(df_good["fpe_C"].median())

    n_fail_improved = int(df_failure_audit["improved_vs_A"].sum())
    n_fail_total = len(df_failure_audit)

    logger.info("\n" + "=" * 95)
    logger.info("PHASE 19B BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 95)
    logger.info(f"Arm A (Baseline Control):   Median FPE = {med_fpe_A:6.2f}m | Mean = {mean_fpe_A:6.2f}m | Drift = {med_drift_A:5.2f}% | <10% Drift = {under10_A}/56")
    logger.info(f"Arm B (Dual-Protected):     Median FPE = {med_fpe_B:6.2f}m | Mean = {mean_fpe_B:6.2f}m | Drift = {med_drift_B:5.2f}% | <10% Drift = {under10_B}/56")
    logger.info(f"Arm C (Heading Gate Alone): Median FPE = {med_fpe_C:6.2f}m | Mean = {mean_fpe_C:6.2f}m | Drift = {med_drift_C:5.2f}% | <10% Drift = {under10_C}/56")

    logger.info("\n--- PERFORMANCE ON 42 ALREADY-GOOD CASES ---")
    logger.info(f"Arm A Good Cases Med FPE: {med_good_A:6.2f}m")
    logger.info(f"Arm B Good Cases Med FPE: {med_good_B:6.2f}m (Δ={med_good_B - med_good_A:+6.2f}m)")
    logger.info(f"Arm C Good Cases Med FPE: {med_good_C:6.2f}m (Δ={med_good_C - med_good_A:+6.2f}m)")

    logger.info(f"\nPhase 18A Failures Recovered (vs Arm A): {n_fail_improved}/{n_fail_total}")

    logger.info("\n--- DURATION BREAKDOWN ---")
    for dur in [30, 60, 120, 180]:
        sub = df_results[df_results["duration_s"] == dur]
        logger.info(f"Duration {dur:3d}s (N={len(sub):2d}): Arm A = {sub['fpe_A'].median():6.2f}m | Arm B = {sub['fpe_B'].median():6.2f}m | Arm C = {sub['fpe_C'].median():6.2f}m")

    # ==========================================================================
    # 5. Generate Diagnostic Plots
    # ==========================================================================
    logger.info("\n--- GENERATING PLOTS ---")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Cumulative Distribution of FPE
    ax = axes[0, 0]
    sorted_A = np.sort(df_results["fpe_A"])
    sorted_B = np.sort(df_results["fpe_B"])
    sorted_C = np.sort(df_results["fpe_C"])
    cdf = np.linspace(0, 1, len(df_results))
    ax.plot(sorted_A, cdf, label=f"Arm A (Baseline, Med={med_fpe_A:.1f}m)", color="salmon", lw=2)
    ax.plot(sorted_C, cdf, label=f"Arm C (Heading Gate Alone, Med={med_fpe_C:.1f}m)", color="green", lw=2, ls="--")
    ax.plot(sorted_B, cdf, label=f"Arm B (Dual Protected, Med={med_fpe_B:.1f}m)", color="royalblue", lw=2.5)
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
    fpe_durs_C = [df_results[df_results["duration_s"] == d]["fpe_C"].median() for d in durs]
    x_pos = np.arange(len(durs))
    width = 0.25
    ax.bar(x_pos - width, fpe_durs_A, width, label="Arm A (Baseline)", color="salmon")
    ax.bar(x_pos, fpe_durs_C, width, label="Arm C (Heading Gate)", color="mediumseagreen")
    ax.bar(x_pos + width, fpe_durs_B, width, label="Arm B (Dual Gate)", color="royalblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{d}s" for d in durs], fontsize=10)
    ax.set_xlabel("Outage Duration", fontsize=11)
    ax.set_ylabel("Median FPE (m)", fontsize=11)
    ax.set_title("Median FPE Across Outage Durations", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Plot 3: Failure Cases Comparison
    ax = axes[1, 0]
    if len(df_failure_audit) > 0:
        y_pos = np.arange(len(df_failure_audit))
        ax.barh(y_pos - width, df_failure_audit["fpe_A"], width, label="Arm A (Baseline)", color="salmon")
        ax.barh(y_pos, df_failure_audit["fpe_C"], width, label="Arm C (Heading Gate)", color="mediumseagreen")
        ax.barh(y_pos + width, df_failure_audit["fpe_B"], width, label="Arm B (Dual Gate)", color="royalblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_failure_audit["scenario"], fontsize=8)
        ax.set_xlabel("FPE (m)", fontsize=11)
        ax.set_title("14 Failure Cases: Arm A vs Arm B vs Arm C", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", ls="--", alpha=0.5)
        ax.legend(fontsize=10)

    # Plot 4: Scatter of Arm A vs Arm B
    ax = axes[1, 1]
    ax.scatter(df_results["fpe_A"], df_results["fpe_B"], color="royalblue", alpha=0.7, s=40, label="Scenarios")
    max_val = max(df_results["fpe_A"].max(), df_results["fpe_B"].max())
    ax.plot([1, max_val], [1, max_val], "k--", alpha=0.5, label="1:1 Parity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arm A FPE (m) [Log Scale]", fontsize=11)
    ax.set_ylabel("Arm B FPE (m) [Log Scale]", fontsize=11)
    ax.set_title("Scenario Error Correlation (Arm A vs Arm B)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = config.BASE_DIR / "results" / "plots" / "phase19b_fig1_dual_protection_benchmark.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved diagnostic figure to {plot_path}")

    # Copy to brain artifact directory
    brain_plot_path = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16/phase19b_fig1_dual_protection_benchmark.png")
    if brain_plot_path.parent.exists():
        import shutil
        shutil.copy(plot_path, brain_plot_path)
        logger.info(f"Copied figure to {brain_plot_path}")

    logger.info("\nPHASE 19B EXPERIMENT COMPLETE!")


if __name__ == "__main__":
    run_phase19b_benchmark()
