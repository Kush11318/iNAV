"""
Phase 19: Map-Matched Heuristic Drift Elimination (MAPHDE) Experiment
Reference: Aggarwal, Thomas, Ojeda, Borenstein (Measurement Science and Technology, 2011)

Strict Constraints:
- DO NOT modify production navigation pipeline
- DO NOT replace existing 7-state UKF
- DO NOT add explicit road-heading Kalman measurement (z = psi_map prohibited)
- DO NOT implement proportional correction (K * E prohibited) or direct heading clamping (psi = psi_map prohibited)
- Controller must respond strictly to SIGN(E), not magnitude
- Binary integrator: I_i = I_{i-1} + SIGN(E_i) * i_c
- Initialized to 0 at the start of each outage; held (not reset) during suspension
- Evaluated on all 56 held-out GNSS outages
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
logger = logging.getLogger("Phase19_MAPHDE")

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
# 1. MAPHDE Controller Implementation (Aggarwal et al. 2011)
# ==============================================================================
class MAPHDEController:
    """
    Binary Integral Controller implementing MAPHDE from Aggarwal et al. 2011:
        E_i = wrapToPi(psi_map - psi_nav)
        I_i = I_{i-1} + SIGN(E_i) * i_c
        r_drift = I_i / dt_map  (applied continuously to gyro yaw rate)

    Strictly responds to SIGN(E), not magnitude.
    When suspended (near junctions, ambiguous map, or sharp maneuvers),
    I is HELD constant (not accumulated and not zeroed).
    """
    def __init__(self, i_c_deg: float = 0.02, max_I_deg: float = 10.0, dt_map: float = 0.5):
        self.i_c_rad = math.radians(i_c_deg)
        self.max_I_rad = math.radians(max_I_deg)
        self.dt_map = dt_map
        self.I = 0.0
        self.history_I = []
        self.history_E = []
        self.history_time = []
        self.history_applied = []
        self.current_time = 0.0

        # Diagnostics counters
        self.total_opportunities = 0
        self.valid_opportunities = 0
        self.suspended_opportunities = 0
        self.conf_rejections = 0
        self.margin_suspensions = 0
        self.dynamic_suspensions = 0

    def reset(self):
        self.I = 0.0
        self.history_I = []
        self.history_E = []
        self.history_time = []
        self.history_applied = []
        self.current_time = 0.0
        self.total_opportunities = 0
        self.valid_opportunities = 0
        self.suspended_opportunities = 0
        self.conf_rejections = 0
        self.margin_suspensions = 0
        self.dynamic_suspensions = 0

    @staticmethod
    def wrap_to_pi(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def update(
        self,
        psi_nav_rad: float,
        psi_map_rad: float,
        is_valid: bool,
        t: float,
        reason: str = ""
    ) -> float:
        self.current_time = t
        self.total_opportunities += 1

        E = self.wrap_to_pi(psi_map_rad - psi_nav_rad)

        if is_valid:
            # Paper mechanism: binary integral feedback
            sign_e = 1.0 if E > 0.0 else (-1.0 if E < 0.0 else 0.0)
            self.I = self.I + sign_e * self.i_c_rad
            # Anti-windup saturation
            self.I = max(-self.max_I_rad, min(self.max_I_rad, self.I))
            self.valid_opportunities += 1
            applied = True
        else:
            # Suspension: hold I constant without accumulation
            self.suspended_opportunities += 1
            applied = False
            if "conf" in reason:
                self.conf_rejections += 1
            if "margin" in reason:
                self.margin_suspensions += 1
            if "dynamic" in reason:
                self.dynamic_suspensions += 1

        self.history_I.append(self.I)
        self.history_E.append(E)
        self.history_time.append(t)
        self.history_applied.append(applied)

        # Return drift rate in rad/s to adjust gyro yaw rate
        r_drift = self.I / self.dt_map
        return r_drift


# ==============================================================================
# 2. DualProtected HMM Map Matcher (providing Viterbi margin & heading)
# ==============================================================================
class DualProtectedHMMMapMatcher(FixedLagHMMMapMatcher):
    def __init__(self, graph: RoadGraph, margin_thresh: float = 3.0, theta_max_deg: float = 30.0):
        super().__init__(graph=graph)
        self.margin_thresh = margin_thresh
        self.theta_max_deg = theta_max_deg
        self.theta_max_rad = math.radians(theta_max_deg)

    def match_with_margin(
        self,
        point_xy: Tuple[float, float],
        heading_deg: float = 0.0,
        speed_ms: float = 0.0,
        travel_dist_m: float = 0.0,
        sigma_pos_m: float = 3.0,
        dt: float = 0.1
    ) -> Tuple[MapMatchResult, float, float, bool]:
        """
        Returns:
          (mres, margin, d_psi_deg, is_margin_ambiguous)
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
            return mres, float("inf"), 0.0, False

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

        hdg_filter_rad = math.radians(heading_deg)
        d_psi_rad = abs(self.wrap_to_pi(hdg_filter_rad - mres.road_heading_rad))
        d_psi_deg = math.degrees(d_psi_rad)

        return mres, margin, d_psi_deg, is_margin_ambiguous


# ==============================================================================
# 3. Three-Arm Simulation Loop
# ==============================================================================
def run_three_arm_simulation(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    alignment: AlignmentEngine,
    graph: RoadGraph,
    r_eff: float = 0.2776,
    r_can_var: float = 0.0325,
    dt: float = config.TARGET_DT,
    i_c_deg: float = 0.02
) -> Dict[str, Dict[str, Any]]:
    """
    Executes:
      - Arm A: Phase 17A Baseline (CAN + 1D road-normal map matching, NO heading feedback)
      - Arm B: Arm A + MAPHDE binary-integral heading drift correction
      - Arm C: Arm B + Viterbi Margin Gating (M >= 3.0) and intersection hold
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

    arms = ["A", "B", "C"]
    ukfs = {}
    matchers = {}
    controllers = {}

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
        if arm in ["B", "C"]:
            controllers[arm] = MAPHDEController(i_c_deg=i_c_deg, max_I_deg=10.0, dt_map=0.5)

    est_lat = {arm: np.zeros(n) for arm in arms}
    est_lon = {arm: np.zeros(n) for arm in arms}
    est_speed = {arm: np.zeros(n) for arm in arms}
    est_heading_deg = {arm: np.zeros(n) for arm in arms}
    est_r_drift = {arm: np.zeros(n) for arm in arms}
    est_I = {arm: np.zeros(n) for arm in arms}

    current_r_drift = {"A": 0.0, "B": 0.0, "C": 0.0}

    map_stats = {
        arm: {
            "attempts": 0,
            "accepted": 0,
            "rejected_gate": 0,
            "margin_gated": 0,
            "heading_gated": 0,
            "maphde_valid": 0,
            "maphde_suspended": 0
        } for arm in arms
    }

    trace_log = {
        arm: {
            "t": [],
            "hdg_nav": [],
            "hdg_road": [],
            "I": [],
            "E_deg": [],
            "edge_id": [],
            "conf": [],
            "active": []
        } for arm in ["B", "C"]
    }

    for k in range(n):
        t_now = k * dt
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        v_val = float(v_rear[k])
        rear_diff = abs(omega_rl[k] - omega_rr[k])
        is_slipping = rear_diff > 25.0

        for arm in arms:
            ukf = ukfs[arm]
            r_drift = current_r_drift[arm]

            # 1. Prediction step (with MAPHDE drift rate for Arms B and C)
            ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2] + r_drift, dt=dt)

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

                mres, margin, d_psi_deg, is_margin_ambiguous = matchers[arm].match_with_margin(
                    point_xy=p_graph,
                    heading_deg=hdg_deg,
                    speed_ms=spd,
                    travel_dist_m=max(spd * 0.5, 0.05),
                    sigma_pos_m=pos_sigma,
                    dt=0.5
                )

                # --- MAPHDE Controller Feedback Update ---
                if arm in ["B", "C"]:
                    ctrl = controllers[arm]
                    # Validity conditions:
                    # 1. Map confidence >= 0.25 and not off-road
                    # 2. Vehicle moving (> 1.5 m/s) and not turning sharply (|w_v[2]| < 0.15 rad/s)
                    # 3. Heading discrepancy < 45 deg
                    # 4. For Arm C: Viterbi margin M >= 3.0
                    conf_ok = (not mres.is_off_road and mres.confidence >= 0.25)
                    dyn_ok = (spd > 1.5 and abs(w_v[2]) < 0.15)
                    angle_ok = (d_psi_deg < 45.0)

                    reason = ""
                    if not conf_ok:
                        reason += "conf "
                    if not dyn_ok or not angle_ok:
                        reason += "dynamic "

                    if arm == "B":
                        is_valid = conf_ok and dyn_ok and angle_ok
                    else: # Arm C
                        if is_margin_ambiguous:
                            reason += "margin "
                        is_valid = conf_ok and dyn_ok and angle_ok and (not is_margin_ambiguous)

                    current_r_drift[arm] = ctrl.update(
                        psi_nav_rad=math.radians(hdg_deg),
                        psi_map_rad=float(mres.road_heading_rad),
                        is_valid=is_valid,
                        t=t_now,
                        reason=reason
                    )

                    trace_log[arm]["t"].append(t_now)
                    trace_log[arm]["hdg_nav"].append(hdg_deg)
                    trace_log[arm]["hdg_road"].append(math.degrees(mres.road_heading_rad) % 360.0)
                    trace_log[arm]["I"].append(math.degrees(ctrl.I))
                    trace_log[arm]["E_deg"].append(math.degrees(MAPHDEController.wrap_to_pi(mres.road_heading_rad - math.radians(hdg_deg))))
                    trace_log[arm]["edge_id"].append(mres.edge_id)
                    trace_log[arm]["conf"].append(mres.confidence)
                    trace_log[arm]["active"].append(is_valid)

                # --- UKF 1D Road-Normal Position Map Update ---
                # Phase 17A rule: apply if confidence >= 0.25 and not off-road
                should_apply = (not mres.is_off_road and mres.confidence >= 0.25)
                if should_apply:
                    snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                    d_lat = math.radians(snap_lat - init_lat)
                    d_lon = math.radians(snap_lon - init_lon)
                    p_N_match = float(ROAD_EARTH_RADIUS * d_lat)
                    p_E_match = float(ROAD_EARTH_RADIUS * d_lon * math.cos(math.radians(init_lat)))

                    applied, _, _ = ukf.update_map_match(
                        p_N_match=p_N_match,
                        p_E_match=p_E_match,
                        psi_road=float(mres.road_heading_rad),
                        confidence=float(mres.confidence),
                        is_heading_valid=False  # NO UKF heading fusion!
                    )
                    if applied:
                        map_stats[arm]["accepted"] += 1
                else:
                    map_stats[arm]["rejected_gate"] += 1

            # Record estimated states
            curr_lat, curr_lon = local_xy_to_latlon(ukf.x[1], ukf.x[0], init_lat, init_lon)
            est_lat[arm][k] = curr_lat
            est_lon[arm][k] = curr_lon
            est_speed[arm][k] = ukf.x[2]
            est_heading_deg[arm][k] = math.degrees(ukf.x[3]) % 360.0
            est_r_drift[arm][k] = current_r_drift[arm]
            if arm in ["B", "C"]:
                est_I[arm][k] = math.degrees(controllers[arm].I)

    res = {}
    for arm in arms:
        res[arm] = {
            "lat": est_lat[arm],
            "lon": est_lon[arm],
            "speed": est_speed[arm],
            "heading_deg": est_heading_deg[arm],
            "r_drift": est_r_drift[arm],
            "I_deg": est_I[arm],
            "map_stats": map_stats[arm],
            "ctrl": controllers.get(arm, None),
            "trace": trace_log.get(arm, None)
        }
    return res


# ==============================================================================
# 4. Master Benchmark Execution Across 56 Held-Out Outages
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


def run_benchmark():
    logger.info("Starting Phase 19 MAPHDE Benchmark on 56 Held-Out Outages...")

    osm_path = BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = BASE_DIR / "data" / "osm_uk_test_roads.json"

    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)
    logger.info(f"Loaded {len(graph.edges)} road edges from {osm_path}")

    test_files = get_benchmark_trajectories()
    logger.info(f"Evaluating across {len(test_files)} trajectories.")

    outage_results = []
    failure_audit = []
    saved_traces = {}

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
            y = np.cross(u_z, ref); y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

        for _, row_out in df_outages.iterrows():
            oid = int(row_out["outage_id"])
            dur = int(row_out["duration_s"])
            dist_gt = float(row_out["distance_travelled_m"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()

            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, _, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values if "gps_speed_ms" in cal_pre_df.columns else np.zeros(len(cal_pre_df)),
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

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

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            sim_res = run_three_arm_simulation(
                df_outage=df_sub,
                init_lat=init_lat,
                init_lon=init_lon,
                init_speed_ms=init_spd,
                init_heading_deg=init_hdg,
                alignment=run_align,
                graph=graph,
                r_eff=r_eff_est,
                dt=0.1,
                i_c_deg=0.02
            )

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
            drift_A = float(scores["A"]["drift_percent"])
            drift_B = float(scores["B"]["drift_percent"])
            drift_C = float(scores["C"]["drift_percent"])

            ctrl_B = sim_res["B"]["ctrl"]
            final_I_B = math.degrees(ctrl_B.I) if ctrl_B else 0.0
            max_I_B = max([abs(math.degrees(x)) for x in ctrl_B.history_I]) if (ctrl_B and ctrl_B.history_I) else 0.0
            valid_opp_B = ctrl_B.valid_opportunities if ctrl_B else 0
            susp_opp_B = ctrl_B.suspended_opportunities if ctrl_B else 0

            row = {
                "scenario": scenario_name,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "distance_m": dist_gt,
                "fpe_A": fpe_A,
                "fpe_B": fpe_B,
                "fpe_C": fpe_C,
                "drift_pct_A": drift_A,
                "drift_pct_B": drift_B,
                "drift_pct_C": drift_C,
                "diff_B_vs_A": fpe_B - fpe_A,
                "diff_C_vs_A": fpe_C - fpe_A,
                "improved_B": fpe_B < fpe_A - 0.5,
                "worsened_B": fpe_B > fpe_A + 0.5,
                "improved_C": fpe_C < fpe_A - 0.5,
                "worsened_C": fpe_C > fpe_A + 0.5,
                "hdg_err_A": float(scores["A"].get("heading_error_deg", 0.0)) if not np.isnan(scores["A"].get("heading_error_deg", np.nan)) else 0.0,
                "hdg_err_B": float(scores["B"].get("heading_error_deg", 0.0)) if not np.isnan(scores["B"].get("heading_error_deg", np.nan)) else 0.0,
                "hdg_err_C": float(scores["C"].get("heading_error_deg", 0.0)) if not np.isnan(scores["C"].get("heading_error_deg", np.nan)) else 0.0,
                "along_err_A": float(scores["A"].get("along_track_m", 0.0)) if not np.isnan(scores["A"].get("along_track_m", np.nan)) else 0.0,
                "along_err_B": float(scores["B"].get("along_track_m", 0.0)) if not np.isnan(scores["B"].get("along_track_m", np.nan)) else 0.0,
                "along_err_C": float(scores["C"].get("along_track_m", 0.0)) if not np.isnan(scores["C"].get("along_track_m", np.nan)) else 0.0,
                "cross_err_A": float(scores["A"].get("cross_track_m", 0.0)) if not np.isnan(scores["A"].get("cross_track_m", np.nan)) else 0.0,
                "cross_err_B": float(scores["B"].get("cross_track_m", 0.0)) if not np.isnan(scores["B"].get("cross_track_m", np.nan)) else 0.0,
                "cross_err_C": float(scores["C"].get("cross_track_m", 0.0)) if not np.isnan(scores["C"].get("cross_track_m", np.nan)) else 0.0,
                "final_I_B_deg": final_I_B,
                "max_I_B_deg": max_I_B,
                "valid_opp_B": valid_opp_B,
                "susp_opp_B": susp_opp_B,
            }
            outage_results.append(row)

            if scenario_name in PHASE18A_FAILURE_CASES:
                failure_audit.append({
                    "scenario": scenario_name,
                    "duration_s": dur,
                    "fpe_A": fpe_A,
                    "fpe_B": fpe_B,
                    "fpe_C": fpe_C,
                    "delta_B_m": fpe_B - fpe_A,
                    "delta_C_m": fpe_C - fpe_A,
                    "hdg_err_A": row["hdg_err_A"],
                    "hdg_err_B": row["hdg_err_B"],
                    "hdg_err_C": row["hdg_err_C"],
                    "cross_err_A": row["cross_err_A"],
                    "cross_err_B": row["cross_err_B"],
                    "cross_err_C": row["cross_err_C"],
                    "along_err_A": row["along_err_A"],
                    "along_err_B": row["along_err_B"],
                    "along_err_C": row["along_err_C"],
                    "final_I_B_deg": final_I_B,
                    "valid_opp_B": valid_opp_B,
                    "susp_opp_B": susp_opp_B,
                })
                saved_traces[scenario_name] = {
                    "sim": sim_res,
                    "gt_lat": gt_lat,
                    "gt_lon": gt_lon,
                    "gt_hdg": gt_hdg,
                    "t": np.arange(len(df_sub)) * 0.1
                }

    df_out = pd.DataFrame(outage_results)
    df_fail = pd.DataFrame(failure_audit)

    eval_dir = BASE_DIR / "eval"
    df_out.to_csv(eval_dir / "phase19_maphde_benchmark_results.csv", index=False)
    df_fail.to_csv(eval_dir / "phase19_maphde_failure_cases_audit.csv", index=False)
    logger.info(f"Saved benchmark CSVs to {eval_dir}")

    print_benchmark_summary(df_out, df_fail)
    generate_figures(df_out, df_fail, saved_traces)


def print_benchmark_summary(df: pd.DataFrame, df_fail: pd.DataFrame):
    print("\n" + "="*80)
    print("PHASE 19 MAPHDE: BENCHMARK SUMMARY ACROSS ALL 56 HELD-OUT OUTAGES")
    print("="*80)
    print(f"Total Outages: {len(df)}")
    print(f"{'Metric':<30} | {'Arm A (Phase 17A)':<18} | {'Arm B (MAPHDE)':<18} | {'Arm C (MAPHDE+Margin)':<20}")
    print("-" * 92)

    for metric, col_A, col_B, col_C, unit in [
        ("Median FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("Mean FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("P95 FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("Max FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("Median Drift %", "drift_pct_A", "drift_pct_B", "drift_pct_C", "%"),
        ("Mean Drift %", "drift_pct_A", "drift_pct_B", "drift_pct_C", "%"),
        ("Mean Heading Error", "hdg_err_A", "hdg_err_B", "hdg_err_C", "deg"),
        ("Mean Cross-Track Error", "cross_err_A", "cross_err_B", "cross_err_C", "m"),
        ("Mean Along-Track Error", "along_err_A", "along_err_B", "along_err_C", "m"),
    ]:
        if "Median" in metric:
            vA, vB, vC = df[col_A].median(), df[col_B].median(), df[col_C].median()
        elif "Mean" in metric:
            vA, vB, vC = df[col_A].mean(), df[col_B].mean(), df[col_C].mean()
        elif "P95" in metric:
            vA, vB, vC = df[col_A].quantile(0.95), df[col_B].quantile(0.95), df[col_C].quantile(0.95)
        elif "Max" in metric:
            vA, vB, vC = df[col_A].max(), df[col_B].max(), df[col_C].max()
        print(f"{metric:<30} | {vA:14.2f} {unit:<3} | {vB:14.2f} {unit:<3} | {vC:16.2f} {unit:<3}")

    print("-" * 92)
    imp_B = (df["fpe_B"] < df["fpe_A"] - 0.5).sum()
    wors_B = (df["fpe_B"] > df["fpe_A"] + 0.5).sum()
    unc_B = len(df) - imp_B - wors_B
    print(f"{'Outages Improved / Worsened (B vs A)':<30} | {'-':<18} | Improved: {imp_B:2d}, Worsened: {wors_B:2d}, Unchanged: {unc_B:2d}")

    imp_C = (df["fpe_C"] < df["fpe_A"] - 0.5).sum()
    wors_C = (df["fpe_C"] > df["fpe_A"] + 0.5).sum()
    unc_C = len(df) - imp_C - wors_C
    print(f"{'Outages Improved / Worsened (C vs A)':<30} | {'-':<18} | Improved: {imp_C:2d}, Worsened: {wors_C:2d}, Unchanged: {unc_C:2d}")

    sih_A = (df["drift_pct_A"] < 10.0).sum()
    sih_B = (df["drift_pct_B"] < 10.0).sum()
    sih_C = (df["drift_pct_C"] < 10.0).sum()
    print(f"{'Satisfying Drift < 10%':<30} | {sih_A:2d}/{len(df)} ({sih_A/len(df)*100:.1f}%)    | {sih_B:2d}/{len(df)} ({sih_B/len(df)*100:.1f}%)    | {sih_C:2d}/{len(df)} ({sih_C/len(df)*100:.1f}%)")

    print("\n" + "="*80)
    print("DURATION BREAKDOWN (Median FPE)")
    print("="*80)
    durations = sorted(df["duration_s"].unique())
    print(f"{'Duration':<10} | {'N':<4} | {'Arm A FPE (m)':<15} | {'Arm B FPE (m)':<15} | {'Arm C FPE (m)':<15} | {'Diff (B - A)':<12}")
    print("-" * 80)
    for dur in durations:
        df_d = df[df["duration_s"] == dur]
        mA = df_d["fpe_A"].median()
        mB = df_d["fpe_B"].median()
        mC = df_d["fpe_C"].median()
        diff = mB - mA
        print(f"{dur:<10.0f} | {len(df_d):<4} | {mA:15.2f} | {mB:15.2f} | {mC:15.2f} | {diff:+12.2f}")

    print("\n" + "="*80)
    print("14 PHASE 18A FAILURE CASES REPLAY AUDIT")
    print("="*80)
    print(f"{'Scenario':<22} | {'Dur':<4} | {'FPE_A (m)':<10} | {'FPE_B (m)':<10} | {'Delta B (m)':<12} | {'FPE_C (m)':<10} | {'Delta C (m)':<12}")
    print("-" * 90)
    for _, r in df_fail.iterrows():
        print(f"{r['scenario']:<22} | {r['duration_s']:<4.0f} | {r['fpe_A']:10.1f} | {r['fpe_B']:10.1f} | {r['delta_B_m']:+12.1f} | {r['fpe_C']:10.1f} | {r['delta_C_m']:+12.1f}")
    print("="*90 + "\n")


def generate_figures(df: pd.DataFrame, df_fail: pd.DataFrame, traces: Dict[str, Any]):
    plots_dir = BASE_DIR / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 19: Map-Matched Heuristic Drift Elimination (MAPHDE) Benchmark", fontsize=15, fontweight="bold")

    # 1. Median & Mean FPE comparison bar chart
    ax = axes[0, 0]
    metrics = ["Median FPE", "Mean FPE", "P95 FPE"]
    arm_A = [df["fpe_A"].median(), df["fpe_A"].mean(), df["fpe_A"].quantile(0.95)]
    arm_B = [df["fpe_B"].median(), df["fpe_B"].mean(), df["fpe_B"].quantile(0.95)]
    arm_C = [df["fpe_C"].median(), df["fpe_C"].mean(), df["fpe_C"].quantile(0.95)]
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, arm_A, width=w, label="Arm A (Phase 17A)", color="#4A90E2")
    ax.bar(x, arm_B, width=w, label="Arm B (MAPHDE ic=0.02°)", color="#E94A4A")
    ax.bar(x + w, arm_C, width=w, label="Arm C (MAPHDE+Margin)", color="#50E3C2")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("FPE (meters)")
    ax.set_title("Aggregate FPE Summary (N=56)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Duration Breakdown
    ax = axes[0, 1]
    durations = sorted(df["duration_s"].unique())
    fpe_d_A = [df[df["duration_s"] == d]["fpe_A"].median() for d in durations]
    fpe_d_B = [df[df["duration_s"] == d]["fpe_B"].median() for d in durations]
    fpe_d_C = [df[df["duration_s"] == d]["fpe_C"].median() for d in durations]
    ax.plot(durations, fpe_d_A, marker="o", linewidth=2, label="Arm A", color="#4A90E2")
    ax.plot(durations, fpe_d_B, marker="s", linewidth=2, label="Arm B (MAPHDE)", color="#E94A4A")
    ax.plot(durations, fpe_d_C, marker="^", linewidth=2, label="Arm C (MAPHDE+Margin)", color="#50E3C2")
    ax.set_xlabel("Outage Duration (s)")
    ax.set_ylabel("Median FPE (m)")
    ax.set_title("Median FPE vs Duration")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 3. Heading Error Distribution (CDF)
    ax = axes[0, 2]
    sorted_A = np.sort(df["hdg_err_A"])
    sorted_B = np.sort(df["hdg_err_B"])
    sorted_C = np.sort(df["hdg_err_C"])
    p = np.linspace(0, 1, len(sorted_A))
    ax.plot(sorted_A, p, label="Arm A", color="#4A90E2", linewidth=2)
    ax.plot(sorted_B, p, label="Arm B (MAPHDE)", color="#E94A4A", linewidth=2)
    ax.plot(sorted_C, p, label="Arm C (MAPHDE+Margin)", color="#50E3C2", linewidth=2)
    ax.set_xlabel("Max Heading Error (deg)")
    ax.set_ylabel("CDF")
    ax.set_title("Max Heading Error CDF")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 4. 14 Phase 18A Failure Cases Delta Bar Chart
    ax = axes[1, 0]
    cases = df_fail["scenario"].tolist()
    delta_B = df_fail["delta_B_m"].tolist()
    delta_C = df_fail["delta_C_m"].tolist()
    x = np.arange(len(cases))
    w = 0.4
    ax.bar(x - w/2, delta_B, width=w, label="Δ FPE (Arm B - A)", color="#E94A4A")
    ax.bar(x + w/2, delta_C, width=w, label="Δ FPE (Arm C - A)", color="#50E3C2")
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("sample_test_trajectory_motorway", "motorway") for c in cases], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ FPE (meters) [Negative = Better]")
    ax.set_title("14 Phase 18A Failures: Impact of MAPHDE")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 5. Representative Failure Case Trace: vw14b_o5 (Heading vs Time)
    ax = axes[1, 1]
    case_key = "vw14b_o5"
    if case_key in traces:
        tr = traces[case_key]
        t = tr["t"]
        gt_hdg = tr["gt_hdg"]
        hdg_A = tr["sim"]["A"]["heading_deg"]
        hdg_B = tr["sim"]["B"]["heading_deg"]
        if gt_hdg is not None:
            ax.plot(t[:len(gt_hdg)], gt_hdg, label="Ground Truth", color="black", linestyle="--", linewidth=1.5)
        ax.plot(t[:len(hdg_A)], hdg_A, label="Arm A (Baseline)", color="#4A90E2", linewidth=1.5)
        ax.plot(t[:len(hdg_B)], hdg_B, label="Arm B (MAPHDE)", color="#E94A4A", linewidth=1.5)
        ax.set_xlabel("Time in Outage (s)")
        ax.set_ylabel("Heading (deg)")
        ax.set_title(f"Case Replay: {case_key} Heading Tracking")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    # 6. Representative MAPHDE Integrator I(t) & Error E(t)
    ax = axes[1, 2]
    if case_key in traces:
        tr = traces[case_key]
        trace_B = tr["sim"]["B"]["trace"]
        if trace_B:
            t_map = trace_B["t"]
            I_vals = trace_B["I"]
            E_vals = trace_B["E_deg"]
            ax.plot(t_map, I_vals, label="Integrator I(t) (deg)", color="#E94A4A", linewidth=2)
            ax.plot(t_map, E_vals, label="Heading Error E(t) (deg)", color="#9013FE", linestyle=":", alpha=0.7)
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Degrees")
            ax.set_title(f"MAPHDE Integrator I(t) & Error: {case_key}")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plot_file = plots_dir / "phase19_fig1_maphde_benchmark.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    logger.info(f"Saved benchmark figure to {plot_file}")


if __name__ == "__main__":
    run_benchmark()
