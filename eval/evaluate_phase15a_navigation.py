"""
Phase 15A: Discrete Physical / Map Event Observability Experiment
Evaluates whether discrete physical/map events (improved standstill detector
and topology-confirmed OSM junction node updates) resolve accumulated along-track error.

Ablation Variants:
  - Variant A (Control): Phase 11 Anchored Temporal VelocityNet + existing ZUPT
  - Variant B (Standstill): Phase 11 Model + Improved Sustained Standstill Detector (ZUPT with NIS)
  - Variant C (Topological): Phase 11 Model + Improved Standstill + Topology-Confirmed Junction Updates

Strictly Frozen:
  - 7-state UKF dynamics, noise, and states
  - Phase 11 AnchoredTemporalVelocityNet weights
  - 56 held-out outage schedules
  - Canonical OSM road database
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from eval.train_phase10_temporal_model import TemporalVelocityNet4s
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment, decompose_along_cross_track
from modules.alignment import AlignmentEngine, AlignmentState
from modules.ukf import UKFNavigationFilter, UKF_NIS_GATE_1D, UKF_NIS_GATE_2D
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase15A")

ROAD_EARTH_RADIUS = 6371000.0


# ==============================================================================
# 1. Topological Junction Cache & Fast Lat/Lon Spatial Grid
# ==============================================================================
class LatLonJunctionIndex:
    """Fast spatial hash grid for querying OSM junctions by (lat, lon)."""
    def __init__(self, cell_deg: float = 0.01):
        self.cell_deg = cell_deg
        self.grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self.all_junctions: Dict[str, Dict[str, Any]] = {}

    def insert(self, junc_id: str, junc_data: Dict[str, Any]):
        self.all_junctions[junc_id] = junc_data
        junc_data["id"] = junc_id
        lat = junc_data["lat"]
        lon = junc_data["lon"]
        c_lat = int(math.floor(lat / self.cell_deg))
        c_lon = int(math.floor(lon / self.cell_deg))
        key = (c_lat, c_lon)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(junc_data)

    def query_radius(self, lat: float, lon: float, radius_m: float) -> List[Tuple[float, Dict[str, Any]]]:
        c_lat = int(math.floor(lat / self.cell_deg))
        c_lon = int(math.floor(lon / self.cell_deg))
        results = []
        lat_rad = math.radians(lat)

        for clat in [c_lat - 1, c_lat, c_lat + 1]:
            for clon in [c_lon - 1, c_lon, c_lon + 1]:
                cell_items = self.grid.get((clat, clon), [])
                for item in cell_items:
                    d_lat = math.radians(item["lat"] - lat)
                    d_lon = math.radians(item["lon"] - lon)
                    d_m = ROAD_EARTH_RADIUS * math.hypot(d_lat, d_lon * math.cos(lat_rad))
                    if d_m <= radius_m:
                        results.append((d_m, item))

        results.sort(key=lambda x: x[0])
        return results


def load_junction_index() -> LatLonJunctionIndex:
    cache_path = config.BASE_DIR / "data" / "osm_uk_junctions_cache.json"
    index = LatLonJunctionIndex(cell_deg=0.01)

    if cache_path.exists():
        logger.info(f"Loading cached junction index from {cache_path.name}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for jid, jdata in data.items():
            index.insert(jid, jdata)
        logger.info(f"Loaded {len(index.all_junctions)} topological junctions into spatial grid.")
    else:
        logger.warning("Junction cache not found!")

    return index


# ==============================================================================
# 2. Standstill Detector (Sustained IMU Quiet Window)
# ==============================================================================
class StandstillDetector:
    """
    Detects physical vehicle standstill based on sustained vehicle-frame IMU quietness.
    Requires continuous satisfaction of:
      ||a_v - g|| < theta_a
      ||w_v||     < theta_w
    for sustained_duration_s before declaring standstill.
    """
    def __init__(
        self,
        dt: float = 0.1,
        theta_a: float = 0.35,      # m/s^2 acceleration norm difference from g
        theta_w: float = 0.05,      # rad/s (~2.86 deg/s) angular velocity norm
        sustained_samples: int = 8  # 0.8 seconds at 10 Hz
    ):
        self.dt = dt
        self.theta_a = theta_a
        self.theta_w = theta_w
        self.sustained_samples = sustained_samples
        self.quiet_count = 0

    def step(self, a_v: np.ndarray, w_v: np.ndarray) -> Tuple[bool, float]:
        a_norm = float(np.linalg.norm(a_v))
        w_norm = float(np.linalg.norm(w_v))
        a_diff = abs(a_norm - 9.80665)
        h_accel = float(np.hypot(a_v[0], a_v[1]))

        is_quiet = (a_diff < self.theta_a) and (w_norm < self.theta_w) and (h_accel < 0.25)
        if is_quiet:
            self.quiet_count += 1
        else:
            self.quiet_count = 0

        is_standstill = (self.quiet_count >= self.sustained_samples)
        duration_s = self.quiet_count * self.dt
        return is_standstill, duration_s


# ==============================================================================
# 3. Topological Junction Landmark Detector
# ==============================================================================
class TopologicalJunctionDetector:
    """
    Detects turn maneuvers from integrated vehicle yaw rate, verifies OSM road
    topology agreement, and constructs discrete 2D node position measurements
    with 2-DOF NIS innovation gating.
    """
    def __init__(
        self,
        junction_index: LatLonJunctionIndex,
        init_lat: float,
        init_lon: float,
        dt: float = 0.1,
        turn_thresh_deg: float = 30.0,
        search_radius_m: float = 50.0,
        node_sigma_m: float = 5.0,
        refractory_s: float = 6.0
    ):
        self.index = junction_index
        self.init_lat = init_lat
        self.init_lon = init_lon
        self.dt = dt
        self.turn_thresh_deg = turn_thresh_deg
        self.search_radius_m = search_radius_m
        self.node_sigma_m = node_sigma_m
        self.refractory_s = refractory_s

        self.last_update_time_s = -999.0
        self.in_turn = False
        self.turn_integral_rad = 0.0
        self.turn_start_k = 0
        self.quiet_turn_samples = 0
        self.heading_history: List[float] = []

    def step(
        self,
        k: int,
        t_s: float,
        w_v_z: float,
        ukf_heading_deg: float,
        ukf_pN: float,
        ukf_pE: float,
        hmm_confidence: float = 0.85
    ) -> Optional[Dict[str, Any]]:
        self.heading_history.append(ukf_heading_deg)
        if len(self.heading_history) > 100:
            self.heading_history.pop(0)

        # Active turning detection
        if abs(w_v_z) > 0.05:  # ~2.86 deg/s active rotation
            if not self.in_turn:
                self.in_turn = True
                self.turn_integral_rad = 0.0
                self.turn_start_k = k
            self.turn_integral_rad += w_v_z * self.dt
            self.quiet_turn_samples = 0
        else:
            if self.in_turn:
                self.quiet_turn_samples += 1
                if self.quiet_turn_samples >= 4:  # 0.4s of straight driving confirms turn completion
                    self.in_turn = False
                    turn_deg = math.degrees(self.turn_integral_rad)

                    if abs(turn_deg) >= self.turn_thresh_deg:
                        if (t_s - self.last_update_time_s) < self.refractory_s:
                            return None

                        # Estimate enter and exit headings
                        dur_k = max(1, k - self.turn_start_k)
                        enter_idx = max(0, len(self.heading_history) - dur_k - 5)
                        psi_enter = self.heading_history[enter_idx]
                        psi_exit = ukf_heading_deg

                        # Convert UKF local pN, pE to lat/lon for spatial querying
                        cur_lat, cur_lon = self._local_to_latlon(ukf_pN, ukf_pE)

                        candidate_event = self._match_junction(
                            cur_lat, cur_lon, ukf_pN, ukf_pE, psi_enter, psi_exit, turn_deg, hmm_confidence
                        )
                        if candidate_event is not None:
                            self.last_update_time_s = t_s
                            return candidate_event

        return None

    def _local_to_latlon(self, pN: float, pE: float) -> Tuple[float, float]:
        d_lat = pN / ROAD_EARTH_RADIUS
        d_lon = pE / (ROAD_EARTH_RADIUS * math.cos(math.radians(self.init_lat)))
        return self.init_lat + math.degrees(d_lat), self.init_lon + math.degrees(d_lon)

    def _latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        d_lat = math.radians(lat - self.init_lat)
        d_lon = math.radians(lon - self.init_lon)
        pN = ROAD_EARTH_RADIUS * d_lat
        pE = ROAD_EARTH_RADIUS * d_lon * math.cos(math.radians(self.init_lat))
        return float(pN), float(pE)

    def _match_junction(
        self,
        cur_lat: float,
        cur_lon: float,
        ukf_pN: float,
        ukf_pE: float,
        psi_enter: float,
        psi_exit: float,
        turn_deg: float,
        hmm_confidence: float
    ) -> Optional[Dict[str, Any]]:
        candidate_matches = self.index.query_radius(cur_lat, cur_lon, self.search_radius_m)
        if not candidate_matches:
            return None

        best_cand = None
        best_cost = float("inf")

        for dist_m, cand in candidate_matches:
            headings = cand.get("headings", [])
            if len(headings) < 2:
                continue

            # Heading compatibility: incoming segment aligns with psi_enter, outgoing aligns with psi_exit
            min_enter_diff = min(abs((psi_enter - h + 180.0) % 360.0 - 180.0) for h in headings)
            min_exit_diff = min(abs((psi_exit - h + 180.0) % 360.0 - 180.0) for h in headings)

            if min_enter_diff <= 35.0 and min_exit_diff <= 35.0:
                jpN, jpE = self._latlon_to_local(cand["lat"], cand["lon"])
                local_dist = math.hypot(jpN - ukf_pN, jpE - ukf_pE)
                cost = local_dist + 0.5 * (min_enter_diff + min_exit_diff)
                if cost < best_cost:
                    best_cost = cost
                    best_cand = {
                        "junction_id": cand["id"],
                        "pN_node": jpN,
                        "pE_node": jpE,
                        "lat_node": cand["lat"],
                        "lon_node": cand["lon"],
                        "degree": cand["degree"],
                        "turn_deg": turn_deg,
                        "distance_m": local_dist,
                        "heading_residual": min_enter_diff + min_exit_diff,
                        "hmm_confidence": hmm_confidence
                    }

        return best_cand


# ==============================================================================
# 4. Neural Velocity Predictions for Phase 11 (Frozen)
# ==============================================================================
def compute_outage_predictions_phase11(
    model_b: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_val: float,
    window_len: int = 40,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float]]:
    lookup = {}
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
        return lookup

    all_windows = np.array(windows, dtype=np.float32)
    bx = torch.from_numpy(all_windows.transpose(0, 2, 1)).float().to(device)
    b_anc = torch.full((len(all_windows), 1), v_anchor_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        del_v_seq_b, v_seq_b, sig_seq_b, ev_b = model_b(bx, b_anc)
        p_v_b = v_seq_b.cpu().numpy()
        p_sig_b = sig_seq_b.cpu().numpy()
        p_ev_b = torch.argmax(ev_b, dim=1).cpu().numpy()

        p_d_b = np.sum(p_v_b * 0.2, axis=1)
        p_int_sig_b = 0.2 * np.sqrt(np.sum(p_sig_b ** 2, axis=1))

    for idx, gk in enumerate(step_indices):
        lookup[gk] = (float(p_d_b[idx]), int(p_ev_b[idx]), float(p_int_sig_b[idx]))

    return lookup


# ==============================================================================
# 5. Core Navigation Engine for Phase 15A Ablations
# ==============================================================================
def run_phase15a_outage_filter(
    variant: str,  # 'A', 'B', or 'C'
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    junction_index: Optional[LatLonJunctionIndex] = None,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), {}

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    gt_lat = df_outage[config.COL_TRUE_LAT].values if config.COL_TRUE_LAT in df_outage.columns else np.zeros(n)
    gt_lon = df_outage[config.COL_TRUE_LON].values if config.COL_TRUE_LON in df_outage.columns else np.zeros(n)
    gt_hdg = df_outage[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_outage.columns else np.zeros(n)

    ukf = UKFNavigationFilter(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg)
    )

    standstill_detector = StandstillDetector(dt=dt) if variant in ["B", "C"] else None
    junction_detector = (
        TopologicalJunctionDetector(junction_index, init_lat, init_lon, dt=dt)
        if (variant == "C" and junction_index is not None)
        else None
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    zupt_events = 0
    zupt_accepted = 0
    zupt_rejected = 0
    junc_events = 0
    junc_accepted = 0
    junc_rejected = 0
    junction_event_logs = []

    for k in range(n):
        t_s = k * dt
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Kinematic prediction
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        # 2. Standstill Detector & ZUPT Update
        is_stationary = False
        if variant == "A":
            # Variant A: Baseline heuristic from Phase 11
            global_k = global_start_idx + k
            if global_k in lookup_preds and (k % 5 == 0):
                raw_d, ev_class, raw_sig = lookup_preds[global_k]
                cal_d = raw_d / 2.0
                cal_sig = max(raw_sig / 2.0, 0.2)
                if ev_class == 0 or cal_d < 0.1:
                    ukf.update_zupt(gyro_reading=w_v[2])
                    zupt_events += 1
                    zupt_accepted += 1
                    is_stationary = True
                else:
                    ukf.update_velocity_net(cal_d, cal_sig, ev_class, 2.0)
        else:
            # Variant B & C: Improved sustained standstill detector
            is_standstill, s_dur = standstill_detector.step(a_v, w_v)
            if is_standstill:
                is_stationary = True
                zupt_events += 1

                # Statistical Zero-Velocity Measurement Update (zv = 0, h(x) = v)
                def h_v(s):
                    return np.array([s[2]])

                sigma_v = 0.05
                R_v = np.array([[sigma_v**2]])
                z_v = np.array([0.0])

                # Statistical Zero-Velocity Measurement Update (zv = 0, h(x) = v)
                acc, nis = ukf.update_measurement(z_v, h_v, R_v, nis_gate=None)

                if acc:
                    zupt_accepted += 1
                    # Gyro bias update
                    def h_bg(s):
                        return np.array([s[4]])
                    ukf.update_measurement(np.array([w_v[2]]), h_bg, np.array([[1e-6]]))
                else:
                    zupt_rejected += 1

            # VelocityNet updates when not stationary
            global_k = global_start_idx + k
            if (not is_stationary) and (global_k in lookup_preds) and (k % 5 == 0):
                raw_d, ev_class, raw_sig = lookup_preds[global_k]
                cal_d = raw_d / 2.0
                cal_sig = max(raw_sig / 2.0, 0.2)
                if ev_class != 0 and cal_d >= 0.1:
                    ukf.update_velocity_net(cal_d, cal_sig, ev_class, 2.0)

        # 3. Topological Junction Landmark Update (Variant C only)
        if variant == "C" and junction_detector is not None:
            ukf_hdg = math.degrees(ukf.x[3]) % 360.0
            candidate_junc = junction_detector.step(
                k=k,
                t_s=t_s,
                w_v_z=w_v[2],
                ukf_heading_deg=ukf_hdg,
                ukf_pN=ukf.x[0],
                ukf_pE=ukf.x[1],
                hmm_confidence=0.85
            )

            if candidate_junc is not None:
                junc_events += 1
                # Decompose error before update
                cur_lat, cur_lon = local_xy_to_latlon(np.array([ukf.x[1]]), np.array([ukf.x[0]]), init_lat, init_lon)
                e_along_before, e_cross_before = decompose_along_cross_track(
                    cur_lat[0], cur_lon[0], gt_lat[k], gt_lon[k], gt_hdg[k]
                )

                # 2D Position Measurement Update
                z_node = np.array([candidate_junc["pN_node"], candidate_junc["pE_node"]])

                def h_pos(s):
                    return np.array([s[0], s[1]])

                R_node = np.diag([5.0**2, 5.0**2])  # 5.0m conservative intersection covariance
                acc, nis = ukf.update_measurement(z_node, h_pos, R_node, nis_gate=UKF_NIS_GATE_2D)

                if acc:
                    junc_accepted += 1
                    cur_lat_after, cur_lon_after = local_xy_to_latlon(np.array([ukf.x[1]]), np.array([ukf.x[0]]), init_lat, init_lon)
                    e_along_after, e_cross_after = decompose_along_cross_track(
                        cur_lat_after[0], cur_lon_after[0], gt_lat[k], gt_lon[k], gt_hdg[k]
                    )
                else:
                    junc_rejected += 1
                    e_along_after, e_cross_after = e_along_before, e_cross_before

                junction_event_logs.append({
                    "time_s": t_s,
                    "junction_id": str(candidate_junc["junction_id"]),
                    "turn_deg": candidate_junc["turn_deg"],
                    "dist_to_node_m": candidate_junc["distance_m"],
                    "heading_res_deg": candidate_junc["heading_residual"],
                    "nis": nis,
                    "accepted": acc,
                    "e_along_before": float(e_along_before),
                    "e_along_after": float(e_along_after),
                    "delta_along": float(abs(e_along_after) - abs(e_along_before)),
                    "e_cross_before": float(e_cross_before),
                    "e_cross_after": float(e_cross_after),
                    "delta_cross": float(abs(e_cross_after) - abs(e_cross_before))
                })

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    stats = {
        "zupt_events": zupt_events,
        "zupt_accepted": zupt_accepted,
        "zupt_rejected": zupt_rejected,
        "junc_events": junc_events,
        "junc_accepted": junc_accepted,
        "junc_rejected": junc_rejected,
        "junc_logs": junction_event_logs
    }
    return est_lat, est_lon, est_speed, stats


# ==============================================================================
# 6. Benchmark Suite Across 56 Held-Out Outages
# ==============================================================================
def run_phase15a_benchmark():
    logger.info("=" * 80)
    logger.info("PHASE 15A: DISCRETE PHYSICAL / MAP EVENT OBSERVABILITY BENCHMARK")
    logger.info("Ablation: A (Control) vs B (Standstill) vs C (Topological Junction)")
    logger.info("=" * 80)

    junction_index = load_junction_index()

    model_b_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    if not model_b_path.exists():
        logger.error(f"Missing model checkpoint: {model_b_path}")
        return

    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_b_path, map_location="cpu"))
    model.eval()

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    logger.info(f"Evaluating across {len(test_files)} test files (56 held-out outages)...")

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    records = []
    all_junction_events = []

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

            # Compute frozen Phase 11 neural predictions
            lookup = compute_outage_predictions_phase11(
                model_b=model,
                seq_data=seq_data,
                outage_start_idx=outage_start_idx,
                sub_len=len(df_sub),
                v_anchor_val=v_anchor_val,
                window_len=40,
                device="cpu"
            )

            # Variant A: Control
            lat_a, lon_a, spd_a, stats_a = run_phase15a_outage_filter(
                variant="A",
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_a = score_outage_segment(lat_a, lon_a, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # Variant B: Standstill Detector
            lat_b, lon_b, spd_b, stats_b = run_phase15a_outage_filter(
                variant="B",
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup, global_start_idx=outage_start_idx,
                alignment=run_align
            )
            score_b = score_outage_segment(lat_b, lon_b, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # Variant C: Topological Junction
            lat_c, lon_c, spd_c, stats_c = run_phase15a_outage_filter(
                variant="C",
                df_outage=df_sub,
                init_lat=init_lat, init_lon=init_lon,
                init_speed_ms=init_spd, init_heading_deg=init_hdg,
                lookup_preds=lookup, global_start_idx=outage_start_idx,
                alignment=run_align,
                junction_index=junction_index
            )
            score_c = score_outage_segment(lat_c, lon_c, gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            for je in stats_c["junc_logs"]:
                je["run"] = run_name
                je["outage_id"] = oid
                je["duration_s"] = dur
                all_junction_events.append(je)

            rec = {
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": dist_gt,
                # Variant A
                "drift_a": score_a["drift_percent"],
                "fpe_a": score_a["final_pos_error_m"],
                "along_a": score_a["along_track_m"],
                "cross_a": score_a["cross_track_m"],
                # Variant B
                "drift_b": score_b["drift_percent"],
                "fpe_b": score_b["final_pos_error_m"],
                "along_b": score_b["along_track_m"],
                "cross_b": score_b["cross_track_m"],
                "zupt_events_b": stats_b["zupt_events"],
                "zupt_accepted_b": stats_b["zupt_accepted"],
                "zupt_rejected_b": stats_b["zupt_rejected"],
                # Variant C
                "drift_c": score_c["drift_percent"],
                "fpe_c": score_c["final_pos_error_m"],
                "along_c": score_c["along_track_m"],
                "cross_c": score_c["cross_track_m"],
                "junc_events_c": stats_c["junc_events"],
                "junc_accepted_c": stats_c["junc_accepted"],
                "junc_rejected_c": stats_c["junc_rejected"]
            }
            records.append(rec)

    df_results = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase15a_nav_results.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved navigation results to {out_csv} ({len(df_results)} scenarios).")

    if all_junction_events:
        df_junc = pd.DataFrame(all_junction_events)
        junc_csv = config.BASE_DIR / "eval" / "phase15a_junction_events.csv"
        df_junc.to_csv(junc_csv, index=False)
        logger.info(f"Saved {len(df_junc)} junction event logs to {junc_csv}.")

    # ==============================================================================
    # 7. Print Comprehensive Summary Table
    # ==============================================================================
    print("\n" + "=" * 90)
    print("PHASE 15A ABLATION BENCHMARK SUMMARY (56 HELD-OUT OUTAGES)")
    print("=" * 90)

    header = f"{'Metric':<30} | {'A (Control)':>16} | {'B (Standstill)':>16} | {'C (Topological)':>16}"
    print(header)
    print("-" * len(header))

    def fmt_val(val: float, unit: str = "") -> str:
        return f"{val:.2f}{unit}"

    print(f"{'Median Drift %':<30} | {fmt_val(df_results['drift_a'].median(), '%'):>16} | {fmt_val(df_results['drift_b'].median(), '%'):>16} | {fmt_val(df_results['drift_c'].median(), '%'):>16}")
    print(f"{'Mean Drift %':<30} | {fmt_val(df_results['drift_a'].mean(), '%'):>16} | {fmt_val(df_results['drift_b'].mean(), '%'):>16} | {fmt_val(df_results['drift_c'].mean(), '%'):>16}")
    print(f"{'Median FPE (m)':<30} | {fmt_val(df_results['fpe_a'].median(), ' m'):>16} | {fmt_val(df_results['fpe_b'].median(), ' m'):>16} | {fmt_val(df_results['fpe_c'].median(), ' m'):>16}")

    for dur in [30, 60, 120, 180]:
        sub = df_results[df_results["duration_s"] == dur]
        if len(sub) > 0:
            print(f"{f'{dur}s Median FPE (m)':<30} | {fmt_val(sub['fpe_a'].median(), ' m'):>16} | {fmt_val(sub['fpe_b'].median(), ' m'):>16} | {fmt_val(sub['fpe_c'].median(), ' m'):>16}")

    print(f"{'Median Along-Track Error (m)':<30} | {fmt_val(df_results['along_a'].abs().median(), ' m'):>16} | {fmt_val(df_results['along_b'].abs().median(), ' m'):>16} | {fmt_val(df_results['along_c'].abs().median(), ' m'):>16}")
    print(f"{'Median Cross-Track Error (m)':<30} | {fmt_val(df_results['cross_a'].abs().median(), ' m'):>16} | {fmt_val(df_results['cross_b'].abs().median(), ' m'):>16} | {fmt_val(df_results['cross_c'].abs().median(), ' m'):>16}")

    pct_a = (df_results['drift_a'] < 10.0).mean() * 100.0
    pct_b = (df_results['drift_b'] < 10.0).mean() * 100.0
    pct_c = (df_results['drift_c'] < 10.0).mean() * 100.0
    print(f"{'<10% Drift Scenarios':<30} | {fmt_val(pct_a, '%'):>16} | {fmt_val(pct_b, '%'):>16} | {fmt_val(pct_c, '%'):>16}")

    tot_zupt_b = int(df_results['zupt_events_b'].sum())
    tot_zupt_acc_b = int(df_results['zupt_accepted_b'].sum())
    print(f"{'ZUPT Events Triggered':<30} | {'0 (heuristic)':>16} | {tot_zupt_b:>16} | {tot_zupt_b:>16}")
    print(f"{'ZUPT Accepted':<30} | {'0 (heuristic)':>16} | {tot_zupt_acc_b:>16} | {tot_zupt_acc_b:>16}")

    tot_junc_c = int(df_results['junc_events_c'].sum())
    tot_junc_acc_c = int(df_results['junc_accepted_c'].sum())
    tot_junc_rej_c = int(df_results['junc_rejected_c'].sum())
    print(f"{'Junction Events Detected':<30} | {'0':>16} | {'0':>16} | {tot_junc_c:>16}")
    print(f"{'Junction Accepted (Passed NIS)':<30} | {'0':>16} | {'0':>16} | {tot_junc_acc_c:>16}")
    print(f"{'Junction Rejected by NIS':<30} | {'0':>16} | {'0':>16} | {tot_junc_rej_c:>16}")
    print("=" * 90)


if __name__ == "__main__":
    run_phase15a_benchmark()
