"""
Phase 18B: Heading-Gated Map Decoupling Benchmark

Evaluates:
  - System A: CAN forward speed only (Phase 17A System B baseline)
  - System B: CAN + existing 1D road-normal map constraint (Phase 17A System C baseline)
  - System C30: CAN + 1D road-normal map constraint + heading-consistency gate (theta_max = 30 deg)
  - System C35: CAN + 1D road-normal map constraint + heading-consistency gate (theta_max = 35 deg)

Evaluates all 56 held-out outages and conducts in-depth analysis on the 14 Phase 17A failure cases.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import time
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
from modules.map_matcher import load_road_graph_from_osm_json, FixedLagHMMMapMatcher, RoadGraph
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase18B_Benchmark")

PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")


def run_outage_simulation_phase18b(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    alignment: AlignmentEngine,
    graph: Optional[RoadGraph] = None,
    mode: str = "B",  # 'A': CAN Only, 'B': CAN+Map, 'C30': CAN+Map (30 deg), 'C35': CAN+Map (35 deg)
    theta_max_deg: Optional[float] = None,
    r_eff: float = 0.2776,
    r_can_var: float = 0.0325,
    dt: float = config.TARGET_DT
) -> Dict[str, Any]:
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

    use_map = (mode != "A" and graph is not None)
    matcher = FixedLagHMMMapMatcher(graph=graph) if use_map else None

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading_deg = np.zeros(n)
    est_bg = np.zeros(n)

    heading_gate_first_t = None
    heading_gate_active_dur = 0.0
    total_map_epochs = 0
    abstained_map_epochs = 0

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
        if use_map and (k % 5 == 0):
            total_map_epochs += 1
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

                prev_hg_rejs = ukf.heading_gate_rejections
                applied, pos_nis, _ = ukf.update_map_match(
                    p_N_match=p_N_match,
                    p_E_match=p_E_match,
                    psi_road=float(mres.road_heading_rad),
                    confidence=float(mres.confidence),
                    is_heading_valid=False,
                    theta_max_deg=theta_max_deg
                )

                if not applied:
                    abstained_map_epochs += 1
                    if ukf.heading_gate_rejections > prev_hg_rejs:
                        if heading_gate_first_t is None:
                            heading_gate_first_t = k * dt
                        heading_gate_active_dur += 0.5
            else:
                abstained_map_epochs += 1

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]
        est_heading_deg[k] = math.degrees(ukf.x[3]) % 360.0
        est_bg[k] = ukf.x[4]

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)

    gt_lat = df_outage[config.COL_TRUE_LAT].values
    gt_lon = df_outage[config.COL_TRUE_LON].values
    R_earth = 6371000.0
    d_lat = np.radians(est_lat - gt_lat) * R_earth
    d_lon = np.radians(est_lon - gt_lon) * R_earth * np.cos(np.radians(gt_lat))
    pos_err_series = np.sqrt(d_lat**2 + d_lon**2)

    abstention_pct = (abstained_map_epochs / max(total_map_epochs, 1)) * 100.0

    return {
        "lat": est_lat,
        "lon": est_lon,
        "speed": est_speed,
        "heading_deg": est_heading_deg,
        "pN": est_pN,
        "pE": est_pE,
        "pos_err": pos_err_series,
        "map_accepted": ukf.accepted_map_updates,
        "map_rejected": ukf.rejected_map_updates,
        "map_total": ukf.total_map_updates,
        "nis_rejections": ukf.map_rejection_reasons["nis_gate"],
        "heading_gate_rejections": ukf.heading_gate_rejections,
        "abstention_pct": abstention_pct,
        "heading_gate_first_t": heading_gate_first_t,
        "heading_gate_active_dur": heading_gate_active_dur,
        "map_nis_hist": ukf.map_nis_history
    }


def run_phase18b_benchmark():
    logger.info("=" * 105)
    logger.info("PHASE 18B: HEADING-GATED MAP DECOUPLING BENCHMARK (A / B / C30 / C35)")
    logger.info("System A   = CAN Speed Only (Phase 17A System B baseline)")
    logger.info("System B   = CAN + 1D Road-Normal Map Constraint (Phase 17A System C baseline)")
    logger.info("System C30 = Phase 17A Map + Heading-Consistency Gate (theta_max = 30 deg)")
    logger.info("System C35 = Phase 17A Map + Heading-Consistency Gate (theta_max = 35 deg)")
    logger.info("=" * 105)

    # 1. Load Road Graph from OSM
    osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    logger.info(f"Loading canonical RoadGraph from {osm_path}...")
    t0 = time.time()
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)
    logger.info(f"Loaded canonical RoadGraph ({len(graph.nodes)} nodes, {len(graph.edges)} edges) in {time.time()-t0:.2f}s.")

    # 2. Test files across benchmark
    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    records = []
    traces_store = {}

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
            y = np.cross(u_z, ref)
            y /= np.linalg.norm(y)
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

            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, _, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values,
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            # 1. System A: CAN Speed Only (Phase 17A System B)
            res_a = run_outage_simulation_phase18b(
                df_sub, init_lat, init_lon, init_spd, init_hdg, run_align, mode="A", r_eff=r_eff_est
            )
            score_a = score_outage_segment(res_a["lat"], res_a["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 2. System B: CAN + Phase 17A Map Constraint (No Heading Gate)
            res_b = run_outage_simulation_phase18b(
                df_sub, init_lat, init_lon, init_spd, init_hdg, run_align, graph=graph, mode="B", theta_max_deg=None, r_eff=r_eff_est
            )
            score_b = score_outage_segment(res_b["lat"], res_b["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 3. System C30: Phase 17A Map + Heading Gate (theta_max = 30 deg)
            res_c30 = run_outage_simulation_phase18b(
                df_sub, init_lat, init_lon, init_spd, init_hdg, run_align, graph=graph, mode="C30", theta_max_deg=30.0, r_eff=r_eff_est
            )
            score_c30 = score_outage_segment(res_c30["lat"], res_c30["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 4. System C35: Phase 17A Map + Heading Gate (theta_max = 35 deg)
            res_c35 = run_outage_simulation_phase18b(
                df_sub, init_lat, init_lon, init_spd, init_hdg, run_align, graph=graph, mode="C35", theta_max_deg=35.0, r_eff=r_eff_est
            )
            score_c35 = score_outage_segment(res_c35["lat"], res_c35["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            sc_id = f"{run_name}_o{oid}"

            traces_store[sc_id] = {
                "t": np.arange(len(df_sub)) * config.TARGET_DT,
                "dur": dur, "run": run_name, "oid": oid,
                "gt_lat": gt_lat, "gt_lon": gt_lon, "gt_hdg": gt_hdg,
                "lat_a": res_a["lat"], "lon_a": res_a["lon"], "pos_err_a": res_a["pos_err"],
                "lat_b": res_b["lat"], "lon_b": res_b["lon"], "pos_err_b": res_b["pos_err"],
                "lat_c30": res_c30["lat"], "lon_c30": res_c30["lon"], "pos_err_c30": res_c30["pos_err"],
                "lat_c35": res_c35["lat"], "lon_c35": res_c35["lon"], "pos_err_c35": res_c35["pos_err"],
            }

            rec = {
                "scenario": sc_id,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                # System A (CAN Only)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                # System B (Phase 17A Map, No Gate)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
                "map_acc_b": res_b["map_accepted"],
                "map_rej_b": res_b["map_rejected"],
                "map_nis_rej_b": res_b["nis_rejections"],
                # System C30 (Phase 18B, theta_max = 30 deg)
                "fpe_c30_m": round(float(score_c30["final_pos_error_m"]), 2),
                "drift_c30_pct": round(float(score_c30["pct_of_distance"]), 2),
                "along_c30_m": round(float(score_c30.get("along_track_m", np.nan)), 2),
                "cross_c30_m": round(float(score_c30.get("cross_track_m", np.nan)), 2),
                "hdg_c30_deg": round(float(score_c30.get("heading_error_deg", np.nan)), 2),
                "map_acc_c30": res_c30["map_accepted"],
                "map_rej_c30": res_c30["map_rejected"],
                "map_nis_rej_c30": res_c30["nis_rejections"],
                "map_hdg_rej_c30": res_c30["heading_gate_rejections"],
                "abstain_pct_c30": round(res_c30["abstention_pct"], 1),
                "hdg_gate_first_t_c30": res_c30["heading_gate_first_t"],
                "hdg_gate_dur_c30": round(res_c30["heading_gate_active_dur"], 1),
                # System C35 (Phase 18B, theta_max = 35 deg)
                "fpe_c35_m": round(float(score_c35["final_pos_error_m"]), 2),
                "drift_c35_pct": round(float(score_c35["pct_of_distance"]), 2),
                "along_c35_m": round(float(score_c35.get("along_track_m", np.nan)), 2),
                "cross_c35_m": round(float(score_c35.get("cross_track_m", np.nan)), 2),
                "hdg_c35_deg": round(float(score_c35.get("heading_error_deg", np.nan)), 2),
                "map_acc_c35": res_c35["map_accepted"],
                "map_rej_c35": res_c35["map_rejected"],
                "map_nis_rej_c35": res_c35["nis_rejections"],
                "map_hdg_rej_c35": res_c35["heading_gate_rejections"],
                "abstain_pct_c35": round(res_c35["abstention_pct"], 1),
                "hdg_gate_first_t_c35": res_c35["heading_gate_first_t"],
                "hdg_gate_dur_c35": round(res_c35["heading_gate_active_dur"], 1),
                # Comparisons
                "c30_vs_b_diff_m": round(float(score_c30["final_pos_error_m"] - score_b["final_pos_error_m"]), 2),
                "c35_vs_b_diff_m": round(float(score_c35["final_pos_error_m"] - score_b["final_pos_error_m"]), 2)
            }
            records.append(rec)
            logger.info(
                f"[{sc_id}] Dur={dur:3d}s | FPE: A={rec['fpe_a_m']:6.1f}m, B={rec['fpe_b_m']:6.1f}m, "
                f"C30={rec['fpe_c30_m']:6.1f}m (Δ={rec['c30_vs_b_diff_m']:+6.1f}m), C35={rec['fpe_c35_m']:6.1f}m | "
                f"HdgRej: C30={rec['map_hdg_rej_c30']}, C35={rec['map_hdg_rej_c35']}"
            )

    df_results = pd.DataFrame(records)
    out_csv = BASE_DIR / "eval" / "phase18b_heading_gate_results.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 18B results to {out_csv}")

    # =========================================================================
    # 3. VERIFICATION: System B Reproduces Phase 17A Baseline
    # =========================================================================
    p17a_csv = BASE_DIR / "eval" / "phase17a_map_fusion_results.csv"
    if p17a_csv.exists():
        df_17a = pd.read_csv(p17a_csv)
        merged = pd.merge(df_results, df_17a, on="scenario", suffixes=("_18b", "_17a"))
        fpe_b_diff = np.abs(merged["fpe_b_m"] - merged["fpe_c_m"]).max()
        logger.info(f"VERIFICATION: Max FPE absolute difference between Phase 18B System B and Phase 17A Reference = {fpe_b_diff:.6f} m")
        if fpe_b_diff < 1e-2:
            logger.info("PASS: System B strictly reproduces Phase 17A reference within numerical precision!")
        else:
            logger.warning(f"CAUTION: Discrepancy observed: {fpe_b_diff:.4f} m")

    return df_results, traces_store


if __name__ == "__main__":
    run_phase18b_benchmark()
