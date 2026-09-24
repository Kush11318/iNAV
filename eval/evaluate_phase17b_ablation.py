"""
Phase 17B: Map Heading Ablation Benchmark

Evaluates:
  - System A: CAN Forward Speed Only (Phase 16A Baseline Control)
  - System B: CAN Forward Speed + Existing Phase 17A Road-Normal HMM/OSM Constraint
              (Exact Phase 17A Reference, is_heading_valid=False)
  - System C: CAN Forward Speed + Road-Normal Map Constraint + Controlled Road-Heading Measurement
              (is_heading_valid=True, tested with canonical sigma=2.0 deg, plus 1.0 deg and 5.0 deg sensitivity)

Primary Research Question:
  Does explicit road-heading information provide additional benefit beyond the Phase 17A
  road-normal constraint (B -> C comparison)?

Strict Constraints:
  - Zero modifications to production code, UKF, Android JNI, model weights.
  - Same 56 held-out IO-VNBD GNSS outage scenarios.
  - Same 7-state UKF, same preprocessing, same outage schedules, same initialization, same OSM road graph.
  - Zero ground truth leakage during outages.
  - System B reproduces Phase 17A results within numerical tolerance.
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
logger = logging.getLogger("iNAV.Phase17B_Ablation")

ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_outage_simulation_phase17b(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    alignment: AlignmentEngine,
    graph: Optional[RoadGraph] = None,
    mode: str = "B",  # 'A': CAN Speed Only, 'B': CAN + Road Normal (Phase 17A), 'C': CAN + Normal + Heading
    sigma_heading_deg: Optional[float] = 2.0,
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
        allow_bias_learning=False  # Keep gyro bias diffusion baseline
    )

    use_map = (mode in ["B", "C"] and graph is not None)
    matcher = FixedLagHMMMapMatcher(graph=graph) if use_map else None

    is_heading_valid = (mode == "C")
    sig_hdg_rad = math.radians(sigma_heading_deg) if (is_heading_valid and sigma_heading_deg is not None) else None

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading_deg = np.zeros(n)
    est_bg = np.zeros(n)

    # Forensic audit logs per step
    map_audit = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Prediction step (IMU Process Model)
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

            step_audit = {
                "step": k,
                "t": k * dt,
                "is_off_road": mres.is_off_road,
                "confidence": float(mres.confidence),
                "edge_id": getattr(mres, "edge_id", 0),
                "road_heading_deg": float(mres.heading_deg),
                "filter_heading_deg": hdg_deg,
                "pos_update_applied": False,
                "pos_nis": None,
                "hdg_update_applied": False,
                "hdg_nis": None,
                "rejection_reason": None
            }

            if not mres.is_off_road and mres.confidence >= 0.25:
                snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                d_lat = math.radians(snap_lat - init_lat)
                d_lon = math.radians(snap_lon - init_lon)
                p_N_match = float(6371000.0 * d_lat)
                p_E_match = float(6371000.0 * d_lon * math.cos(math.radians(init_lat)))

                applied, pos_nis, hdg_nis = ukf.update_map_match(
                    p_N_match=p_N_match,
                    p_E_match=p_E_match,
                    psi_road=float(mres.road_heading_rad),
                    confidence=float(mres.confidence),
                    is_heading_valid=is_heading_valid,
                    sigma_heading_rad=sig_hdg_rad
                )
                step_audit["pos_update_applied"] = applied
                step_audit["pos_nis"] = pos_nis
                step_audit["hdg_nis"] = hdg_nis
                if not applied:
                    step_audit["rejection_reason"] = "pos_nis_gated"
                if hdg_nis is not None:
                    step_audit["hdg_update_applied"] = True
            else:
                if mres.is_off_road:
                    step_audit["rejection_reason"] = "off_road"
                else:
                    step_audit["rejection_reason"] = "low_confidence"

            map_audit.append(step_audit)

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

    return {
        "lat": est_lat,
        "lon": est_lon,
        "speed": est_speed,
        "heading_deg": est_heading_deg,
        "pN": est_pN,
        "pE": est_pE,
        "bg": est_bg,
        "pos_err": pos_err_series,
        "map_accepted": ukf.accepted_map_updates,
        "map_rejected": ukf.rejected_map_updates,
        "map_total": ukf.total_map_updates,
        "map_nis_hist": ukf.map_nis_history,
        "map_rejection_reasons": ukf.map_rejection_reasons,
        "map_heading_accepted": ukf.accepted_map_heading_updates,
        "map_heading_rejected": ukf.rejected_map_heading_updates,
        "map_heading_total": ukf.total_map_heading_updates,
        "map_heading_nis_hist": ukf.map_heading_nis_history,
        "map_audit": map_audit
    }


def run_phase17b_benchmark():
    logger.info("=" * 95)
    logger.info("PHASE 17B: MAP HEADING ABLATION BENCHMARK (A / B / C)")
    logger.info("System A = CAN Forward Speed Only (Phase 16A Baseline Control)")
    logger.info("System B = CAN Speed + Existing Phase 17A Road-Normal HMM/OSM Constraint (Reference)")
    logger.info("System C = CAN Speed + Road-Normal Constraint + Controlled Road-Heading (Phase 17B)")
    logger.info("=" * 95)

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

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    records = []
    traces_store = {}
    forensic_audits = {}

    # Aggregators for Map updates & NIS
    pos_nis_all = []
    hdg_nis_all = []
    total_pos_acc = 0
    total_pos_tot = 0
    total_hdg_acc = 0
    total_hdg_tot = 0

    # Sensitivity storage
    sensitivity_records = []

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

            # Pre-outage CAN speed calibration
            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, cal_method_can, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values,
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            # 1. System A: Phase 16A CAN Speed Only (Control)
            res_a = run_outage_simulation_phase17b(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                run_align, mode="A", r_eff=r_eff_est
            )
            score_a = score_outage_segment(res_a["lat"], res_a["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 2. System B: CAN Speed + Road-Normal Map Constraint (Phase 17A Reference)
            res_b = run_outage_simulation_phase17b(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                run_align, graph=graph, mode="B", r_eff=r_eff_est
            )
            score_b = score_outage_segment(res_b["lat"], res_b["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 3. System C: CAN Speed + Road-Normal + Controlled Heading (Canonical sigma=2.0 deg)
            res_c = run_outage_simulation_phase17b(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                run_align, graph=graph, mode="C", sigma_heading_deg=2.0, r_eff=r_eff_est
            )
            score_c = score_outage_segment(res_c["lat"], res_c["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            # 4. Sensitivity variants for System C: sigma = 1.0 deg and sigma = 5.0 deg
            res_c_1deg = run_outage_simulation_phase17b(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                run_align, graph=graph, mode="C", sigma_heading_deg=1.0, r_eff=r_eff_est
            )
            score_c_1deg = score_outage_segment(res_c_1deg["lat"], res_c_1deg["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            res_c_5deg = run_outage_simulation_phase17b(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                run_align, graph=graph, mode="C", sigma_heading_deg=5.0, r_eff=r_eff_est
            )
            score_c_5deg = score_outage_segment(res_c_5deg["lat"], res_c_5deg["lon"], gt_lat, gt_lon, dist_gt, dur, gt_heading_deg=gt_hdg)

            sc_id = f"{run_name}_o{oid}"

            pos_nis_all.extend(res_c["map_nis_hist"])
            hdg_nis_all.extend(res_c["map_heading_nis_hist"])
            total_pos_acc += res_c["map_accepted"]
            total_pos_tot += res_c["map_total"]
            total_hdg_acc += res_c["map_heading_accepted"]
            total_hdg_tot += res_c["map_heading_total"]

            forensic_audits[sc_id] = {
                "dur": dur,
                "dist_gt": dist_gt,
                "fpe_a": float(score_a["final_pos_error_m"]),
                "fpe_b": float(score_b["final_pos_error_m"]),
                "fpe_c": float(score_c["final_pos_error_m"]),
                "hdg_err_b": float(score_b.get("heading_error_deg", np.nan)),
                "hdg_err_c": float(score_c.get("heading_error_deg", np.nan)),
                "audit": res_c["map_audit"]
            }

            traces_store[sc_id] = {
                "t": np.arange(len(df_sub)) * config.TARGET_DT,
                "dur": dur, "run": run_name, "oid": oid,
                "gt_lat": gt_lat, "gt_lon": gt_lon, "gt_hdg": gt_hdg,
                "lat_a": res_a["lat"], "lon_a": res_a["lon"], "hdg_a": res_a["heading_deg"], "pos_err_a": res_a["pos_err"],
                "lat_b": res_b["lat"], "lon_b": res_b["lon"], "hdg_b": res_b["heading_deg"], "pos_err_b": res_b["pos_err"],
                "lat_c": res_c["lat"], "lon_c": res_c["lon"], "hdg_c": res_c["heading_deg"], "pos_err_c": res_c["pos_err"],
            }

            pos_acc_rate = (res_c["map_accepted"] / max(res_c["map_total"], 1)) * 100.0
            hdg_acc_rate = (res_c["map_heading_accepted"] / max(res_c["map_heading_total"], 1)) * 100.0
            mean_pos_nis = float(np.mean(res_c["map_nis_hist"])) if res_c["map_nis_hist"] else 0.0
            mean_hdg_nis = float(np.mean(res_c["map_heading_nis_hist"])) if res_c["map_heading_nis_hist"] else 0.0

            fpe_diff_b_to_c = float(score_c["final_pos_error_m"] - score_b["final_pos_error_m"])
            b_to_c_status = "IMPROVED" if fpe_diff_b_to_c < -0.05 else ("WORSENED" if fpe_diff_b_to_c > 0.05 else "UNCHANGED")

            rec = {
                "scenario": sc_id,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(dist_gt, 2),
                "r_eff": round(r_eff_est, 4),
                # System A: CAN Speed Only (Phase 16A Baseline)
                "fpe_a_m": round(float(score_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(score_a["pct_of_distance"]), 2),
                "along_a_m": round(float(score_a.get("along_track_m", np.nan)), 2),
                "cross_a_m": round(float(score_a.get("cross_track_m", np.nan)), 2),
                "hdg_a_deg": round(float(score_a.get("heading_error_deg", np.nan)), 2),
                # System B: CAN Speed + Road Normal (Phase 17A Reference)
                "fpe_b_m": round(float(score_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(score_b["pct_of_distance"]), 2),
                "along_b_m": round(float(score_b.get("along_track_m", np.nan)), 2),
                "cross_b_m": round(float(score_b.get("cross_track_m", np.nan)), 2),
                "hdg_b_deg": round(float(score_b.get("heading_error_deg", np.nan)), 2),
                # System C: CAN Speed + Road Normal + Road Heading (Canonical sigma=2 deg)
                "fpe_c_m": round(float(score_c["final_pos_error_m"]), 2),
                "drift_c_pct": round(float(score_c["pct_of_distance"]), 2),
                "along_c_m": round(float(score_c.get("along_track_m", np.nan)), 2),
                "cross_c_m": round(float(score_c.get("cross_track_m", np.nan)), 2),
                "hdg_c_deg": round(float(score_c.get("heading_error_deg", np.nan)), 2),
                # Map Statistics (Position)
                "pos_map_accepted": res_c["map_accepted"],
                "pos_map_rejected": res_c["map_rejected"],
                "pos_map_acc_pct": round(pos_acc_rate, 1),
                "mean_pos_nis": round(mean_pos_nis, 2),
                # Map Statistics (Heading)
                "hdg_map_accepted": res_c["map_heading_accepted"],
                "hdg_map_rejected": res_c["map_heading_rejected"],
                "hdg_map_acc_pct": round(hdg_acc_rate, 1),
                "mean_hdg_nis": round(mean_hdg_nis, 2),
                # Sensitivity FPE
                "fpe_c_1deg_m": round(float(score_c_1deg["final_pos_error_m"]), 2),
                "fpe_c_5deg_m": round(float(score_c_5deg["final_pos_error_m"]), 2),
                # Primary Comparison: B -> C
                "b_to_c_fpe_diff_m": round(fpe_diff_b_to_c, 2),
                "b_to_c_status": b_to_c_status
            }
            records.append(rec)

            logger.info(
                f"[{sc_id}] Dur={dur:3d}s | FPE: A={rec['fpe_a_m']:6.1f}m, B={rec['fpe_b_m']:6.1f}m, C={rec['fpe_c_m']:6.1f}m "
                f"({b_to_c_status}, Δ={rec['b_to_c_fpe_diff_m']:+6.1f}m) | "
                f"HdgErr: B={rec['hdg_b_deg']:4.1f}°, C={rec['hdg_c_deg']:4.1f}° | HdgAcc={res_c['map_heading_accepted']}/{res_c['map_heading_total']}"
            )

    df_results = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase17b_heading_ablation_results.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved Phase 17B results to {out_csv}")

    # =========================================================================
    # VERIFICATION: System B Reproduces Phase 17A Baseline
    # =========================================================================
    p17a_csv = config.BASE_DIR / "eval" / "phase17a_map_fusion_results.csv"
    if p17a_csv.exists():
        df_17a = pd.read_csv(p17a_csv)
        merged = pd.merge(df_results, df_17a, on="scenario", suffixes=("_17b", "_17a"))
        fpe_b_diff = np.abs(merged["fpe_b_m_17b"] - merged["fpe_c_m_17a"]).max()
        logger.info(f"VERIFICATION: Max FPE absolute difference between Phase 17B System B and Phase 17A Reference = {fpe_b_diff:.4f} m")
        if fpe_b_diff < 1e-2:
            logger.info("PASS: System B strictly reproduces Phase 17A reference within numerical precision!")
        else:
            logger.warning(f"CAUTION: Discrepancy observed: {fpe_b_diff:.4f} m")

    # =========================================================================
    # AGGREGATE SUMMARY
    # =========================================================================
    print("\n" + "=" * 135)
    print("PHASE 17B: MAP HEADING ABLATION BENCHMARK RESULTS (A vs B vs C)")
    print("=" * 135)

    def format_row(title, val_a, val_b, val_c, unit=""):
        delta_bc = val_c - val_b
        pct_bc = (delta_bc / val_b * 100.0) if abs(val_b) > 1e-6 else 0.0
        sign = "+" if delta_bc > 0 else ""
        return f"{title:<32} | {val_a:8.2f} {unit:<3} | {val_b:8.2f} {unit:<3} | {val_c:8.2f} {unit:<3} | {sign}{delta_bc:6.2f} {unit:<3} ({sign}{pct_bc:5.1f}%)"

    med_fpe_a = df_results["fpe_a_m"].median()
    med_fpe_b = df_results["fpe_b_m"].median()
    med_fpe_c = df_results["fpe_c_m"].median()

    mean_fpe_a = df_results["fpe_a_m"].mean()
    mean_fpe_b = df_results["fpe_b_m"].mean()
    mean_fpe_c = df_results["fpe_c_m"].mean()

    med_drift_a = df_results["drift_a_pct"].median()
    med_drift_b = df_results["drift_b_pct"].median()
    med_drift_c = df_results["drift_c_pct"].median()

    mean_drift_a = df_results["drift_a_pct"].mean()
    mean_drift_b = df_results["drift_b_pct"].mean()
    mean_drift_c = df_results["drift_c_pct"].mean()

    med_along_a = df_results["along_a_m"].median()
    med_along_b = df_results["along_b_m"].median()
    med_along_c = df_results["along_c_m"].median()

    mean_along_a = df_results["along_a_m"].mean()
    mean_along_b = df_results["along_b_m"].mean()
    mean_along_c = df_results["along_c_m"].mean()

    med_cross_a = df_results["cross_a_m"].median()
    med_cross_b = df_results["cross_b_m"].median()
    med_cross_c = df_results["cross_c_m"].median()

    mean_cross_a = df_results["cross_a_m"].mean()
    mean_cross_b = df_results["cross_b_m"].mean()
    mean_cross_c = df_results["cross_c_m"].mean()

    med_hdg_a = df_results["hdg_a_deg"].median()
    med_hdg_b = df_results["hdg_b_deg"].median()
    med_hdg_c = df_results["hdg_c_deg"].median()

    mean_hdg_a = df_results["hdg_a_deg"].mean()
    mean_hdg_b = df_results["hdg_b_deg"].mean()
    mean_hdg_c = df_results["hdg_c_deg"].mean()

    n_below_10_a = (df_results["drift_a_pct"] < 10.0).sum()
    n_below_10_b = (df_results["drift_b_pct"] < 10.0).sum()
    n_below_10_c = (df_results["drift_c_pct"] < 10.0).sum()

    n_improved_bc = (df_results["b_to_c_status"] == "IMPROVED").sum()
    n_worsened_bc = (df_results["b_to_c_status"] == "WORSENED").sum()
    n_unchanged_bc = (df_results["b_to_c_status"] == "UNCHANGED").sum()

    print(f"{'Metric':<32} | {'System A':<12} | {'System B':<12} | {'System C':<12} | {'B -> C Delta':<18}")
    print("-" * 135)
    print(format_row("Median FPE", med_fpe_a, med_fpe_b, med_fpe_c, "m"))
    print(format_row("Mean FPE", mean_fpe_a, mean_fpe_b, mean_fpe_c, "m"))
    print(format_row("Median Drift %", med_drift_a, med_drift_b, med_drift_c, "%"))
    print(format_row("Mean Drift %", mean_drift_a, mean_drift_b, mean_drift_c, "%"))
    print(format_row("Median Along-Track Error", med_along_a, med_along_b, med_along_c, "m"))
    print(format_row("Mean Along-Track Error", mean_along_a, mean_along_b, mean_along_c, "m"))
    print(format_row("Median Cross-Track Error", med_cross_a, med_cross_b, med_cross_c, "m"))
    print(format_row("Mean Cross-Track Error", mean_cross_a, mean_cross_b, mean_cross_c, "m"))
    print(format_row("Median Heading Error", med_hdg_a, med_hdg_b, med_hdg_c, "°"))
    print(format_row("Mean Heading Error", mean_hdg_a, mean_hdg_b, mean_hdg_c, "°"))
    print("-" * 135)
    print(f"Scenarios Below 10% Drift: A = {n_below_10_a}/56 ({n_below_10_a/56*100:.1f}%), B = {n_below_10_b}/56 ({n_below_10_b/56*100:.1f}%), C = {n_below_10_c}/56 ({n_below_10_c/56*100:.1f}%)")
    print(f"B -> C Comparison: {n_improved_bc} Improved, {n_worsened_bc} Worsened, {n_unchanged_bc} Unchanged (Net: {n_improved_bc - n_worsened_bc:+d})")
    print(f"Map Position Updates: {total_pos_acc}/{total_pos_tot} ({total_pos_acc/max(total_pos_tot, 1)*100:.1f}%) accepted")
    print(f"Map Heading Updates:  {total_hdg_acc}/{total_hdg_tot} ({total_hdg_acc/max(total_hdg_tot, 1)*100:.1f}%) accepted")

    # NIS summary
    if pos_nis_all:
        print(f"Position NIS: Mean = {np.mean(pos_nis_all):.2f}, Median = {np.median(pos_nis_all):.2f}, 95th = {np.percentile(pos_nis_all, 95):.2f}")
    if hdg_nis_all:
        print(f"Heading NIS:  Mean = {np.mean(hdg_nis_all):.2f}, Median = {np.median(hdg_nis_all):.2f}, 95th = {np.percentile(hdg_nis_all, 95):.2f}")

    # =========================================================================
    # DURATION BREAKDOWN TABLE
    # =========================================================================
    print("\n" + "=" * 135)
    print("DURATION BREAKDOWN (Median FPE / Drift % / Heading Error)")
    print("=" * 135)
    print(f"{'Duration':<10} | {'Count':<6} | {'A FPE (m)':<10} | {'B FPE (m)':<10} | {'C FPE (m)':<10} | {'B->C Δ FPE':<12} | {'B Drift%':<10} | {'C Drift%':<10} | {'B Hdg(°)':<10} | {'C Hdg(°)':<10} | {'B->C Imp/Wor'}")
    print("-" * 135)

    duration_stats = {}
    for dur_val in [10, 30, 60, 120, 180]:
        sub = df_results[df_results["duration_s"] == dur_val]
        cnt = len(sub)
        if cnt == 0:
            continue
        fpe_a = sub["fpe_a_m"].median()
        fpe_b = sub["fpe_b_m"].median()
        fpe_c = sub["fpe_c_m"].median()
        dfpe = fpe_c - fpe_b
        drf_b = sub["drift_b_pct"].median()
        drf_c = sub["drift_c_pct"].median()
        hdg_b = sub["hdg_b_deg"].median()
        hdg_c = sub["hdg_c_deg"].median()
        n_imp = (sub["b_to_c_status"] == "IMPROVED").sum()
        n_wor = (sub["b_to_c_status"] == "WORSENED").sum()
        print(f"{dur_val:3d}s        | {cnt:<6} | {fpe_a:10.2f} | {fpe_b:10.2f} | {fpe_c:10.2f} | {dfpe:+10.2f}m  | {drf_b:9.2f}% | {drf_c:9.2f}% | {hdg_b:9.2f}° | {hdg_c:9.2f}° | {n_imp} / {n_wor}")
        duration_stats[dur_val] = {
            "cnt": cnt, "fpe_a": fpe_a, "fpe_b": fpe_b, "fpe_c": fpe_c, "dfpe": dfpe,
            "drf_b": drf_b, "drf_c": drf_c, "hdg_b": hdg_b, "hdg_c": hdg_c, "n_imp": n_imp, "n_wor": n_wor
        }

    # =========================================================================
    # SENSITIVITY TABLE
    # =========================================================================
    print("\n" + "=" * 135)
    print("PARAMETER SENSITIVITY: sigma_heading = {1.0°, 2.0°, 5.0°}")
    print("=" * 135)
    print(f"System B (No Heading):         Median FPE = {med_fpe_b:.2f} m | Mean FPE = {mean_fpe_b:.2f} m")
    print(f"System C (sigma = 1.0 deg):    Median FPE = {df_results['fpe_c_1deg_m'].median():.2f} m | Mean FPE = {df_results['fpe_c_1deg_m'].mean():.2f} m")
    print(f"System C (sigma = 2.0 deg):    Median FPE = {med_fpe_c:.2f} m | Mean FPE = {mean_fpe_c:.2f} m")
    print(f"System C (sigma = 5.0 deg):    Median FPE = {df_results['fpe_c_5deg_m'].median():.2f} m | Mean FPE = {df_results['fpe_c_5deg_m'].mean():.2f} m")

    # =========================================================================
    # FORENSIC FAILURE AUDIT (120s and 180s Outages)
    # =========================================================================
    print("\n" + "=" * 135)
    print("FORENSIC AUDIT: 120s & 180s SCENARIOS (Phase 17A vs Phase 17B)")
    print("=" * 135)
    long_outages = df_results[df_results["duration_s"] >= 120].sort_values("fpe_b_m", ascending=False)
    for _, row in long_outages.iterrows():
        sc = row["scenario"]
        dur = row["duration_s"]
        audit_info = forensic_audits.get(sc, {})
        aud_steps = audit_info.get("audit", [])
        
        # Categorize failure / behavior
        n_steps = len(aud_steps)
        n_off_road = sum(1 for s in aud_steps if s["is_off_road"])
        n_low_conf = sum(1 for s in aud_steps if not s["is_off_road"] and s["confidence"] < 0.25)
        n_pos_nis_rej = sum(1 for s in aud_steps if s["rejection_reason"] == "pos_nis_gated")
        n_hdg_applied = sum(1 for s in aud_steps if s["hdg_update_applied"])
        n_hdg_rej = sum(1 for s in aud_steps if s["pos_update_applied"] and not s["hdg_update_applied"])

        print(f"[{sc}] Dur={dur}s | FPE: B={row['fpe_b_m']:6.1f}m, C={row['fpe_c_m']:6.1f}m (Δ={row['b_to_c_fpe_diff_m']:+6.1f}m) | "
              f"HdgErr: B={row['hdg_b_deg']:4.1f}°, C={row['hdg_c_deg']:4.1f}° | "
              f"Steps={n_steps}, OffRoad={n_off_road}, LowConf={n_low_conf}, PosNISRej={n_pos_nis_rej}, HdgAcc={n_hdg_applied}/{n_hdg_applied+n_hdg_rej}")

    # =========================================================================
    # GENERATE PUBLICATION-GRADE PLOTS
    # =========================================================================
    logger.info("Generating publication-grade figures...")

    # Figure 1: Summary Comparison Bar Charts (FPE, Drift %, Along, Cross, Heading)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Phase 17B: Map Heading Ablation Summary (56 Held-Out Outages)", fontsize=15, fontweight="bold")

    metrics_to_plot = [
        ("Median FPE (m)", [med_fpe_a, med_fpe_b, med_fpe_c], axes[0, 0]),
        ("Median Drift (%)", [med_drift_a, med_drift_b, med_drift_c], axes[0, 1]),
        ("Median Heading Error (°)", [med_hdg_a, med_hdg_b, med_hdg_c], axes[0, 2]),
        ("Median Along-Track Error (m)", [med_along_a, med_along_b, med_along_c], axes[1, 0]),
        ("Median Cross-Track Error (m)", [med_cross_a, med_cross_b, med_cross_c], axes[1, 1]),
        ("Scenarios < 10% Drift", [n_below_10_a, n_below_10_b, n_below_10_c], axes[1, 2])
    ]

    colors = ["#4A90E2", "#E2844A", "#2ECC71"]
    systems = ["Sys A\n(CAN Only)", "Sys B\n(CAN+Normal)", "Sys C\n(CAN+Normal+Hdg)"]

    for title, vals, ax in metrics_to_plot:
        bars = ax.bar(systems, vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            yval = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, yval * 1.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig1_path = PLOTS_DIR / "phase17b_fig1_ablation_summary.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Also save to artifacts directory
    fig1_art = ARTIFACTS_DIR / "phase17b_fig1_ablation_summary.png"
    plt.figure()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Phase 17B: Map Heading Ablation Summary (56 Held-Out Outages)", fontsize=15, fontweight="bold")
    for title, vals, ax in metrics_to_plot:
        bars = ax.bar(systems, vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            yval = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, yval * 1.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig1_art, dpi=300)
    plt.close()

    # Figure 2: Duration Breakdown (FPE and Heading Error vs Outage Duration)
    durs = [10, 30, 60, 120, 180]
    fpe_a_durs = [df_results[df_results["duration_s"] == d]["fpe_a_m"].median() for d in durs]
    fpe_b_durs = [df_results[df_results["duration_s"] == d]["fpe_b_m"].median() for d in durs]
    fpe_c_durs = [df_results[df_results["duration_s"] == d]["fpe_c_m"].median() for d in durs]

    hdg_a_durs = [df_results[df_results["duration_s"] == d]["hdg_a_deg"].median() for d in durs]
    hdg_b_durs = [df_results[df_results["duration_s"] == d]["hdg_b_deg"].median() for d in durs]
    hdg_c_durs = [df_results[df_results["duration_s"] == d]["hdg_c_deg"].median() for d in durs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phase 17B: Performance by Outage Duration (10s to 180s)", fontsize=15, fontweight="bold")

    x = np.arange(len(durs))
    width = 0.25

    ax1.bar(x - width, fpe_a_durs, width, label="System A (CAN Only)", color="#4A90E2", edgecolor="black")
    ax1.bar(x, fpe_b_durs, width, label="System B (CAN + Road-Normal)", color="#E2844A", edgecolor="black")
    ax1.bar(x + width, fpe_c_durs, width, label="System C (CAN + Normal + Heading)", color="#2ECC71", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}s" for d in durs])
    ax1.set_ylabel("Median FPE (m)", fontsize=12)
    ax1.set_title("Median FPE vs Outage Duration", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(x - width, hdg_a_durs, width, label="System A (CAN Only)", color="#4A90E2", edgecolor="black")
    ax2.bar(x, hdg_b_durs, width, label="System B (CAN + Road-Normal)", color="#E2844A", edgecolor="black")
    ax2.bar(x + width, hdg_c_durs, width, label="System C (CAN + Normal + Heading)", color="#2ECC71", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d}s" for d in durs])
    ax2.set_ylabel("Median Heading Error (°)", fontsize=12)
    ax2.set_title("Median Heading Error vs Outage Duration", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = PLOTS_DIR / "phase17b_fig2_duration_breakdown.png"
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(ARTIFACTS_DIR / "phase17b_fig2_duration_breakdown.png", dpi=300)
    plt.close()

    # Figure 3: Representative / Diagnostic Traces (Showing Trajectory and Heading Convergence)
    # Pick representative cases: a motorway case, an urban turn case, and a long 120s case
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 17B: Trajectory and Heading Comparison on Key Scenarios", fontsize=15, fontweight="bold")

    # Find two representative scenarios
    rep_scenarios = ["vw11_o2", "sample_test_trajectory_motorway_o2"]
    available_keys = [k for k in rep_scenarios if k in traces_store]
    if len(available_keys) < 2:
        available_keys = list(traces_store.keys())[:2]

    sc1, sc2 = available_keys[0], available_keys[1]

    for col_idx, sc_name in enumerate([sc1, sc2]):
        tr = traces_store[sc_name]
        # Trajectory (East vs North)
        ref_lat, ref_lon = tr["gt_lat"][0], tr["gt_lon"][0]
        R_e = 6371000.0
        gt_N = np.radians(tr["gt_lat"] - ref_lat) * R_e
        gt_E = np.radians(tr["gt_lon"] - ref_lon) * R_e * np.cos(np.radians(ref_lat))
        a_N = np.radians(tr["lat_a"] - ref_lat) * R_e
        a_E = np.radians(tr["lon_a"] - ref_lon) * R_e * np.cos(np.radians(ref_lat))
        b_N = np.radians(tr["lat_b"] - ref_lat) * R_e
        b_E = np.radians(tr["lon_b"] - ref_lon) * R_e * np.cos(np.radians(ref_lat))
        c_N = np.radians(tr["lat_c"] - ref_lat) * R_e
        c_E = np.radians(tr["lon_c"] - ref_lon) * R_e * np.cos(np.radians(ref_lat))

        ax_pos = axes[0, col_idx]
        ax_pos.plot(gt_E, gt_N, "k--", label="Ground Truth", linewidth=2.0)
        ax_pos.plot(a_E, a_N, "b-.", label="Sys A (CAN Only)", alpha=0.7)
        ax_pos.plot(b_E, b_N, "r:", label="Sys B (CAN+Normal)", linewidth=2.0)
        ax_pos.plot(c_E, c_N, "g-", label="Sys C (CAN+Normal+Hdg)", linewidth=2.0)
        ax_pos.set_title(f"Trajectory: {sc_name} ({tr['dur']}s)", fontsize=12, fontweight="bold")
        ax_pos.set_xlabel("East Position (m)")
        ax_pos.set_ylabel("North Position (m)")
        ax_pos.legend()
        ax_pos.grid(True, linestyle="--", alpha=0.5)

        # Heading Error vs Time
        ax_hdg = axes[1, col_idx]
        gt_h = tr["gt_hdg"]
        t_arr = tr["t"]
        err_a = np.abs((tr["hdg_a"] - gt_h + 180.0) % 360.0 - 180.0)
        err_b = np.abs((tr["hdg_b"] - gt_h + 180.0) % 360.0 - 180.0)
        err_c = np.abs((tr["hdg_c"] - gt_h + 180.0) % 360.0 - 180.0)

        ax_hdg.plot(t_arr, err_a, "b-.", label="Sys A Error", alpha=0.7)
        ax_hdg.plot(t_arr, err_b, "r:", label="Sys B Error", linewidth=2.0)
        ax_hdg.plot(t_arr, err_c, "g-", label="Sys C Error", linewidth=2.0)
        ax_hdg.set_title(f"Heading Error vs Time: {sc_name}", fontsize=12, fontweight="bold")
        ax_hdg.set_xlabel("Time into Outage (s)")
        ax_hdg.set_ylabel("Absolute Heading Error (°)")
        ax_hdg.legend()
        ax_hdg.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig3_path = PLOTS_DIR / "phase17b_fig3_failure_traces.png"
    plt.savefig(fig3_path, dpi=300)
    plt.savefig(ARTIFACTS_DIR / "phase17b_fig3_failure_traces.png", dpi=300)
    plt.close()

    logger.info("Saved all Phase 17B diagnostic figures.")
    return df_results


if __name__ == "__main__":
    run_phase17b_benchmark()
