"""
Phase 19C: Paper-Faithful MAPHDE Benchmark (56 Outages)

Comparing:
  Arm A: Phase 17A Baseline (CAN speed + 7-state UKF + 1D road-normal map matching)
  Arm B: Phase 19 Implementation (Continuous r_drift = I / dt injected into predict, even when suspended)
  Arm C: Paper-Faithful MAPHDE (Aggarwal et al. 2011):
         - When valid: I_i = I_{i-1} + SIGN(E_i) * i_c, applied as heading increment at map update
         - When suspended: Delta_psi_MAPHDE = 0 (no correction applied)
         - In IMU predict: r_drift = 0 at all times (pure natural IMU dead reckoning)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
    ROAD_EARTH_RADIUS
)
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase19C")

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

def wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

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

def run_simulation(
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
    n = len(df_outage)
    if n == 0:
        return {}

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    omega_rl = df_outage[config.COL_TRUE_WHEEL_RL].values
    omega_rr = df_outage[config.COL_TRUE_WHEEL_RR].values
    v_rear = 0.5 * (omega_rl * r_eff + omega_rr * r_eff)

    arms = ["A", "B", "C"]
    ukfs = {}
    matchers = {}
    for arm in arms:
        u = CANFusionUKF(dt=dt)
        u.initialize(
            init_lat=init_lat,
            init_lon=init_lon,
            init_speed_ms=init_speed_ms,
            init_heading_rad=math.radians(init_heading_deg),
            allow_bias_learning=False
        )
        ukfs[arm] = u
        matchers[arm] = FixedLagHMMMapMatcher(graph=graph)

    i_c_rad = math.radians(i_c_deg)
    max_I_rad = math.radians(10.0)

    # Controller state
    I_B = 0.0
    r_drift_B = 0.0
    I_C = 0.0

    est_lat = {arm: np.zeros(n) for arm in arms}
    est_lon = {arm: np.zeros(n) for arm in arms}
    est_hdg = {arm: np.zeros(n) for arm in arms}
    trace_I = {"B": [], "C": []}
    trace_E = {"B": [], "C": []}
    trace_active = {"B": [], "C": []}
    trace_t = []

    valid_count = {"B": 0, "C": 0}
    susp_count = {"B": 0, "C": 0}

    for k in range(n):
        t_now = k * dt
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        v_val = float(v_rear[k])
        rear_diff = abs(omega_rl[k] - omega_rr[k])
        is_slipping = rear_diff > 25.0

        # 1. IMU Prediction step
        # Arm A: No drift injection
        ukfs["A"].predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)
        # Arm B: Continuous r_drift injection (Phase 19)
        ukfs["B"].predict(acc_fwd=a_v[0], gyro_yaw=w_v[2] + r_drift_B, dt=dt)
        # Arm C: Paper-Faithful: Pure natural IMU dead reckoning in predict (r_drift = 0)
        ukfs["C"].predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        # 2. CAN forward speed / ZUPT update
        for arm in arms:
            if v_val < 0.15 and not is_slipping:
                ukfs[arm].update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukfs[arm].update_can_speed(v_can=v_val, r_var=r_can_var)

        # 3. Map update (2 Hz)
        if k % 5 == 0:
            trace_t.append(t_now)
            for arm in arms:
                u = ukfs[arm]
                p_N_curr = float(u.x[0])
                p_E_curr = float(u.x[1])
                curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
                p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
                hdg_deg = math.degrees(u.x[3]) % 360.0
                spd = float(u.x[2])
                pos_sigma = float(math.sqrt(max(0.01, u.P[0, 0] + u.P[1, 1])))

                mres = matchers[arm].match(
                    point_xy=p_graph,
                    heading_deg=hdg_deg,
                    speed_ms=spd,
                    travel_dist_m=max(spd * 0.5, 0.05),
                    sigma_pos_m=pos_sigma,
                    dt=0.5
                )

                diff_hdg = abs(math.degrees(wrap_to_pi(math.radians(hdg_deg) - mres.road_heading_rad)))
                is_valid = (not mres.is_off_road and mres.confidence >= 0.25 and spd > 1.5 and abs(w_v[2]) < 0.15 and diff_hdg < 45.0)

                if arm == "B":
                    # Arm B: Phase 19 semantics
                    E_B = wrap_to_pi(mres.road_heading_rad - math.radians(hdg_deg))
                    if is_valid:
                        valid_count["B"] += 1
                        sign_e = 1.0 if E_B > 0 else (-1.0 if E_B < 0 else 0.0)
                        I_B = max(-max_I_rad, min(max_I_rad, I_B + sign_e * i_c_rad))
                    else:
                        susp_count["B"] += 1
                    # Continuous injection even when suspended
                    r_drift_B = I_B / 0.5
                    trace_I["B"].append(math.degrees(I_B))
                    trace_E["B"].append(math.degrees(E_B))
                    trace_active["B"].append(is_valid)

                elif arm == "C":
                    # Arm C: Paper-Faithful MAPHDE semantics
                    E_C = wrap_to_pi(mres.road_heading_rad - math.radians(hdg_deg))
                    if is_valid:
                        valid_count["C"] += 1
                        sign_e = 1.0 if E_C > 0 else (-1.0 if E_C < 0 else 0.0)
                        I_C = max(-max_I_rad, min(max_I_rad, I_C + sign_e * i_c_rad))
                        # Apply heading increment directly at this map update
                        u.x[3] = wrap_to_pi(u.x[3] + I_C)
                    else:
                        susp_count["C"] += 1
                        # Suspended: Delta_psi_MAPHDE = 0 (do nothing to heading!)
                    trace_I["C"].append(math.degrees(I_C))
                    trace_E["C"].append(math.degrees(E_C))
                    trace_active["C"].append(is_valid)

                # 1D position map update (same for all arms)
                if not mres.is_off_road and mres.confidence >= 0.25:
                    snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                    d_lat = math.radians(snap_lat - init_lat)
                    d_lon = math.radians(snap_lon - init_lon)
                    p_N_match = float(ROAD_EARTH_RADIUS * d_lat)
                    p_E_match = float(ROAD_EARTH_RADIUS * d_lon * math.cos(math.radians(init_lat)))
                    u.update_map_match(
                        p_N_match=p_N_match,
                        p_E_match=p_E_match,
                        psi_road=float(mres.road_heading_rad),
                        confidence=float(mres.confidence),
                        is_heading_valid=False
                    )

        for arm in arms:
            lat_k, lon_k = local_xy_to_latlon(ukfs[arm].x[1], ukfs[arm].x[0], init_lat, init_lon)
            est_lat[arm][k] = lat_k
            est_lon[arm][k] = lon_k
            est_hdg[arm][k] = math.degrees(ukfs[arm].x[3]) % 360.0

    res = {}
    for arm in arms:
        res[arm] = {
            "lat": est_lat[arm],
            "lon": est_lon[arm],
            "hdg": est_hdg[arm],
            "valid": valid_count.get(arm, 0),
            "susp": susp_count.get(arm, 0),
            "final_I": trace_I[arm][-1] if (arm in trace_I and trace_I[arm]) else 0.0,
            "trace_I": trace_I.get(arm, []),
            "trace_E": trace_E.get(arm, []),
            "trace_t": trace_t,
            "trace_active": trace_active.get(arm, [])
        }
    return res

def run_phase19c_benchmark():
    logger.info("Starting Phase 19C: Paper-Faithful MAPHDE Benchmark (56 Outages)...")

    osm_path = BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = BASE_DIR / "data" / "osm_uk_test_roads.json"
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)
    logger.info(f"Loaded {len(graph.edges)} road edges from {osm_path}")

    test_files = get_benchmark_trajectories()
    logger.info(f"Evaluating across {len(test_files)} trajectories.")

    benchmark_rows = []
    failure_rows = []
    traces_to_save = {}

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

            sim_res = run_simulation(
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
                    pred_heading_deg=sim_res[arm]["hdg"]
                )

            scenario_name = f"{run_name}_o{oid}"
            fpe_A = float(scores["A"]["fpe_m"])
            fpe_B = float(scores["B"]["fpe_m"])
            fpe_C = float(scores["C"]["fpe_m"])
            drift_A = float(scores["A"]["drift_percent"])
            drift_B = float(scores["B"]["drift_percent"])
            drift_C = float(scores["C"]["drift_percent"])

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
                "diff_C_vs_A": fpe_C - fpe_A,
                "diff_C_vs_B": fpe_C - fpe_B,
                "diff_B_vs_A": fpe_B - fpe_A,
                "improved_C_vs_A": fpe_C < fpe_A - 0.5,
                "worsened_C_vs_A": fpe_C > fpe_A + 0.5,
                "improved_C_vs_B": fpe_C < fpe_B - 0.5,
                "worsened_C_vs_B": fpe_C > fpe_B + 0.5,
                "hdg_err_A": float(scores["A"].get("heading_error_deg", 0.0)) if not np.isnan(scores["A"].get("heading_error_deg", np.nan)) else 0.0,
                "hdg_err_B": float(scores["B"].get("heading_error_deg", 0.0)) if not np.isnan(scores["B"].get("heading_error_deg", np.nan)) else 0.0,
                "hdg_err_C": float(scores["C"].get("heading_error_deg", 0.0)) if not np.isnan(scores["C"].get("heading_error_deg", np.nan)) else 0.0,
                "cross_err_A": float(scores["A"].get("cross_track_m", 0.0)) if not np.isnan(scores["A"].get("cross_track_m", np.nan)) else 0.0,
                "cross_err_B": float(scores["B"].get("cross_track_m", 0.0)) if not np.isnan(scores["B"].get("cross_track_m", np.nan)) else 0.0,
                "cross_err_C": float(scores["C"].get("cross_track_m", 0.0)) if not np.isnan(scores["C"].get("cross_track_m", np.nan)) else 0.0,
                "final_I_C_deg": sim_res["C"]["final_I"],
                "valid_C": sim_res["C"]["valid"],
                "susp_C": sim_res["C"]["susp"],
            }
            benchmark_rows.append(row)

            if scenario_name in PHASE18A_FAILURE_CASES:
                failure_rows.append({
                    "scenario": scenario_name,
                    "duration_s": dur,
                    "fpe_A": fpe_A,
                    "fpe_B": fpe_B,
                    "fpe_C": fpe_C,
                    "delta_C_vs_A": fpe_C - fpe_A,
                    "delta_C_vs_B": fpe_C - fpe_B,
                    "delta_B_vs_A": fpe_B - fpe_A,
                    "hdg_err_A": row["hdg_err_A"],
                    "hdg_err_B": row["hdg_err_B"],
                    "hdg_err_C": row["hdg_err_C"],
                    "cross_err_A": row["cross_err_A"],
                    "cross_err_B": row["cross_err_B"],
                    "cross_err_C": row["cross_err_C"],
                    "final_I_C_deg": row["final_I_C_deg"],
                    "valid_C": row["valid_C"],
                    "susp_C": row["susp_C"]
                })
                traces_to_save[scenario_name] = {
                    "sim": sim_res,
                    "gt_lat": gt_lat,
                    "gt_lon": gt_lon,
                    "gt_hdg": gt_hdg,
                    "t": np.arange(len(df_sub)) * 0.1
                }

    df_bench = pd.DataFrame(benchmark_rows)
    df_fail = pd.DataFrame(failure_rows)

    eval_dir = BASE_DIR / "eval"
    df_bench.to_csv(eval_dir / "phase19c_paper_faithful_benchmark.csv", index=False)
    df_fail.to_csv(eval_dir / "phase19c_failure_cases_audit.csv", index=False)

    print_summary(df_bench, df_fail)
    generate_figures(df_bench, df_fail, traces_to_save)

def print_summary(df: pd.DataFrame, df_fail: pd.DataFrame):
    print("\n" + "="*95)
    print("PHASE 19C: PAPER-FAITHFUL MAPHDE MASTER BENCHMARK SUMMARY (N = 56)")
    print("="*95)
    print(f"{'Metric':<30} | {'Arm A (Phase 17A)':<18} | {'Arm B (Phase 19)':<18} | {'Arm C (Paper-Faithful)':<22}")
    print("-" * 95)

    for metric, col_A, col_B, col_C, unit in [
        ("Median FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("Mean FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("P95 FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("Max FPE", "fpe_A", "fpe_B", "fpe_C", "m"),
        ("Median Drift %", "drift_pct_A", "drift_pct_B", "drift_pct_C", "%"),
        ("Mean Drift %", "drift_pct_A", "drift_pct_B", "drift_pct_C", "%"),
        ("Mean Heading Error", "hdg_err_A", "hdg_err_B", "hdg_err_C", "deg"),
        ("Mean Cross-Track Error", "cross_err_A", "cross_err_B", "cross_err_C", "m"),
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

    print("-" * 95)
    imp_CA = (df["fpe_C"] < df["fpe_A"] - 0.5).sum()
    wors_CA = (df["fpe_C"] > df["fpe_A"] + 0.5).sum()
    unc_CA = len(df) - imp_CA - wors_CA
    print(f"{'Outages Improved / Worsened (C vs A)':<30} | Improved: {imp_CA:2d}, Worsened: {wors_CA:2d}, Unchanged: {unc_CA:2d}")

    imp_CB = (df["fpe_C"] < df["fpe_B"] - 0.5).sum()
    wors_CB = (df["fpe_C"] > df["fpe_B"] + 0.5).sum()
    unc_CB = len(df) - imp_CB - wors_CB
    print(f"{'Outages Improved / Worsened (C vs B)':<30} | Improved: {imp_CB:2d}, Worsened: {wors_CB:2d}, Unchanged: {unc_CB:2d}")

    sih_A = (df["drift_pct_A"] < 10.0).sum()
    sih_B = (df["drift_pct_B"] < 10.0).sum()
    sih_C = (df["drift_pct_C"] < 10.0).sum()
    print(f"{'Satisfying Drift < 10%':<30} | {sih_A:2d}/{len(df)} ({sih_A/len(df)*100:.1f}%)    | {sih_B:2d}/{len(df)} ({sih_B/len(df)*100:.1f}%)    | {sih_C:2d}/{len(df)} ({sih_C/len(df)*100:.1f}%)")

    print("\n" + "="*95)
    print("DURATION BREAKDOWN (Median FPE)")
    print("="*95)
    durations = sorted(df["duration_s"].unique())
    print(f"{'Duration':<10} | {'N':<4} | {'Arm A FPE (m)':<15} | {'Arm B FPE (m)':<15} | {'Arm C FPE (m)':<15} | {'Diff (C - A)':<12} | {'Diff (C - B)':<12}")
    print("-" * 95)
    for dur in durations:
        df_d = df[df["duration_s"] == dur]
        mA = df_d["fpe_A"].median()
        mB = df_d["fpe_B"].median()
        mC = df_d["fpe_C"].median()
        print(f"{dur:<10.0f} | {len(df_d):<4} | {mA:15.2f} | {mB:15.2f} | {mC:15.2f} | {mC - mA:+12.2f} | {mC - mB:+12.2f}")

    print("\n" + "="*95)
    print("INSPECTION OF THE 14 PHASE 18A FAILURE CASES (Focusing on Critical 4)")
    print("="*95)
    print(f"{'Scenario':<22} | {'Dur':<4} | {'FPE_A (m)':<10} | {'FPE_B (m)':<10} | {'FPE_C (m)':<10} | {'Δ(C-A) m':<10} | {'Δ(C-B) m':<10} | {'Valid/Susp':<10}")
    print("-" * 95)
    for _, r in df_fail.iterrows():
        highlight = " <-- CRITICAL" if r['scenario'] in ["vw14b_o5", "vw4_o1", "vw16a_o4", "vw14c_o4"] else ""
        print(f"{r['scenario']:<22} | {r['duration_s']:<4.0f} | {r['fpe_A']:10.1f} | {r['fpe_B']:10.1f} | {r['fpe_C']:10.1f} | {r['delta_C_vs_A']:+10.1f} | {r['delta_C_vs_B']:+10.1f} | {int(r['valid_C'])}/{int(r['susp_C'])}{highlight}")
    print("="*95 + "\n")

def generate_figures(df: pd.DataFrame, df_fail: pd.DataFrame, traces: Dict[str, Any]):
    plots_dir = BASE_DIR / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Phase 19C: Paper-Faithful MAPHDE vs Continuous Bias Injection", fontsize=14, fontweight="bold")

    # 1. Median & Mean FPE comparison
    ax = axes[0, 0]
    metrics = ["Median FPE", "Mean FPE", "P95 FPE"]
    vA = [df["fpe_A"].median(), df["fpe_A"].mean(), df["fpe_A"].quantile(0.95)]
    vB = [df["fpe_B"].median(), df["fpe_B"].mean(), df["fpe_B"].quantile(0.95)]
    vC = [df["fpe_C"].median(), df["fpe_C"].mean(), df["fpe_C"].quantile(0.95)]
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, vA, width=w, label="Arm A (Phase 17A Baseline)", color="#4A90E2")
    ax.bar(x, vB, width=w, label="Arm B (Phase 19 Continuous)", color="#E94A4A")
    ax.bar(x + w, vC, width=w, label="Arm C (Paper-Faithful)", color="#50E3C2")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("FPE (meters)")
    ax.set_title("Aggregate FPE Metrics (N = 56)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Duration Breakdown
    ax = axes[0, 1]
    durations = sorted(df["duration_s"].unique())
    fpe_dA = [df[df["duration_s"] == d]["fpe_A"].median() for d in durations]
    fpe_dB = [df[df["duration_s"] == d]["fpe_B"].median() for d in durations]
    fpe_dC = [df[df["duration_s"] == d]["fpe_C"].median() for d in durations]
    ax.plot(durations, fpe_dA, marker="o", linewidth=2, label="Arm A (Baseline)", color="#4A90E2")
    ax.plot(durations, fpe_dB, marker="s", linewidth=2, label="Arm B (Phase 19)", color="#E94A4A")
    ax.plot(durations, fpe_dC, marker="^", linewidth=2, label="Arm C (Paper-Faithful)", color="#50E3C2")
    ax.set_xlabel("Outage Duration (s)")
    ax.set_ylabel("Median FPE (m)")
    ax.set_title("Median FPE vs Duration")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 3. Critical 4 Cases Comparison
    ax = axes[1, 0]
    crit_cases = ["vw14b_o5", "vw4_o1", "vw16a_o4", "vw14c_o4"]
    df_crit = df_fail[df_fail["scenario"].isin(crit_cases)].copy()
    xc = np.arange(len(crit_cases))
    wc = 0.25
    fpe_cA = [df_crit[df_crit["scenario"] == c]["fpe_A"].values[0] for c in crit_cases]
    fpe_cB = [df_crit[df_crit["scenario"] == c]["fpe_B"].values[0] for c in crit_cases]
    fpe_cC = [df_crit[df_crit["scenario"] == c]["fpe_C"].values[0] for c in crit_cases]
    ax.bar(xc - wc, fpe_cA, width=wc, label="Arm A (Baseline)", color="#4A90E2")
    ax.bar(xc, fpe_cB, width=wc, label="Arm B (Phase 19)", color="#E94A4A")
    ax.bar(xc + wc, fpe_cC, width=wc, label="Arm C (Paper-Faithful)", color="#50E3C2")
    ax.set_xticks(xc)
    ax.set_xticklabels(crit_cases, rotation=15)
    ax.set_ylabel("FPE (meters)")
    ax.set_title("The 4 Critical Scenarios: Impact of Paper Semantics")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 4. Forensic Trajectory: vw14b_o5
    ax = axes[1, 1]
    if "vw14b_o5" in traces:
        tr = traces["vw14b_o5"]
        init_lat = tr["gt_lat"][0]
        init_lon = tr["gt_lon"][0]
        lat0_rad = math.radians(init_lat)
        EARTH_R = 6371000.0

        gt_dN = (tr["gt_lat"] - init_lat) * (math.pi / 180.0) * EARTH_R
        gt_dE = (tr["gt_lon"] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
        A_dN = (tr["sim"]["A"]["lat"] - init_lat) * (math.pi / 180.0) * EARTH_R
        A_dE = (tr["sim"]["A"]["lon"] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
        B_dN = (tr["sim"]["B"]["lat"] - init_lat) * (math.pi / 180.0) * EARTH_R
        B_dE = (tr["sim"]["B"]["lon"] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
        C_dN = (tr["sim"]["C"]["lat"] - init_lat) * (math.pi / 180.0) * EARTH_R
        C_dE = (tr["sim"]["C"]["lon"] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)

        ax.plot(gt_dE, gt_dN, label="Ground Truth", color="black", linestyle="--", linewidth=1.5)
        ax.plot(A_dE, A_dN, label=f"Arm A ({fpe_cA[0]:.0f}m)", color="#4A90E2", linewidth=1.2)
        ax.plot(B_dE, B_dN, label=f"Arm B ({fpe_cB[0]:.0f}m)", color="#E94A4A", linewidth=1.2)
        ax.plot(C_dE, C_dN, label=f"Arm C ({fpe_cC[0]:.0f}m)", color="#50E3C2", linewidth=1.5)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("vw14b_o5 Trajectory Ground Plane: False Latch Removal")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_file = plots_dir / "phase19c_fig1_paper_faithful.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info(f"Saved figure to {out_file}")

if __name__ == "__main__":
    run_phase19c_benchmark()
