"""
Quick diagnostic test for Phase 19C paper-faithful MAPHDE semantics on the 4 critical cases:
vw14b_o5, vw4_o1, vw16a_o4, vw14c_o4.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
from pathlib import Path
import numpy as np
import pandas as pd

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
    ROAD_EARTH_RADIUS
)
from eval.baseline import local_xy_to_latlon

def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def test_critical_cases():
    osm_path = BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = BASE_DIR / "data" / "osm_uk_test_roads.json"
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)

    cases = ["vw14b_o5", "vw4_o1", "vw16a_o4", "vw14c_o4"]
    test_trajs = ["vw14b", "vw4", "vw16a", "vw14c"]

    print("=" * 80)
    print("PHASE 19C: DIAGNOSTIC TEST ON 4 CRITICAL CASES")
    print("=" * 80)

    for traj_name in test_trajs:
        tf = config.SYNC_PROCESSED_DIR / f"sync_{traj_name}.parquet"
        if not tf.exists():
            continue
        df = pd.read_parquet(tf)
        total_dur = len(df) * config.TARGET_DT
        schedule = generate_outage_schedule(total_dur, run_id=traj_name)
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
            scenario_name = f"{traj_name}_o{oid}"
            if scenario_name not in cases:
                continue

            dur = int(row_out["duration_s"])
            dist_gt = float(row_out["distance_travelled_m"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
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

            # Run 3 Arms:
            # Arm A: Phase 17A Baseline
            # Arm B: Phase 19 (Continuous r_drift injected even during suspension)
            # Arm C: Phase 19C Paper-Faithful (Heading increment applied ONLY at valid map update; delta_psi=0 during suspension; r_drift=0 in predict)
            n = len(df_sub)
            acc_raw = df_sub[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            gyro_raw = df_sub[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
            omega_rl = df_sub[config.COL_TRUE_WHEEL_RL].values
            omega_rr = df_sub[config.COL_TRUE_WHEEL_RR].values
            v_rear = 0.5 * (omega_rl * r_eff_est + omega_rr * r_eff_est)

            arms = ["A", "B", "C"]
            ukfs = {}
            matchers = {}
            for arm in arms:
                u = CANFusionUKF(dt=0.1)
                u.initialize(init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_spd, init_heading_rad=math.radians(init_hdg), allow_bias_learning=False)
                ukfs[arm] = u
                matchers[arm] = FixedLagHMMMapMatcher(graph=graph)

            I_B = 0.0
            r_drift_B = 0.0
            I_C = 0.0
            i_c_rad = math.radians(0.02)
            max_I_rad = math.radians(10.0)

            est_lat = {arm: np.zeros(n) for arm in arms}
            est_lon = {arm: np.zeros(n) for arm in arms}
            est_hdg = {arm: np.zeros(n) for arm in arms}

            valid_count_C = 0
            susp_count_C = 0

            for k in range(n):
                a_b = acc_raw[k]
                w_b = gyro_raw[k]
                a_v, w_v = run_align.transform_imu(a_b, w_b)

                v_val = float(v_rear[k])
                rear_diff = abs(omega_rl[k] - omega_rr[k])
                is_slipping = rear_diff > 25.0

                # 1. IMU Predict
                # Arm A: no drift injection
                ukfs["A"].predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=0.1)
                # Arm B: continuous r_drift injection (Phase 19)
                ukfs["B"].predict(acc_fwd=a_v[0], gyro_yaw=w_v[2] + r_drift_B, dt=0.1)
                # Arm C: Paper-Faithful: NO drift rate in predict, purely normal IMU prediction
                ukfs["C"].predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=0.1)

                # 2. CAN speed / ZUPT update
                for arm in arms:
                    if v_val < 0.15 and not is_slipping:
                        ukfs[arm].update_zupt(gyro_reading=w_v[2])
                    elif not is_slipping:
                        ukfs[arm].update_can_speed(v_can=v_val, r_var=0.0325)

                # 3. Map update (2 Hz)
                if k % 5 == 0:
                    for arm in arms:
                        u = ukfs[arm]
                        p_N_curr = float(u.x[0])
                        p_E_curr = float(u.x[1])
                        curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
                        p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
                        hdg_deg = math.degrees(u.x[3]) % 360.0
                        spd = float(u.x[2])
                        pos_sigma = float(math.sqrt(max(0.01, u.P[0, 0] + u.P[1, 1])))

                        mres = matchers[arm].match(point_xy=p_graph, heading_deg=hdg_deg, speed_ms=spd, travel_dist_m=max(spd * 0.5, 0.05), sigma_pos_m=pos_sigma, dt=0.5)

                        diff_hdg = abs(math.degrees(wrap_to_pi(math.radians(hdg_deg) - mres.road_heading_rad)))
                        is_valid = (not mres.is_off_road and mres.confidence >= 0.25 and spd > 1.5 and abs(w_v[2]) < 0.15 and diff_hdg < 45.0)

                        if arm == "B":
                            E_B = wrap_to_pi(mres.road_heading_rad - math.radians(hdg_deg))
                            if is_valid:
                                sign_e = 1.0 if E_B > 0 else (-1.0 if E_B < 0 else 0.0)
                                I_B = max(-max_I_rad, min(max_I_rad, I_B + sign_e * i_c_rad))
                            # Arm B: continuous injection even when suspended
                            r_drift_B = I_B / 0.5

                        elif arm == "C":
                            # Arm C: Paper-Faithful MAPHDE
                            if is_valid:
                                valid_count_C += 1
                                E_C = wrap_to_pi(mres.road_heading_rad - math.radians(hdg_deg))
                                sign_e = 1.0 if E_C > 0 else (-1.0 if E_C < 0 else 0.0)
                                I_C = max(-max_I_rad, min(max_I_rad, I_C + sign_e * i_c_rad))
                                # Apply heading increment at this map update
                                u.x[3] = wrap_to_pi(u.x[3] + I_C)
                            else:
                                susp_count_C += 1
                                # When suspended: delta_psi_MAPHDE = 0 (do nothing!)

                        # 1D position map update (same for all arms)
                        if not mres.is_off_road and mres.confidence >= 0.25:
                            snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                            d_lat = math.radians(snap_lat - init_lat)
                            d_lon = math.radians(snap_lon - init_lon)
                            p_N_match = float(ROAD_EARTH_RADIUS * d_lat)
                            p_E_match = float(ROAD_EARTH_RADIUS * d_lon * math.cos(math.radians(init_lat)))
                            u.update_map_match(p_N_match=p_N_match, p_E_match=p_E_match, psi_road=float(mres.road_heading_rad), confidence=float(mres.confidence), is_heading_valid=False)

                for arm in arms:
                    lat_k, lon_k = local_xy_to_latlon(ukfs[arm].x[1], ukfs[arm].x[0], init_lat, init_lon)
                    est_lat[arm][k] = lat_k
                    est_lon[arm][k] = lon_k
                    est_hdg[arm][k] = math.degrees(ukfs[arm].x[3]) % 360.0

            scores = {}
            for arm in arms:
                scores[arm] = score_outage_segment(
                    pred_lat=est_lat[arm],
                    pred_lon=est_lon[arm],
                    gt_lat=gt_lat,
                    gt_lon=gt_lon,
                    distance_travelled_m=dist_gt,
                    duration_s=dur,
                    gt_heading_deg=gt_hdg,
                    pred_heading_deg=est_hdg[arm]
                )

            fpe_A = scores["A"]["fpe_m"]
            fpe_B = scores["B"]["fpe_m"]
            fpe_C = scores["C"]["fpe_m"]
            print(f"Scenario: {scenario_name:<10} (Dur {dur}s) | Arm A (Base): {fpe_A:7.1f} m | Arm B (Ph19): {fpe_B:7.1f} m | Arm C (Paper): {fpe_C:7.1f} m | Delta C-A: {fpe_C-fpe_A:+7.1f} m | Delta C-B: {fpe_C-fpe_B:+7.1f} m (Valid: {valid_count_C}, Susp: {susp_count_C})")

if __name__ == "__main__":
    test_critical_cases()
