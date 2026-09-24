"""
Phase 19 MAPHDE Parameter Sweep Script.

Sweeps i_c in {0.01 deg, 0.02 deg, 0.05 deg, 0.10 deg} on development trajectories
(vw2 and vw10) to evaluate convergence, stability, and FPE before frozen held-out benchmark.
"""
import sys
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
    RoadGraph,
    ROAD_EARTH_RADIUS
)
from eval.baseline import local_xy_to_latlon

class MAPHDEController:
    def __init__(self, i_c_deg: float = 0.02, max_I_deg: float = 5.0, dt_map: float = 0.5):
        self.i_c_rad = math.radians(i_c_deg)
        self.max_I_rad = math.radians(max_I_deg)
        self.dt_map = dt_map
        self.I = 0.0
        self.history_I = []
        self.history_E = []
        self.total_updates = 0
        self.active_updates = 0
        self.suspended_updates = 0

    def reset(self):
        self.I = 0.0
        self.history_I.clear()
        self.history_E.clear()
        self.total_updates = 0
        self.active_updates = 0
        self.suspended_updates = 0

    def update(self, psi_nav_rad: float, psi_map_rad: float, is_valid: bool) -> float:
        self.total_updates += 1
        if not is_valid:
            self.suspended_updates += 1
            self.history_I.append(self.I)
            self.history_E.append(np.nan)
            return self.I / self.dt_map

        diff = psi_map_rad - psi_nav_rad
        E = math.atan2(math.sin(diff), math.cos(diff))
        self.history_E.append(E)

        if abs(E) < 1e-5:
            sign_E = 0.0
        elif E > 0:
            sign_E = 1.0
        else:
            sign_E = -1.0

        self.I += sign_E * self.i_c_rad
        self.I = float(np.clip(self.I, -self.max_I_rad, self.max_I_rad))
        self.active_updates += 1
        self.history_I.append(self.I)

        return self.I / self.dt_map


def run_sweep():
    osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)

    dev_runs = ["vw2", "vw10"]
    test_files = [config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet" for r in dev_runs]

    candidate_ic = [0.01, 0.02, 0.05, 0.10]
    results = {ic: [] for ic in candidate_ic}
    baseline_results = []

    print("Running parameter sweep across development trajectories (vw2, vw10)...")

    for tf in test_files:
        run_name = tf.stem.replace("sync_", "")
        df = pd.read_parquet(tf)
        total_dur = len(df) * config.TARGET_DT
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

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            dist_gt = float(row["distance_travelled_m"])
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
            init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
            init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
            init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean()) if config.COL_TRUE_SPEED_MS in pre_df.columns else 0.0
            h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values) if config.COL_TRUE_HEADING in pre_df.columns else np.zeros(len(pre_df))
            init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            # Test Baseline
            fpe_base = run_single_sim(df_sub, init_lat, init_lon, init_spd, init_hdg, run_align, graph, r_eff_est, None, gt_lat, gt_lon, dist_gt, dur, gt_hdg)
            baseline_results.append(fpe_base)

            # Test each ic
            for ic in candidate_ic:
                controller = MAPHDEController(i_c_deg=ic, max_I_deg=5.0, dt_map=0.5)
                fpe_ic = run_single_sim(df_sub, init_lat, init_lon, init_spd, init_hdg, run_align, graph, r_eff_est, controller, gt_lat, gt_lon, dist_gt, dur, gt_hdg)
                results[ic].append(fpe_ic)

    print("\n=== PARAMETER SWEEP RESULTS (N = %d outages) ===" % len(baseline_results))
    print("Baseline (Phase 17A): Median FPE = %.2f m | Mean = %.2f m" % (np.median(baseline_results), np.mean(baseline_results)))
    for ic in candidate_ic:
        med = float(np.median(results[ic]))
        mean = float(np.mean(results[ic]))
        improved = sum(1 for b, c in zip(baseline_results, results[ic]) if c < b)
        print("i_c = %4.2f deg: Median FPE = %6.2f m | Mean = %6.2f m | Improved: %d/%d" % (ic, med, mean, improved, len(baseline_results)))


def run_single_sim(df_outage, init_lat, init_lon, init_speed_ms, init_heading_deg, alignment, graph, r_eff, controller, gt_lat, gt_lon, dist_gt, dur, gt_hdg):
    n = len(df_outage)
    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    omega_rl = df_outage[config.COL_TRUE_WHEEL_RL].values
    omega_rr = df_outage[config.COL_TRUE_WHEEL_RR].values
    v_rear = 0.5 * (omega_rl + omega_rr) * r_eff

    ukf = CANFusionUKF(dt=0.1)
    ukf.initialize(init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_speed_ms, init_heading_rad=math.radians(init_heading_deg), allow_bias_learning=False)
    matcher = FixedLagHMMMapMatcher(graph=graph)

    if controller is not None:
        controller.reset()

    r_drift = 0.0
    pred_lats = []
    pred_lons = []
    pred_hdgs = []
    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Prediction with external drift rate correction
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2] + r_drift, dt=0.1)

        # 2. CAN forward speed / ZUPT update
        v_val = float(v_rear[k])
        rear_diff = abs(omega_rl[k] - omega_rr[k])
        is_slipping = rear_diff > 25.0
        if v_val < 0.15 and not is_slipping:
            ukf.update_zupt(gyro_reading=w_v[2])
        elif not is_slipping:
            ukf.update_can_speed(v_can=v_val, r_var=0.0325)

        # 3. Map update (2 Hz)
        if k % 5 == 0:
            p_N_curr = float(ukf.x[0])
            p_E_curr = float(ukf.x[1])
            curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
            p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
            hdg_deg = math.degrees(ukf.x[3]) % 360.0
            spd = float(ukf.x[2])
            pos_sigma = float(math.sqrt(max(0.01, ukf.P[0, 0] + ukf.P[1, 1])))

            mres = matcher.match(point_xy=p_graph, heading_deg=hdg_deg, speed_ms=spd, travel_dist_m=max(spd * 0.5, 0.05), sigma_pos_m=pos_sigma, dt=0.5)

            # MAPHDE Controller Update
            if controller is not None:
                # Valid gating check:
                # Map confidence >= 0.25, not off road, vehicle moving > 1.5 m/s, not turning sharply
                diff_hdg = abs((hdg_deg - mres.heading_deg + 180.0) % 360.0 - 180.0)
                is_valid = (not mres.is_off_road and mres.confidence >= 0.25 and spd > 1.5 and abs(w_v[2]) < 0.15 and diff_hdg < 45.0)
                r_drift = controller.update(psi_nav_rad=math.radians(hdg_deg), psi_map_rad=mres.road_heading_rad, is_valid=is_valid)

            if not mres.is_off_road and mres.confidence >= 0.25:
                snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                d_lat = math.radians(snap_lat - init_lat)
                d_lon = math.radians(snap_lon - init_lon)
                p_N_match = float(ROAD_EARTH_RADIUS * d_lat)
                p_E_match = float(ROAD_EARTH_RADIUS * d_lon * math.cos(math.radians(init_lat)))
                ukf.update_map_match(p_N_match=p_N_match, p_E_match=p_E_match, psi_road=float(mres.road_heading_rad), confidence=float(mres.confidence), is_heading_valid=False)

        lat_k, lon_k = local_xy_to_latlon(ukf.x[1], ukf.x[0], init_lat, init_lon)
        pred_lats.append(lat_k)
        pred_lons.append(lon_k)
        pred_hdgs.append(math.degrees(ukf.x[3]) % 360.0)

    metrics = score_outage_segment(
        pred_lat=np.array(pred_lats),
        pred_lon=np.array(pred_lons),
        gt_lat=gt_lat,
        gt_lon=gt_lon,
        distance_travelled_m=dist_gt,
        duration_s=dur,
        gt_heading_deg=gt_hdg,
        pred_heading_deg=np.array(pred_hdgs)
    )
    return float(metrics["fpe_m"])

if __name__ == "__main__":
    run_sweep()
