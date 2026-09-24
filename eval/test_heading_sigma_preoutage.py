"""
Pre-outage validation script for Phase 17B sigma_heading selection.
Tests sigma_heading in {1.0 deg, 2.0 deg, 5.0 deg, adaptive} on healthy PRE-OUTAGE data
to select the optimal canonical value without any outage ground truth leakage.
"""
import math
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from modules.alignment import AlignmentEngine, AlignmentState
from modules.can_fusion_ukf import CANFusionUKF
from modules.map_matcher import load_road_graph_from_osm_json, FixedLagHMMMapMatcher
from eval.baseline import local_xy_to_latlon

osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)

# Load a sample trajectory to test pre-outage tracking
tf = config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet"
df = pd.read_parquet(tf)

# Use first 30 seconds of pre-outage driving (healthy GNSS reference)
df_pre = df.iloc[:300].copy()

results = {}
for sig_choice in [1.0, 2.0, 5.0, "adaptive"]:
    ukf = CANFusionUKF(dt=0.1)
    init_lat = float(df_pre[config.COL_TRUE_LAT].iloc[0])
    init_lon = float(df_pre[config.COL_TRUE_LON].iloc[0])
    init_spd = float(df_pre[config.COL_TRUE_SPEED_MS].iloc[0])
    init_hdg = float(df_pre[config.COL_TRUE_HEADING].iloc[0])
    ukf.initialize(init_lat, init_lon, init_spd, math.radians(init_hdg))

    acc_raw = df_pre[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_pre[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    w_rl = df_pre[config.COL_TRUE_WHEEL_RL].values
    w_rr = df_pre[config.COL_TRUE_WHEEL_RR].values
    v_rear = 0.5 * (w_rl + w_rr) * 0.2879

    align = AlignmentEngine()
    align.result.R_b_to_v = np.eye(3)
    align.result.state = AlignmentState.FULL_ALIGNED
    align.result.confidence = 1.0

    matcher = FixedLagHMMMapMatcher(graph=graph)

    nis_list = []
    applied_count = 0
    total_count = 0

    sig_rad = math.radians(sig_choice) if isinstance(sig_choice, float) else None

    for k in range(len(df_pre)):
        ukf.predict(acc_raw[k, 0], gyro_raw[k, 2], dt=0.1)
        ukf.update_can_speed(float(v_rear[k]))

        if k % 5 == 0:
            curr_lat, curr_lon = local_xy_to_latlon(ukf.x[1], ukf.x[0], init_lat, init_lon)
            p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
            hdg_deg = math.degrees(ukf.x[3]) % 360.0
            spd = float(ukf.x[2])
            pos_sigma = float(math.sqrt(max(0.01, ukf.P[0, 0] + ukf.P[1, 1])))

            mres = matcher.match(point_xy=p_graph, heading_deg=hdg_deg, speed_ms=spd, travel_dist_m=max(spd*0.5, 0.05), sigma_pos_m=pos_sigma, dt=0.5)
            if not mres.is_off_road and mres.confidence >= 0.25:
                snap_lat, snap_lon = graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                d_lat = math.radians(snap_lat - init_lat)
                d_lon = math.radians(snap_lon - init_lon)
                pN = float(6371000.0 * d_lat)
                pE = float(6371000.0 * d_lon * math.cos(math.radians(init_lat)))

                applied, pos_nis, hdg_nis = ukf.update_map_match(
                    p_N_match=pN,
                    p_E_match=pE,
                    psi_road=float(mres.road_heading_rad),
                    confidence=float(mres.confidence),
                    is_heading_valid=True,
                    sigma_heading_rad=sig_rad
                )
                if hdg_nis is not None:
                    nis_list.append(hdg_nis)

    mean_nis = np.mean(nis_list) if nis_list else 0.0
    med_nis = np.median(nis_list) if nis_list else 0.0
    p95_nis = np.percentile(nis_list, 95) if nis_list else 0.0
    print(f"Sigma: {sig_choice} | Accepted: {ukf.accepted_map_heading_updates}/{ukf.total_map_heading_updates} | Mean NIS: {mean_nis:.3f} | Med NIS: {med_nis:.3f} | 95th NIS: {p95_nis:.3f}")
