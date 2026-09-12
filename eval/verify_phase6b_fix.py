import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.ukf import UKFNavigationFilter
from modules.alignment import AlignmentEngine, AlignmentState
from modules.velocity_net import VelocityNetPredictor
from modules.map_matcher import FixedLagHMMMapMatcher, load_road_graph_from_osm_json
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from eval.baseline import local_xy_to_latlon

def run_evaluation(use_uk_map: bool = True):
    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)
    uk_osm = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    indore_osm = config.BASE_DIR / "android" / "app" / "src" / "main" / "assets" / "osm_roads_cache.json"

    sync_files = sorted(config.SYNC_PROCESSED_DIR.glob("sync_vw*.parquet"))
    test_files = [f for f in sync_files if any(f.stem.replace("sync_", "").lower().startswith(p) for p in config.TEST_DRIVERS)][:5]

    scores_nomap = []
    scores_withmap = []

    for test_file in test_files:
        run_name = test_file.stem.replace("sync_", "")
        df_raw = pd.read_parquet(test_file)
        df_sim, df_outages = inject_outages(df_raw, generate_outage_schedule(len(df_raw)))

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df_sim.loc[mask]
            if len(df_sub) < 5: continue

            outage_start_idx = mask.idxmax()
            pre_win = df_sim.iloc[max(0, outage_start_idx-10):outage_start_idx]
            init_lat = float(pre_win[config.COL_GPS_LAT].iloc[-1])
            init_lon = float(pre_win[config.COL_GPS_LON].iloc[-1])
            init_spd = float(pre_win[config.COL_GPS_SPEED_MS].iloc[-1])
            init_hdg = float(pre_win[config.COL_GPS_BEARING].iloc[-1])

            # Pre-calibrate run alignment
            run_align = AlignmentEngine()
            warmup = df_raw.iloc[:min(50, len(df_raw))]
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            gyro_w = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref); y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

            # 1. WITHOUT MAP
            ukf_nomap = UKFNavigationFilter(dt=0.1)
            ukf_nomap.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            nomap_pN, nomap_pE = [], []

            # 2. WITH MAP (1D Cross-Track Constraint)
            osm_file = uk_osm if use_uk_map else indore_osm
            graph = load_road_graph_from_osm_json(str(osm_file), ref_lat=init_lat, ref_lon=init_lon)
            matcher = FixedLagHMMMapMatcher(graph=graph)

            ukf_map = UKFNavigationFilter(dt=0.1)
            ukf_map.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            map_pN, map_pE = [], []

            win_buf = []

            for step_i in range(len(df_sub)):
                r = df_sub.iloc[step_i]
                ab = np.array([r[config.COL_ACC_X], r[config.COL_ACC_Y], r[config.COL_ACC_Z]])
                wb = np.array([r[config.COL_GYRO_X], r[config.COL_GYRO_Y], r[config.COL_GYRO_Z]])
                av, wv = run_align.transform_imu(ab, wb)

                ukf_nomap.predict(av[0], wv[2], dt=0.1)
                ukf_map.predict(av[0], wv[2], dt=0.1)

                win_buf.append(np.concatenate([ab, wb]))
                if len(win_buf) >= config.WINDOW_SIZE and (step_i % 5 == 0):
                    warr = np.ascontiguousarray(np.array(win_buf[-config.WINDOW_SIZE:], dtype=np.float32))
                    pred_d, ev_class, sigma = predictor.predict(warr)
                    if ev_class == 0 or pred_d < 0.1:
                        ukf_nomap.update_zupt(wv[2])
                        ukf_map.update_zupt(wv[2])
                    else:
                        ukf_nomap.update_velocity_net(pred_d * 1.15, max(sigma * 1.15, 0.2), ev_class, 2.0)
                        ukf_map.update_velocity_net(pred_d * 1.15, max(sigma * 1.15, 0.2), ev_class, 2.0)
                if len(win_buf) > 40: win_buf.pop(0)

                # 1D Cross-Track Map measurement update
                if step_i % 5 == 0:
                    pos = np.array([ukf_map.x[0], ukf_map.x[1]])
                    hdg = math.degrees(ukf_map.x[3]) % 360.0
                    spd = ukf_map.x[2]
                    sig = math.sqrt(ukf_map.P[0, 0] + ukf_map.P[1, 1])

                    # Check if position is within map graph coverage
                    lat_cur, lon_cur = local_xy_to_latlon(np.array([pos[1]]), np.array([pos[0]]), init_lat, init_lon)
                    if hasattr(graph, 'is_in_bounds') and not graph.is_in_bounds(lat_cur[0], lon_cur[0]):
                        mres_is_off_road = True
                    else:
                        mres = matcher.match(pos, hdg, spd, max(spd * 0.5, 0.05), sig, 0.5)
                        mres_is_off_road = mres.is_off_road

                    if not mres_is_off_road and mres.confidence >= 0.25:
                        psi_r = mres.road_heading_rad
                        n_N = -math.sin(psi_r)
                        n_E =  math.cos(psi_r)
                        y_ct = n_N * (mres.snapped_point[0] - ukf_map.x[0]) + n_E * (mres.snapped_point[1] - ukf_map.x[1])
                        sig_meas = 2.5 / max(mres.confidence, 0.25)
                        R_1d = sig_meas ** 2
                        S = n_N**2 * ukf_map.P[0, 0] + 2.0 * n_N * n_E * ukf_map.P[0, 1] + n_E**2 * ukf_map.P[1, 1] + R_1d
                        nis = (y_ct ** 2) / S

                        if nis <= 6.635:
                            K_N = (ukf_map.P[0, 0] * n_N + ukf_map.P[0, 1] * n_E) / S
                            K_E = (ukf_map.P[1, 0] * n_N + ukf_map.P[1, 1] * n_E) / S
                            ukf_map.x[0] += K_N * y_ct
                            ukf_map.x[1] += K_E * y_ct
                            K_vec = np.zeros(7); K_vec[0] = K_N; K_vec[1] = K_E
                            ukf_map.P -= S * np.outer(K_vec, K_vec)
                            ukf_map.P = 0.5 * (ukf_map.P + ukf_map.P.T)

                nomap_pN.append(ukf_nomap.x[0]); nomap_pE.append(ukf_nomap.x[1])
                map_pN.append(ukf_map.x[0]); map_pE.append(ukf_map.x[1])

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_pN = (gt_lat - init_lat) * (np.pi / 180.0) * 6371000.0
            gt_pE = (gt_lon - init_lon) * (np.pi / 180.0) * 6371000.0 * np.cos(np.radians(init_lat))
            dist_m = float(np.sum(np.hypot(np.diff(gt_pN), np.diff(gt_pE))))

            lat_nomap, lon_nomap = local_xy_to_latlon(np.array(nomap_pE), np.array(nomap_pN), init_lat, init_lon)
            lat_map, lon_map = local_xy_to_latlon(np.array(map_pE), np.array(map_pN), init_lat, init_lon)

            sc_no = score_outage_segment(lat_nomap, lon_nomap, gt_lat, gt_lon, dist_m, dur)
            sc_map = score_outage_segment(lat_map, lon_map, gt_lat, gt_lon, dist_m, dur)

            sc_no["run"] = run_name
            sc_map["run"] = run_name
            scores_nomap.append(sc_no)
            scores_withmap.append(sc_map)

    df_no = pd.DataFrame(scores_nomap)
    df_map = pd.DataFrame(scores_withmap)

    print(f"\n--- SUMMARY ({'REAL UK MAP' if use_uk_map else 'INDORE MAP (Coverage Gated)'}) ---")
    print(f"WITHOUT MAP Median Drift : {df_no['pct_of_distance'].median():.2f}%")
    print(f"WITH MAP    Median Drift : {df_map['pct_of_distance'].median():.2f}%")
    print(f"WITHOUT MAP Median CT Err: {df_no['cross_track_m'].median():.2f}m")
    print(f"WITH MAP    Median CT Err: {df_map['cross_track_m'].median():.2f}m")
    print(f"WITHOUT MAP Median FPE   : {df_no['final_pos_error_m'].median():.2f}m")
    print(f"WITH MAP    Median FPE   : {df_map['final_pos_error_m'].median():.2f}m")

if __name__ == "__main__":
    print("Testing with REAL UK MAP (1D Cross-Track Update):")
    run_evaluation(use_uk_map=True)
    print("\nTesting with INDORE MAP (Coverage Gated fallback):")
    run_evaluation(use_uk_map=False)
