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

def test_uk_map_runs():
    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)
    uk_osm = config.BASE_DIR / "data" / "osm_uk_test_roads.json"

    sync_files = sorted(config.SYNC_PROCESSED_DIR.glob("sync_vw*.parquet"))
    test_files = [f for f in sync_files if any(f.stem.replace("sync_", "").lower().startswith(p) for p in config.TEST_DRIVERS)][:5]

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

            # 1. NO MAP
            ukf_nomap = UKFNavigationFilter(dt=0.1)
            ukf_nomap.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            nomap_pN, nomap_pE = [], []
            win_buf = []

            for step_i in range(len(df_sub)):
                r = df_sub.iloc[step_i]
                ab = np.array([r[config.COL_ACC_X], r[config.COL_ACC_Y], r[config.COL_ACC_Z]])
                wb = np.array([r[config.COL_GYRO_X], r[config.COL_GYRO_Y], r[config.COL_GYRO_Z]])
                av, wv = run_align.transform_imu(ab, wb)
                ukf_nomap.predict(av[0], wv[2], dt=0.1)
                win_buf.append(np.concatenate([ab, wb]))
                if len(win_buf) >= config.WINDOW_SIZE and (step_i % 5 == 0):
                    warr = np.ascontiguousarray(np.array(win_buf[-config.WINDOW_SIZE:], dtype=np.float32))
                    pred_d, ev_class, sigma = predictor.predict(warr)
                    if ev_class == 0 or pred_d < 0.1:
                        ukf_nomap.update_zupt(wv[2])
                    else:
                        ukf_nomap.update_velocity_net(pred_d * 1.15, max(sigma * 1.15, 0.2), ev_class, 2.0)
                if len(win_buf) > 40: win_buf.pop(0)
                nomap_pN.append(ukf_nomap.x[0])
                nomap_pE.append(ukf_nomap.x[1])

            # 2. REAL UK MAP
            graph = load_road_graph_from_osm_json(str(uk_osm), ref_lat=init_lat, ref_lon=init_lon)
            matcher = FixedLagHMMMapMatcher(graph=graph)

            ukf_map = UKFNavigationFilter(dt=0.1)
            ukf_map.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            map_pN, map_pE = [], []
            win_buf.clear()

            for step_i in range(len(df_sub)):
                r = df_sub.iloc[step_i]
                ab = np.array([r[config.COL_ACC_X], r[config.COL_ACC_Y], r[config.COL_ACC_Z]])
                wb = np.array([r[config.COL_GYRO_X], r[config.COL_GYRO_Y], r[config.COL_GYRO_Z]])
                av, wv = run_align.transform_imu(ab, wb)
                ukf_map.predict(av[0], wv[2], dt=0.1)
                win_buf.append(np.concatenate([ab, wb]))
                if len(win_buf) >= config.WINDOW_SIZE and (step_i % 5 == 0):
                    warr = np.ascontiguousarray(np.array(win_buf[-config.WINDOW_SIZE:], dtype=np.float32))
                    pred_d, ev_class, sigma = predictor.predict(warr)
                    if ev_class == 0 or pred_d < 0.1:
                        ukf_map.update_zupt(wv[2])
                    else:
                        ukf_map.update_velocity_net(pred_d * 1.15, max(sigma * 1.15, 0.2), ev_class, 2.0)
                if len(win_buf) > 40: win_buf.pop(0)

                if step_i % 5 == 0:
                    pos = np.array([ukf_map.x[0], ukf_map.x[1]])
                    hdg = math.degrees(ukf_map.x[3]) % 360.0
                    spd = ukf_map.x[2]
                    sig = math.sqrt(ukf_map.P[0, 0] + ukf_map.P[1, 1])
                    mres = matcher.match(pos, hdg, spd, max(spd * 0.5, 0.05), sig, 0.5)
                    if not mres.is_off_road and mres.confidence >= 0.25:
                        ukf_map.update_map_match(
                            float(mres.snapped_point[0]),
                            float(mres.snapped_point[1]),
                            float(mres.road_heading_rad),
                            float(mres.confidence),
                            is_heading_valid=False # Do not force heading
                        )
                map_pN.append(ukf_map.x[0])
                map_pE.append(ukf_map.x[1])

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_pN = (gt_lat - init_lat) * (np.pi / 180.0) * 6371000.0
            gt_pE = (gt_lon - init_lon) * (np.pi / 180.0) * 6371000.0 * np.cos(np.radians(init_lat))
            dist_m = float(np.sum(np.hypot(np.diff(gt_pN), np.diff(gt_pE))))

            lat_nomap, lon_nomap = local_xy_to_latlon(np.array(nomap_pE), np.array(nomap_pN), init_lat, init_lon)
            lat_map, lon_map = local_xy_to_latlon(np.array(map_pE), np.array(map_pN), init_lat, init_lon)

            sc_no = score_outage_segment(lat_nomap, lon_nomap, gt_lat, gt_lon, dist_m, dur)
            sc_uk = score_outage_segment(lat_map, lon_map, gt_lat, gt_lon, dist_m, dur)

            print(f"[{run_name} {dur}s outage {oid}] WITHOUT MAP Drift: {sc_no['pct_of_distance']:.2f}% (CT: {sc_no['cross_track_m']:.1f}m) | REAL UK MAP Drift: {sc_uk['pct_of_distance']:.2f}% (CT: {sc_uk['cross_track_m']:.1f}m)")

if __name__ == "__main__":
    test_uk_map_runs()
