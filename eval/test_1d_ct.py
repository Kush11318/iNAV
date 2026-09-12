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

def run_1d_cross_track_comparison():
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
            ukf_no = UKFNavigationFilter(dt=0.1)
            ukf_no.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            no_pN, no_pE = [], []

            # 2. 1D CROSS TRACK MAP
            graph = load_road_graph_from_osm_json(str(uk_osm), ref_lat=init_lat, ref_lon=init_lon)
            matcher = FixedLagHMMMapMatcher(graph=graph)
            ukf_ct = UKFNavigationFilter(dt=0.1)
            ukf_ct.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            ct_pN, ct_pE = [], []

            win_buf = []

            for step_i in range(len(df_sub)):
                r = df_sub.iloc[step_i]
                ab = np.array([r[config.COL_ACC_X], r[config.COL_ACC_Y], r[config.COL_ACC_Z]])
                wb = np.array([r[config.COL_GYRO_X], r[config.COL_GYRO_Y], r[config.COL_GYRO_Z]])
                av, wv = run_align.transform_imu(ab, wb)

                ukf_no.predict(av[0], wv[2], dt=0.1)
                ukf_ct.predict(av[0], wv[2], dt=0.1)

                win_buf.append(np.concatenate([ab, wb]))
                if len(win_buf) >= config.WINDOW_SIZE and (step_i % 5 == 0):
                    warr = np.ascontiguousarray(np.array(win_buf[-config.WINDOW_SIZE:], dtype=np.float32))
                    pred_d, ev_class, sigma = predictor.predict(warr)
                    if ev_class == 0 or pred_d < 0.1:
                        ukf_no.update_zupt(wv[2])
                        ukf_ct.update_zupt(wv[2])
                    else:
                        ukf_no.update_velocity_net(pred_d * 1.15, max(sigma * 1.15, 0.2), ev_class, 2.0)
                        ukf_ct.update_velocity_net(pred_d * 1.15, max(sigma * 1.15, 0.2), ev_class, 2.0)
                if len(win_buf) > 40: win_buf.pop(0)

                # 1D Cross-track measurement update
                if step_i % 5 == 0:
                    pos = np.array([ukf_ct.x[0], ukf_ct.x[1]])
                    hdg = math.degrees(ukf_ct.x[3]) % 360.0
                    spd = ukf_ct.x[2]
                    sig = math.sqrt(ukf_ct.P[0, 0] + ukf_ct.P[1, 1])
                    mres = matcher.match(pos, hdg, spd, max(spd * 0.5, 0.05), sig, 0.5)

                    if not mres.is_off_road and mres.confidence >= 0.25:
                        psi_r = mres.road_heading_rad
                        n_N = -math.sin(psi_r)
                        n_E =  math.cos(psi_r)
                        # Scalar innovation along normal:
                        y_res = n_N * (mres.snapped_point[0] - ukf_ct.x[0]) + n_E * (mres.snapped_point[1] - ukf_ct.x[1])
                        sig_meas = 2.5 / max(mres.confidence, 0.25)
                        R_1d = sig_meas ** 2
                        S = n_N**2 * ukf_ct.P[0, 0] + 2.0 * n_N * n_E * ukf_ct.P[0, 1] + n_E**2 * ukf_ct.P[1, 1] + R_1d
                        nis = (y_res ** 2) / S

                        if nis <= 6.635:
                            K_N = (ukf_ct.P[0, 0] * n_N + ukf_ct.P[0, 1] * n_E) / S
                            K_E = (ukf_ct.P[1, 0] * n_N + ukf_ct.P[1, 1] * n_E) / S
                            ukf_ct.x[0] += K_N * y_res
                            ukf_ct.x[1] += K_E * y_res
                            # 1D Covariance reduction strictly perpendicular to road
                            K_arr = np.zeros(7)
                            K_arr[0] = K_N
                            K_arr[1] = K_E
                            ukf_ct.P -= S * np.outer(K_arr, K_arr)
                            ukf_ct.P = 0.5 * (ukf_ct.P + ukf_ct.P.T)

                no_pN.append(ukf_no.x[0]); no_pE.append(ukf_no.x[1])
                ct_pN.append(ukf_ct.x[0]); ct_pE.append(ukf_ct.x[1])

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_pN = (gt_lat - init_lat) * (np.pi / 180.0) * 6371000.0
            gt_pE = (gt_lon - init_lon) * (np.pi / 180.0) * 6371000.0 * np.cos(np.radians(init_lat))
            dist_m = float(np.sum(np.hypot(np.diff(gt_pN), np.diff(gt_pE))))

            lat_no, lon_no = local_xy_to_latlon(np.array(no_pE), np.array(no_pN), init_lat, init_lon)
            lat_ct, lon_ct = local_xy_to_latlon(np.array(ct_pE), np.array(ct_pN), init_lat, init_lon)

            sc_no = score_outage_segment(lat_no, lon_no, gt_lat, gt_lon, dist_m, dur)
            sc_ct = score_outage_segment(lat_ct, lon_ct, gt_lat, gt_lon, dist_m, dur)

            d_no = sc_no["pct_of_distance"]
            d_ct = sc_ct["pct_of_distance"]
            fpe_no = sc_no["final_pos_error_m"]
            fpe_ct = sc_ct["final_pos_error_m"]
            ct_no = sc_no["cross_track_m"]
            ct_ct = sc_ct["cross_track_m"]

            print(f"[{run_name} {dur}s outage {oid}] WITHOUT MAP Drift: {d_no:.2f}% (FPE: {fpe_no:.1f}m, CT: {ct_no:.1f}m) | 1D CT MAP Drift: {d_ct:.2f}% (FPE: {fpe_ct:.1f}m, CT: {ct_ct:.1f}m)")

if __name__ == "__main__":
    run_1d_cross_track_comparison()
