import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from eval.baseline import local_xy_to_latlon
from modules.ukf import UKFNavigationFilter
from modules.alignment import AlignmentEngine, AlignmentState
from modules.velocity_net import VelocityNetPredictor
from modules.map_matcher import FixedLagHMMMapMatcher, load_road_graph_from_osm_json

def generate_verification_plots():
    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)
    uk_osm = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    indore_osm = config.BASE_DIR / "android" / "app" / "src" / "main" / "assets" / "osm_roads_cache.json"

    graph_uk = load_road_graph_from_osm_json(str(uk_osm), ref_lat=52.20, ref_lon=-2.19)
    matcher_uk = FixedLagHMMMapMatcher(graph=graph_uk)

    sync_files = sorted(config.SYNC_PROCESSED_DIR.glob("sync_vw*.parquet"))
    target_stems = {"sync_vw10", "sync_vw11", "sync_vw12", "sync_vw14a"}
    test_files = [f for f in sync_files if f.stem in target_stems]

    for test_file in test_files:
        run_name = test_file.stem.replace("sync_", "")
        df_raw = pd.read_parquet(test_file)
        total_dur = df_raw[config.COL_TIME].iloc[-1] - df_raw[config.COL_TIME].iloc[0]
        sched = generate_outage_schedule(total_dur, run_id=run_name)
        df_sim, df_outages = inject_outages(df_raw, sched)

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            if dur < 30 and run_name != "vw10":
                continue

            mask = (df_sim["outage_id"] == oid)
            df_sub = df_sim.loc[mask]
            if len(df_sub) < 10:
                continue

            outage_start_idx = mask.idxmax()
            pre_win = df_sim.iloc[max(0, outage_start_idx-10):outage_start_idx]
            init_lat = float(pre_win[config.COL_GPS_LAT].iloc[-1])
            init_lon = float(pre_win[config.COL_GPS_LON].iloc[-1])
            init_spd = float(pre_win[config.COL_GPS_SPEED_MS].iloc[-1]) if config.COL_GPS_SPEED_MS in pre_win.columns else 0.0
            init_hdg = float(pre_win[config.COL_GPS_BEARING].iloc[-1]) if config.COL_GPS_BEARING in pre_win.columns else 0.0

            run_align = AlignmentEngine()
            warmup = df_raw.iloc[:min(50, len(df_raw))]
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref); y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

            k_scale = 1.15

            # 1. Ground Truth in local meters
            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_pN = 6371000.0 * np.radians(gt_lat - init_lat)
            gt_pE = 6371000.0 * np.radians(gt_lon - init_lon) * np.cos(np.radians(init_lat))

            # 2. WITHOUT MAP
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
                        ukf_nomap.update_velocity_net(pred_d * k_scale, max(sigma * k_scale, 0.2), ev_class, 2.0)
                if len(win_buf) > 40:
                    win_buf.pop(0)
                nomap_pN.append(ukf_nomap.x[0])
                nomap_pE.append(ukf_nomap.x[1])

            # 3. WITH MAP (Fixed UK Map with 1D Cross-Track Constraint)
            matcher_uk.reset()
            ukf_map = UKFNavigationFilter(dt=0.1)
            ukf_map.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            map_pN, map_pE = [], []
            matched_pN, matched_pE = [], []
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
                        ukf_map.update_velocity_net(pred_d * k_scale, max(sigma * k_scale, 0.2), ev_class, 2.0)
                if len(win_buf) > 40:
                    win_buf.pop(0)

                # Map update
                if step_i % 5 == 0:
                    pos_before = np.array([ukf_map.x[0], ukf_map.x[1]])
                    hdg_deg = math.degrees(ukf_map.x[3]) % 360.0
                    spd = ukf_map.x[2]
                    sig_p = math.sqrt(ukf_map.P[0, 0] + ukf_map.P[1, 1])

                    # Local to lat/lon
                    cur_lat = init_lat + math.degrees(pos_before[0] / 6371000.0)
                    cur_lon = init_lon + math.degrees(pos_before[1] / (6371000.0 * math.cos(math.radians(init_lat))))

                    if matcher_uk.graph.is_in_bounds(cur_lat, cur_lon):
                        p_graph = matcher_uk.graph.latlon_to_local(cur_lat, cur_lon)
                        mres = matcher_uk.match(p_graph, hdg_deg, spd, max(spd * 0.5, 0.05), sig_p, 0.5)
                        if not mres.is_off_road and mres.confidence >= 0.25:
                            s_lat, s_lon = matcher_uk.graph.local_to_latlon(mres.snapped_point[0], mres.snapped_point[1])
                            p_N_m = 6371000.0 * math.radians(s_lat - init_lat)
                            p_E_m = 6371000.0 * math.radians(s_lon - init_lon) * math.cos(math.radians(init_lat))
                            ukf_map.update_map_match(p_N_m, p_E_m, mres.road_heading_rad, mres.confidence, mres.is_heading_valid)
                            matched_pN.append(p_N_m)
                            matched_pE.append(p_E_m)

                map_pN.append(ukf_map.x[0])
                map_pE.append(ukf_map.x[1])

            # Generate Trajectory Comparison Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(gt_pE, gt_pN, 'k--', linewidth=2.5, label="1. Ground Truth")
            ax.plot(nomap_pE, nomap_pN, 'b-', linewidth=2.0, label="2. WITHOUT MAP (inav_ai_ukf_v1)")
            ax.plot(map_pE, map_pN, 'g-', linewidth=2.0, label="3. WITH MAP (inav_ukf_map 1D Constraint)")
            if matched_pE:
                ax.scatter(matched_pE, matched_pN, color='orange', s=25, alpha=0.7, zorder=5, label="4. Map-Matched Road Position")
            ax.scatter([0], [0], color='red', marker='*', s=120, zorder=10, label="Outage Start")

            ax.set_title(f"Phase 6B Trajectory Comparison: {run_name} Outage {oid} ({dur}s)", fontsize=14, fontweight='bold')
            ax.set_xlabel("East Position (m)", fontsize=12)
            ax.set_ylabel("North Position (m)", fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axis('equal')

            plot_path = config.RESULTS_DIR / f"phase6b_verified_{run_name}_outage_{oid}_{dur}s.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Generated verification plot: {plot_path}")
            break # 1 outage per run is sufficient for plot

if __name__ == "__main__":
    generate_verification_plots()
