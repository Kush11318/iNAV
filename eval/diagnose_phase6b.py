import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from eval.baseline import local_xy_to_latlon
from modules.ukf import UKFNavigationFilter
from modules.alignment import AlignmentEngine, AlignmentState
from modules.velocity_net import VelocityNetPredictor
from modules.map_matcher import FixedLagHMMMapMatcher, load_road_graph_from_osm_json

def run_diagnostic():
    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    # 1. Inspect both OSM files: Indore cache vs UK test roads
    indore_osm = config.BASE_DIR / "android" / "app" / "src" / "main" / "assets" / "osm_roads_cache.json"
    uk_osm = config.BASE_DIR / "data" / "osm_uk_test_roads.json"

    print("==================================================")
    print("PHASE 6B FORENSIC DIAGNOSTIC")
    print("==================================================")
    print(f"Indore OSM path exists: {indore_osm.exists()}")
    print(f"UK OSM path exists: {uk_osm.exists()}")

    # Find test runs
    sync_files = sorted(config.SYNC_PROCESSED_DIR.glob("sync_vw*.parquet"))
    test_files = [f for f in sync_files if any(f.stem.replace("sync_", "").lower().startswith(p) for p in config.TEST_DRIVERS)][:5]
    print(f"Test files ({len(test_files)}): {[f.stem for f in test_files]}")

    log_records = []
    
    # Run comparison on each test run
    for test_file in test_files:
        run_name = test_file.stem.replace("sync_", "")
        df_raw = pd.read_parquet(test_file)
        df_sim, df_outages = inject_outages(df_raw, generate_outage_schedule(len(df_raw)))

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df_sim.loc[mask]
            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()
            pre_idx_start = max(0, outage_start_idx - 10)
            pre_win = df_sim.iloc[pre_idx_start:outage_start_idx]

            init_lat = float(pre_win[config.COL_GPS_LAT].iloc[-1])
            init_lon = float(pre_win[config.COL_GPS_LON].iloc[-1])
            init_spd = float(pre_win[config.COL_GPS_SPEED_MS].iloc[-1]) if config.COL_GPS_SPEED_MS in pre_win.columns else 0.0
            init_hdg = float(pre_win[config.COL_GPS_BEARING].iloc[-1]) if config.COL_GPS_BEARING in pre_win.columns else 0.0

            # Pre-calibrate run alignment using warmup acceleration & gyroscope
            run_align = AlignmentEngine()
            warmup = df_raw.iloc[:min(50, len(df_raw))]
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            gyro_w = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref)
            y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

            # Scale factor k
            k_scale = 1.15

            # A. RUN WITHOUT MAP
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

            # B. RUN WITH OLD INDORE MAP CACHE (Phase 6 as committed)
            graph_indore = load_road_graph_from_osm_json(str(indore_osm), ref_lat=22.72, ref_lon=75.83)
            matcher_indore = FixedLagHMMMapMatcher(graph=graph_indore)

            ukf_map = UKFNavigationFilter(dt=0.1)
            ukf_map.initialize(init_lat, init_lon, init_spd, np.radians(init_hdg))
            map_pN, map_pE = [], []
            matched_pts = []
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

                # Map update every 5 steps
                if step_i % 5 == 0:
                    pos_before = np.array([ukf_map.x[0], ukf_map.x[1]])
                    cov_before = float(ukf_map.P[0, 0] + ukf_map.P[1, 1])
                    hdg_deg = math.degrees(ukf_map.x[3]) % 360.0
                    spd = ukf_map.x[2]
                    sig_p = math.sqrt(cov_before)

                    mres = matcher_indore.match(pos_before, hdg_deg, spd, max(spd * 0.5, 0.05), sig_p, 0.5)
                    accepted = False
                    pos_nis = 0.0

                    if not mres.is_off_road and mres.confidence >= 0.25:
                        accepted, pos_nis, _ = ukf_map.update_map_match(
                            float(mres.snapped_point[0]),
                            float(mres.snapped_point[1]),
                            float(mres.road_heading_rad),
                            float(mres.confidence),
                            bool(mres.is_heading_valid)
                        )
                        matched_pts.append((mres.snapped_point[0], mres.snapped_point[1]))
                    else:
                        matched_pts.append((np.nan, np.nan))

                    pos_after = np.array([ukf_map.x[0], ukf_map.x[1]])
                    cov_after = float(ukf_map.P[0, 0] + ukf_map.P[1, 1])
                    hdg_resid = math.degrees(mres.road_heading_rad - ukf_map.x[3])

                    log_records.append({
                        "run": run_name,
                        "outage_id": oid,
                        "dur": dur,
                        "step": step_i,
                        "ukf_pN_before": pos_before[0],
                        "ukf_pE_before": pos_before[1],
                        "map_pN": mres.snapped_point[0],
                        "map_pE": mres.snapped_point[1],
                        "edge_id": mres.edge_id,
                        "confidence": mres.confidence,
                        "cross_track_m": mres.cross_track_error_m,
                        "hdg_residual_deg": hdg_resid,
                        "pos_nis": pos_nis,
                        "accepted": accepted,
                        "ukf_pN_after": pos_after[0],
                        "ukf_pE_after": pos_after[1],
                        "cov_trace_before": cov_before,
                        "cov_trace_after": cov_after,
                        "gnss_health": "PURE_DR"
                    })

                map_pN.append(ukf_map.x[0])
                map_pE.append(ukf_map.x[1])

            # Evaluate scores
            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_pN = (gt_lat - init_lat) * (np.pi / 180.0) * 6371000.0
            gt_pE = (gt_lon - init_lon) * (np.pi / 180.0) * 6371000.0 * np.cos(np.radians(init_lat))

            dist_m = float(np.sum(np.hypot(np.diff(gt_pN), np.diff(gt_pE)))) if len(gt_pN) > 1 else 10.0
            lat_nomap, lon_nomap = local_xy_to_latlon(np.array(nomap_pE), np.array(nomap_pN), init_lat, init_lon)
            lat_map, lon_map = local_xy_to_latlon(np.array(map_pE), np.array(map_pN), init_lat, init_lon)

            sc_nomap = score_outage_segment(lat_nomap, lon_nomap, gt_lat, gt_lon, dist_m, dur)
            sc_map = score_outage_segment(lat_map, lon_map, gt_lat, gt_lon, dist_m, dur)

            print(f"[{run_name} {dur}s outage {oid}] WITHOUT MAP Drift: {sc_nomap['pct_of_distance']:.2f}%, CT: {sc_nomap['cross_track_m']:.2f}m | WITH MAP Drift: {sc_map['pct_of_distance']:.2f}%, CT: {sc_map['cross_track_m']:.2f}m")

            # Generate Plot for this outage
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(gt_pE, gt_pN, 'k-', linewidth=2.5, label="Ground Truth (VBOX)")
            ax.plot(nomap_pE, nomap_pN, 'b--', linewidth=2.0, label="WITHOUT MAP (inav_ai_ukf_v1)")
            ax.plot(map_pE, map_pN, 'r-', linewidth=2.0, label="WITH MAP (inav_ukf_map)")
            
            mE = [p[1] for p in matched_pts if not np.isnan(p[1])]
            mN = [p[0] for p in matched_pts if not np.isnan(p[0])]
            if mE:
                ax.scatter(mE, mN, c='green', marker='x', s=40, label="Matched Road Snaps")

            ax.set_title(f"Forensic Map Matching: {run_name} ({dur}s Outage {oid})\nWITHOUT MAP CT: {sc_nomap['cross_track_m']:.1f}m vs WITH MAP CT: {sc_map['cross_track_m']:.1f}m")
            ax.set_xlabel("Local East [m]")
            ax.set_ylabel("Local North [m]")
            ax.legend(loc="best")
            ax.grid(True, linestyle=":", alpha=0.6)
            plot_p = config.RESULTS_DIR / f"diagnostic_{run_name}_outage_{oid}_{dur}s.png"
            fig.savefig(plot_p, dpi=120, bbox_inches="tight")
            plt.close(fig)

    df_log = pd.DataFrame(log_records)
    log_csv = config.RESULTS_DIR / "phase6b_map_update_log.csv"
    df_log.to_csv(log_csv, index=False)
    print(f"\nSaved detailed log ({len(df_log)} records) to {log_csv}")

if __name__ == "__main__":
    run_diagnostic()
