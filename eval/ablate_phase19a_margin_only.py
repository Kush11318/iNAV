"""
Ablation study: Evaluate Viterbi Margin Gate (M >= 3.0) ALONE (without Doppler bgz lock).
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
from modules.can_fusion_ukf import compute_pre_outage_can_calibration
from modules.map_matcher import load_road_graph_from_osm_json
from eval.evaluate_phase19a_viterbi_margin import run_single_outage_simulation, get_benchmark_trajectories

def run_ablation():
    osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)

    test_files = get_benchmark_trajectories()
    rows = []

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

            # Arm A: Baseline
            res_A = run_single_outage_simulation(
                df_outage=df_sub, init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_spd,
                init_heading_deg=init_hdg, init_gyro_bias_rad=0.0, alignment=run_align,
                graph=graph, mode="A", r_eff=r_eff_est
            )

            # Arm B_margin_only: Standard init, but with Viterbi Margin Gate M >= 3.0
            res_B_margin = run_single_outage_simulation(
                df_outage=df_sub, init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_spd,
                init_heading_deg=init_hdg, init_gyro_bias_rad=0.0, alignment=run_align,
                graph=graph, mode="B", margin_thresh=3.0, r_eff=r_eff_est
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            metrics_A = score_outage_segment(res_A["lat"], res_A["lon"], gt_lat, gt_lon, dist_gt, dur, gt_hdg, res_A["heading_deg"])
            metrics_B_margin = score_outage_segment(res_B_margin["lat"], res_B_margin["lon"], gt_lat, gt_lon, dist_gt, dur, gt_hdg, res_B_margin["heading_deg"])

            scenario_name = f"{run_name}_o{oid}"
            fpe_A = float(metrics_A["fpe_m"])
            fpe_Bm = float(metrics_B_margin["fpe_m"])

            rows.append({
                "scenario": scenario_name,
                "duration_s": dur,
                "fpe_A": fpe_A,
                "fpe_B_margin": fpe_Bm,
                "delta": fpe_Bm - fpe_A,
                "gated": res_B_margin["map_margin_gated"]
            })

    df_ab = pd.DataFrame(rows)
    print("\n=== ABLATION RESULTS: VITERBI MARGIN GATE ALONE (NO DOPPLER BIAS LOCK) ===")
    print(f"Median FPE Arm A: {df_ab['fpe_A'].median():.2f}m")
    print(f"Median FPE Arm B (Margin Only): {df_ab['fpe_B_margin'].median():.2f}m")
    print(f"Mean FPE Arm A: {df_ab['fpe_A'].mean():.2f}m")
    print(f"Mean FPE Arm B (Margin Only): {df_ab['fpe_B_margin'].mean():.2f}m")
    print(f"Improved: {(df_ab['delta'] < 0).sum()}, Worsened: {(df_ab['delta'] > 0).sum()}, Identical: {(df_ab['delta'] == 0).sum()}")

    # Duration breakdown
    for d in [30, 60, 120, 180]:
        sub = df_ab[df_ab['duration_s'] == d]
        print(f"Duration {d:3d}s: A={sub['fpe_A'].median():.2f}m -> B_margin={sub['fpe_B_margin'].median():.2f}m (delta={sub['fpe_B_margin'].median() - sub['fpe_A'].median():+.2f}m)")

    # Save to CSV
    df_ab.to_csv("eval/phase19a_margin_gate_ablation.csv", index=False)

if __name__ == "__main__":
    run_ablation()
