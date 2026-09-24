"""
Plotting script for Phase 19 MAPHDE forensic analysis on representative failure cases:
vw14b_o5, vw14c_o4, and vw4_o1.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from modules.alignment import AlignmentEngine, AlignmentState
from modules.can_fusion_ukf import CANFusionUKF, compute_pre_outage_can_calibration
from modules.map_matcher import (
    load_road_graph_from_osm_json,
    RoadGraph
)
from eval.evaluate_phase19_maphde import (
    run_three_arm_simulation,
    get_benchmark_trajectories,
    PHASE18A_FAILURE_CASES
)
from eval.baseline import local_xy_to_latlon

def plot_case_forensics():
    osm_path = BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = BASE_DIR / "data" / "osm_uk_test_roads.json"
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)

    cases_to_plot = ["vw14b_o5", "vw14c_o4", "vw4_o1"]
    test_files = get_benchmark_trajectories()

    plots_dir = BASE_DIR / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
            scenario_name = f"{run_name}_o{oid}"
            if scenario_name not in cases_to_plot:
                continue

            dur = int(row_out["duration_s"])
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

            sim_res = run_three_arm_simulation(
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

            # Compute FPE time series
            t_steps = np.arange(len(df_sub)) * 0.1
            fpe_A_t = []
            fpe_B_t = []
            EARTH_R = 6371000.0
            lat0_rad = math.radians(init_lat)
            for k in range(len(df_sub)):
                # GT pos in local meters
                dN_gt = (gt_lat[k] - init_lat) * (math.pi / 180.0) * EARTH_R
                dE_gt = (gt_lon[k] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
                # Arm A
                dNA = (sim_res["A"]["lat"][k] - init_lat) * (math.pi / 180.0) * EARTH_R
                dEA = (sim_res["A"]["lon"][k] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
                fpe_A_t.append(math.hypot(dNA - dN_gt, dEA - dE_gt))
                # Arm B
                dNB = (sim_res["B"]["lat"][k] - init_lat) * (math.pi / 180.0) * EARTH_R
                dEB = (sim_res["B"]["lon"][k] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
                fpe_B_t.append(math.hypot(dNB - dN_gt, dEB - dE_gt))

            # Multi-panel forensic plot
            fig, axs = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f"MAPHDE Forensic Analysis: {scenario_name} (Duration {dur}s)", fontsize=14, fontweight="bold")

            # Panel 1: Heading vs Time
            ax = axs[0, 0]
            if gt_hdg is not None:
                ax.plot(t_steps, gt_hdg, label="Ground Truth", color="black", linestyle="--", linewidth=1.5)
            ax.plot(t_steps, sim_res["A"]["heading_deg"], label="Arm A (Baseline)", color="#4A90E2", linewidth=1.5)
            ax.plot(t_steps, sim_res["B"]["heading_deg"], label="Arm B (MAPHDE)", color="#E94A4A", linewidth=1.5)
            ax.set_ylabel("Heading (deg)")
            ax.set_title("Navigation Heading vs Time")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

            # Panel 2: Road Heading vs Navigation Heading
            ax = axs[0, 1]
            tr_B = sim_res["B"]["trace"]
            if tr_B:
                ax.plot(tr_B["t"], tr_B["hdg_nav"], label="Nav Heading", color="#E94A4A", linewidth=1.5)
                ax.plot(tr_B["t"], tr_B["hdg_road"], label="Road Heading", color="#7ED321", linestyle="-.", linewidth=1.2)
                ax.set_ylabel("Heading (deg)")
                ax.set_title("Road Heading vs Nav Heading (Arm B)")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8)

            # Panel 3: MAPHDE Integrator I(t) & Error E(t)
            ax = axs[1, 0]
            if tr_B:
                ax.plot(tr_B["t"], tr_B["I"], label="Integrator I(t) (deg)", color="#E94A4A", linewidth=2)
                ax.plot(tr_B["t"], tr_B["E_deg"], label="Heading Error E(t) (deg)", color="#9013FE", linestyle=":", alpha=0.7)
                ax.axhline(0, color="gray", linestyle="--")
                ax.set_ylabel("Degrees")
                ax.set_title("MAPHDE Integrator I(t) & Heading Error")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8)

            # Panel 4: Map Association Status & Confidence
            ax = axs[1, 1]
            if tr_B:
                ax.plot(tr_B["t"], tr_B["conf"], label="Map Match Confidence", color="#F5A623", linewidth=1.5)
                ax.plot(tr_B["t"], [1.0 if act else 0.0 for act in tr_B["active"]], label="MAPHDE Active (1=Active, 0=Susp)", color="#4A90E2", linestyle="--", alpha=0.8)
                ax.set_ylabel("Confidence / Active")
                ax.set_ylim(-0.1, 1.1)
                ax.set_title("Map Match Confidence & MAPHDE Status")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8)

            # Panel 5: Trajectory (pE vs pN)
            ax = axs[2, 0]
            # Convert to local meters
            gt_dN = (gt_lat - init_lat) * (math.pi / 180.0) * EARTH_R
            gt_dE = (gt_lon - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
            A_dN = (sim_res["A"]["lat"] - init_lat) * (math.pi / 180.0) * EARTH_R
            A_dE = (sim_res["A"]["lon"] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)
            B_dN = (sim_res["B"]["lat"] - init_lat) * (math.pi / 180.0) * EARTH_R
            B_dE = (sim_res["B"]["lon"] - init_lon) * (math.pi / 180.0) * EARTH_R * math.cos(lat0_rad)

            ax.plot(gt_dE, gt_dN, label="Ground Truth", color="black", linestyle="--", linewidth=1.5)
            ax.plot(A_dE, A_dN, label=f"Arm A (FPE={fpe_A_t[-1]:.1f}m)", color="#4A90E2", linewidth=1.5)
            ax.plot(B_dE, B_dN, label=f"Arm B (FPE={fpe_B_t[-1]:.1f}m)", color="#E94A4A", linewidth=1.5)
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")
            ax.set_title("Trajectory Ground Plane")
            ax.axis("equal")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

            # Panel 6: FPE vs Time
            ax = axs[2, 1]
            ax.plot(t_steps, fpe_A_t, label=f"Arm A Final={fpe_A_t[-1]:.1f}m", color="#4A90E2", linewidth=1.5)
            ax.plot(t_steps, fpe_B_t, label=f"Arm B Final={fpe_B_t[-1]:.1f}m", color="#E94A4A", linewidth=1.5)
            ax.set_xlabel("Time in Outage (s)")
            ax.set_ylabel("Position Error (m)")
            ax.set_title("Position Error (FPE) vs Time")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

            plt.tight_layout()
            out_png = plots_dir / f"phase19_forensic_{scenario_name}.png"
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"Generated forensic plot for {scenario_name} -> {out_png}")

if __name__ == "__main__":
    plot_case_forensics()
