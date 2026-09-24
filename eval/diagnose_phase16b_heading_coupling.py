"""
Phase 16B: CAN + Heading Diagnostic Script

Performs deep forensic investigation of:
1. CAN speed error vs Ground Truth (MAE, RMSE, Bias) across all 56 outages.
2. Heading error vs Ground Truth (Final and Mean) across all 56 outages.
3. Along-track error, Cross-track error, and FPE.
4. Pearson and Spearman correlations:
   - Heading error vs Cross-track error
   - Heading error vs FPE degradation (FPE_CAN - FPE_Ctrl)
   - CAN speed error vs Along-track error
   - CAN speed error vs FPE
   Separated by duration: 10s, 30s, 60s, 120s, 180s.
5. In-depth analysis and 9-panel multi-trace figures for the Worst 5 CAN failures:
   - GT speed vs Control speed vs CAN speed
   - GT heading vs Control heading vs CAN heading
   - Along-track, Cross-track, and Total FPE evolution over time
   - 2D trajectory spatial comparison
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment, decompose_along_cross_track
from modules.alignment import AlignmentEngine, AlignmentState
from modules.can_fusion_ukf import CANFusionUKF, compute_pre_outage_can_calibration
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase16B_Diagnostic")

ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")


def compute_baseline_a_predictions(
    model: AnchoredTemporalVelocityNet,
    seq_data: np.ndarray,
    outage_start_idx: int,
    sub_len: int,
    v_anchor_frozen: float,
    window_len: int = 40,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float, float]]:
    lookup = {}
    model.eval()
    windows = []
    step_indices = []

    for k in range(0, sub_len):
        global_k = outage_start_idx + k
        if global_k >= window_len - 1 and (k % 5 == 0):
            win = seq_data[global_k - window_len + 1 : global_k + 1]
            windows.append(win)
            step_indices.append(global_k)

    if not windows:
        return lookup

    all_windows = np.array(windows, dtype=np.float32)
    bx = torch.from_numpy(all_windows.transpose(0, 2, 1)).float().to(device)
    b_anc = torch.full((len(all_windows), 1), v_anchor_frozen, dtype=torch.float32).to(device)

    with torch.no_grad():
        del_v_seq, v_seq, sig_seq, ev = model(bx, b_anc)
        p_v = v_seq.cpu().numpy()
        p_sig = sig_seq.cpu().numpy()
        p_ev = torch.argmax(ev, dim=1).cpu().numpy()
        p_v_mean = np.mean(p_v, axis=1)
        p_v_end = p_v[:, -1]
        p_sig_mean = np.mean(p_sig, axis=1)

    for idx, gk in enumerate(step_indices):
        lookup[gk] = (
            float(p_v_mean[idx]),
            int(p_ev[idx]),
            float(p_sig_mean[idx]),
            float(p_v_end[idx])
        )
    return lookup


def run_single_outage_detailed(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    lookup_preds: Dict[int, Tuple[float, int, float, float]],
    global_start_idx: int,
    alignment: AlignmentEngine,
    mode: str = "A",
    r_eff: float = 0.2776,
    r_can_var: float = 0.0325,
    dt: float = config.TARGET_DT
) -> Dict[str, Any]:
    n = len(df_outage)
    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    omega_rl = df_outage[config.COL_TRUE_WHEEL_RL].values
    omega_rr = df_outage[config.COL_TRUE_WHEEL_RR].values
    omega_rear = 0.5 * (omega_rl + omega_rr)
    v_can_arr = omega_rear * r_eff

    ukf = CANFusionUKF(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=math.radians(init_heading_deg)
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_heading_deg = np.zeros(n)

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)
        global_k = global_start_idx + k

        if mode == "A":
            if global_k in lookup_preds and (k % 5 == 0):
                v_meas, ev_class, sig_meas, _ = lookup_preds[global_k]
                cal_d = v_meas * 2.0
                cal_sig = max(sig_meas * 2.0, 0.2)
                if ev_class == 0 or v_meas < 0.15:
                    ukf.update_zupt(gyro_reading=w_v[2])
                else:
                    ukf.update_velocity_net(cal_d, cal_sig, ev_class, window_dur=2.0)
        elif mode == "B1":
            v_can_val = float(v_can_arr[k])
            rear_diff = abs(omega_rl[k] - omega_rr[k])
            is_slipping = rear_diff > 25.0
            if v_can_val < 0.15 and not is_slipping:
                ukf.update_zupt(gyro_reading=w_v[2])
            elif not is_slipping:
                ukf.update_can_speed(v_can=v_can_val, r_var=r_can_var)

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]
        est_heading_deg[k] = math.degrees(ukf.x[3]) % 360.0

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)

    # Time-series decomposition
    gt_lat = df_outage[config.COL_TRUE_LAT].values
    gt_lon = df_outage[config.COL_TRUE_LON].values
    gt_hdg = df_outage[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_outage.columns else None

    # Error series
    R_earth = 6371000.0
    d_lat = np.radians(est_lat - gt_lat) * R_earth
    d_lon = np.radians(est_lon - gt_lon) * R_earth * np.cos(np.radians(gt_lat))
    pos_err_series = np.sqrt(d_lat**2 + d_lon**2)

    along_series = np.zeros(n)
    cross_series = np.zeros(n)
    hdg_err_series = np.zeros(n)

    for k in range(n):
        th = gt_hdg[k] if gt_hdg is not None else est_heading_deg[k]
        psi = np.radians(th)
        along_series[k] = d_lon[k] * np.sin(psi) + d_lat[k] * np.cos(psi)
        cross_series[k] = d_lon[k] * np.cos(psi) - d_lat[k] * np.sin(psi)
        if gt_hdg is not None:
            dh = (est_heading_deg[k] - gt_hdg[k] + 180.0) % 360.0 - 180.0
            hdg_err_series[k] = abs(dh)

    return {
        "lat": est_lat,
        "lon": est_lon,
        "speed": est_speed,
        "heading_deg": est_heading_deg,
        "pN": est_pN,
        "pE": est_pE,
        "pos_err": pos_err_series,
        "along": along_series,
        "cross": cross_series,
        "hdg_err": hdg_err_series,
        "v_can_arr": v_can_arr
    }


def run_phase16b_analysis():
    logger.info("=" * 90)
    logger.info("PHASE 16B: CAN + HEADING DIAGNOSTIC & OBSERVABILITY ANALYSIS")
    logger.info("=" * 90)

    model_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    records = []
    traces_store = {}

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
            y = np.cross(u_z, ref)
            y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

        for c in cols:
            df[c] = df[c].interpolate().bfill().ffill().fillna(0.0)

        seq_data = df[cols].values.astype(np.float32)

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            dist_gt = float(row["distance_travelled_m"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()
            pre_idx_start = max(0, outage_start_idx - 10)
            pre_df = df.iloc[pre_idx_start:outage_start_idx]

            if len(pre_df) > 0:
                init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
                init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
                init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
                h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
                init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
                v_anchor_val = float(pre_df[config.COL_TRUE_SPEED_MS].iloc[-1])
            else:
                init_lat = float(df[config.COL_TRUE_LAT].iloc[outage_start_idx])
                init_lon = float(df[config.COL_TRUE_LON].iloc[outage_start_idx])
                init_spd = float(df[config.COL_TRUE_SPEED_MS].iloc[outage_start_idx])
                init_hdg = float(df[config.COL_TRUE_HEADING].iloc[outage_start_idx])
                v_anchor_val = init_spd

            cal_pre_start = max(0, outage_start_idx - 300)
            cal_pre_df = df.iloc[cal_pre_start:outage_start_idx]
            r_eff_est, cal_method, _ = compute_pre_outage_can_calibration(
                pre_gps_speed=cal_pre_df["gps_speed_ms"].values,
                pre_wheel_rl=cal_pre_df[config.COL_TRUE_WHEEL_RL].values,
                pre_wheel_rr=cal_pre_df[config.COL_TRUE_WHEEL_RR].values
            )

            gt_lat = df_sub[config.COL_TRUE_LAT].values
            gt_lon = df_sub[config.COL_TRUE_LON].values
            gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None
            gt_speed = df_sub[config.COL_TRUE_SPEED_MS].values

            lookup_a = compute_baseline_a_predictions(model, seq_data, outage_start_idx, len(df_sub), v_anchor_val)

            res_a = run_single_outage_detailed(
                df_sub, init_lat, init_lon, init_spd, init_hdg, lookup_a,
                outage_start_idx, run_align, mode="A"
            )
            res_b1 = run_single_outage_detailed(
                df_sub, init_lat, init_lon, init_spd, init_hdg, lookup_a,
                outage_start_idx, run_align, mode="B1", r_eff=r_eff_est
            )

            # Accuracy of CAN speed vs Ground Truth
            v_can = res_b1["v_can_arr"]
            speed_err = v_can - gt_speed
            can_speed_mae = float(np.mean(np.abs(speed_err)))
            can_speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
            can_speed_bias = float(np.mean(speed_err))

            # Accuracy of Heading vs Ground Truth
            final_hdg_err = float(res_b1["hdg_err"][-1])
            mean_hdg_err = float(np.mean(res_b1["hdg_err"]))

            fpe_a = float(res_a["pos_err"][-1])
            fpe_b1 = float(res_b1["pos_err"][-1])
            fpe_deg = fpe_b1 - fpe_a  # Positive means CAN worsened FPE

            along_a = float(res_a["along"][-1])
            along_b1 = float(res_b1["along"][-1])
            cross_a = float(res_a["cross"][-1])
            cross_b1 = float(res_b1["cross"][-1])

            sc_id = f"{run_name}_o{oid}"
            traces_store[sc_id] = {
                "t": np.arange(len(df_sub)) * config.TARGET_DT,
                "gt_speed": gt_speed,
                "ctrl_speed": res_a["speed"],
                "can_speed": res_b1["speed"],
                "v_can_raw": v_can,
                "gt_heading": gt_hdg,
                "ctrl_heading": res_a["heading_deg"],
                "can_heading": res_b1["heading_deg"],
                "along_a": res_a["along"],
                "along_b1": res_b1["along"],
                "cross_a": res_a["cross"],
                "cross_b1": res_b1["cross"],
                "pos_err_a": res_a["pos_err"],
                "pos_err_b1": res_b1["pos_err"],
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "lat_a": res_a["lat"],
                "lon_a": res_a["lon"],
                "lat_b1": res_b1["lat"],
                "lon_b1": res_b1["lon"],
                "fpe_deg": fpe_deg,
                "dur": dur,
                "run": run_name,
                "oid": oid
            }

            records.append({
                "scenario": sc_id,
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": dist_gt,
                # CAN speed error vs GT
                "can_speed_mae": can_speed_mae,
                "can_speed_rmse": can_speed_rmse,
                "can_speed_bias": can_speed_bias,
                # Heading error vs GT
                "final_hdg_err_deg": final_hdg_err,
                "mean_hdg_err_deg": mean_hdg_err,
                # Position errors
                "along_a_m": along_a,
                "along_b1_m": along_b1,
                "cross_a_m": cross_a,
                "cross_b1_m": cross_b1,
                "fpe_a_m": fpe_a,
                "fpe_b1_m": fpe_b1,
                "fpe_deg_m": fpe_deg,  # positive -> worsened
                "along_deg_m": abs(along_b1) - abs(along_a),
                "cross_deg_m": abs(cross_b1) - abs(cross_a),
            })

    df_diag = pd.DataFrame(records)
    out_csv = config.BASE_DIR / "eval" / "phase16b_diagnostic_metrics.csv"
    df_diag.to_csv(out_csv, index=False)
    logger.info(f"Saved diagnostic metrics to {out_csv} ({len(df_diag)} scenarios)")

    # -------------------------------------------------------------
    # 1. CAN Speed Error Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("1. CAN SPEED ERROR VERSUS GROUND TRUTH (ACROSS 56 OUTAGES)")
    print("=" * 95)
    print(f"Mean Absolute Error (MAE) : {df_diag['can_speed_mae'].mean():.4f} m/s (median: {df_diag['can_speed_mae'].median():.4f} m/s)")
    print(f"Root Mean Square Error    : {df_diag['can_speed_rmse'].mean():.4f} m/s (median: {df_diag['can_speed_rmse'].median():.4f} m/s)")
    print(f"Mean Speed Bias           : {df_diag['can_speed_bias'].mean():+.4f} m/s")
    print(f"P95 Speed RMSE            : {df_diag['can_speed_rmse'].quantile(0.95):.4f} m/s")
    print(f"P99 Speed RMSE            : {df_diag['can_speed_rmse'].quantile(0.99):.4f} m/s")

    # -------------------------------------------------------------
    # 2. Heading Error Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("2. HEADING ERROR VERSUS GROUND TRUTH (ACROSS 56 OUTAGES)")
    print("=" * 95)
    print(f"Median Final Heading Error: {df_diag['final_hdg_err_deg'].median():.2f} deg (mean: {df_diag['final_hdg_err_deg'].mean():.2f} deg)")
    print(f"Median Mean Heading Error : {df_diag['mean_hdg_err_deg'].median():.2f} deg (mean: {df_diag['mean_hdg_err_deg'].mean():.2f} deg)")
    print(f"P95 Final Heading Error   : {df_diag['final_hdg_err_deg'].quantile(0.95):.2f} deg")
    print(f"Scenarios with Hdg Err > 20 deg: {(df_diag['final_hdg_err_deg'] > 20.0).sum()}/{len(df_diag)} ({(df_diag['final_hdg_err_deg'] > 20.0).mean()*100:.1f}%)")

    # -------------------------------------------------------------
    # 3. Correlation Matrix (Global & By Duration)
    # -------------------------------------------------------------
    def compute_corrs(sub_df, label):
        if len(sub_df) < 3:
            return
        r_hdg_cross, p1 = pearsonr(sub_df["final_hdg_err_deg"], sub_df["cross_b1_m"].abs())
        r_hdg_fpedeg, p2 = pearsonr(sub_df["final_hdg_err_deg"], sub_df["fpe_deg_m"])
        r_spd_along, p3 = pearsonr(sub_df["can_speed_rmse"], sub_df["along_b1_m"].abs())
        r_spd_fpe, p4 = pearsonr(sub_df["can_speed_rmse"], sub_df["fpe_b1_m"])

        rho_hdg_cross, _ = spearmanr(sub_df["final_hdg_err_deg"], sub_df["cross_b1_m"].abs())
        rho_hdg_fpedeg, _ = spearmanr(sub_df["final_hdg_err_deg"], sub_df["fpe_deg_m"])
        rho_spd_along, _ = spearmanr(sub_df["can_speed_rmse"], sub_df["along_b1_m"].abs())
        rho_spd_fpe, _ = spearmanr(sub_df["can_speed_rmse"], sub_df["fpe_b1_m"])

        print(f"| {label:<12} | {len(sub_df):<4} | {r_hdg_cross:+.3f} ({rho_hdg_cross:+.3f}) | {r_hdg_fpedeg:+.3f} ({rho_hdg_fpedeg:+.3f}) | {r_spd_along:+.3f} ({rho_spd_along:+.3f}) | {r_spd_fpe:+.3f} ({rho_spd_fpe:+.3f}) |")

    print("\n" + "=" * 105)
    print("3. CORRELATION ANALYSIS (Pearson r / Spearman rho)")
    print("=" * 105)
    print(f"| {'Subset':<12} | {'N':<4} | {'Hdg vs CrossErr':<17} | {'Hdg vs FPEDegrad':<17} | {'SpdErr vs Along':<17} | {'SpdErr vs FPE':<17} |")
    print("-" * 105)
    compute_corrs(df_diag, "ALL (Global)")
    for d in sorted(df_diag["duration_s"].unique()):
        sub = df_diag[df_diag["duration_s"] == d]
        compute_corrs(sub, f"{d}s Outages")
    print("=" * 105)

    # -------------------------------------------------------------
    # 4. Identification of Worst 5 CAN Failures
    # -------------------------------------------------------------
    worst_5 = df_diag.sort_values(by="fpe_deg_m", ascending=False).head(5)
    print("\n" + "=" * 125)
    print("4. TOP 5 WORST CAN FAILURES (Ranked by FPE Degradation: FPE_CAN - FPE_Ctrl)")
    print("=" * 125)
    print(f"{'Scenario':<18} | {'Dur':<4} | {'Ctrl FPE':<10} | {'CAN FPE':<10} | {'Degradation':<12} | {'CAN Spd RMSE':<14} | {'Final Hdg Err':<14} | {'Along Err':<11} | {'Cross Err':<11}")
    print("-" * 125)
    for _, r in worst_5.iterrows():
        print(f"{r['scenario']:<18} | {int(r['duration_s']):<4} | {r['fpe_a_m']:<10.2f} | {r['fpe_b1_m']:<10.2f} | {r['fpe_deg_m']:+11.2f}m | {r['can_speed_rmse']:<14.4f} | {r['final_hdg_err_deg']:<14.2f} | {r['along_b1_m']:<11.2f} | {r['cross_b1_m']:<11.2f}")
    print("=" * 125)

    # -------------------------------------------------------------
    # 5. Generate Multi-Panel Traces for the Worst 5 Failures
    # -------------------------------------------------------------
    logger.info("Generating multi-panel trace figures for the worst 5 failures...")

    for rank, (_, r) in enumerate(worst_5.iterrows(), 1):
        sc_id = r["scenario"]
        tdata = traces_store[sc_id]
        t = tdata["t"]

        fig, axs = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            f"Worst Failure #{rank}: {sc_id} ({int(r['duration_s'])}s Outage) — FPE Degradation: {r['fpe_deg_m']:+.1f}m\n"
            f"CAN Speed RMSE: {r['can_speed_rmse']:.3f} m/s | Final Heading Error: {r['final_hdg_err_deg']:.1f}°",
            fontsize=13, fontweight="bold"
        )

        # Panel 1: Speed Profiles
        ax_spd = axs[0, 0]
        ax_spd.plot(t, tdata["gt_speed"], "k-", linewidth=2.5, label="GT Speed")
        ax_spd.plot(t, tdata["ctrl_speed"], "r--", linewidth=1.8, label="Control Speed (Frozen/VNet)")
        ax_spd.plot(t, tdata["can_speed"], "g-", linewidth=2.0, label="CAN-Fused Speed")
        ax_spd.set_xlabel("Time (s)", fontweight="bold")
        ax_spd.set_ylabel("Speed (m/s)", fontweight="bold")
        ax_spd.set_title("Forward Velocity Tracking", fontweight="bold")
        ax_spd.grid(True, linestyle=":", alpha=0.6)
        ax_spd.legend(loc="best")

        # Panel 2: Heading Profiles
        ax_hdg = axs[0, 1]
        if tdata["gt_heading"] is not None:
            ax_hdg.plot(t, tdata["gt_heading"], "k-", linewidth=2.5, label="GT Heading")
        ax_hdg.plot(t, tdata["ctrl_heading"], "r--", linewidth=1.8, label="Control Heading")
        ax_hdg.plot(t, tdata["can_heading"], "m-", linewidth=2.0, label="CAN-Fused Heading")
        ax_hdg.set_xlabel("Time (s)", fontweight="bold")
        ax_hdg.set_ylabel("Heading (°)", fontweight="bold")
        ax_hdg.set_title("Heading Evolution", fontweight="bold")
        ax_hdg.grid(True, linestyle=":", alpha=0.6)
        ax_hdg.legend(loc="best")

        # Panel 3: Along vs Cross vs Total Error over Time
        ax_err = axs[1, 0]
        ax_err.plot(t, np.abs(tdata["along_b1"]), "b-", linewidth=2.0, label="|Along-Track Error| (CAN)")
        ax_err.plot(t, np.abs(tdata["cross_b1"]), "r-", linewidth=2.0, label="|Cross-Track Error| (CAN)")
        ax_err.plot(t, tdata["pos_err_b1"], "k-", linewidth=2.5, label="Total FPE (CAN)")
        ax_err.plot(t, tdata["pos_err_a"], "gray", linestyle="--", linewidth=1.8, label="Total FPE (Control)")
        ax_err.set_xlabel("Time (s)", fontweight="bold")
        ax_err.set_ylabel("Error (m)", fontweight="bold")
        ax_err.set_title("Along-Track vs Cross-Track Error Evolution", fontweight="bold")
        ax_err.grid(True, linestyle=":", alpha=0.6)
        ax_err.legend(loc="best")

        # Panel 4: 2D Spatial Trajectory
        ax_traj = axs[1, 1]
        # Local tangent ENU from start
        lat0, lon0 = tdata["gt_lat"][0], tdata["gt_lon"][0]
        R_e = 6371000.0
        gt_E = (tdata["gt_lon"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        gt_N = (tdata["gt_lat"] - lat0) * np.radians(1.0) * R_e
        a_E = (tdata["lon_a"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        a_N = (tdata["lat_a"] - lat0) * np.radians(1.0) * R_e
        b1_E = (tdata["lon_b1"] - lon0) * np.radians(1.0) * R_e * np.cos(np.radians(lat0))
        b1_N = (tdata["lat_b1"] - lat0) * np.radians(1.0) * R_e

        ax_traj.plot(gt_E, gt_N, "k-", linewidth=2.5, label="Ground Truth")
        ax_traj.plot(a_E, a_N, "r--", linewidth=1.8, label="Control")
        ax_traj.plot(b1_E, b1_N, "g-", linewidth=2.0, label="CAN-Fused")
        ax_traj.scatter([0], [0], color="black", s=80, marker="o", label="Start")
        ax_traj.scatter([gt_E[-1]], [gt_N[-1]], color="black", s=100, marker="*", label="GT End")
        ax_traj.scatter([a_E[-1]], [a_N[-1]], color="red", s=80, marker="x", label="Ctrl End")
        ax_traj.scatter([b1_E[-1]], [b1_N[-1]], color="green", s=80, marker="^", label="CAN End")
        ax_traj.set_xlabel("East (m)", fontweight="bold")
        ax_traj.set_ylabel("North (m)", fontweight="bold")
        ax_traj.set_title("2D Tangent Plane Trajectory", fontweight="bold")
        ax_traj.grid(True, linestyle=":", alpha=0.6)
        ax_traj.axis("equal")
        ax_traj.legend(loc="best")

        plt.tight_layout()
        out_img = ARTIFACTS_DIR / f"phase16b_worst_failure_{rank}_{sc_id}.png"
        plt.savefig(out_img, dpi=180)
        plt.close()
        logger.info(f"Saved: {out_img}")

    return df_diag, worst_5


if __name__ == "__main__":
    run_phase16b_analysis()
