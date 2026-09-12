"""
Phase 10: Critical Model-Level Benchmark
Compares Model A (Phase 8A 4s Direct Displacement Net) vs Model B (Phase 10 4s Temporal Velocity Net)
on held-out test dataset (test_windows_phase10.npz, 60,084 windows from 'vw' campaign).

Evaluates:
1. Model-Level Metrics:
   - Velocity MAE, RMSE, Bias, Pearson r, R^2
   - Integrated 4s Displacement MAE, RMSE, Bias
2. Speed-Bin Analysis (0-5, 5-10, 10-15, 15-20, 20-25, 25-30, >30 m/s):
   - Compares GT vs Model A vs Model B
   - Computes Predicted / GT scale ratios
3. Temporal Profile Plots on representative driving segments:
   - Acceleration, Cruising, Braking, Turning, Rough Road
4. Cumulative Longitudinal Distance Error Tracking:
   - s_GT(t), s_A(t), s_B(t) across full test trajectories
"""

import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase8a_4s_model import VelocityNet4s
from eval.train_phase10_temporal_model import TemporalVelocityNet4s

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase10_Benchmark")

VIZ_DIR = BASE_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_models() -> Tuple[VelocityNet4s, TemporalVelocityNet4s]:
    model_a_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    model_b_path = config.MODELS_DIR / "phase10_temporal_velocity_net.pt"

    if not model_a_path.exists():
        raise FileNotFoundError(f"Model A checkpoint not found: {model_a_path}")
    if not model_b_path.exists():
        raise FileNotFoundError(f"Model B checkpoint not found: {model_b_path}")

    model_a = VelocityNet4s(in_channels=6, num_events=5)
    model_a.load_state_dict(torch.load(model_a_path, map_location="cpu"))
    model_a.eval()

    model_b = TemporalVelocityNet4s(in_channels=6, num_events=5)
    model_b.load_state_dict(torch.load(model_b_path, map_location="cpu"))
    model_b.eval()

    return model_a, model_b


def run_model_benchmark():
    logger.info("=" * 80)
    logger.info("PHASE 10: CRITICAL MODEL-LEVEL BENCHMARK (HELD-OUT TEST SET)")
    logger.info("=" * 80)

    test_path = config.WINDOWED_DIR / "test_windows_phase10.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    logger.info(f"Loading test dataset from {test_path}...")
    test_npz = np.load(test_path)
    X_test = test_npz["X_seq"]          # (N, 40, 6)
    y_vel_test = test_npz["y_vel_20"]   # (N, 20)
    y_delta_d_test = test_npz["y_delta_d"] # (N,)
    y_spd_mean_test = test_npz["y_speed_mean"] # (N,)
    y_ev_test = test_npz["y_event"]     # (N,)

    N = len(X_test)
    logger.info(f"Total test windows to evaluate: {N:,}")

    model_a, model_b = load_models()

    batch_size = 2048
    pred_d_a_list = []
    pred_v_seq_b_list = []

    logger.info("Running batched inference for Model A and Model B...")
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X_test[i : i + batch_size]  # (B, 40, 6)
            bx = torch.from_numpy(batch.transpose(0, 2, 1)).float()  # (B, 6, 40)

            # Model A
            d_a, _, _ = model_a(bx)
            pred_d_a_list.append(d_a.squeeze(-1).cpu().numpy())

            # Model B
            v_b, _, _ = model_b(bx)
            pred_v_seq_b_list.append(v_b.cpu().numpy())

    pred_d_a = np.concatenate(pred_d_a_list, axis=0)        # (N,)
    pred_v_seq_b = np.concatenate(pred_v_seq_b_list, axis=0) # (N, 20)

    # Implied velocity for Model A: v_A = Delta d_A / 4.0
    pred_v_mean_a = pred_d_a / 4.0

    # Integrated displacement for Model B: Delta d_B = sum(v_b * 0.2s)
    pred_d_b = np.sum(pred_v_seq_b * 0.2, axis=1)           # (N,)
    pred_v_mean_b = np.mean(pred_v_seq_b, axis=1)          # (N,)

    # Ground truth speed
    gt_spd_mean = y_spd_mean_test                           # (N,)
    gt_delta_d = y_delta_d_test                             # (N,)

    # =========================================================================
    # 1. Overall Metrics
    # =========================================================================
    def calc_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        err = pred - gt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        r = float(np.corrcoef(pred, gt)[0, 1]) if np.std(pred) > 1e-6 and np.std(gt) > 1e-6 else 0.0
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))
        return {"mae": mae, "rmse": rmse, "bias": bias, "r": r, "r2": r2}

    m_vel_a = calc_metrics(pred_v_mean_a, gt_spd_mean)
    m_vel_b = calc_metrics(pred_v_mean_b, gt_spd_mean)

    m_dist_a = calc_metrics(pred_d_a, gt_delta_d)
    m_dist_b = calc_metrics(pred_d_b, gt_delta_d)

    # Sequence-level velocity metrics for Model B (all 20*N points)
    all_v_b_flat = pred_v_seq_b.flatten()
    all_v_gt_flat = y_vel_test.flatten()
    m_vel_seq_b = calc_metrics(all_v_b_flat, all_v_gt_flat)

    print("\n" + "=" * 80)
    print("                MODEL-LEVEL BENCHMARK RESULTS (TEST SET, N=60,084)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Model A (Phase 8A 4s)':<22} | {'Model B (Phase 10 Velocity)':<25}")
    print("-" * 80)
    print(f"{'Velocity MAE (m/s)':<30} | {m_vel_a['mae']:<22.4f} | {m_vel_b['mae']:<25.4f}")
    print(f"{'Velocity RMSE (m/s)':<30} | {m_vel_a['rmse']:<22.4f} | {m_vel_b['rmse']:<25.4f}")
    print(f"{'Velocity Bias (m/s)':<30} | {m_vel_a['bias']:<22.4f} | {m_vel_b['bias']:<25.4f}")
    print(f"{'Velocity Pearson r':<30} | {m_vel_a['r']:<22.4f} | {m_vel_b['r']:<25.4f}")
    print(f"{'Velocity R^2':<30} | {m_vel_a['r2']:<22.4f} | {m_vel_b['r2']:<25.4f}")
    print(f"{'Seq Velocity RMSE (20-step)':<30} | {'N/A (Scalar Only)':<22} | {m_vel_seq_b['rmse']:<25.4f}")
    print("-" * 80)
    print(f"{'Integrated 4s Dist MAE (m)':<30} | {m_dist_a['mae']:<22.2f} | {m_dist_b['mae']:<25.2f}")
    print(f"{'Integrated 4s Dist RMSE (m)':<30} | {m_dist_a['rmse']:<22.2f} | {m_dist_b['rmse']:<25.2f}")
    print(f"{'Integrated 4s Dist Bias (m)':<30} | {m_dist_a['bias']:<22.2f} | {m_dist_b['bias']:<25.2f}")
    print(f"{'Displacement Pearson r':<30} | {m_dist_a['r']:<22.4f} | {m_dist_b['r']:<25.4f}")

    # =========================================================================
    # 2. Speed-Bin Analysis
    # =========================================================================
    bins = [0, 5, 10, 15, 20, 25, 30, 1000]
    bin_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '>30']

    bin_cats = pd.cut(gt_spd_mean, bins=bins, labels=bin_labels, right=False)

    print("\n" + "=" * 95)
    print("                     SPEED-BIN ANALYSIS: RATIOS & DISPLACEMENT")
    print("=" * 95)
    print(f"{'Speed Bin (m/s)':<15} | {'N':<6} | {'Mean GT (m/s)':<13} | {'Model A (m/s)':<13} | {'Model B (m/s)':<13} | {'Ratio A/GT':<11} | {'Ratio B/GT':<11}")
    print("-" * 95)

    bin_rows = []
    for bl in bin_labels:
        mask = (bin_cats == bl)
        count = int(np.sum(mask))
        if count == 0:
            continue

        gt_b_mean = float(np.mean(gt_spd_mean[mask]))
        va_b_mean = float(np.mean(pred_v_mean_a[mask]))
        vb_b_mean = float(np.mean(pred_v_mean_b[mask]))

        ratio_a = va_b_mean / max(gt_b_mean, 1e-3)
        ratio_b = vb_b_mean / max(gt_b_mean, 1e-3)

        print(f"{bl:<15} | {count:<6} | {gt_b_mean:<13.2f} | {va_b_mean:<13.2f} | {vb_b_mean:<13.2f} | {ratio_a:<11.3f} | {ratio_b:<11.3f}")

        bin_rows.append({
            "speed_bin": bl,
            "count": count,
            "gt_mean_ms": round(gt_b_mean, 2),
            "model_a_ms": round(va_b_mean, 2),
            "model_b_ms": round(vb_b_mean, 2),
            "ratio_a": round(ratio_a, 4),
            "ratio_b": round(ratio_b, 4)
        })

    # =========================================================================
    # 3. Temporal Profile Plots on Representative Segments
    # =========================================================================
    logger.info("Generating temporal profile plots across driving conditions...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    regimes = [
        (0, "Stationary / Near Rest"),
        (1, "Steady Cruising"),
        (2, "Braking Event"),
        (3, "Turning / Curve"),
        (4, "Rough Road / High Vibration")
    ]

    for plot_idx, (ev_id, title) in enumerate(regimes):
        ev_mask = (y_ev_test == ev_id)
        indices = np.where(ev_mask)[0]
        ax = axes[plot_idx]

        if len(indices) > 0:
            # Pick a representative window with good dynamics
            cand_idx = indices[len(indices) // 3]

            time_20 = np.linspace(0.1, 3.9, 20)
            gt_v20 = y_vel_test[cand_idx]
            b_v20 = pred_v_seq_b[cand_idx]
            a_const_v = pred_v_mean_a[cand_idx]

            ax.plot(time_20, gt_v20, 'k-', linewidth=2.5, label="GT Velocity (m/s)")
            ax.plot(time_20, b_v20, 'b-o', linewidth=2.0, markersize=4, label="Model B Predicted (5 Hz)")
            ax.axhline(a_const_v, color='r', linestyle='--', linewidth=2.0, label=f"Model A Implied ({a_const_v:.1f} m/s)")

            ax.set_title(f"Regime: {title} (Idx {cand_idx})", fontsize=11, fontweight='bold')
            ax.set_xlabel("Time in 4s Window (seconds)", fontsize=9)
            ax.set_ylabel("Speed (m/s)", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=8, loc='best')

    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    prof_plot_path = VIZ_DIR / "phase10_temporal_profiles.png"
    plt.savefig(prof_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved temporal profile plots to: {prof_plot_path}")

    # =========================================================================
    # 4. Longitudinal Cumulative Distance Error Tracking
    # =========================================================================
    logger.info("Computing cumulative distance curves on representative full test trajectory...")
    sample_tf = config.SYNC_PROCESSED_DIR / "sync_vw14b.parquet"
    if not sample_tf.exists():
        sample_tf = config.SYNC_PROCESSED_DIR / "sync_vw2.parquet"

    if sample_tf.exists():
        df_traj = pd.read_parquet(sample_tf)
        n_traj = len(df_traj)
        dt = config.TARGET_DT

        # Run model predictions across trajectory with stride 10 (every 1.0s)
        stride_s = 10
        traj_times = []
        cum_dist_gt = [0.0]
        cum_dist_a = [0.0]
        cum_dist_b = [0.0]

        cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
        for c in cols:
            df_traj[c] = df_traj[c].interpolate().bfill().ffill().fillna(0.0)
        seq_full = df_traj[cols].values.astype(np.float32)
        spd_gt_full = df_traj[config.COL_TRUE_SPEED_MS].values

        curr_gt = 0.0
        curr_a = 0.0
        curr_b = 0.0

        for k in range(39, n_traj, stride_s):
            t_s = k * dt
            traj_times.append(t_s)

            # Ground truth integrated over the stride
            d_gt_stride = float(np.sum(spd_gt_full[k - stride_s + 1 : k + 1]) * dt)
            curr_gt += d_gt_stride

            win = seq_full[k - 39 : k + 1]  # (40, 6)
            bx = torch.from_numpy(win.T).unsqueeze(0).float()  # (1, 6, 40)

            with torch.no_grad():
                d_a, _, _ = model_a(bx)
                v_b, _, _ = model_b(bx)

            # Stride distance: scale 4s window prediction by stride / 4.0
            scale_fac = (stride_s * dt) / 4.0
            d_a_stride = float(d_a.item()) * scale_fac
            d_b_stride = float(torch.sum(v_b * 0.2).item()) * scale_fac

            curr_a += d_a_stride
            curr_b += d_b_stride

            cum_dist_gt.append(curr_gt)
            cum_dist_a.append(curr_a)
            cum_dist_b.append(curr_b)

        cum_dist_gt = np.array(cum_dist_gt[1:])
        cum_dist_a = np.array(cum_dist_a[1:])
        cum_dist_b = np.array(cum_dist_b[1:])
        traj_times = np.array(traj_times)

        err_a = cum_dist_a - cum_dist_gt
        err_b = cum_dist_b - cum_dist_gt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(traj_times, cum_dist_gt, 'k-', linewidth=2.0, label="Ground Truth Distance")
        ax1.plot(traj_times, cum_dist_a, 'r--', linewidth=1.8, label="Model A Integrated Distance")
        ax1.plot(traj_times, cum_dist_b, 'b-.', linewidth=1.8, label="Model B Integrated Distance")
        ax1.set_ylabel("Cumulative Distance (m)", fontsize=11)
        ax1.set_title(f"Cumulative Distance Comparison ({sample_tf.stem})", fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=10)

        ax2.plot(traj_times, err_a, 'r-', linewidth=2.0, label=f"Model A Error (Final: {err_a[-1]:.1f}m)")
        ax2.plot(traj_times, err_b, 'b-', linewidth=2.0, label=f"Model B Error (Final: {err_b[-1]:.1f}m)")
        ax2.axhline(0, color='k', linestyle=':', alpha=0.7)
        ax2.set_xlabel("Time (seconds)", fontsize=11)
        ax2.set_ylabel("Distance Error (m)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        dist_plot_path = VIZ_DIR / "phase10_cumulative_distance_error.png"
        plt.savefig(dist_plot_path, dpi=200)
        plt.close()
        logger.info(f"Saved cumulative distance error plot to: {dist_plot_path}")

    return {
        "m_vel_a": m_vel_a,
        "m_vel_b": m_vel_b,
        "m_dist_a": m_dist_a,
        "m_dist_b": m_dist_b,
        "m_vel_seq_b": m_vel_seq_b,
        "bin_rows": bin_rows
    }


if __name__ == "__main__":
    run_model_benchmark()
