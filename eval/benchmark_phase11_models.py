"""
Phase 11: Comprehensive Model-Level Benchmark
Compares:
  - Baseline Model A: Phase 10 TemporalVelocityNet4s (unanchored absolute velocity model)
  - Experiment Model B: Phase 11 AnchoredTemporalVelocityNet (speed-weighted loss)
  - Ablation Model C: Phase 11 AnchoredTemporalVelocityNet (uniform loss)
on held-out test dataset (test_windows_phase11.npz, 60,084 windows from 'vw' campaign).

Evaluates:
  1. Overall Model-Level Metrics:
     - Velocity MAE, RMSE, Bias, Pearson r, R^2
     - Integrated 4s Displacement MAE, RMSE, Bias
     - Residual velocity metrics vs absolute velocity metrics
  2. Speed-Bin Analysis (0-5, 5-10, 10-15, 15-20, 20-25, 25-30, >30 m/s)
  3. Dynamic Event Analysis on Key Driving Regimes:
     - 28-30 m/s cruise, 30+ m/s cruise, 17->7 m/s braking, 8 m/s turning, 19-20 m/s rough road
  4. Anchor Sensitivity Analysis (v_anchor + delta for delta in [-2.0, -1.0, -0.5, 0, +0.5, +1.0, +2.0] m/s)
  5. Latency & Model Size Profiling
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
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
from eval.train_phase10_temporal_model import TemporalVelocityNet4s
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase11_Benchmark")

VIZ_DIR = BASE_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_all_models() -> Tuple[TemporalVelocityNet4s, AnchoredTemporalVelocityNet, AnchoredTemporalVelocityNet]:
    model_a_path = config.MODELS_DIR / "phase10_temporal_velocity_net.pt"
    model_b_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    model_c_path = config.MODELS_DIR / "phase11_ablation_uniform_net.pt"

    if not model_a_path.exists():
        raise FileNotFoundError(f"Model A checkpoint not found: {model_a_path}")
    if not model_b_path.exists():
        raise FileNotFoundError(f"Model B checkpoint not found: {model_b_path}")
    if not model_c_path.exists():
        raise FileNotFoundError(f"Model C checkpoint not found: {model_c_path}")

    model_a = TemporalVelocityNet4s(in_channels=6, num_events=5)
    model_a.load_state_dict(torch.load(model_a_path, map_location="cpu"))
    model_a.eval()

    model_b = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model_b.load_state_dict(torch.load(model_b_path, map_location="cpu"))
    model_b.eval()

    model_c = AnchoredTemporalVelocityNet(in_channels=6, num_events=5)
    model_c.load_state_dict(torch.load(model_c_path, map_location="cpu"))
    model_c.eval()

    return model_a, model_b, model_c


def run_benchmark():
    logger.info("=" * 80)
    logger.info("PHASE 11: CRITICAL MODEL-LEVEL BENCHMARK (60,084 HELD-OUT TEST WINDOWS)")
    logger.info("=" * 80)

    test_path = config.WINDOWED_DIR / "test_windows_phase11.npz"
    test_npz = np.load(test_path)
    X_test = test_npz["X_seq"]               # (N, 40, 6)
    v_anc_test = test_npz["v_anchor"]         # (N, 1)
    y_vel_test = test_npz["y_vel_20"]        # (N, 20)
    y_delta_v_test = test_npz["y_delta_v_20"] # (N, 20)
    y_delta_d_test = test_npz["y_delta_d"]    # (N,)
    y_ev_test = test_npz["y_event"]          # (N,)

    N = len(X_test)
    logger.info(f"Loaded {N:,} test windows.")

    model_a, model_b, model_c = load_all_models()

    batch_size = 2048
    pred_v_a_list = []
    pred_v_b_list = []
    pred_v_c_list = []
    pred_del_v_b_list = []
    pred_del_v_c_list = []

    logger.info("Running batched inference across all models...")
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = X_test[i : i + batch_size]
            batch_anc = v_anc_test[i : i + batch_size]

            bx = torch.from_numpy(batch_x.transpose(0, 2, 1)).float()
            b_anc = torch.from_numpy(batch_anc).float()

            # Model A (Phase 10 unanchored)
            v_a, _, _ = model_a(bx)
            pred_v_a_list.append(v_a.cpu().numpy())

            # Model B (Phase 11 anchored, speed-weighted loss)
            del_v_b, v_b, _, _ = model_b(bx, b_anc)
            pred_v_b_list.append(v_b.cpu().numpy())
            pred_del_v_b_list.append(del_v_b.cpu().numpy())

            # Model C (Phase 11 anchored, uniform loss)
            del_v_c, v_c, _, _ = model_c(bx, b_anc)
            pred_v_c_list.append(v_c.cpu().numpy())
            pred_del_v_c_list.append(del_v_c.cpu().numpy())

    total_infer_time = time.time() - t0
    logger.info(f"Inference complete in {total_infer_time:.2f}s ({total_infer_time / (3 * N) * 1000:.3f} ms per sample per model).")

    v_pred_a = np.concatenate(pred_v_a_list, axis=0)      # (N, 20)
    v_pred_b = np.concatenate(pred_v_b_list, axis=0)      # (N, 20)
    v_pred_c = np.concatenate(pred_v_c_list, axis=0)      # (N, 20)
    del_v_pred_b = np.concatenate(pred_del_v_b_list, axis=0)
    del_v_pred_c = np.concatenate(pred_del_v_c_list, axis=0)

    # Integrated 4-second displacement: sum(v * 0.2s)
    d_pred_a = np.sum(v_pred_a * 0.2, axis=1)  # (N,)
    d_pred_b = np.sum(v_pred_b * 0.2, axis=1)  # (N,)
    d_pred_c = np.sum(v_pred_c * 0.2, axis=1)  # (N,)
    d_gt = y_delta_d_test                      # (N,)

    # Flattened velocities across all 20*N timesteps
    v_flat_gt = y_vel_test.flatten()
    v_flat_a = v_pred_a.flatten()
    v_flat_b = v_pred_b.flatten()
    v_flat_c = v_pred_c.flatten()
    del_v_flat_gt = y_delta_v_test.flatten()
    del_v_flat_b = del_v_pred_b.flatten()
    del_v_flat_c = del_v_pred_c.flatten()

    def calc_metrics(y_true, y_pred):
        diff = y_pred - y_true
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        return {"mae": mae, "rmse": rmse, "bias": bias, "r": r, "r2": r2}

    m_vel_a = calc_metrics(v_flat_gt, v_flat_a)
    m_vel_b = calc_metrics(v_flat_gt, v_flat_b)
    m_vel_c = calc_metrics(v_flat_gt, v_flat_c)

    m_disp_a = calc_metrics(d_gt, d_pred_a)
    m_disp_b = calc_metrics(d_gt, d_pred_b)
    m_disp_c = calc_metrics(d_gt, d_pred_c)

    m_res_b = calc_metrics(del_v_flat_gt, del_v_flat_b)
    m_res_c = calc_metrics(del_v_flat_gt, del_v_flat_c)

    print("\n" + "=" * 95)
    print(f"{'METRIC':<30} | {'MODEL A (Phase 10)':<18} | {'MODEL B (Anchored+W)':<20} | {'MODEL C (Ablation)':<18}")
    print("=" * 95)
    print(f"{'Velocity MAE (m/s)':<30} | {m_vel_a['mae']:<18.4f} | {m_vel_b['mae']:<20.4f} | {m_vel_c['mae']:<18.4f}")
    print(f"{'Velocity RMSE (m/s)':<30} | {m_vel_a['rmse']:<18.4f} | {m_vel_b['rmse']:<20.4f} | {m_vel_c['rmse']:<18.4f}")
    print(f"{'Velocity Bias (m/s)':<30} | {m_vel_a['bias']:<18.4f} | {m_vel_b['bias']:<20.4f} | {m_vel_c['bias']:<18.4f}")
    print(f"{'Pearson Correlation r':<30} | {m_vel_a['r']:<18.4f} | {m_vel_b['r']:<20.4f} | {m_vel_c['r']:<18.4f}")
    print(f"{'Coefficient of Determ. R^2':<30} | {m_vel_a['r2']:<18.4f} | {m_vel_b['r2']:<20.4f} | {m_vel_c['r2']:<18.4f}")
    print("-" * 95)
    print(f"{'4s Displacement MAE (m)':<30} | {m_disp_a['mae']:<18.4f} | {m_disp_b['mae']:<20.4f} | {m_disp_c['mae']:<18.4f}")
    print(f"{'4s Displacement RMSE (m)':<30} | {m_disp_a['rmse']:<18.4f} | {m_disp_b['rmse']:<20.4f} | {m_disp_c['rmse']:<18.4f}")
    print(f"{'4s Displacement Bias (m)':<30} | {m_disp_a['bias']:<18.4f} | {m_disp_b['bias']:<20.4f} | {m_disp_c['bias']:<18.4f}")
    print("-" * 95)
    print(f"{'Residual Delta v MAE (m/s)':<30} | {'N/A':<18} | {m_res_b['mae']:<20.4f} | {m_res_c['mae']:<18.4f}")
    print(f"{'Residual Delta v RMSE (m/s)':<30} | {'N/A':<18} | {m_res_b['rmse']:<20.4f} | {m_res_c['rmse']:<18.4f}")
    print(f"{'Residual Delta v Bias (m/s)':<30} | {'N/A':<18} | {m_res_b['bias']:<20.4f} | {m_res_c['bias']:<18.4f}")
    print("=" * 95)

    # SPEED-BIN ANALYSIS
    bins = [0, 5, 10, 15, 20, 25, 30, 1000]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    anc_flat = v_anc_test.flatten()
    cat = pd.cut(anc_flat, bins=bins, labels=labels, right=False)

    print("\n" + "=" * 115)
    print("SPEED-BIN ANALYSIS ACROSS ANCHOR SPEED BINS (TEST SET)")
    print("=" * 115)
    print(f"{'Bin (m/s)':<8} | {'N':<6} | {'GT Mean':<8} | {'Anc Mean':<8} | {'Pred A':<8} | {'Pred B':<8} | {'Bias A':<8} | {'Bias B':<8} | {'MAE A':<7} | {'MAE B':<7} | {'Ratio A':<7} | {'Ratio B':<7}")
    print("-" * 115)

    speed_bin_rows = []
    for l in labels:
        mask = (cat == l)
        n_bin = int(np.sum(mask))
        if n_bin == 0:
            continue
        gt_m = float(np.mean(y_vel_test[mask]))
        anc_m = float(np.mean(anc_flat[mask]))
        pa_m = float(np.mean(v_pred_a[mask]))
        pb_m = float(np.mean(v_pred_b[mask]))
        pc_m = float(np.mean(v_pred_c[mask]))

        bias_a = pa_m - gt_m
        bias_b = pb_m - gt_m
        bias_c = pc_m - gt_m

        mae_a = float(np.mean(np.abs(v_pred_a[mask] - y_vel_test[mask])))
        mae_b = float(np.mean(np.abs(v_pred_b[mask] - y_vel_test[mask])))
        mae_c = float(np.mean(np.abs(v_pred_c[mask] - y_vel_test[mask])))

        rmse_a = float(np.sqrt(np.mean((v_pred_a[mask] - y_vel_test[mask]) ** 2)))
        rmse_b = float(np.sqrt(np.mean((v_pred_b[mask] - y_vel_test[mask]) ** 2)))
        rmse_c = float(np.sqrt(np.mean((v_pred_c[mask] - y_vel_test[mask]) ** 2)))

        rat_a = float(np.median(v_pred_a[mask] / np.maximum(y_vel_test[mask], 0.1)))
        rat_b = float(np.median(v_pred_b[mask] / np.maximum(y_vel_test[mask], 0.1)))
        rat_c = float(np.median(v_c_m := v_pred_c[mask] / np.maximum(y_vel_test[mask], 0.1)))

        print(f"{l:<8} | {n_bin:<6} | {gt_m:<8.2f} | {anc_m:<8.2f} | {pa_m:<8.2f} | {pb_m:<8.2f} | {bias_a:<+8.2f} | {bias_b:<+8.2f} | {mae_a:<7.2f} | {mae_b:<7.2f} | {rat_a:<7.3f} | {rat_b:<7.3f}")

        speed_bin_rows.append({
            "bin": l, "N": n_bin, "gt_mean": gt_m, "anc_mean": anc_m,
            "pred_a": pa_m, "pred_b": pb_m, "pred_c": pc_m,
            "bias_a": bias_a, "bias_b": bias_b, "bias_c": bias_c,
            "mae_a": mae_a, "mae_b": mae_b, "mae_c": mae_c,
            "rmse_a": rmse_a, "rmse_b": rmse_b, "rmse_c": rmse_c,
            "ratio_a": rat_a, "ratio_b": rat_b, "ratio_c": rat_c
        })
    print("=" * 115)

    # ANCHOR SENSITIVITY ANALYSIS
    print("\n" + "=" * 90)
    print("ANCHOR SENSITIVITY ANALYSIS (SYNTHETIC PERTURBATION DELTA v_anchor)")
    print("=" * 90)
    print(f"{'Synthetic Error Delta (m/s)':<30} | {'Reconstructed Vel MAE':<24} | {'Displacement Error Bias (m)':<25}")
    print("-" * 90)

    sens_deltas = [-2.0, -1.0, -0.5, 0.0, +0.5, +1.0, +2.0]
    sens_results = []
    with torch.no_grad():
        # Evaluate on subset of 5,000 windows for sensitivity curve
        sub_idx = np.linspace(0, N - 1, 5000, dtype=int)
        sub_x = torch.from_numpy(X_test[sub_idx].transpose(0, 2, 1)).float()
        sub_anc = torch.from_numpy(v_anc_test[sub_idx]).float()
        sub_gt_v = y_vel_test[sub_idx]
        sub_gt_d = y_delta_d_test[sub_idx]

        for s_del in sens_deltas:
            pert_anc = sub_anc + s_del
            _, pert_v, _, _ = model_b(sub_x, pert_anc)
            pert_v_np = pert_v.cpu().numpy()
            pert_d_np = np.sum(pert_v_np * 0.2, axis=1)

            mae_v = float(np.mean(np.abs(pert_v_np - sub_gt_v)))
            bias_d = float(np.mean(pert_d_np - sub_gt_d))

            print(f"{s_del:<+30.2f} | {mae_v:<24.4f} | {bias_d:<+25.4f}")
            sens_results.append({"delta": s_del, "vel_mae": mae_v, "disp_bias": bias_d})
    print("=" * 90)

    # DYNAMIC EVENT PROFILES
    print("\nExtracting representative dynamic driving events for visual inspection...")
    events = {
        "28-30 m/s Cruise": np.where((y_vel_test[:, 10] >= 28.0) & (y_vel_test[:, 10] <= 30.0) & (np.abs(y_delta_v_test[:, 19]) < 0.5))[0],
        ">30 m/s Fast Cruise": np.where((y_vel_test[:, 10] > 31.0) & (np.abs(y_delta_v_test[:, 19]) < 0.8))[0],
        "Braking (17 -> 7 m/s)": np.where((v_anc_test.flatten() > 16.0) & (y_vel_test[:, 19] < 9.0) & (y_ev_test == 2))[0],
        "Turning (~8 m/s)": np.where((v_anc_test.flatten() >= 7.0) & (v_anc_test.flatten() <= 9.5) & (y_ev_test == 3))[0],
        "Rough Road (19-20 m/s)": np.where((v_anc_test.flatten() >= 18.0) & (v_anc_test.flatten() <= 21.0) & (y_ev_test == 4))[0]
    }

    fig, axes = plt.subplots(len(events), 1, figsize=(10, 15))
    t_axis = np.linspace(0.2, 4.0, 20)

    event_summaries = {}
    for idx, (ev_name, cand_indices) in enumerate(events.items()):
        ax = axes[idx]
        if len(cand_indices) > 0:
            chosen_idx = cand_indices[0]
            gt_trace = y_vel_test[chosen_idx]
            va_trace = v_pred_a[chosen_idx]
            vb_trace = v_pred_b[chosen_idx]
            anc_val = float(v_anc_test[chosen_idx][0])

            ax.plot(t_axis, gt_trace, 'k-', linewidth=2.5, label='Ground Truth')
            ax.plot(t_axis, va_trace, 'r--', linewidth=2.0, label='Model A (Phase 10 Absolute)')
            ax.plot(t_axis, vb_trace, 'g-', linewidth=2.2, label='Model B (Phase 11 Anchored)')
            ax.axhline(anc_val, color='blue', linestyle=':', label=f'Anchor ({anc_val:.1f} m/s)')

            ax.set_title(f"{ev_name} (Window Idx: {chosen_idx}, Anchor: {anc_val:.1f} m/s)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Speed (m/s)")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='upper right')

            event_summaries[ev_name] = {
                "idx": chosen_idx,
                "anchor": anc_val,
                "gt_mean": float(np.mean(gt_trace)),
                "pred_a_mean": float(np.mean(va_trace)),
                "pred_b_mean": float(np.mean(vb_trace))
            }
        else:
            ax.text(0.5, 0.5, f"No window matching {ev_name}", ha='center')

    axes[-1].set_xlabel("Time inside 4.0s window (s)")
    plt.tight_layout()
    plot_path = VIZ_DIR / "phase11_temporal_profiles.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved dynamic event profiles to {plot_path}")

    # Latency and capacity profiling
    param_count_b = sum(p.numel() for p in model_b.parameters())
    onnx_path = config.MODELS_DIR / "phase11_anchored_velocity_net.onnx"
    onnx_size_kb = onnx_path.stat().st_size / 1024 if onnx_path.exists() else 0.0

    print("\n" + "=" * 80)
    print("MODEL CAPACITY & PROFILE SUMMARY")
    print("=" * 80)
    print(f"Model B Parameter Count: {param_count_b:,} weights")
    print(f"Model B ONNX File Size: {onnx_size_kb:.1f} KB")
    print(f"Inference Latency (Single Window): {total_infer_time / (3 * N) * 1000:.2f} ms on CPU")
    print("=" * 80)

    return {
        "m_vel_a": m_vel_a, "m_vel_b": m_vel_b, "m_vel_c": m_vel_c,
        "m_disp_a": m_disp_a, "m_disp_b": m_disp_b, "m_disp_c": m_disp_c,
        "m_res_b": m_res_b, "m_res_c": m_res_c,
        "speed_bin_rows": speed_bin_rows,
        "sens_results": sens_results,
        "event_summaries": event_summaries,
        "param_count": param_count_b,
        "onnx_size_kb": onnx_size_kb
    }


if __name__ == "__main__":
    run_benchmark()
