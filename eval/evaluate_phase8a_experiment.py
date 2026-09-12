"""
Phase 8A-1: Model-Level and Navigation Evaluation Script (ASCII Safe)
Compares:
- Ground Truth
- Phase 7D Clean 2-Second Model (phase7d_velocity_net_clean.pt)
- Phase 8A-1 4-Second Context Model (phase8a_4s_velocity_net.pt)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from modules.velocity_net import VelocityNet, VelocityNetPredictor
from eval.train_phase8a_4s_model import VelocityNet4s
from eval.replay import evaluate_run_outages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase8A_Eval")


class VelocityNetPredictor4s:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = device
        self.model = VelocityNet4s(in_channels=6, num_events=5).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, window_imu: np.ndarray) -> Tuple[float, int, float]:
        if len(window_imu) < 40:
            pad_len = 40 - len(window_imu)
            window_imu = np.pad(window_imu, ((pad_len, 0), (0, 0)), mode="edge")
        elif len(window_imu) > 40:
            window_imu = window_imu[-40:]

        with torch.no_grad():
            tensor_x = torch.from_numpy(np.ascontiguousarray(window_imu.T)).unsqueeze(0).float().to(self.device)
            d_d, ev_logits, sig = self.model(tensor_x)
            pred_d = float(d_d.cpu().item())
            pred_ev = int(torch.argmax(ev_logits, dim=1).cpu().item())
            pred_sig = float(sig.cpu().item())
            return pred_d, pred_ev, pred_sig


def run_model_diagnostics_4s():
    print("="*80)
    print("PHASE 8A-1: MODEL-LEVEL EVALUATION (2.0s CONTEXT vs 4.0s CONTEXT)")
    print("="*80)

    p2s_path = config.MODELS_DIR / "phase7d_velocity_net_clean.pt"
    p4s_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"

    if not p4s_path.exists():
        print(f"Error: {p4s_path} does not exist yet!")
        return

    m2s = VelocityNet(in_channels=6, num_events=5)
    m2s.load_state_dict(torch.load(p2s_path, map_location="cpu"))
    m2s.eval()

    m4s = VelocityNet4s(in_channels=6, num_events=5)
    m4s.load_state_dict(torch.load(p4s_path, map_location="cpu"))
    m4s.eval()

    test_2s = np.load(config.WINDOWED_DIR / "test_windows.npz")
    test_4s = np.load(config.WINDOWED_DIR / "test_windows_4s.npz")

    # Evaluate 2s model on 2s test windows
    X_2s = test_2s["X_seq"]
    y_d_2s = test_2s["y_delta_d"]
    y_spd_2s = test_2s["y_speed"]

    batch_sz = 2048
    preds_2s = []
    with torch.no_grad():
        for i in range(0, len(X_2s), batch_sz):
            bx = torch.from_numpy(X_2s[i:i+batch_sz].transpose(0, 2, 1)).float()
            p_d, _, _ = m2s(bx)
            preds_2s.append(p_d.squeeze().numpy())
    pred_d_2s = np.concatenate(preds_2s)
    pred_spd_2s = pred_d_2s / 2.0

    # Evaluate 4s model on 4s test windows
    X_4s = test_4s["X_seq"]
    y_d_4s = test_4s["y_delta_d"]
    y_spd_4s = test_4s["y_speed"]

    preds_4s = []
    with torch.no_grad():
        for i in range(0, len(X_4s), batch_sz):
            bx = torch.from_numpy(X_4s[i:i+batch_sz].transpose(0, 2, 1)).float()
            p_d, _, _ = m4s(bx)
            preds_4s.append(p_d.squeeze().numpy())
    pred_d_4s = np.concatenate(preds_4s)
    pred_spd_4s = pred_d_4s / 4.0

    def dist_summary(arr, name):
        return {
            "Metric": name,
            "Min": round(float(np.min(arr)), 2),
            "P5": round(float(np.percentile(arr, 5)), 2),
            "P25": round(float(np.percentile(arr, 25)), 2),
            "Median": round(float(np.percentile(arr, 50)), 2),
            "Mean": round(float(np.mean(arr)), 2),
            "P75": round(float(np.percentile(arr, 75)), 2),
            "P95": round(float(np.percentile(arr, 95)), 2),
            "Max": round(float(np.max(arr)), 2)
        }

    print("\n--- DISTRIBUTION COMPARISON (DISPLACEMENT & IMPLIED SPEED) ---")
    df_dists = pd.DataFrame([
        dist_summary(y_d_2s, "2s GT Delta d (m)"),
        dist_summary(pred_d_2s, "2s Model AI Delta d (m)"),
        dist_summary(y_d_4s, "4s GT Delta d (m)"),
        dist_summary(pred_d_4s, "4s Model AI Delta d (m)"),
        dist_summary(y_spd_4s, "GT Speed (m/s)"),
        dist_summary(pred_spd_2s, "2s Model Implied Speed (m/s)"),
        dist_summary(pred_spd_4s, "4s Model Implied Speed (m/s)")
    ])
    print(df_dists.to_string(index=False))

    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
    bin_rows = []

    for b_l, b_h in bins:
        m2 = (y_spd_2s >= b_l) & (y_spd_2s < b_h)
        m4 = (y_spd_4s >= b_l) & (y_spd_4s < b_h)

        if np.sum(m4) > 0 and np.sum(m2) > 0:
            gt_spd_2 = y_spd_2s[m2]
            ai_spd_2 = pred_spd_2s[m2]
            ratio_2s_mean = float(np.mean(pred_d_2s[m2])) / float(np.mean(y_d_2s[m2]))
            ratio_2s_med = float(np.median(pred_d_2s[m2] / np.maximum(y_d_2s[m2], 1e-4)))
            mae_2s = float(np.mean(np.abs(pred_spd_2s[m2] - gt_spd_2)))
            rmse_2s = float(np.sqrt(np.mean((pred_spd_2s[m2] - gt_spd_2)**2)))

            gt_spd_4 = y_spd_4s[m4]
            ai_spd_4 = pred_spd_4s[m4]
            ratio_4s_mean = float(np.mean(pred_d_4s[m4])) / float(np.mean(y_d_4s[m4]))
            ratio_4s_med = float(np.median(pred_d_4s[m4] / np.maximum(y_d_4s[m4], 1e-4)))
            mae_4s = float(np.mean(np.abs(pred_spd_4s[m4] - gt_spd_4)))
            rmse_4s = float(np.sqrt(np.mean((pred_spd_4s[m4] - gt_spd_4)**2)))

            bin_rows.append({
                "Speed Bin": f"{b_l}-{b_h} m/s",
                "N (4s)": int(np.sum(m4)),
                "GT Spd": round(float(np.mean(gt_spd_4)), 2),
                "2s AI Spd": round(float(np.mean(ai_spd_2)), 2),
                "2s Ratio": round(ratio_2s_med, 4),
                "2s MAE": round(mae_2s, 2),
                "4s AI Spd": round(float(np.mean(ai_spd_4)), 2),
                "4s Ratio": round(ratio_4s_med, 4),
                "4s MAE": round(mae_4s, 2)
            })

    print("\n--- SPEED-BIN A/B COMPARISON (2s vs 4s CONTEXT) ---")
    df_bins = pd.DataFrame(bin_rows)
    print(df_bins.to_string(index=False))

    mov_4 = y_spd_4s > 1.0
    s_4, i_4, r_4, _, _ = stats.linregress(y_d_4s[mov_4], pred_d_4s[mov_4])
    s_spd_4, i_spd_4, r_spd_4, _, _ = stats.linregress(y_spd_4s[mov_4], pred_spd_4s[mov_4])

    mov_2 = y_spd_2s > 1.0
    s_2, i_2, r_2, _, _ = stats.linregress(y_d_2s[mov_2], pred_d_2s[mov_2])
    s_spd_2, i_spd_2, r_spd_2, _, _ = stats.linregress(y_spd_2s[mov_2], pred_spd_2s[mov_2])

    print("\n--- REGRESSION ANALYSIS ---")
    print(f"2s Model (dd): a={s_2:.4f}, b={i_2:.2f}m, R^2={r_2**2:.4f}, MAE={np.mean(np.abs(pred_d_2s[mov_2]-y_d_2s[mov_2])):.2f}m")
    print(f"4s Model (dd): a={s_4:.4f}, b={i_4:.2f}m, R^2={r_4**2:.4f}, MAE={np.mean(np.abs(pred_d_4s[mov_4]-y_d_4s[mov_4])):.2f}m")
    print(f"2s Model (Speed): a={s_spd_2:.4f}, b={i_spd_2:.2f}m/s, R^2={r_spd_2**2:.4f}, Correlation r={r_spd_2:+.4f}")
    print(f"4s Model (Speed): a={s_spd_4:.4f}, b={i_spd_4:.2f}m/s, R^2={r_spd_4**2:.4f}, Correlation r={r_spd_4:+.4f}")


def run_navigation_ab_test_4s():
    print("\n" + "="*80)
    print("PHASE 8A-1: NAVIGATION A/B BENCHMARK (2.0s vs 4.0s CONTEXT)")
    print("="*80)

    p2s_path = config.MODELS_DIR / "phase7d_velocity_net_clean.pt"
    p4s_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"

    pred_2s = VelocityNetPredictor(model_path=str(p2s_path))
    pred_4s = VelocityNetPredictor4s(model_path=p4s_path)

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    ab_results = []
    for tf in test_files:
        scores_2s = evaluate_run_outages(tf, method="inav_ukf", predictor=pred_2s)
        scores_4s = evaluate_run_outages(tf, method="inav_ukf", predictor=pred_4s)

        for s2, s4 in zip(scores_2s, scores_4s):
            dur = s2.get("outage_s", 0)
            dist_gt = s2.get("distance_m", 0)

            fpe_2s = s2.get("final_pos_error_m", 0)
            drift_2s = s2.get("pct_of_distance", 0)
            hdg_2s = s2.get("heading_error_deg", 0)

            fpe_4s = s4.get("final_pos_error_m", 0)
            drift_4s = s4.get("pct_of_distance", 0)
            hdg_4s = s4.get("heading_error_deg", 0)

            ab_results.append({
                "run": tf.stem.replace("sync_", ""),
                "duration_s": dur,
                "dist_gt_m": dist_gt,
                "fpe_2s_m": fpe_2s,
                "fpe_4s_m": fpe_4s,
                "drift_2s_pct": drift_2s,
                "drift_4s_pct": drift_4s,
                "hdg_2s_deg": hdg_2s,
                "hdg_4s_deg": hdg_4s,
            })

    df_ab = pd.DataFrame(ab_results)
    print("\n--- OUTAGE BY OUTAGE A/B COMPARISON (2s vs 4s) ---")
    print(df_ab.to_string(index=False))

    print("\n--- SUMMARY BY OUTAGE DURATION ---")
    print(df_ab.groupby("duration_s")[["fpe_2s_m", "fpe_4s_m", "drift_2s_pct", "drift_4s_pct", "hdg_2s_deg", "hdg_4s_deg"]].median().to_string())

    print("\n--- OVERALL MEDIANS ---")
    print(f"Overall Median Drift 2s : {df_ab['drift_2s_pct'].median():.2f}%")
    print(f"Overall Median Drift 4s : {df_ab['drift_4s_pct'].median():.2f}%")
    print(f"Overall Median FPE 2s   : {df_ab['fpe_2s_m'].median():.2f} m")
    print(f"Overall Median FPE 4s   : {df_ab['fpe_4s_m'].median():.2f} m")


if __name__ == "__main__":
    run_model_diagnostics_4s()
    run_navigation_ab_test_4s()
