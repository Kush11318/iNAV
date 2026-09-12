"""
Phase 7D: Evaluation and Navigation A/B Comparison Script
Compares OLD VelocityNet (velocity_net_best.pt) vs NEW Clean VelocityNet (phase7d_velocity_net_clean.pt)
Across:
1. Model Diagnostics (GT vs Old AI vs New AI Distributions)
2. Speed-Bin Analysis (0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+)
3. Offline Linear Regressions
4. Complete Navigation A/B Benchmark across Outage Scenarios
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from modules.velocity_net import VelocityNet, VelocityNetPredictor
from eval.replay import evaluate_run_outages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase7D_Eval")


def run_model_diagnostics():
    print("="*80)
    print("PHASE 7D: MODEL-LEVEL DIAGNOSTICS (OLD vs NEW)")
    print("="*80)
    
    old_pt_path = config.MODELS_DIR / "velocity_net_best.pt"
    new_pt_path = config.MODELS_DIR / "phase7d_velocity_net_clean.pt"
    
    if not new_pt_path.exists():
        print(f"Error: {new_pt_path} does not exist yet!")
        return
        
    old_model = VelocityNet(in_channels=6, num_events=5)
    old_model.load_state_dict(torch.load(old_pt_path, map_location="cpu"))
    old_model.eval()
    
    new_model = VelocityNet(in_channels=6, num_events=5)
    new_model.load_state_dict(torch.load(new_pt_path, map_location="cpu"))
    new_model.eval()
    
    train_npz = np.load(config.WINDOWED_DIR / "train_windows.npz")
    val_npz = np.load(config.WINDOWED_DIR / "val_windows.npz")
    test_npz = np.load(config.WINDOWED_DIR / "test_windows.npz")
    
    splits = {"Train": train_npz, "Val": val_npz, "Test": test_npz}
    
    def get_dist_row(arr, name):
        return {
            "name": name,
            "min": float(np.min(arr)),
            "P5": float(np.percentile(arr, 5)),
            "P25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "mean": float(np.mean(arr)),
            "P75": float(np.percentile(arr, 75)),
            "P95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr))
        }
    
    # 1. Distribution summary
    for s_name, s_data in splits.items():
        X = s_data["X_seq"]
        y_d = s_data["y_delta_d"]
        
        # Inference
        batch_sz = 2048
        old_preds = []
        new_preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_sz):
                bx = torch.from_numpy(X[i:i+batch_sz].transpose(0, 2, 1)).float()
                p_old, _, _ = old_model(bx)
                p_new, _, _ = new_model(bx)
                old_preds.append(p_old.squeeze().numpy())
                new_preds.append(p_new.squeeze().numpy())
        old_p = np.concatenate(old_preds)
        new_p = np.concatenate(new_preds)
        
        print(f"\n--- SPLIT: {s_name} (N={len(y_d)}) ---")
        df_dists = pd.DataFrame([
            get_dist_row(y_d, f"{s_name} GT dd"),
            get_dist_row(old_p, f"{s_name} OLD AI dd"),
            get_dist_row(new_p, f"{s_name} NEW AI dd")
        ])
        print(df_dists.to_string(index=False))
    
    # 2. Speed-bin analysis on Test Set
    X_test = test_npz["X_seq"]
    y_d_test = test_npz["y_delta_d"]
    y_spd_test = test_npz["y_speed"]
    
    batch_sz = 2048
    old_preds_test = []
    new_preds_test = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_sz):
            bx = torch.from_numpy(X_test[i:i+batch_sz].transpose(0, 2, 1)).float()
            p_old, _, _ = old_model(bx)
            p_new, _, _ = new_model(bx)
            old_preds_test.append(p_old.squeeze().numpy())
            new_preds_test.append(p_new.squeeze().numpy())
    old_test = np.concatenate(old_preds_test)
    new_test = np.concatenate(new_preds_test)
    
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
    bin_rows = []
    for b_l, b_h in bins:
        m = (y_spd_test >= b_l) & (y_spd_test < b_h)
        if np.sum(m) > 0:
            gt_b = y_d_test[m]
            old_b = old_test[m]
            new_b = new_test[m]
            
            bin_rows.append({
                "bin": f"{b_l}-{b_h} m/s",
                "N": int(np.sum(m)),
                "mean_gt_spd": float(np.mean(y_spd_test[m])),
                "old_mean_ai_spd": float(np.mean(old_b) / 2.0),
                "old_ratio": float(np.median(old_b / np.maximum(gt_b, 1e-3))),
                "old_mae": float(np.mean(np.abs(old_b - gt_b))),
                "new_mean_ai_spd": float(np.mean(new_b) / 2.0),
                "new_ratio": float(np.median(new_b / np.maximum(gt_b, 1e-3))),
                "new_mae": float(np.mean(np.abs(new_b - gt_b)))
            })
    
    print("\n--- TEST SET SPEED-BIN A/B COMPARISON ---")
    df_bins = pd.DataFrame(bin_rows)
    print(df_bins.to_string(index=False))
    
    # 3. Linear Regressions
    mov_m = y_d_test > 2.0
    gt_mov = y_d_test[mov_m]
    old_mov = old_test[mov_m]
    new_mov = new_test[mov_m]
    
    # Old regression
    s_old, i_old, r_old, _, _ = stats.linregress(gt_mov, old_mov)
    # New regression
    s_new, i_new, r_new, _, _ = stats.linregress(gt_mov, new_mov)
    
    print("\n--- REGRESSION ANALYSIS: AI_dd = a * GT_dd + b ---")
    print(f"OLD MODEL: a={s_old:.4f}, b={i_old:.4f}m, R^2={r_old**2:.4f}, MAE={np.mean(np.abs(old_mov - gt_mov)):.3f}m, RMSE={np.sqrt(np.mean((old_mov - gt_mov)**2)):.3f}m, Median Signed={np.median(old_mov - gt_mov):.3f}m")
    print(f"NEW MODEL: a={s_new:.4f}, b={i_new:.4f}m, R^2={r_new**2:.4f}, MAE={np.mean(np.abs(new_mov - gt_mov)):.3f}m, RMSE={np.sqrt(np.mean((new_mov - gt_mov)**2)):.3f}m, Median Signed={np.median(new_mov - gt_mov):.3f}m")


def run_navigation_ab_test():
    print("\n" + "="*80)
    print("PHASE 7D: NAVIGATION A/B BENCHMARK (OUTAGE REPLAY)")
    print("="*80)
    
    old_pt_path = config.MODELS_DIR / "velocity_net_best.pt"
    new_pt_path = config.MODELS_DIR / "phase7d_velocity_net_clean.pt"
    
    old_predictor = VelocityNetPredictor(model_path=str(old_pt_path))
    new_predictor = VelocityNetPredictor(model_path=str(new_pt_path))
    
    # Evaluate across sample and held-out test parquet files
    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    
    # Also add standard test sync runs if present
    for r in ["vw10", "vw11", "vw12", "vw13"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)
            
    ab_results = []
    
    for tf in test_files:
        print(f"\nEvaluating outages on: {tf.name}...")
        scores_old = evaluate_run_outages(tf, method="inav_ukf", predictor=old_predictor)
        scores_new = evaluate_run_outages(tf, method="inav_ukf", predictor=new_predictor)
        
        for so, sn in zip(scores_old, scores_new):
            dur = so.get("outage_s", 0)
            dist_gt = so.get("distance_m", 0)
            
            fpe_old = so.get("final_pos_error_m", 0)
            drift_old = so.get("pct_of_distance", 0)
            hdg_old = so.get("heading_error_deg", 0)
            
            fpe_new = sn.get("final_pos_error_m", 0)
            drift_new = sn.get("pct_of_distance", 0)
            hdg_new = sn.get("heading_error_deg", 0)
            
            ab_results.append({
                "run": tf.stem.replace("sync_", ""),
                "duration_s": dur,
                "dist_gt_m": dist_gt,
                "fpe_old_m": fpe_old,
                "fpe_new_m": fpe_new,
                "drift_old_pct": drift_old,
                "drift_new_pct": drift_new,
                "hdg_old_deg": hdg_old,
                "hdg_new_deg": hdg_new,
            })
            
    df_ab = pd.DataFrame(ab_results)
    print("\n--- OUTAGE BY OUTAGE A/B COMPARISON ---")
    print(df_ab.to_string(index=False))
    
    print("\n--- SUMMARY BY OUTAGE DURATION ---")
    print(df_ab.groupby("duration_s")[["fpe_old_m", "fpe_new_m", "drift_old_pct", "drift_new_pct", "hdg_old_deg", "hdg_new_deg"]].median().to_string())
    
    print("\n--- OVERALL MEDIANS ---")
    print(f"Overall Median Drift OLD: {df_ab['drift_old_pct'].median():.2f}%")
    print(f"Overall Median Drift NEW: {df_ab['drift_new_pct'].median():.2f}%")
    print(f"Overall Median FPE OLD  : {df_ab['fpe_old_m'].median():.2f} m")
    print(f"Overall Median FPE NEW  : {df_ab['fpe_new_m'].median():.2f} m")


if __name__ == "__main__":
    run_model_diagnostics()
    run_navigation_ab_test()
