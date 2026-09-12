"""
Phase 8B Pre-Flight: RLS Calibration Forensic Diagnostic
Strict Audit / Diagnostic Only.

Inspects offline oracle scale k = Delta_d_GT / Delta_d_AI
across all held-out test trajectories using frozen Phase 8A 4s model.
"""

import sys
import glob
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
from eval.train_phase8a_4s_model import VelocityNet4s
from eval.windowize_4s_experiment import classify_window_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase8B_Diag")

WINDOW_SIZE_4S = 40  # 4.0s @ 10 Hz
WINDOW_STRIDE_4S = 4  # 0.4s step (90% overlap)

SEQ_CHANNELS = [
    config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z,
    config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z
]

def load_4s_model(device: str = "cpu") -> VelocityNet4s:
    model_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found!")
    model = VelocityNet4s(in_channels=6, num_events=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_file_with_metadata(parquet_path: Path, run_id: str):
    df = pd.read_parquet(parquet_path)
    n_samples = len(df)
    if n_samples < WINDOW_SIZE_4S:
        return None

    for c in SEQ_CHANNELS + [config.COL_TRUE_SPEED_MS]:
        if c not in df.columns:
            return None

    for c in SEQ_CHANNELS:
        df[c] = df[c].interpolate().bfill().ffill().fillna(0.0)

    seq_data = df[SEQ_CHANNELS].values.astype(np.float32)
    gt_speed = df[config.COL_TRUE_SPEED_MS].clip(lower=0.0).values.astype(np.float32)
    gt_delta_d = (gt_speed * config.TARGET_DT).astype(np.float32)
    time_arr = df[config.COL_TIME].values if config.COL_TIME in df.columns else np.arange(n_samples) * 0.1

    yaw_rate = df[config.COL_TRUE_YAW_RATE].values * config.DEG_TO_RAD if config.COL_TRUE_YAW_RATE in df.columns else np.zeros(n_samples)
    acc_long = df[config.COL_TRUE_ACCEL_LONG].values if config.COL_TRUE_ACCEL_LONG in df.columns else np.zeros(n_samples)

    windows_seq = []
    labels_delta_d = []
    labels_speed = []
    labels_event = []
    time_mid = []

    for start_idx in range(0, n_samples - WINDOW_SIZE_4S + 1, WINDOW_STRIDE_4S):
        end_idx = start_idx + WINDOW_SIZE_4S
        win_seq = seq_data[start_idx:end_idx]

        win_delta_d = float(np.sum(gt_delta_d[start_idx:end_idx]))
        win_mean_spd = float(np.mean(gt_speed[start_idx:end_idx]))
        mean_acc_long = float(np.mean(acc_long[start_idx:end_idx]))
        mean_yaw = float(np.mean(yaw_rate[start_idx:end_idx]))
        acc_z_std = float(np.std(win_seq[:, 2]))
        ev_class = classify_window_event(win_mean_spd, mean_acc_long, mean_yaw, acc_z_std)

        windows_seq.append(win_seq)
        labels_delta_d.append(win_delta_d)
        labels_speed.append(win_mean_spd)
        labels_event.append(ev_class)
        time_mid.append(float(time_arr[start_idx + WINDOW_SIZE_4S // 2]))

    if not windows_seq:
        return None

    return {
        "run_id": run_id,
        "X_seq": np.array(windows_seq, dtype=np.float32),
        "y_delta_d": np.array(labels_delta_d, dtype=np.float32),
        "y_speed": np.array(labels_speed, dtype=np.float32),
        "y_event": np.array(labels_event, dtype=np.int64),
        "time_s": np.array(time_mid, dtype=np.float32)
    }

def run_diagnostic():
    print("="*80)
    print("PHASE 8B PRE-FLIGHT: VELOCITYNET SCALE FACTOR & RLS CALIBRATION DIAGNOSTIC")
    print("="*80)

    model = load_4s_model(device="cpu")

    # Identify all test files
    sync_files = sorted(glob.glob(str(config.SYNC_PROCESSED_DIR / "sync_*.parquet")))
    test_files = [
        Path(f) for f in sync_files
        if not any(Path(f).stem.replace("sync_", "").lower().startswith(p.lower())
                   for p in config.TRAIN_DRIVERS + config.VAL_DRIVERS)
    ]

    all_records = []

    print(f"\nProcessing {len(test_files)} held-out test runs...")
    batch_sz = 2048

    for tf in test_files:
        run_name = tf.stem.replace("sync_", "")
        data = process_file_with_metadata(tf, run_name)
        if data is None:
            continue

        X = data["X_seq"]
        y_d = data["y_delta_d"]
        y_spd = data["y_speed"]
        y_ev = data["y_event"]
        t_s = data["time_s"]

        # Run inference in batches
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_sz):
                bx = torch.from_numpy(X[i:i+batch_sz].transpose(0, 2, 1)).float()
                p_d, _, _ = model(bx)
                preds.append(p_d.squeeze().numpy())
        pred_d = np.concatenate(preds)
        pred_spd = pred_d / 4.0

        for j in range(len(pred_d)):
            all_records.append({
                "run": run_name,
                "time_s": t_s[j],
                "event": y_ev[j],
                "d_gt": y_d[j],
                "d_ai": pred_d[j],
                "spd_gt": y_spd[j],
                "spd_ai": pred_spd[j]
            })

    df = pd.DataFrame(all_records)
    print(f"Total window samples extracted across test set: {len(df)}")

    # Add Road Type metadata if known (all VW runs are Driver E West Midlands Motorways campaign)
    # Some runs like vw10, vw11, vw12, vw13 have distinct speed profiles (e.g. urban start vs highway cruise)
    df["road_type"] = np.where(df["spd_gt"] > 15.0, "Motorway/HighSpeed",
                      np.where(df["spd_gt"] > 5.0, "Arterial/Suburban", "Urban/Stationary"))

    # EXCLUSION THRESHOLD FOR k_oracle:
    # When vehicle is stationary or moving at crawl speed (d_ai < 1.0m or d_gt < 1.0m, i.e. < 0.25 m/s),
    # division by d_ai produces unbounded, noisy singularities.
    # We define:
    # Stationary / Crawl: spd_gt < 1.0 m/s OR spd_ai < 0.25 m/s OR d_ai < 1.0 m
    # Moving threshold: spd_gt >= 1.0 m/s AND d_ai >= 1.0 m
    threshold_min_d_ai = 1.0  # meters in 4.0s (0.25 m/s)
    threshold_min_spd_gt = 1.0  # m/s

    valid_mask = (df["d_ai"] >= threshold_min_d_ai) & (df["spd_gt"] >= threshold_min_spd_gt)
    print(f"\nExclusion Threshold Applied:")
    print(f"- d_ai >= {threshold_min_d_ai} m (implied speed >= {threshold_min_d_ai/4.0} m/s)")
    print(f"- spd_gt >= {threshold_min_spd_gt} m/s")
    print(f"- Samples retained: {valid_mask.sum()} / {len(df)} ({valid_mask.sum()/len(df)*100:.1f}%)")
    print(f"- Excluded stationary / low-speed crawl samples: {(~valid_mask).sum()}")

    df_valid = df[valid_mask].copy()
    df_valid["k_oracle"] = df_valid["d_gt"] / df_valid["d_ai"]

    # 1. OVERALL k DISTRIBUTION
    k_arr = df_valid["k_oracle"].values
    p5, p25, p50, p75, p95 = np.percentile(k_arr, [5, 25, 50, 75, 95])
    mean_k = np.mean(k_arr)
    std_k = np.std(k_arr)
    iqr_k = p75 - p25
    cv_k = std_k / mean_k

    print("\n" + "="*80)
    print("1. OVERALL ORACLE SCALE FACTOR (k_oracle) DISTRIBUTION")
    print("="*80)
    print(f"Sample Count (N)        : {len(k_arr)}")
    print(f"Mean k                  : {mean_k:.4f}")
    print(f"Std Dev                 : {std_k:.4f}")
    print(f"Coefficient of Var (CV) : {cv_k:.4f} ({cv_k*100:.2f}%)")
    print(f"Median k (P50)          : {p50:.4f}")
    print(f"P5                      : {p5:.4f}")
    print(f"P25                     : {p25:.4f}")
    print(f"P75                     : {p75:.4f}")
    print(f"P95                     : {p95:.4f}")
    print(f"IQR (P75 - P25)         : {iqr_k:.4f}")

    # 2. k BY SPEED BIN
    print("\n" + "="*80)
    print("2. k_oracle BY SPEED BIN")
    print("="*80)
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
    bin_rows = []
    for b_l, b_h in bins:
        m = (df_valid["spd_gt"] >= b_l) & (df_valid["spd_gt"] < b_h)
        n = m.sum()
        if n > 0:
            sub_k = df_valid.loc[m, "k_oracle"].values
            sub_gt_spd = df_valid.loc[m, "spd_gt"].values
            sub_ai_spd = df_valid.loc[m, "spd_ai"].values
            bin_rows.append({
                "Speed Bin": f"{b_l}-{b_h} m/s",
                "N": int(n),
                "Mean GT Spd": round(float(np.mean(sub_gt_spd)), 2),
                "Mean AI Spd": round(float(np.mean(sub_ai_spd)), 2),
                "Mean k": round(float(np.mean(sub_k)), 4),
                "Median k": round(float(np.median(sub_k)), 4),
                "Std k": round(float(np.std(sub_k)), 4),
                "P25 k": round(float(np.percentile(sub_k, 25)), 4),
                "P75 k": round(float(np.percentile(sub_k, 75)), 4),
                "IQR k": round(float(np.percentile(sub_k, 75) - np.percentile(sub_k, 25)), 4),
            })
    df_bins = pd.DataFrame(bin_rows)
    print(df_bins.to_string(index=False))

    # 3. k BY TRAJECTORY / RUN
    print("\n" + "="*80)
    print("3. k_oracle BY HELD-OUT TEST TRAJECTORY")
    print("="*80)
    run_rows = []
    for r, grp in df_valid.groupby("run"):
        sub_k = grp["k_oracle"].values
        sub_spd = grp["spd_gt"].values
        sub_ai = grp["spd_ai"].values
        run_rows.append({
            "Run": r,
            "N": len(grp),
            "Mean GT Spd (m/s)": round(float(np.mean(sub_spd)), 2),
            "Mean AI Spd (m/s)": round(float(np.mean(sub_ai)), 2),
            "Mean k": round(float(np.mean(sub_k)), 4),
            "Median k": round(float(np.median(sub_k)), 4),
            "Std k": round(float(np.std(sub_k)), 4),
            "P25 k": round(float(np.percentile(sub_k, 25)), 4),
            "P75 k": round(float(np.percentile(sub_k, 75)), 4),
            "IQR k": round(float(np.percentile(sub_k, 75) - np.percentile(sub_k, 25)), 4),
        })
    df_runs = pd.DataFrame(run_rows)
    print(df_runs.to_string(index=False))

    # 4. k BY ROAD TYPE
    print("\n" + "="*80)
    print("4. k_oracle BY ROAD TYPE / REGIME")
    print("="*80)
    road_rows = []
    for rt, grp in df_valid.groupby("road_type"):
        sub_k = grp["k_oracle"].values
        road_rows.append({
            "Road Type": rt,
            "N": len(grp),
            "Mean GT Spd": round(float(grp["spd_gt"].mean()), 2),
            "Mean k": round(float(np.mean(sub_k)), 4),
            "Median k": round(float(np.median(sub_k)), 4),
            "Std k": round(float(np.std(sub_k)), 4),
            "IQR k": round(float(np.percentile(sub_k, 75) - np.percentile(sub_k, 25)), 4)
        })
    df_roads = pd.DataFrame(road_rows)
    print(df_roads.to_string(index=False))

    # 5. FIT AND COMPARE CALIBRATIONS
    print("\n" + "="*80)
    print("5. FITTING CALIBRATION MODELS")
    print("="*80)

    # Method A: Global constant k
    # Option 1: Median k of moving windows
    k_global_median = float(np.median(df_valid["k_oracle"]))
    # Option 2: Least-squares scalar k: argmin sum (k * d_ai - d_gt)^2 -> k = sum(d_ai * d_gt) / sum(d_ai^2)
    k_global_ls = float(np.sum(df_valid["d_ai"] * df_valid["d_gt"]) / np.sum(df_valid["d_ai"]**2))

    print(f"Global Constant k (Median)      : {k_global_median:.4f}")
    print(f"Global Constant k (Least-Squares): {k_global_ls:.4f}")

    # Method B: Speed-dependent linear calibration on AI predicted speed
    # k(v_ai) = a_cal * spd_ai + b_cal
    # Alternatively, direct linear regression: d_cal = alpha * d_ai + beta
    slope_k, intercept_k, r_k, _, _ = stats.linregress(df_valid["spd_ai"], df_valid["k_oracle"])
    print(f"Linear k(v_ai) fit              : k(v_ai) = {slope_k:.4f} * v_ai + {intercept_k:.4f} (r={r_k:.4f})")

    slope_d, intercept_d, r_d, _, _ = stats.linregress(df_valid["d_ai"], df_valid["d_gt"])
    print(f"Direct Linear d_cal(d_ai) fit   : d_cal = {slope_d:.4f} * d_ai + {intercept_d:.4f} (r={r_d:.4f})")

    # Method C: Speed-binned calibration table
    bin_k_map = {}
    for b_l, b_h in bins:
        m = (df_valid["spd_ai"] >= b_l) & (df_valid["spd_ai"] < b_h)
        if m.sum() > 0:
            bin_k_map[(b_l, b_h)] = float(np.median(df_valid.loc[m, "k_oracle"]))
        else:
            bin_k_map[(b_l, b_h)] = k_global_median
    print("Speed-Binned Calibration Map (based on v_ai):")
    for b, kv in bin_k_map.items():
        print(f"  v_ai in [{b[0]}, {b[1]}): k = {kv:.4f}")

    # Helper for binned calibration
    def apply_binned_k(v_ai_arr):
        k_out = np.full_like(v_ai_arr, k_global_median)
        for (bl, bh), kval in bin_k_map.items():
            mask = (v_ai_arr >= bl) & (v_ai_arr < bh)
            k_out[mask] = kval
        return k_out

    # 6. SIMULATED CALIBRATION PERFORMANCE ON TEST SET
    print("\n" + "="*80)
    print("6. THEORETICAL CALIBRATED DISPLACEMENT ERROR COMPARISON")
    print("="*80)

    d_gt = df_valid["d_gt"].values
    d_ai = df_valid["d_ai"].values
    v_ai = df_valid["spd_ai"].values

    # Predictions under various calibration schemes:
    d_uncal = d_ai
    d_cal_global_med = d_ai * k_global_median
    d_cal_global_ls = d_ai * k_global_ls
    d_cal_linear_k = d_ai * np.clip(slope_k * v_ai + intercept_k, 0.5, 3.0)
    d_cal_direct_reg = np.maximum(slope_d * d_ai + intercept_d, 0.0)
    d_cal_binned = d_ai * apply_binned_k(v_ai)

    def evaluate_errors(y_true, y_pred, name):
        diff = y_pred - y_true
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        p_err = np.percentile(np.abs(diff), [50, 75, 95])
        slope, inter, r_val, _, _ = stats.linregress(y_true, y_pred)
        return {
            "Method": name,
            "MAE (m)": round(float(mae), 3),
            "RMSE (m)": round(float(rmse), 3),
            "P50 Err (m)": round(float(p_err[0]), 3),
            "P75 Err (m)": round(float(p_err[1]), 3),
            "P95 Err (m)": round(float(p_err[2]), 3),
            "Corr r": round(float(r_val), 4),
            "Slope a": round(float(slope), 4),
            "Intercept b": round(float(inter), 3),
            "R^2": round(float(r_val**2), 4)
        }

    comp_rows = [
        evaluate_errors(d_gt, d_uncal, "1. Uncalibrated Phase 8A (Raw)"),
        evaluate_errors(d_gt, d_cal_global_med, f"2. Global k (Median = {k_global_median:.3f})"),
        evaluate_errors(d_gt, d_cal_global_ls, f"3. Global k (Least-Squares = {k_global_ls:.3f})"),
        evaluate_errors(d_gt, d_cal_linear_k, "4. Speed-Dependent Linear k(v_ai)"),
        evaluate_errors(d_gt, d_cal_direct_reg, "5. Direct Linear Regression d_cal"),
        evaluate_errors(d_gt, d_cal_binned, "6. Speed-Binned Table k(v_ai)")
    ]
    df_comp = pd.DataFrame(comp_rows)
    print(df_comp.to_string(index=False))

    # Also check high-speed bins (>20 m/s) under each calibration
    print("\n--- HIGH-SPEED PERFORMANCE (GT Speed >= 20.0 m/s, Highway Regime) ---")
    hs_mask = df_valid["spd_gt"] >= 20.0
    print(f"High-Speed Sample Count: {hs_mask.sum()}")
    hs_gt = d_gt[hs_mask]
    hs_rows = [
        evaluate_errors(hs_gt, d_uncal[hs_mask], "Uncalibrated 8A"),
        evaluate_errors(hs_gt, d_cal_global_med[hs_mask], f"Global k (Median={k_global_median:.3f})"),
        evaluate_errors(hs_gt, d_cal_global_ls[hs_mask], f"Global k (LS={k_global_ls:.3f})"),
        evaluate_errors(hs_gt, d_cal_linear_k[hs_mask], "Speed-Dependent k(v)"),
        evaluate_errors(hs_gt, d_cal_direct_reg[hs_mask], "Direct Linear Regression"),
        evaluate_errors(hs_gt, d_cal_binned[hs_mask], "Speed-Binned Table")
    ]
    df_hs = pd.DataFrame(hs_rows)
    print(df_hs.to_string(index=False))

    # 7. TEMPORAL STABILITY & RUN-BY-RUN RLS PLAUSIBILITY
    print("\n" + "="*80)
    print("7. TEMPORAL VARIATION & RLS PLAUSIBILITY AUDIT")
    print("="*80)
    # Check if k varies smoothly or wildly within individual runs
    run_cvs = []
    for r, grp in df_valid.groupby("run"):
        if len(grp) > 100:
            k_vals = grp["k_oracle"].values
            med_r = np.median(k_vals)
            iqr_r = np.percentile(k_vals, 75) - np.percentile(k_vals, 25)
            cv_r = np.std(k_vals) / np.mean(k_vals)
            run_cvs.append({"Run": r, "Median k": med_r, "IQR": iqr_r, "CV": cv_r})
    df_run_cvs = pd.DataFrame(run_cvs)
    print("Within-run dispersion:")
    print(df_run_cvs.to_string(index=False))
    print(f"Mean within-run CV : {df_run_cvs['CV'].mean():.4f}")
    print(f"Mean within-run IQR: {df_run_cvs['IQR'].mean():.4f}")

if __name__ == "__main__":
    run_diagnostic()
