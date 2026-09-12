"""
Phase 11: Dataset Windowizer for GNSS-Anchored Forward-Velocity Residual Experiment
Extracts 4.0-second sliding windows (40 samples @ 10 Hz, 0.4s stride = 4 samples)
from synchronized Parquet files.

Outputs for each window:
  - X_seq: (40, 6) raw IMU sequence [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  - v_anchor: (1,) velocity anchor = speed immediately before the window (m/s)
  - y_vel_20: (20,) ground-truth forward speed sequence @ 5 Hz (m/s)
  - y_delta_v_20: (20,) ground-truth velocity residual Delta v(t) = v_GT(t) - v_anchor (m/s)
  - y_delta_d: (1,) integrated 4.0-second ground-truth displacement (m)
  - y_speed_mean: (1,) mean ground-truth speed across window (m/s)
  - y_event: (1,) driving condition class

Saves strictly to:
  - data/processed/windowized/train_windows_phase11.npz
  - data/processed/windowized/val_windows_phase11.npz
  - data/processed/windowized/test_windows_phase11.npz
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase11_Windowize")

WINDOW_SIZE_4S = 40   # 4.0s @ 10 Hz
WINDOW_STRIDE_4S = 4  # 0.4s step

SEQ_CHANNELS = [
    config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z,
    config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z
]


def classify_window_event(
    mean_spd: float,
    mean_acc_long: float,
    mean_yaw_rate: float,
    acc_z_std: float
) -> int:
    if mean_spd < 0.15:
        return 0  # Stationary
    if mean_acc_long < -2.2:
        return 2  # Hard Braking
    if abs(mean_yaw_rate) >= 0.15:
        return 3  # Turning
    if acc_z_std > 1.8:
        return 4  # Rough road
    return 1  # Cruise


def process_synchronized_file(parquet_path: Path) -> Optional[Dict[str, np.ndarray]]:
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
    gt_delta_d = (gt_speed * config.TARGET_DT).astype(np.float32)  # 0.1s step

    yaw_rate = df[config.COL_TRUE_YAW_RATE].values * config.DEG_TO_RAD if config.COL_TRUE_YAW_RATE in df.columns else np.zeros(n_samples)
    acc_long = df[config.COL_TRUE_ACCEL_LONG].values if config.COL_TRUE_ACCEL_LONG in df.columns else np.zeros(n_samples)

    windows_seq = []
    anchors = []
    labels_vel_20 = []
    labels_delta_v_20 = []
    labels_delta_d = []
    labels_speed_mean = []
    labels_event = []

    for start_idx in range(0, n_samples - WINDOW_SIZE_4S + 1, WINDOW_STRIDE_4S):
        end_idx = start_idx + WINDOW_SIZE_4S
        win_seq = seq_data[start_idx:end_idx]

        # Velocity anchor: last observable speed immediately before window onset
        v_anchor = float(gt_speed[start_idx - 1]) if start_idx > 0 else float(gt_speed[0])

        # 40-sample speed sequence
        win_speed_40 = gt_speed[start_idx:end_idx]

        # 20-sample velocity sequence @ 5 Hz: average pairs of consecutive 10 Hz samples
        win_vel_20 = win_speed_40.reshape(20, 2).mean(axis=1)

        # Residual target Delta v(t) = v_GT(t) - v_anchor
        win_delta_v_20 = win_vel_20 - v_anchor

        # 4.0s integrated displacement: sum(speed * 0.1s)
        win_delta_d = float(np.sum(gt_delta_d[start_idx:end_idx]))
        win_mean_spd = float(np.mean(win_speed_40))

        mean_acc_long = float(np.mean(acc_long[start_idx:end_idx]))
        mean_yaw = float(np.mean(yaw_rate[start_idx:end_idx]))
        acc_z_std = float(np.std(win_seq[:, 2]))
        ev_class = classify_window_event(win_mean_spd, mean_acc_long, mean_yaw, acc_z_std)

        windows_seq.append(win_seq)
        anchors.append(v_anchor)
        labels_vel_20.append(win_vel_20)
        labels_delta_v_20.append(win_delta_v_20)
        labels_delta_d.append(win_delta_d)
        labels_speed_mean.append(win_mean_spd)
        labels_event.append(ev_class)

    if not windows_seq:
        return None

    return {
        "X_seq": np.array(windows_seq, dtype=np.float32),
        "v_anchor": np.array(anchors, dtype=np.float32).reshape(-1, 1),
        "y_vel_20": np.array(labels_vel_20, dtype=np.float32),
        "y_delta_v_20": np.array(labels_delta_v_20, dtype=np.float32),
        "y_delta_d": np.array(labels_delta_d, dtype=np.float32),
        "y_speed_mean": np.array(labels_speed_mean, dtype=np.float32),
        "y_event": np.array(labels_event, dtype=np.int64),
    }


def partition_campaign(run_name: str) -> str:
    name_lower = run_name.replace("sync_", "").split("_")[0].lower()
    for c in config.TRAIN_CAMPAIGNS:
        if name_lower.startswith(c):
            return "train"
    for c in config.VAL_CAMPAIGNS:
        if name_lower.startswith(c):
            return "val"
    for c in config.TEST_CAMPAIGNS:
        if name_lower.startswith(c):
            return "test"
    return "unknown"


def main():
    logger.info("=" * 80)
    logger.info("PHASE 11: 4.0s GNSS-ANCHORED VELOCITY RESIDUAL WINDOWIZATION")
    logger.info("=" * 80)

    all_files = sorted(config.SYNC_PROCESSED_DIR.glob("sync_*.parquet"))
    logger.info(f"Found {len(all_files)} synchronized Parquet files.")

    split_data = {
        "train": {"X_seq": [], "v_anchor": [], "y_vel_20": [], "y_delta_v_20": [], "y_delta_d": [], "y_speed_mean": [], "y_event": []},
        "val": {"X_seq": [], "v_anchor": [], "y_vel_20": [], "y_delta_v_20": [], "y_delta_d": [], "y_speed_mean": [], "y_event": []},
        "test": {"X_seq": [], "v_anchor": [], "y_vel_20": [], "y_delta_v_20": [], "y_delta_d": [], "y_speed_mean": [], "y_event": []}
    }

    for p in all_files:
        run_name = p.stem.replace("sync_", "")
        split = partition_campaign(run_name)
        if split not in split_data:
            continue

        res = process_synchronized_file(p)
        if res is not None:
            split_data[split]["X_seq"].append(res["X_seq"])
            split_data[split]["v_anchor"].append(res["v_anchor"])
            split_data[split]["y_vel_20"].append(res["y_vel_20"])
            split_data[split]["y_delta_v_20"].append(res["y_delta_v_20"])
            split_data[split]["y_delta_d"].append(res["y_delta_d"])
            split_data[split]["y_speed_mean"].append(res["y_speed_mean"])
            split_data[split]["y_event"].append(res["y_event"])

    for split_name in ["train", "val", "test"]:
        if not split_data[split_name]["X_seq"]:
            logger.warning(f"No data for split: {split_name}")
            continue

        cat_X_seq = np.concatenate(split_data[split_name]["X_seq"], axis=0)
        cat_v_anchor = np.concatenate(split_data[split_name]["v_anchor"], axis=0)
        cat_vel_20 = np.concatenate(split_data[split_name]["y_vel_20"], axis=0)
        cat_delta_v_20 = np.concatenate(split_data[split_name]["y_delta_v_20"], axis=0)
        cat_delta_d = np.concatenate(split_data[split_name]["y_delta_d"], axis=0)
        cat_spd = np.concatenate(split_data[split_name]["y_speed_mean"], axis=0)
        cat_ev = np.concatenate(split_data[split_name]["y_event"], axis=0)

        out_path = config.WINDOWED_DIR / f"{split_name}_windows_phase11.npz"
        np.savez_compressed(
            out_path,
            X_seq=cat_X_seq,
            v_anchor=cat_v_anchor,
            y_vel_20=cat_vel_20,
            y_delta_v_20=cat_delta_v_20,
            y_delta_d=cat_delta_d,
            y_speed_mean=cat_spd,
            y_event=cat_ev
        )
        logger.info(
            f"Saved {split_name.upper()} dataset to {out_path}: "
            f"X_seq={cat_X_seq.shape}, v_anchor={cat_v_anchor.shape}, "
            f"y_delta_v_20={cat_delta_v_20.shape}, mean_residual={cat_delta_v_20.mean():.4f} m/s, "
            f"residual_std={cat_delta_v_20.std():.4f} m/s"
        )


if __name__ == "__main__":
    main()
