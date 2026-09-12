"""
Phase 8A-1: Dataset Windowizer for 4.0-Second Context Experiment
Extracts 4.0-second sliding windows (40 samples @ 10 Hz, 0.4s stride = 4 samples)
from synchronized Parquet files without altering existing 2.0-second datasets.

Saves strictly to:
- data/processed/windowized/train_windows_4s.npz
- data/processed/windowized/val_windows_4s.npz
- data/processed/windowized/test_windows_4s.npz
"""

import sys
import glob
import logging
from pathlib import Path
from typing import Dict, Optional
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
logger = logging.getLogger("iNAV.Phase8A_Windowize")

WINDOW_SIZE_4S = 40  # 4.0s @ 10 Hz
WINDOW_STRIDE_4S = 4  # 0.4s step (90% overlap, manageable dataset size)

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


def process_synchronized_file_4s(parquet_path: Path) -> Optional[Dict[str, np.ndarray]]:
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
    labels_delta_d = []
    labels_event = []
    labels_speed = []

    for start_idx in range(0, n_samples - WINDOW_SIZE_4S + 1, WINDOW_STRIDE_4S):
        end_idx = start_idx + WINDOW_SIZE_4S
        win_seq = seq_data[start_idx:end_idx]

        # 4.0s integrated ground truth displacement: sum of 40 samples * 0.1s
        win_delta_d = float(np.sum(gt_delta_d[start_idx:end_idx]))
        win_mean_spd = float(np.mean(gt_speed[start_idx:end_idx]))

        mean_acc_long = float(np.mean(acc_long[start_idx:end_idx]))
        mean_yaw = float(np.mean(yaw_rate[start_idx:end_idx]))
        acc_z_std = float(np.std(win_seq[:, 2]))
        ev_class = classify_window_event(win_mean_spd, mean_acc_long, mean_yaw, acc_z_std)

        windows_seq.append(win_seq)
        labels_delta_d.append(win_delta_d)
        labels_event.append(ev_class)
        labels_speed.append(win_mean_spd)

    if not windows_seq:
        return None

    return {
        "X_seq": np.array(windows_seq, dtype=np.float32),
        "y_delta_d": np.array(labels_delta_d, dtype=np.float32),
        "y_event": np.array(labels_event, dtype=np.int64),
        "y_speed": np.array(labels_speed, dtype=np.float32)
    }


def build_4s_datasets() -> None:
    sync_files = sorted(glob.glob(str(config.SYNC_PROCESSED_DIR / "sync_*.parquet")))
    logger.info(f"Building 4-second windowized datasets from {len(sync_files)} synchronized files...")

    splits = {"train": [], "val": [], "test": []}
    for f in sync_files:
        run_key = Path(f).stem.replace("sync_", "").lower()
        if any(run_key.startswith(p.lower()) for p in config.TRAIN_DRIVERS):
            splits["train"].append(Path(f))
        elif any(run_key.startswith(p.lower()) for p in config.VAL_DRIVERS):
            splits["val"].append(Path(f))
        else:
            splits["test"].append(Path(f))

    logger.info(f"Split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    for split_name, files in splits.items():
        all_seq = []
        all_delta_d = []
        all_event = []
        all_speed = []

        logger.info(f"Processing split '{split_name}' with {len(files)} files...")
        for p in files:
            res = process_synchronized_file_4s(p)
            if res is not None:
                all_seq.append(res["X_seq"])
                all_delta_d.append(res["y_delta_d"])
                all_event.append(res["y_event"])
                all_speed.append(res["y_speed"])

        if all_seq:
            cat_seq = np.concatenate(all_seq, axis=0)
            cat_delta_d = np.concatenate(all_delta_d, axis=0)
            cat_event = np.concatenate(all_event, axis=0)
            cat_speed = np.concatenate(all_speed, axis=0)

            out_npz = config.WINDOWED_DIR / f"{split_name}_windows_4s.npz"
            np.savez_compressed(
                out_npz,
                X_seq=cat_seq,
                y_delta_d=cat_delta_d,
                y_event=cat_event,
                y_speed=cat_speed
            )
            logger.info(f"Saved {split_name}_windows_4s.npz: {len(cat_seq)} windows, shape {cat_seq.shape}")


if __name__ == "__main__":
    build_4s_datasets()
