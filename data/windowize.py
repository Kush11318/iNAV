"""
iNAV Windowing and Feature Engineering Pipeline
Extracts 2-second sliding windows (20 samples @ 10 Hz, 0.2s stride) with IMU sequences,
statistical/spectral features, and displacement/event labels, partitioned into train,
validation, and test sets.
"""

import sys
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.windowize")

# Primary IMU input sequence channels
SEQ_CHANNELS = [
    config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z,
    config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z
]

# Optional auxiliary channels (gravity, magnetic field)
AUX_CHANNELS = [
    config.COL_GRAV_X, config.COL_GRAV_Y, config.COL_GRAV_Z,
    config.COL_MAG_X, config.COL_MAG_Y, config.COL_MAG_Z
]

ALL_INPUT_CHANNELS = SEQ_CHANNELS + AUX_CHANNELS


def compute_spectral_features(x: np.ndarray, fs: float = 10.0) -> Tuple[float, float]:
    """
    Compute low-band (0-2.5 Hz) and high-band (2.5-5.0 Hz) spectral energy using FFT.
    """
    n = len(x)
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    low_mask = (freqs >= 0.0) & (freqs < 2.5)
    high_mask = (freqs >= 2.5) & (freqs <= 5.0)

    low_power = float(np.sum(fft_vals[low_mask])) if np.any(low_mask) else 0.0
    high_power = float(np.sum(fft_vals[high_mask])) if np.any(high_mask) else 0.0

    return low_power, high_power


def classify_window_event(
    mean_spd: float,
    mean_acc_long: float,
    mean_yaw_rate: float,
    acc_z_std: float
) -> int:
    """
    Categorize window into driving event class:
    0: Stationary / Idle (< 0.15 m/s)
    1: Normal Cruise (constant speed / gentle acceleration)
    2: Hard Braking (decel < -2.2 m/s^2)
    3: Turning (|yaw rate| >= 0.15 rad/s or ~8.6 deg/s)
    4: Rough Road / Bump / Pothole (high vertical acceleration variance)
    """
    if mean_spd < 0.15:
        return 0  # Stationary
    if mean_acc_long < -2.2:
        return 2  # Hard Braking
    if abs(mean_yaw_rate) >= 0.15:
        return 3  # Turning
    if acc_z_std > 1.8:
        return 4  # Rough road
    return 1  # Cruise


def extract_features_from_window(
    imu_win: np.ndarray,
    aux_win: np.ndarray,
    dt: float = config.TARGET_DT
) -> np.ndarray:
    """
    Extract engineered summary feature vector for a 2.0s window.
    Features include mean, std, RMS, jerk, and spectral band power.
    """
    # imu_win: shape (20, 6) -> acc(3), gyro(3)
    acc = imu_win[:, 0:3]
    gyro = imu_win[:, 3:6]

    # Means and STDs
    acc_mean = np.mean(acc, axis=0)
    acc_std = np.std(acc, axis=0)
    gyro_mean = np.mean(gyro, axis=0)
    gyro_std = np.std(gyro, axis=0)

    # RMS energy
    acc_rms = np.sqrt(np.mean(acc**2, axis=0))
    gyro_rms = np.sqrt(np.mean(gyro**2, axis=0))

    # Jerk (first difference of acceleration)
    jerk = np.diff(acc, axis=0) / dt
    jerk_max = np.max(np.abs(jerk), axis=0)

    # Spectral band powers for vertical acceleration (z-axis) and forward (x-axis)
    acc_z_low, acc_z_high = compute_spectral_features(acc[:, 2])
    acc_x_low, acc_x_high = compute_spectral_features(acc[:, 0])

    features = np.concatenate([
        acc_mean, acc_std, acc_rms,
        gyro_mean, gyro_std, gyro_rms,
        jerk_max,
        [acc_z_low, acc_z_high, acc_x_low, acc_x_high]
    ])

    return features.astype(np.float32)


def process_synchronized_file(
    parquet_path: Path,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.WINDOW_STRIDE
) -> Optional[Dict[str, np.ndarray]]:
    """
    Process a single synchronized parquet file into windowed dataset.
    """
    df = pd.read_parquet(parquet_path)
    n_samples = len(df)
    if n_samples < window_size:
        return None

    # Check required columns
    for c in SEQ_CHANNELS + [config.COL_TRUE_SPEED_MS]:
        if c not in df.columns:
            logger.warning(f"Missing column {c} in {parquet_path.name}")
            return None

    # Fill NaNs
    for c in ALL_INPUT_CHANNELS:
        if c in df.columns:
            df[c] = df[c].interpolate().bfill().ffill().fillna(0.0)
        else:
            df[c] = 0.0

    seq_data = df[SEQ_CHANNELS].values.astype(np.float32)
    aux_data = df[AUX_CHANNELS].values.astype(np.float32)

    gt_speed = df[config.COL_TRUE_SPEED_MS].clip(lower=0.0).values.astype(np.float32)
    gt_delta_d = (gt_speed * config.TARGET_DT).astype(np.float32)

    # Yaw rate for event detection
    yaw_rate = df[config.COL_TRUE_YAW_RATE].values * config.DEG_TO_RAD if config.COL_TRUE_YAW_RATE in df.columns else np.zeros(n_samples)
    acc_long = df[config.COL_TRUE_ACCEL_LONG].values if config.COL_TRUE_ACCEL_LONG in df.columns else np.zeros(n_samples)

    windows_seq = []
    windows_feat = []
    labels_delta_d = []
    labels_event = []
    labels_speed = []

    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        win_seq = seq_data[start_idx:end_idx]
        win_aux = aux_data[start_idx:end_idx]

        # Target: integrated true displacement over window
        win_delta_d = float(np.sum(gt_delta_d[start_idx:end_idx]))
        win_mean_spd = float(np.mean(gt_speed[start_idx:end_idx]))

        # Engineered features
        feat = extract_features_from_window(win_seq, win_aux)

        # Event classification
        mean_acc_long = float(np.mean(acc_long[start_idx:end_idx]))
        mean_yaw = float(np.mean(yaw_rate[start_idx:end_idx]))
        acc_z_std = float(np.std(win_seq[:, 2]))
        ev_class = classify_window_event(win_mean_spd, mean_acc_long, mean_yaw, acc_z_std)

        windows_seq.append(win_seq)
        windows_feat.append(feat)
        labels_delta_d.append(win_delta_d)
        labels_event.append(ev_class)
        labels_speed.append(win_mean_spd)

    if not windows_seq:
        return None

    return {
        "X_seq": np.array(windows_seq, dtype=np.float32),
        "X_feat": np.array(windows_feat, dtype=np.float32),
        "y_delta_d": np.array(labels_delta_d, dtype=np.float32),
        "y_event": np.array(labels_event, dtype=np.int64),
        "y_speed": np.array(labels_speed, dtype=np.float32)
    }


def build_split_datasets(max_per_split: Optional[int] = None) -> None:
    """
    Iterate over all synchronized Parquets, partition into Train/Val/Test by driver,
    and save compressed .npz files.
    """
    sync_files = glob.glob(str(config.SYNC_PROCESSED_DIR / "sync_*.parquet"))
    logger.info(f"Found {len(sync_files)} synchronized files.")

    splits = {"train": [], "val": [], "test": []}

    for f in sync_files:
        run_key = Path(f).stem.replace("sync_", "").lower()
        if any(run_key.startswith(p.lower()) for p in config.TRAIN_DRIVERS):
            splits["train"].append(Path(f))
        elif any(run_key.startswith(p.lower()) for p in config.VAL_DRIVERS):
            splits["val"].append(Path(f))
        else:
            splits["test"].append(Path(f))

    logger.info(
        f"Partitioning: Train={len(splits['train'])} runs, Val={len(splits['val'])} runs, Test={len(splits['test'])} runs"
    )

    for split_name, files in splits.items():
        if max_per_split:
            files = files[:max_per_split]

        all_seq = []
        all_feat = []
        all_delta_d = []
        all_event = []
        all_speed = []

        logger.info(f"Processing split '{split_name}' with {len(files)} files...")
        for p in files:
            res = process_synchronized_file(p)
            if res is not None:
                all_seq.append(res["X_seq"])
                all_feat.append(res["X_feat"])
                all_delta_d.append(res["y_delta_d"])
                all_event.append(res["y_event"])
                all_speed.append(res["y_speed"])

        if all_seq:
            cat_seq = np.concatenate(all_seq, axis=0)
            cat_feat = np.concatenate(all_feat, axis=0)
            cat_delta_d = np.concatenate(all_delta_d, axis=0)
            cat_event = np.concatenate(all_event, axis=0)
            cat_speed = np.concatenate(all_speed, axis=0)

            out_npz = config.WINDOWED_DIR / f"{split_name}_windows.npz"
            np.savez_compressed(
                out_npz,
                X_seq=cat_seq,
                X_feat=cat_feat,
                y_delta_d=cat_delta_d,
                y_event=cat_event,
                y_speed=cat_speed
            )
            logger.info(
                f"Split '{split_name}' saved: {cat_seq.shape[0]} windows, seq shape: {cat_seq.shape} -> {out_npz.name}"
            )
        else:
            logger.warning(f"No windows generated for split '{split_name}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV Sliding Window Feature Extractor")
    parser.add_argument("--max-per-split", type=int, default=None, help="Limit files per split for testing")
    args = parser.parse_args()

    build_split_datasets(max_per_split=args.max_per_split)
