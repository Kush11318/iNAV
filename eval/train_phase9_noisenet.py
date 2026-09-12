"""
Phase 9: NoiseNet Training Script for AI-Adaptive NHC Covariance Prediction
Inspired by Brossard et al., 'AI-IMU Dead-Reckoning' (IEEE T-RO 2020).

Predicts measurement noise covariance:
    R_NHC = diag(sigma^2_lat, sigma^2_vert)
from 6-channel raw IMU window (40 samples @ 10 Hz = 4.0s).

Training target:
    Derived strictly from IO-VNBD training campaigns (TRAIN_CAMPAIGNS)
    using reference ground truth velocity projected into vehicle chassis frame:
        v_lat_gt = -v_N * sin(psi) + v_E * cos(psi)
        v_vert_gt = d(height)/dt
    sigma^2_lat_target = mean(v_lat_gt^2) over window
    sigma^2_vert_target = mean(v_vert_gt^2) over window

Strict Data Integrity:
    - Zero test set data used in training, validation, or tuning.
    - Model weights saved strictly to models/phase9_noisenet.pt.
"""

import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase9_Train")

WINDOW_LEN = 40      # 4.0 seconds @ 10 Hz
WINDOW_STRIDE = 4    # 0.4 seconds step
MIN_VAR = 1e-4       # Lower bound on variance (0.0001 m^2/s^2)
MAX_VAR = 10.0       # Upper bound on variance (10.0 m^2/s^2)


class NoiseNet(nn.Module):
    """
    Lightweight 1D Temporal CNN for IMU Measurement Noise Covariance Estimation.
    Input: (B, 6, 40) - 6-channel IMU window [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Output: (B, 2) - [sigma^2_lat, sigma^2_vert] strictly positive and bounded
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 40)
        feat = self.conv(x).squeeze(-1)  # (B, 64)
        raw = self.fc(feat)              # (B, 2)
        # Softplus ensures strict positivity; clamp prevents pathological values
        cov = torch.clamp(nn.functional.softplus(raw) + MIN_VAR, MIN_VAR, MAX_VAR)
        return cov


class IMUNoiseDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        # windows: (N, 40, 6) -> transpose to (N, 6, 40)
        self.windows = torch.from_numpy(windows.transpose(0, 2, 1)).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]


def extract_windows_from_parquet(
    parquet_path: Path,
    window_len: int = WINDOW_LEN,
    stride: int = WINDOW_STRIDE
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extracts sliding IMU windows and corresponding ground-truth variance targets.
    """
    df = pd.read_parquet(parquet_path)
    n = len(df)
    if n < window_len:
        return [], []

    cols = [
        config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z,
        config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z
    ]
    for c in cols:
        if c not in df.columns:
            return [], []
        df[c] = df[c].interpolate().bfill().ffill().fillna(0.0)

    if config.COL_TRUE_LAT not in df.columns or config.COL_TRUE_LON not in df.columns:
        return [], []

    dt = config.TARGET_DT
    lat_rad = np.radians(df[config.COL_TRUE_LAT].values)
    lon_rad = np.radians(df[config.COL_TRUE_LON].values)
    ref_lat = lat_rad[0]
    pN = 6371000.0 * (lat_rad - ref_lat)
    pE = 6371000.0 * (lon_rad - lon_rad[0]) * np.cos(ref_lat)

    vN = np.gradient(pN, dt)
    vE = np.gradient(pE, dt)

    if config.COL_TRUE_HEADING in df.columns:
        psi = np.radians(df[config.COL_TRUE_HEADING].values)
    else:
        psi = np.arctan2(vE, vN)

    # Lateral velocity in vehicle chassis frame
    vy_b = -vN * np.sin(psi) + vE * np.cos(psi)

    # Vertical velocity in vehicle chassis frame
    if "gt_height_m" in df.columns:
        vz_b = np.gradient(df["gt_height_m"].values, dt)
    elif config.COL_GPS_ALT in df.columns:
        vz_b = np.gradient(df[config.COL_GPS_ALT].values, dt)
    else:
        vz_b = np.zeros_like(vy_b)

    # Clean any NaNs or infinities in ground truth
    vy_b = np.nan_to_num(vy_b, nan=0.0, posinf=0.0, neginf=0.0)
    vz_b = np.nan_to_num(vz_b, nan=0.0, posinf=0.0, neginf=0.0)
    vy_b = np.clip(vy_b, -10.0, 10.0)
    vz_b = np.clip(vz_b, -5.0, 5.0)

    imu_data = df[cols].values.astype(np.float32)

    windows = []
    targets = []

    for start in range(0, n - window_len + 1, stride):
        end = start + window_len
        win_imu = imu_data[start:end]

        # Empirical variance of NHC residuals over the window
        var_lat = float(np.mean(vy_b[start:end]**2))
        var_vert = float(np.mean(vz_b[start:end]**2))

        var_lat = np.clip(var_lat, MIN_VAR, MAX_VAR)
        var_vert = np.clip(var_vert, MIN_VAR, MAX_VAR)

        windows.append(win_imu)
        targets.append(np.array([var_lat, var_vert], dtype=np.float32))

    return windows, targets


def gaussian_nll_loss(pred_var: torch.Tensor, target_var: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Log-Likelihood loss for variance estimation:
        Loss = 0.5 * (target_var / pred_var + log(pred_var))
    Equivalently, this is the MLE loss for estimating zero-mean Gaussian measurement noise.
    """
    loss = 0.5 * (target_var / pred_var + torch.log(pred_var))
    return torch.mean(loss)


def train_noisenet():
    logger.info("=" * 80)
    logger.info("PHASE 9: NOISENET COVARIANCE ADAPTER TRAINING")
    logger.info("=" * 80)

    all_files = list(config.SYNC_PROCESSED_DIR.glob("sync_*.parquet"))
    logger.info(f"Total synchronized files found: {len(all_files)}")

    train_files = []
    val_files = []

    for p in all_files:
        run_name = p.stem.replace("sync_", "").split("_")[0].lower()
        if any(run_name.startswith(c) for c in config.TRAIN_CAMPAIGNS):
            train_files.append(p)
        elif any(run_name.startswith(c) for c in config.VAL_CAMPAIGNS):
            val_files.append(p)

    logger.info(f"Training files: {len(train_files)} | Validation files: {len(val_files)}")

    # Extract training windows
    train_wins, train_tgts = [], []
    for p in train_files:
        w, t = extract_windows_from_parquet(p)
        train_wins.extend(w)
        train_tgts.extend(t)

    # Extract validation windows
    val_wins, val_tgts = [], []
    for p in val_files:
        w, t = extract_windows_from_parquet(p)
        val_wins.extend(w)
        val_tgts.extend(t)

    train_wins = np.array(train_wins, dtype=np.float32)
    train_tgts = np.array(train_tgts, dtype=np.float32)
    val_wins = np.array(val_wins, dtype=np.float32)
    val_tgts = np.array(val_tgts, dtype=np.float32)

    logger.info(f"Extracted Training Windows: {train_wins.shape}")
    logger.info(f"Extracted Validation Windows: {val_wins.shape}")

    logger.info("--- Train Target Statistics (m^2/s^2) ---")
    logger.info(
        f"sigma^2_lat  : Min={train_tgts[:, 0].min():.5f}, "
        f"Median={np.median(train_tgts[:, 0]):.5f}, "
        f"Mean={train_tgts[:, 0].mean():.5f}, "
        f"P95={np.percentile(train_tgts[:, 0], 95):.5f}, "
        f"Max={train_tgts[:, 0].max():.5f}"
    )
    logger.info(
        f"sigma^2_vert : Min={train_tgts[:, 1].min():.5f}, "
        f"Median={np.median(train_tgts[:, 1]):.5f}, "
        f"Mean={train_tgts[:, 1].mean():.5f}, "
        f"P95={np.percentile(train_tgts[:, 1], 95):.5f}, "
        f"Max={train_tgts[:, 1].max():.5f}"
    )

    train_dataset = IMUNoiseDataset(train_wins, train_tgts)
    val_dataset = IMUNoiseDataset(val_wins, val_tgts)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = NoiseNet(in_channels=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            pred_cov = model(bx)
            loss = gaussian_nll_loss(pred_cov, by)
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item()) * len(bx)
            n_train += len(bx)

        train_loss = train_loss_total / max(1, n_train)

        # Validation
        model.eval()
        val_loss_total = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                pred_cov = model(bx)
                loss = gaussian_nll_loss(pred_cov, by)
                val_loss_total += float(loss.item()) * len(bx)
                n_val += len(bx)

        val_loss = val_loss_total / max(1, n_val)

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train NLL Loss: {train_loss:.4f} | Val NLL Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    # Save model
    save_path = config.MODELS_DIR / "phase9_noisenet.pt"
    if best_model_state is not None:
        torch.save(best_model_state, save_path)
    else:
        torch.save(model.state_dict(), save_path)

    logger.info(f"Successfully trained and saved NoiseNet model to: {save_path}")

    # Compute and log mean train covariance for Ablation C
    mean_train_lat_var = float(np.mean(train_tgts[:, 0]))
    mean_train_vert_var = float(np.mean(train_tgts[:, 1]))
    median_train_lat_var = float(np.median(train_tgts[:, 0]))
    median_train_vert_var = float(np.median(train_tgts[:, 1]))

    logger.info("--- Ablation C Parameters (Derived strictly from Train Split) ---")
    logger.info(f"Mean Train sigma^2_lat  : {mean_train_lat_var:.5f} m^2/s^2")
    logger.info(f"Mean Train sigma^2_vert : {mean_train_vert_var:.5f} m^2/s^2")
    logger.info(f"Median Train sigma^2_lat: {median_train_lat_var:.5f} m^2/s^2")
    logger.info(f"Median Train sigma^2_vert: {median_train_vert_var:.5f} m^2/s^2")

    return save_path


if __name__ == "__main__":
    train_noisenet()
