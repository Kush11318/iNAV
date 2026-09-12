"""
Phase 10: Training Script for Temporal Forward-Velocity Neural Model
Trains TemporalVelocityNet4s on 4.0-second IMU windows (40 samples @ 10 Hz)
predicting 20-step forward velocity profile (m/s) @ 5 Hz.

Loss: Standard Huber loss on temporal velocity sequence:
    Loss_vel = Huber(v_pred[t], v_gt[t])
Auxiliary Loss: Event classification (CrossEntropy)
Data Augmentation: SO(3) 3D spatial coordinate rotation

Strict Data Integrity:
- Trained strictly on train_windows_phase10.npz (TRAIN_CAMPAIGNS)
- Evaluated for model checkpointing on val_windows_phase10.npz (VAL_CAMPAIGNS)
- Zero test data used in training, tuning, or checkpoint selection
- Saves strictly to models/phase10_temporal_velocity_net.pt and .onnx
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from modules.augmentation import apply_batch_3d_spatial_rotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase10_Train")


class TemporalVelocityNet4s(nn.Module):
    """
    Temporal Forward-Velocity Sequence Model for 4.0-Second Context.
    Input: Tensor of shape (Batch, Channels=6, Time=40)
    Outputs:
        v_seq: (Batch, 20) forward speed sequence @ 5 Hz (dt = 0.2s)
        sig_seq: (Batch, 20) speed uncertainty sequence (m/s)
        event_logits: (Batch, 5) driving condition class
    """

    def __init__(self, in_channels: int = 6, num_events: int = 5):
        super().__init__()
        self.in_bn = nn.BatchNorm1d(in_channels)

        # Multi-scale 1D Convolutions
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)

        # Bidirectional GRU maintaining temporal sequence
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Temporal downsampling from 10 Hz (40 samples) to 5 Hz (20 samples)
        self.pool_time = nn.AvgPool1d(kernel_size=2, stride=2)

        # Per-timestep sequence velocity head (20 steps)
        self.head_velocity = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.ReLU()  # Speed is strictly non-negative
        )
        # Initialize bias to 7.5 m/s (~27 km/h)
        nn.init.constant_(self.head_velocity[-2].bias, 7.5)

        # Per-timestep uncertainty head
        self.head_uncertainty = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        nn.init.constant_(self.head_uncertainty[-2].bias, 0.5)

        # Global event classifier head
        self.head_event = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_events)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (Batch, 6, 40)
        x_norm = self.in_bn(x)

        h = F.leaky_relu(self.bn1(self.conv1(x_norm)), 0.1)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.1)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.1)

        h_perm = h.permute(0, 2, 1)  # (Batch, 40, 128)
        gru_out, _ = self.gru(h_perm)  # (Batch, 40, 128)

        # Downsample along temporal axis to 20 steps (5 Hz)
        gru_time = gru_out.permute(0, 2, 1)  # (Batch, 128, 40)
        gru_pooled = self.pool_time(gru_time).permute(0, 2, 1)  # (Batch, 20, 128)

        v_seq = self.head_velocity(gru_pooled).squeeze(-1)  # (Batch, 20)
        sig_seq = self.head_uncertainty(gru_pooled).squeeze(-1) + 1e-3  # (Batch, 20)

        # Global event feature
        feat_global = torch.mean(gru_out, dim=1)  # (Batch, 128)
        event_logits = self.head_event(feat_global)

        return v_seq, sig_seq, event_logits


def compute_speed_balanced_weights(train_speeds: np.ndarray) -> Tuple[torch.Tensor, pd.Series]:
    bins = [0, 5, 10, 15, 20, 25, 30, 1000]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    cat = pd.cut(train_speeds, bins=bins, labels=labels, right=False)
    counts = pd.Series(cat).value_counts().sort_index()

    bin_weights = {label: 1.0 / np.sqrt(count) if count > 0 else 0.0 for label, count in counts.items()}
    sample_weights = np.array([bin_weights[c] for c in cat], dtype=np.float32)
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

    return torch.from_numpy(sample_weights), counts


def train_phase10_model(
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42
) -> Tuple[Path, Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str)
    logger.info("=" * 80)
    logger.info(f"PHASE 10: TRAINING TEMPORAL VELOCITY MODEL (Device: {device}, Epochs: {epochs}, Batch Size: {batch_size})")
    logger.info("=" * 80)

    train_path = config.WINDOWED_DIR / "train_windows_phase10.npz"
    val_path = config.WINDOWED_DIR / "val_windows_phase10.npz"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing windowed dataset: {train_path} or {val_path}")

    logger.info(f"Loading training data from {train_path}...")
    train_npz = np.load(train_path)
    X_train = train_npz["X_seq"]        # (N, 40, 6)
    y_vel_train = train_npz["y_vel_20"] # (N, 20)
    y_spd_train = train_npz["y_speed_mean"]
    y_ev_train = train_npz["y_event"]

    logger.info(f"Loading validation data from {val_path}...")
    val_npz = np.load(val_path)
    X_val = val_npz["X_seq"]
    y_vel_val = val_npz["y_vel_20"]
    y_spd_val = val_npz["y_speed_mean"]
    y_ev_val = val_npz["y_event"]

    logger.info(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    # Speed-balanced sampling
    sample_weights, bin_counts = compute_speed_balanced_weights(y_spd_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Tensor datasets: (N, 40, 6) -> transposed to (N, 6, 40)
    train_x_tensor = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    train_vel_tensor = torch.from_numpy(y_vel_train).float()
    train_ev_tensor = torch.from_numpy(y_ev_train).long()

    val_x_tensor = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    val_vel_tensor = torch.from_numpy(y_vel_val).float()
    val_ev_tensor = torch.from_numpy(y_ev_val).long()

    train_dataset = TensorDataset(train_x_tensor, train_vel_tensor, train_ev_tensor)
    val_dataset = TensorDataset(val_x_tensor, val_vel_tensor, val_ev_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TemporalVelocityNet4s(in_channels=6, num_events=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_vel = nn.HuberLoss(delta=1.0)
    criterion_event = nn.CrossEntropyLoss()

    best_val_rmse = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_vel_loss = 0.0
        total_ev_loss = 0.0
        n_batches = 0

        for bx, b_vel, b_ev in train_loader:
            bx, b_vel, b_ev = bx.to(device), b_vel.to(device), b_ev.to(device)

            # Apply SO(3) 3D spatial rotation augmentation
            bx = apply_batch_3d_spatial_rotation(bx, prob=0.5)

            optimizer.zero_grad()
            v_pred, sig_pred, ev_logits = model(bx)

            # Sequence Huber loss across all 20 time steps
            l_vel = criterion_vel(v_pred, b_vel)
            l_ev = criterion_event(ev_logits, b_ev)

            # Uncertainty calibration loss: encourage sigma to track absolute velocity residual
            res = torch.abs(v_pred - b_vel).detach()
            l_sig = criterion_vel(sig_pred, res)

            loss = l_vel + 0.1 * l_ev + 0.2 * l_sig
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_vel_loss += l_vel.item()
            total_ev_loss += l_ev.item()
            n_batches += 1

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_vel_sq_err = []
        val_vel_abs_err = []
        val_ev_correct = 0
        total_val_pts = 0

        with torch.no_grad():
            for bx, b_vel, b_ev in val_loader:
                bx, b_vel, b_ev = bx.to(device), b_vel.to(device), b_ev.to(device)
                v_pred, _, ev_logits = model(bx)

                diff = v_pred - b_vel
                val_vel_sq_err.append(torch.mean(diff ** 2).item() * len(bx))
                val_vel_abs_err.append(torch.mean(torch.abs(diff)).item() * len(bx))

                preds_ev = torch.argmax(ev_logits, dim=1)
                val_ev_correct += (preds_ev == b_ev).sum().item()
                total_val_pts += len(bx)

        val_rmse = np.sqrt(np.sum(val_vel_sq_err) / total_val_pts)
        val_mae = np.sum(val_vel_abs_err) / total_val_pts
        val_acc = val_ev_correct / total_val_pts * 100.0

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {total_loss/n_batches:.4f} (Vel: {total_vel_loss/n_batches:.4f}) | "
            f"Val Vel RMSE: {val_rmse:.4f} m/s | Val Vel MAE: {val_mae:.4f} m/s | "
            f"Val Ev Acc: {val_acc:.1f}%"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()

    # Save PyTorch checkpoint
    pt_save_path = config.MODELS_DIR / "phase10_temporal_velocity_net.pt"
    if best_model_state is not None:
        torch.save(best_model_state, pt_save_path)
        model.load_state_dict(best_model_state)
    else:
        torch.save(model.state_dict(), pt_save_path)

    logger.info(f"Saved best PyTorch model checkpoint to {pt_save_path} (Best Val RMSE: {best_val_rmse:.4f} m/s)")

    # Export to ONNX
    onnx_save_path = config.MODELS_DIR / "phase10_temporal_velocity_net.onnx"
    dummy_input = torch.randn(1, 6, 40, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        input_names=["imu_window"],
        output_names=["v_seq", "sig_seq", "event_logits"],
        dynamic_axes={
            "imu_window": {0: "batch_size"},
            "v_seq": {0: "batch_size"},
            "sig_seq": {0: "batch_size"},
            "event_logits": {0: "batch_size"}
        },
        opset_version=14
    )
    logger.info(f"Successfully exported ONNX model to {onnx_save_path}")

    # Model profiling
    param_count = sum(p.numel() for p in model.parameters())
    pt_size_kb = pt_save_path.stat().st_size / 1024.0
    onnx_size_kb = onnx_save_path.stat().st_size / 1024.0

    # Measure CPU inference latency
    t0 = time.perf_counter()
    n_benchmark = 100
    with torch.no_grad():
        for _ in range(n_benchmark):
            _ = model(dummy_input)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) / n_benchmark * 1000.0

    logger.info("=" * 80)
    logger.info("MODEL PROFILING SUMMARY:")
    logger.info(f"Total Parameters : {param_count:,}")
    logger.info(f"PyTorch File Size: {pt_size_kb:.1f} KB")
    logger.info(f"ONNX File Size   : {onnx_size_kb:.1f} KB")
    logger.info(f"CPU Latency / Win: {latency_ms:.2f} ms")
    logger.info("=" * 80)

    return pt_save_path, onnx_save_path


if __name__ == "__main__":
    train_phase10_model()
