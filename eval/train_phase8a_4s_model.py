"""
Phase 8A-1: Training and Evaluation Pipeline for 4.0-Second Context VelocityNet
Trains an experimental 4-second model on train_windows_4s.npz using:
- Standard Huber displacement loss
- Speed-balanced WeightedRandomSampler on training distribution
- SO(3) 3D spatial rotation augmentation
- Saves strictly to models/phase8a_4s_velocity_net.pt and models/phase8a_4s_velocity_net.onnx
"""

import sys
import logging
from pathlib import Path
from typing import Tuple
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
logger = logging.getLogger("iNAV.Phase8A_Train")


class VelocityNet4s(nn.Module):
    """
    VelocityNet Adapted for 4.0-Second Context Window (40 samples @ 10 Hz).
    Input: Tensor of shape (Batch, Channels=6, Time=40)
    """
    def __init__(self, in_channels: int = 6, num_events: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.num_events = num_events

        self.in_bn = nn.BatchNorm1d(in_channels)

        # Multi-scale Dilated 1D Convolutions
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Head 1: Forward Displacement Regression (meters over 4.0s window)
        self.head_displacement = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        # Initialize bias to 30.0m (typical 4s displacement at 27 km/h)
        nn.init.constant_(self.head_displacement[-1].bias, 30.0)

        # Head 2: Driving Event Classifier
        self.head_event = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_events)
        )

        # Head 3: Learned Uncertainty sigma(Delta d) > 0
        self.head_uncertainty = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        nn.init.constant_(self.head_uncertainty[-2].bias, 2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (Batch, 6, 40)
        x_norm = self.in_bn(x)

        h = F.leaky_relu(self.bn1(self.conv1(x_norm)), 0.1)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.1)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.1)

        # Permute for GRU: (Batch, 128, 40) -> (Batch, 40, 128)
        h_perm = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h_perm)

        # Global average pooling across time
        feat = torch.mean(gru_out, dim=1)  # (Batch, 128)

        delta_d = torch.clamp(self.head_displacement(feat), min=0.0)
        event_logits = self.head_event(feat)
        sigma = self.head_uncertainty(feat) + 1e-3

        return delta_d, event_logits, sigma


def compute_speed_balanced_weights_4s(train_speeds: np.ndarray) -> Tuple[torch.Tensor, pd.Series]:
    bins = [0, 5, 10, 15, 20, 25, 30, 1000]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    cat = pd.cut(train_speeds, bins=bins, labels=labels, right=False)
    counts = pd.Series(cat).value_counts().sort_index()

    bin_weights = {label: 1.0 / np.sqrt(count) if count > 0 else 0.0 for label, count in counts.items()}
    sample_weights = np.array([bin_weights[c] for c in cat], dtype=np.float32)
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

    return torch.from_numpy(sample_weights), counts


def train_4s_model(
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42
) -> Tuple[Path, Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str)
    logger.info(f"Phase 8A-1 Training on device: {device} | Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size}")

    train_path = config.WINDOWED_DIR / "train_windows_4s.npz"
    val_path = config.WINDOWED_DIR / "val_windows_4s.npz"

    train_npz = np.load(train_path)
    val_npz = np.load(val_path)

    X_train = train_npz["X_seq"]  # (N, 40, 6)
    y_d_train = train_npz["y_delta_d"]
    y_ev_train = train_npz["y_event"]
    y_spd_train = train_npz["y_speed"]

    X_val = val_npz["X_seq"]
    y_d_val = val_npz["y_delta_d"]
    y_ev_val = val_npz["y_event"]
    y_spd_val = val_npz["y_speed"]

    # Transpose for Conv1d: (N, 40, 6) -> (N, 6, 40)
    X_train_t = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    y_d_train_t = torch.from_numpy(y_d_train).float().unsqueeze(1)
    y_ev_train_t = torch.from_numpy(y_ev_train).long()

    X_val_t = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    y_d_val_t = torch.from_numpy(y_d_val).float().unsqueeze(1)
    y_ev_val_t = torch.from_numpy(y_ev_val).long()

    sample_weights, bin_counts = compute_speed_balanced_weights_4s(y_spd_train)
    logger.info("4S Training set speed distribution:")
    for b_lbl, b_cnt in bin_counts.items():
        logger.info(f"  Speed bin {b_lbl} m/s: {b_cnt} windows ({b_cnt/len(y_spd_train)*100:.1f}%)")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(X_train_t, y_d_train_t, y_ev_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val_t, y_d_val_t, y_ev_val_t), batch_size=batch_size * 2, shuffle=False)

    model = VelocityNet4s(in_channels=6, num_events=5).to(device)

    pt_path_4s = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    onnx_path_4s = config.MODELS_DIR / "phase8a_4s_velocity_net.onnx"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_mae = float("inf")

    logger.info("Beginning Phase 8A-1 (4-Second Context) Training...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_d_mae = 0.0

        for batch_x, batch_d, batch_ev in train_loader:
            batch_x = batch_x.to(device)
            batch_d = batch_d.to(device)
            batch_ev = batch_ev.to(device)

            batch_x = apply_batch_3d_spatial_rotation(batch_x, prob=0.5)

            optimizer.zero_grad()
            pred_d, pred_ev, pred_sig = model(batch_x)

            loss_d = huber_loss_fn(pred_d, batch_d)
            loss_nll = torch.mean(0.5 * ((batch_d - pred_d.detach())**2 / (pred_sig**2)) + torch.log(pred_sig))
            loss_ev = ce_loss_fn(pred_ev, batch_ev)

            total_loss = 0.7 * loss_d + 0.05 * loss_nll + 0.1 * loss_ev
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += total_loss.item() * len(batch_x)
            train_d_mae += torch.sum(torch.abs(pred_d - batch_d)).item()

        scheduler.step()
        train_loss /= len(X_train_t)
        train_d_mae /= len(X_train_t)

        model.eval()
        val_loss = 0.0
        val_d_mae = 0.0
        val_ev_acc = 0.0
        val_mean_sig = 0.0

        with torch.no_grad():
            for batch_x, batch_d, batch_ev in val_loader:
                batch_x = batch_x.to(device)
                batch_d = batch_d.to(device)
                batch_ev = batch_ev.to(device)

                pred_d, pred_ev, pred_sig = model(batch_x)
                loss_d = huber_loss_fn(pred_d, batch_d)
                loss_nll = torch.mean(0.5 * ((batch_d - pred_d)**2 / (pred_sig**2)) + torch.log(pred_sig))
                loss_ev = ce_loss_fn(pred_ev, batch_ev)

                total_loss = 0.7 * loss_d + 0.05 * loss_nll + 0.1 * loss_ev
                val_loss += total_loss.item() * len(batch_x)
                val_d_mae += torch.sum(torch.abs(pred_d - batch_d)).item()
                val_ev_acc += torch.sum(torch.argmax(pred_ev, dim=1) == batch_ev).item()
                val_mean_sig += torch.sum(pred_sig).item()

        val_loss /= len(X_val_t)
        val_d_mae /= len(X_val_t)
        val_ev_acc /= len(X_val_t)
        val_mean_sig /= len(X_val_t)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f}, Train MAE: {train_d_mae:.3f}m | "
            f"Val Loss: {val_loss:.4f}, Val MAE: {val_d_mae:.3f}m, Ev Acc: {val_ev_acc*100:.1f}%, Mean sig: {val_mean_sig:.2f}m"
        )

        if val_d_mae < best_val_mae:
            best_val_mae = val_d_mae
            torch.save(model.state_dict(), pt_path_4s)
            logger.info(f"Saved new best Phase 8A-1 checkpoint to {pt_path_4s.name} (Val MAE: {val_d_mae:.3f}m)")

    if not pt_path_4s.exists():
        torch.save(model.state_dict(), pt_path_4s)

    # Export to ONNX
    try:
        model.load_state_dict(torch.load(pt_path_4s, map_location=device))
        model.eval()
        dummy_input = torch.randn(1, 6, 40, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path_4s),
            input_names=["imu_window"],
            output_names=["delta_d", "event_logits", "sigma"],
            dynamic_axes={"imu_window": {0: "batch_size"}, "delta_d": {0: "batch_size"}},
            opset_version=14
        )
        logger.info(f"Exported Phase 8A-1 ONNX to {onnx_path_4s.name}")
    except Exception as e:
        logger.warning(f"ONNX export exception: {e}")

    return pt_path_4s, onnx_path_4s


if __name__ == "__main__":
    train_4s_model(epochs=8, batch_size=256)
