"""
Phase 11: Training Script for GNSS-Anchored Forward-Velocity Residual Model
Trains AnchoredTemporalVelocityNet on 4.0-second IMU windows (40 samples @ 10 Hz)
conditioned on scalar velocity anchor v_anchor (m/s).

Predicts 20-step velocity residual sequence Delta v(t) @ 5 Hz,
and reconstructs:
    v_pred(t) = clamp(v_anchor + Delta v_pred(t), min=0.0)

Includes:
  - Experiment B: Anchored model with speed-aware bounded inverse-frequency loss w(v_anchor)
  - Ablation C: Anchored model with uniform loss weighting (w = 1.0)

Strict Data Integrity:
  - Trained strictly on train_windows_phase11.npz (TRAIN_CAMPAIGNS)
  - Evaluated for checkpointing on val_windows_phase11.npz (VAL_CAMPAIGNS)
  - Zero test data used in training, tuning, or weight derivation
  - Saves to:
      models/phase11_anchored_velocity_net.pt
      models/phase11_anchored_velocity_net.onnx
      models/phase11_ablation_uniform_net.pt
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

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
logger = logging.getLogger("iNAV.Phase11_Train")


class AnchoredTemporalVelocityNet(nn.Module):
    """
    Anchored Temporal Forward-Velocity Sequence Model for 4.0-Second Context.
    Inputs:
        x: Tensor of shape (Batch, Channels=6, Time=40)
        v_anchor: Tensor of shape (Batch, 1) in m/s
    Outputs:
        delta_v_seq: (Batch, 20) velocity residual sequence @ 5 Hz (dt = 0.2s)
        v_pred_seq: (Batch, 20) reconstructed forward speed sequence @ 5 Hz
        sig_seq: (Batch, 20) speed residual uncertainty sequence (m/s)
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

        # Anchor speed embedding: scalar v_anchor -> 16-dim feature
        self.embed_anchor = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.1)
        )

        # Concatenated temporal feature dimension: 128 (GRU) + 16 (anchor) = 144
        # Per-timestep sequence residual velocity head (20 steps)
        self.head_delta_v = nn.Sequential(
            nn.Linear(144, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
            # Linear activation: Delta v can be negative (braking) or positive (acceleration)
        )
        nn.init.constant_(self.head_delta_v[-1].bias, 0.0)

        # Per-timestep uncertainty head
        self.head_uncertainty = nn.Sequential(
            nn.Linear(144, 32),
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

    def forward(
        self,
        x: torch.Tensor,
        v_anchor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (Batch, 6, 40)
        # v_anchor shape: (Batch, 1)
        x_norm = self.in_bn(x)

        h = F.leaky_relu(self.bn1(self.conv1(x_norm)), 0.1)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.1)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.1)

        h_perm = h.permute(0, 2, 1)  # (Batch, 40, 128)
        gru_out, _ = self.gru(h_perm)  # (Batch, 40, 128)

        # Downsample along temporal axis to 20 steps (5 Hz)
        gru_time = gru_out.permute(0, 2, 1)  # (Batch, 128, 40)
        gru_pooled = self.pool_time(gru_time).permute(0, 2, 1)  # (Batch, 20, 128)

        # Embed anchor and tile along temporal dimension
        emb_anc = self.embed_anchor(v_anchor)  # (Batch, 16)
        emb_anc_tiled = emb_anc.unsqueeze(1).expand(-1, 20, -1)  # (Batch, 20, 16)

        # Concatenate temporal features with anchor representation
        feat_concat = torch.cat([gru_pooled, emb_anc_tiled], dim=-1)  # (Batch, 20, 144)

        delta_v_seq = self.head_delta_v(feat_concat).squeeze(-1)  # (Batch, 20)
        # Reconstruct non-negative forward velocity
        v_pred_seq = torch.clamp(v_anchor + delta_v_seq, min=0.0)  # (Batch, 20)

        sig_seq = self.head_uncertainty(feat_concat).squeeze(-1) + 1e-3  # (Batch, 20)

        # Global event feature
        feat_global = torch.mean(gru_out, dim=1)  # (Batch, 128)
        event_logits = self.head_event(feat_global)

        return delta_v_seq, v_pred_seq, sig_seq, event_logits


def compute_training_anchor_weights(train_v_anchors: np.ndarray) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes bounded inverse-frequency speed weights strictly from training data.
    """
    bins = [0, 5, 10, 15, 20, 25, 30, 1000]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    cat = pd.cut(train_v_anchors.flatten(), bins=bins, labels=labels, right=False)
    counts = pd.Series(cat).value_counts().sort_index()

    n_bins = len(labels)
    n_total = len(train_v_anchors)
    raw_weights = {l: (n_total / (n_bins * max(counts[l], 1)))**0.5 for l in labels}

    sample_raw_w = np.array([raw_weights[c] for c in cat], dtype=np.float32)
    sample_norm_w = sample_raw_w / np.mean(sample_raw_w)
    sample_capped_w = np.clip(sample_norm_w, 0.4, 3.5)
    sample_final_w = sample_capped_w / np.mean(sample_capped_w)

    bin_weight_summary = {l: float(np.mean(sample_final_w[cat == l])) for l in labels}
    return torch.from_numpy(sample_final_w), bin_weight_summary


def train_single_model(
    experiment_name: str,
    use_speed_weighted_loss: bool,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42
) -> Tuple[AnchoredTemporalVelocityNet, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    logger.info("=" * 80)
    logger.info(f"PHASE 11: TRAINING {experiment_name.upper()} (Speed-Weighted Loss: {use_speed_weighted_loss})")
    logger.info("=" * 80)

    train_path = config.WINDOWED_DIR / "train_windows_phase11.npz"
    val_path = config.WINDOWED_DIR / "val_windows_phase11.npz"

    train_npz = np.load(train_path)
    X_train = train_npz["X_seq"]              # (N, 40, 6)
    v_anc_train = train_npz["v_anchor"]        # (N, 1)
    y_delta_v_train = train_npz["y_delta_v_20"] # (N, 20)
    y_vel_train = train_npz["y_vel_20"]         # (N, 20)
    y_ev_train = train_npz["y_event"]          # (N,)

    val_npz = np.load(val_path)
    X_val = val_npz["X_seq"]
    v_anc_val = val_npz["v_anchor"]
    y_delta_v_val = val_npz["y_delta_v_20"]
    y_vel_val = val_npz["y_vel_20"]
    y_ev_val = val_npz["y_event"]

    # Compute training sample loss weights
    sample_loss_weights, bin_weight_summary = compute_training_anchor_weights(v_anc_train)
    logger.info(f"Derived Bounded Training Anchor Loss Weights: {bin_weight_summary}")

    # Dataset tensors
    train_x_t = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    train_anc_t = torch.from_numpy(v_anc_train).float()
    train_del_v_t = torch.from_numpy(y_delta_v_train).float()
    train_vel_t = torch.from_numpy(y_vel_train).float()
    train_w_t = sample_loss_weights.float().unsqueeze(1) if use_speed_weighted_loss else torch.ones((len(train_x_t), 1)).float()
    train_ev_t = torch.from_numpy(y_ev_train).long()

    val_x_t = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    val_anc_t = torch.from_numpy(v_anc_val).float()
    val_del_v_t = torch.from_numpy(y_delta_v_val).float()
    val_vel_t = torch.from_numpy(y_vel_val).float()
    val_ev_t = torch.from_numpy(y_ev_val).long()

    train_dataset = TensorDataset(train_x_t, train_anc_t, train_del_v_t, train_vel_t, train_w_t, train_ev_t)
    val_dataset = TensorDataset(val_x_t, val_anc_t, val_del_v_t, val_vel_t, val_ev_t)

    # Use standard random shuffle for clean representation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_event = nn.CrossEntropyLoss()
    criterion_sig = nn.SmoothL1Loss(beta=1.0)

    best_val_vel_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_vel_loss = 0.0
        n_batches = 0

        for bx, b_anc, b_del_v, b_vel, b_w, b_ev in train_loader:
            bx = bx.to(device)
            b_anc = b_anc.to(device)
            b_del_v = b_del_v.to(device)
            b_vel = b_vel.to(device)
            b_w = b_w.to(device)
            b_ev = b_ev.to(device)

            # Apply SO(3) spatial rotation augmentation
            bx = apply_batch_3d_spatial_rotation(bx, prob=0.5)

            optimizer.zero_grad()
            del_v_pred, v_pred, sig_pred, ev_logits = model(bx, b_anc)

            # SmoothL1 sequence residual loss with speed weighting
            raw_res_loss = F.smooth_l1_loss(del_v_pred, b_del_v, beta=1.0, reduction='none')  # (Batch, 20)
            if use_speed_weighted_loss:
                l_vel = torch.mean(b_w * raw_res_loss)
            else:
                l_vel = torch.mean(raw_res_loss)

            l_ev = criterion_event(ev_logits, b_ev)
            res_mag = torch.abs(del_v_pred - b_del_v).detach()
            l_sig = criterion_sig(sig_pred, res_mag)

            loss = l_vel + 0.1 * l_ev + 0.2 * l_sig
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_vel_loss += l_vel.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_sq_err = []
        val_abs_err = []
        val_del_abs_err = []
        val_pts = 0

        with torch.no_grad():
            for bx, b_anc, b_del_v, b_vel, b_ev in val_loader:
                bx = bx.to(device)
                b_anc = b_anc.to(device)
                b_del_v = b_del_v.to(device)
                b_vel = b_vel.to(device)

                del_v_pred, v_pred, _, _ = model(bx, b_anc)

                diff_v = v_pred - b_vel
                val_sq_err.append(torch.mean(diff_v ** 2).item() * len(bx))
                val_abs_err.append(torch.mean(torch.abs(diff_v)).item() * len(bx))

                diff_del = del_v_pred - b_del_v
                val_del_abs_err.append(torch.mean(torch.abs(diff_del)).item() * len(bx))
                val_pts += len(bx)

        val_rmse = np.sqrt(np.sum(val_sq_err) / val_pts)
        val_mae = np.sum(val_abs_err) / val_pts
        val_del_mae = np.sum(val_del_abs_err) / val_pts

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/n_batches:.4f} (Vel: {total_vel_loss/n_batches:.4f}) | "
            f"Val Reconstructed Vel RMSE: {val_rmse:.4f} m/s | MAE: {val_mae:.4f} m/s | Residual MAE: {val_del_mae:.4f} m/s"
        )

        if val_rmse < best_val_vel_rmse:
            best_val_vel_rmse = val_rmse
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_vel_rmse


def main():
    logger.info("=" * 80)
    logger.info("PHASE 11: TRAINING EXPERIMENT B (ANCHORED + SPEED WEIGHTED LOSS)")
    logger.info("=" * 80)

    model_b, b_val_rmse = train_single_model(
        experiment_name="Experiment B (Anchored + Speed-Weighted Loss)",
        use_speed_weighted_loss=True,
        epochs=8,
        batch_size=256,
        learning_rate=1e-3,
        device_str="cpu"
    )

    pt_b_path = config.MODELS_DIR / "phase11_anchored_velocity_net.pt"
    torch.save(model_b.state_dict(), pt_b_path)
    logger.info(f"Saved Experiment B checkpoint to {pt_b_path} (Best Val RMSE: {b_val_rmse:.4f} m/s)")

    # Export Model B to ONNX
    onnx_b_path = config.MODELS_DIR / "phase11_anchored_velocity_net.onnx"
    dummy_x = torch.randn(1, 6, 40)
    dummy_anc = torch.tensor([[15.0]], dtype=torch.float32)
    model_b.eval()
    torch.onnx.export(
        model_b,
        (dummy_x, dummy_anc),
        onnx_b_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["imu_window_40", "v_anchor"],
        output_names=["delta_v_seq", "v_pred_seq", "uncertainty_seq", "event_logits"],
        dynamic_axes={
            "imu_window_40": {0: "batch_size"},
            "v_anchor": {0: "batch_size"},
            "delta_v_seq": {0: "batch_size"},
            "v_pred_seq": {0: "batch_size"},
            "uncertainty_seq": {0: "batch_size"},
            "event_logits": {0: "batch_size"}
        }
    )
    logger.info(f"Exported Experiment B to ONNX: {onnx_b_path} ({onnx_b_path.stat().st_size / 1024:.1f} KB)")

    logger.info("=" * 80)
    logger.info("PHASE 11: TRAINING ABLATION C (ANCHORED + UNIFORM LOSS)")
    logger.info("=" * 80)

    model_c, c_val_rmse = train_single_model(
        experiment_name="Ablation C (Anchored + Uniform Loss)",
        use_speed_weighted_loss=False,
        epochs=8,
        batch_size=256,
        learning_rate=1e-3,
        device_str="cpu"
    )

    pt_c_path = config.MODELS_DIR / "phase11_ablation_uniform_net.pt"
    torch.save(model_c.state_dict(), pt_c_path)
    logger.info(f"Saved Ablation C checkpoint to {pt_c_path} (Best Val RMSE: {c_val_rmse:.4f} m/s)")


if __name__ == "__main__":
    main()
