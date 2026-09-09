"""
iNAV VelocityNet End-to-End Training Pipeline
Trains the multi-head displacement, event classifier, and learned uncertainty
network using Huber + Gaussian NLL + Cross-Entropy loss on IO-VNBD windowed datasets.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config
from modules.velocity_net import VelocityNet
from modules.augmentation import apply_batch_3d_spatial_rotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.train")


def dynamic_latency_huber_loss(
    pred_d: "torch.Tensor",
    target_d: "torch.Tensor",
    huber_fn: "torch.nn.SmoothL1Loss"
) -> "torch.Tensor":
    """
    Sliding-window dynamic loss matcher for GNSS latency (Pillar 3).
    Prevents penalizing the network when 1Hz GNSS logs lag behind 10Hz IMU events by ~1s.
    L = min(L(x[1:], y[1:]), L(x[:-1], y[1:]))
    """
    import torch
    if len(pred_d) > 2:
        # Standard unshifted loss
        l_curr = huber_fn(pred_d[1:], target_d[1:])
        # Shifted by 1 step (0.2s or 0.1s lag compensation)
        l_lag = huber_fn(pred_d[:-1], target_d[1:])
        return torch.min(l_curr, l_lag)
    return huber_fn(pred_d, target_d)


def train_velocity_net(
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    subsample_ratio: float = 0.5,
    device_str: str = "cpu",
    architecture: str = "spectra",  # 'spectra' (Pillar 2) or 'baseline_vnet'
    augment_3d_rot: bool = True     # Pillar 1 3D SO(3) rotation augmentation
) -> Path:
    """
    Train VelocityNet or SpectraNet on train_windows.npz and evaluate on val_windows.npz.
    Includes 3D spatial rotation augmentation and dynamic GNSS latency loss matching.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device(device_str)
    logger.info(f"Training on device: {device} | Architecture: {architecture} | 3D Augmentation: {augment_3d_rot}")

    train_path = config.WINDOWED_DIR / "train_windows.npz"
    val_path = config.WINDOWED_DIR / "val_windows.npz"

    logger.info(f"Loading data from {train_path.name} and {val_path.name}...")
    train_npz = np.load(train_path)
    val_npz = np.load(val_path)

    # Subsample if requested for fast screening iteration
    n_train = len(train_npz["y_delta_d"])
    if subsample_ratio < 1.0:
        idx_train = np.random.choice(n_train, int(n_train * subsample_ratio), replace=False)
    else:
        idx_train = np.arange(n_train)

    X_train = train_npz["X_seq"][idx_train]
    y_d_train = train_npz["y_delta_d"][idx_train]
    y_ev_train = train_npz["y_event"][idx_train]

    X_val = val_npz["X_seq"]
    y_d_val = val_npz["y_delta_d"]
    y_ev_val = val_npz["y_event"]

    # Transpose for Conv1d / Conv2d: (N, Time=20, Channels=6) -> (N, Channels=6, Time=20)
    X_train_t = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    y_d_train_t = torch.from_numpy(y_d_train).float().unsqueeze(1)
    y_ev_train_t = torch.from_numpy(y_ev_train).long()

    X_val_t = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    y_d_val_t = torch.from_numpy(y_d_val).float().unsqueeze(1)
    y_ev_val_t = torch.from_numpy(y_ev_val).long()

    logger.info(f"Train samples: {len(X_train_t)}, Validation samples: {len(X_val_t)}")

    train_loader = DataLoader(TensorDataset(X_train_t, y_d_train_t, y_ev_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_d_val_t, y_ev_val_t), batch_size=batch_size * 2, shuffle=False)

    model = VelocityNet(in_channels=6, num_events=5).to(device)
    best_model_path = config.MODELS_DIR / "velocity_net_best.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_d_mae = 0.0

        for batch_x, batch_d, batch_ev in train_loader:
            batch_x = batch_x.to(device)
            batch_d = batch_d.to(device)
            batch_ev = batch_ev.to(device)

            # Pillar 1: Apply 3D Spatial Rotation Augmentation in SO(3)
            if augment_3d_rot:
                batch_x = apply_batch_3d_spatial_rotation(batch_x, prob=0.5)

            optimizer.zero_grad()
            pred_d, pred_ev, pred_sig = model(batch_x)

            # Pillar 3: Dynamic latency-tolerant Huber displacement loss
            loss_d = dynamic_latency_huber_loss(pred_d, batch_d, huber_loss_fn)

            # Head 3: Gaussian Negative Log-Likelihood for uncertainty (pred_d detached)
            loss_nll = torch.mean(0.5 * ((batch_d - pred_d.detach())**2 / (pred_sig**2)) + torch.log(pred_sig))

            # Head 2: Cross-entropy event classification
            loss_ev = ce_loss_fn(pred_ev, batch_ev)

            # Combined multi-task loss (lambda = 0.7 on displacement/velocity)
            total_loss = 0.7 * loss_d + 0.05 * loss_nll + 0.1 * loss_ev
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += total_loss.item() * len(batch_x)
            train_d_mae += torch.sum(torch.abs(pred_d - batch_d)).item()

        scheduler.step()
        train_loss /= len(X_train_t)
        train_d_mae /= len(X_train_t)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_d_mae = 0.0
        val_ev_acc = 0.0
        mean_sig = 0.0

        with torch.no_grad():
            for batch_x, batch_d, batch_ev in val_loader:
                batch_x = batch_x.to(device)
                batch_d = batch_d.to(device)
                batch_ev = batch_ev.to(device)

                pred_d, pred_ev, pred_sig = model(batch_x)
                loss_d = huber_loss_fn(pred_d, batch_d)
                loss_nll = torch.mean(0.5 * ((batch_d - pred_d)**2 / (pred_sig**2)) + torch.log(pred_sig))
                loss_ev = ce_loss_fn(pred_ev, batch_ev)

                total_loss = loss_d + 0.05 * loss_nll + 0.1 * loss_ev
                val_loss += total_loss.item() * len(batch_x)
                val_d_mae += torch.sum(torch.abs(pred_d - batch_d)).item()
                val_ev_acc += torch.sum(torch.argmax(pred_ev, dim=1) == batch_ev).item()
                mean_sig += torch.sum(pred_sig).item()

        val_loss /= len(X_val_t)
        val_d_mae /= len(X_val_t)
        val_ev_acc /= len(X_val_t)
        mean_sig /= len(X_val_t)

        logger.info(
            f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f}, Train Δd MAE: {train_d_mae:.3f}m | "
            f"Val Loss: {val_loss:.4f}, Val Δd MAE: {val_d_mae:.3f}m, Event Acc: {val_ev_acc*100:.1f}%, Mean σ: {mean_sig:.2f}m"
        )

        if val_d_mae < best_val_mae:
            best_val_mae = val_d_mae
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model checkpoint to {best_model_path.name} (Val MAE: {val_d_mae:.3f}m)")

    # Export to ONNX for edge deployment
    onnx_path = config.MODELS_DIR / "velocity_net.onnx"
    try:
        dummy_input = torch.randn(1, 6, 20, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["imu_window"],
            output_names=["delta_d", "event_logits", "sigma"],
            dynamic_axes={"imu_window": {0: "batch_size"}, "delta_d": {0: "batch_size"}},
            opset_version=14
        )
        logger.info(f"Exported model to ONNX: {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX export deferred: {e}")

    return best_model_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV VelocityNet Training")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--subsample", type=float, default=0.5, help="Fraction of train data to use for fast screening")
    parser.add_argument("--no-augment", action="store_true", help="Disable 3D SO(3) rotation augmentation")
    args = parser.parse_args()

    train_velocity_net(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subsample_ratio=args.subsample,
        augment_3d_rot=not args.no_augment
    )
