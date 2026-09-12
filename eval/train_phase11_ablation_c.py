"""
Phase 11: Training Script for Ablation C (Anchored Model with Uniform Loss Weighting)
Distinguishes the effect of speed anchoring from speed-aware loss weighting.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from modules.augmentation import apply_batch_3d_spatial_rotation
from eval.train_phase11_anchored_model import AnchoredTemporalVelocityNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase11_AblationC")


def train_ablation_c(epochs: int = 4, batch_size: int = 512, lr: float = 1e-3, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    logger.info("=" * 80)
    logger.info(f"TRAINING PHASE 11 ABLATION C: ANCHORED + UNIFORM LOSS (Epochs: {epochs}, Batch Size: {batch_size})")
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

    train_x_t = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    train_anc_t = torch.from_numpy(v_anc_train).float()
    train_del_v_t = torch.from_numpy(y_delta_v_train).float()
    train_vel_t = torch.from_numpy(y_vel_train).float()
    train_ev_t = torch.from_numpy(y_ev_train).long()

    val_x_t = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    val_anc_t = torch.from_numpy(v_anc_val).float()
    val_del_v_t = torch.from_numpy(y_delta_v_val).float()
    val_vel_t = torch.from_numpy(y_vel_val).float()
    val_ev_t = torch.from_numpy(y_ev_val).long()

    train_dataset = TensorDataset(train_x_t, train_anc_t, train_del_v_t, train_vel_t, train_ev_t)
    val_dataset = TensorDataset(val_x_t, val_anc_t, val_del_v_t, val_vel_t, val_ev_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AnchoredTemporalVelocityNet(in_channels=6, num_events=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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

        for bx, b_anc, b_del_v, b_vel, b_ev in train_loader:
            bx = apply_batch_3d_spatial_rotation(bx.to(device), prob=0.5)
            b_anc = b_anc.to(device)
            b_del_v = b_del_v.to(device)
            b_ev = b_ev.to(device)

            optimizer.zero_grad()
            del_v_pred, v_pred, sig_pred, ev_logits = model(bx, b_anc)

            # Uniform loss (w = 1.0)
            l_vel = F.smooth_l1_loss(del_v_pred, b_del_v, beta=1.0, reduction='mean')
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
                val_pts += len(bx)

        val_rmse = np.sqrt(np.sum(val_sq_err) / val_pts)
        val_mae = np.sum(val_abs_err) / val_pts

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/n_batches:.4f} (Vel: {total_vel_loss/n_batches:.4f}) | "
            f"Val Vel RMSE: {val_rmse:.4f} m/s | MAE: {val_mae:.4f} m/s"
        )

        if val_rmse < best_val_vel_rmse:
            best_val_vel_rmse = val_rmse
            best_state = model.state_dict().copy()

    save_path = config.MODELS_DIR / "phase11_ablation_uniform_net.pt"
    if best_state is not None:
        torch.save(best_state, save_path)
    else:
        torch.save(model.state_dict(), save_path)

    logger.info(f"Saved Ablation C checkpoint to {save_path} (Best Val RMSE: {best_val_vel_rmse:.4f} m/s)")


if __name__ == "__main__":
    train_ablation_c(epochs=4, batch_size=512)
