"""
Phase 7D: Controlled VelocityNet Retraining Experiment
Trains an experimental VelocityNet without dynamic_latency_huber_loss,
using standard Huber loss and a speed-balanced training sampler.

Strict Isolation:
- Does NOT overwrite models/velocity_net_best.pt
- Does NOT overwrite models/velocity_net.onnx
- Saves to models/phase7d_velocity_net_clean.pt and models/phase7d_velocity_net_clean.onnx
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
from modules.velocity_net import VelocityNet
from modules.augmentation import apply_batch_3d_spatial_rotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase7D")


def compute_speed_balanced_weights(train_speeds: np.ndarray) -> Tuple[torch.Tensor, pd.Series]:
    """
    Compute sample weights based strictly on training set speed distribution.
    Uses inverse frequency of discrete speed bins.
    """
    bins = [0, 5, 10, 15, 20, 25, 30, 1000]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    cat = pd.cut(train_speeds, bins=bins, labels=labels, right=False)
    counts = pd.Series(cat).value_counts().sort_index()
    
    # Calculate inverse-frequency weight per bin
    # We use inverse square root of counts to avoid over-amplifying noise in sparse bins
    # while providing strong boost to high-speed regimes
    bin_weights = {label: 1.0 / np.sqrt(count) if count > 0 else 0.0 for label, count in counts.items()}
    
    # Map back to each individual sample
    sample_weights = np.array([bin_weights[c] for c in cat], dtype=np.float32)
    # Normalize weights so sum equals number of samples
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    
    return torch.from_numpy(sample_weights), counts


def run_phase7d_training(
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42
) -> Tuple[Path, Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(device_str)
    logger.info(f"Phase 7D Training on device: {device} | Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size}")
    
    train_path = config.WINDOWED_DIR / "train_windows.npz"
    val_path = config.WINDOWED_DIR / "val_windows.npz"
    
    train_npz = np.load(train_path)
    val_npz = np.load(val_path)
    
    X_train = train_npz["X_seq"] # (N, 20, 6)
    y_d_train = train_npz["y_delta_d"]
    y_ev_train = train_npz["y_event"]
    y_spd_train = train_npz["y_speed"]
    
    X_val = val_npz["X_seq"]
    y_d_val = val_npz["y_delta_d"]
    y_ev_val = val_npz["y_event"]
    y_spd_val = val_npz["y_speed"]
    
    # Transpose for Conv1d: (N, 20, 6) -> (N, 6, 20)
    X_train_t = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    y_d_train_t = torch.from_numpy(y_d_train).float().unsqueeze(1)
    y_ev_train_t = torch.from_numpy(y_ev_train).long()
    
    X_val_t = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    y_d_val_t = torch.from_numpy(y_d_val).float().unsqueeze(1)
    y_ev_val_t = torch.from_numpy(y_ev_val).long()
    
    # Compute speed-balanced sampling weights
    sample_weights, bin_counts = compute_speed_balanced_weights(y_spd_train)
    logger.info("Training set raw speed distribution:")
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
    
    # Instantiate identical VelocityNet architecture
    model = VelocityNet(in_channels=6, num_events=5).to(device)
    
    clean_pt_path = config.MODELS_DIR / "phase7d_velocity_net_clean.pt"
    clean_onnx_path = config.MODELS_DIR / "phase7d_velocity_net_clean.onnx"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # STANDARD HUBER LOSS (Beta = 1.0) - REMOVED dynamic_latency_huber_loss
    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    best_val_mae = float("inf")
    
    logger.info("Beginning Phase 7D Training Epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_d_mae = 0.0
        
        for batch_x, batch_d, batch_ev in train_loader:
            batch_x = batch_x.to(device)
            batch_d = batch_d.to(device)
            batch_ev = batch_ev.to(device)
            
            # Apply SO(3) 3D rotation augmentation
            batch_x = apply_batch_3d_spatial_rotation(batch_x, prob=0.5)
            
            optimizer.zero_grad()
            pred_d, pred_ev, pred_sig = model(batch_x)
            
            # STANDARD UNCORRUPTED HUBER LOSS ON DISPLACEMENT
            loss_d = huber_loss_fn(pred_d, batch_d)
            
            # Gaussian NLL for uncertainty head
            loss_nll = torch.mean(0.5 * ((batch_d - pred_d.detach())**2 / (pred_sig**2)) + torch.log(pred_sig))
            
            # Cross-Entropy for event classification
            loss_ev = ce_loss_fn(pred_ev, batch_ev)
            
            # Multitask loss
            total_loss = 0.7 * loss_d + 0.05 * loss_nll + 0.1 * loss_ev
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += total_loss.item() * len(batch_x)
            train_d_mae += torch.sum(torch.abs(pred_d - batch_d)).item()
        
        scheduler.step()
        train_loss /= len(X_train_t)
        train_d_mae /= len(X_train_t)
        
        # Validation Evaluation
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
        
        # Save best model checkpoint
        if val_d_mae < best_val_mae:
            best_val_mae = val_d_mae
            torch.save(model.state_dict(), clean_pt_path)
            logger.info(f"Saved new best Phase 7D checkpoint to {clean_pt_path.name} (Val MAE: {val_d_mae:.3f}m)")
    
    # Save final model state if needed
    if not clean_pt_path.exists():
        torch.save(model.state_dict(), clean_pt_path)
    
    # Export to ONNX
    try:
        model.load_state_dict(torch.load(clean_pt_path, map_location=device))
        model.eval()
        dummy_input = torch.randn(1, 6, 20, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            str(clean_onnx_path),
            input_names=["imu_window"],
            output_names=["delta_d", "event_logits", "sigma"],
            dynamic_axes={"imu_window": {0: "batch_size"}, "delta_d": {0: "batch_size"}},
            opset_version=14
        )
        logger.info(f"Exported Phase 7D clean ONNX to {clean_onnx_path.name}")
    except Exception as e:
        logger.warning(f"ONNX export exception: {e}")
        
    return clean_pt_path, clean_onnx_path


if __name__ == "__main__":
    run_phase7d_training(epochs=8, batch_size=256)
