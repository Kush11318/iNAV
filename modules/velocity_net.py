"""
iNAV VelocityNet Neural Architecture (Module B)
Multi-head deep learning displacement and uncertainty predictor:
- 1D Dilated Convolutions (multi-scale vibration features)
- Bidirectional GRU (temporal sequence context)
- Head 1: Forward displacement Δd regression (Huber loss)
- Head 2: Event classification (5 driving states)
- Head 3: Learned uncertainty σ(Δd) (Gaussian negative log-likelihood)
"""

import logging
from typing import Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger("iNAV.velocity_net")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not yet imported; VelocityNet PyTorch model class defined conditionally.")


if HAS_TORCH:
    class VelocityNet(nn.Module):
        """
        Multi-Head 1D-CNN + GRU Displacement Network.
        Input: Tensor of shape (Batch, Channels=6, Time=20)
        """

        def __init__(self, in_channels: int = 6, num_events: int = 5):
            super().__init__()
            self.in_channels = in_channels
            self.num_events = num_events

            # Input normalization
            self.in_bn = nn.BatchNorm1d(in_channels)

            # Multi-scale Dilated 1D Convolutions
            self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, dilation=1)
            self.bn1 = nn.BatchNorm1d(32)

            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2)
            self.bn2 = nn.BatchNorm1d(64)

            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4)
            self.bn3 = nn.BatchNorm1d(128)

            # Bidirectional GRU for sequential temporal dynamics
            self.gru = nn.GRU(
                input_size=128,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            # Output features from bidirectional GRU: 64 * 2 = 128

            # Head 1: Forward Displacement Regression (LeakyReLU -> Linear with positive bias init)
            self.head_displacement = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 1)
            )
            # Initialize displacement bias to 15.0m (typical 2s window displacement at 27 km/h)
            nn.init.constant_(self.head_displacement[-1].bias, 15.0)

            # Head 2: Driving Event Classifier
            self.head_event = nn.Sequential(
                nn.Linear(128, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, num_events)
            )

            # Head 3: Learned Uncertainty σ(Δd) > 0
            self.head_uncertainty = nn.Sequential(
                nn.Linear(128, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1),
                nn.Softplus()  # Strictly positive standard deviation
            )
            nn.init.constant_(self.head_uncertainty[-2].bias, 1.0)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            x shape: (Batch, 6, 20)
            Returns:
              - delta_d: (Batch, 1) in meters
              - event_logits: (Batch, 5)
              - sigma: (Batch, 1) standard deviation in meters
            """
            # Normalize input channels
            x_norm = self.in_bn(x)

            # Convolutions with dilation
            h = F.leaky_relu(self.bn1(self.conv1(x_norm)), 0.1)
            h = F.leaky_relu(self.bn2(self.conv2(h)), 0.1)
            h = F.leaky_relu(self.bn3(self.conv3(h)), 0.1)

            # Permute for GRU: (Batch, 128, 20) -> (Batch, 20, 128)
            h_perm = h.permute(0, 2, 1)
            gru_out, _ = self.gru(h_perm)

            # Global average pooling across time
            feat = torch.mean(gru_out, dim=1)  # (Batch, 128)

            delta_d = torch.clamp(self.head_displacement(feat), min=0.0)
            event_logits = self.head_event(feat)
            sigma = self.head_uncertainty(feat) + 1e-3  # Numerical stability floor

            return delta_d, event_logits, sigma


class VelocityNetPredictor:
    """
    Inference Wrapper for VelocityNet.
    Accepts NumPy IMU window (20, 6) and outputs (delta_d, event_class, sigma).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None

        if HAS_TORCH:
            self.model = VelocityNet().to(self.device)
            if model_path:
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded trained VelocityNet weights from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load weights from {model_path}: {e}")
            self.model.eval()

    def predict(self, window_imu: np.ndarray) -> Tuple[float, int, float]:
        """
        Predict for a single window:
        window_imu: shape (20, 6) -> acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        Returns:
          (delta_d_meters, event_class_id, uncertainty_sigma_meters)
        """
        if self.model is not None:
            with torch.no_grad():
                # Prepare tensor: (1, 6, 20)
                tensor_x = torch.from_numpy(np.ascontiguousarray(window_imu.T)).unsqueeze(0).float().to(self.device)
                d_d, ev_logits, sig = self.model(tensor_x)

                pred_d = float(d_d.cpu().item())
                pred_ev = int(torch.argmax(ev_logits, dim=1).cpu().item())
                pred_sig = float(sig.cpu().item())
                return pred_d, pred_ev, pred_sig
        else:
            # Analytical fallback: RMS energy based estimate
            acc_rms = np.sqrt(np.mean(window_imu[:, 0]**2 + window_imu[:, 1]**2))
            est_speed = np.clip(acc_rms * 1.5, 0.0, 30.0)
            pred_d = est_speed * 2.0
            pred_ev = 1 if est_speed > 1.0 else 0
            pred_sig = 0.5 + 0.1 * pred_d
            return float(pred_d), int(pred_ev), float(pred_sig)
