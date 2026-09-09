"""
iNAV SPECTRA Neural Architecture (Pillar 2)
State-of-the-Art Deep Inertial Dead-Reckoning Engine:
- Quadrature STFT Time-Frequency Front-End (Gabor wavelet filterbank)
- Depthwise Separable 2D Convolutions (90% MAC reduction vs dense 1D convs)
- Channel-Wise Self-Attention across 6 IMU axes (O(C^2 D) cross-axis coupling)
- Bidirectional GRU with Learned Attention Pooling (dynamic focus on maneuvers)
- Multi-Head Prediction: Displacement Δd, 5-Class Event, Learned Uncertainty σ(Δd)
"""

import logging
from typing import Tuple, Optional, Dict
import math
import numpy as np

logger = logging.getLogger("iNAV.spectra_net")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed; SpectraNet defined conditionally.")


if HAS_TORCH:
    class QuadratureSTFTFrontEnd(nn.Module):
        """
        Differentiable Quadrature STFT Filter Bank Layer.
        Transforms raw (Batch, 6, Time) IMU sequences into (Batch, 6, N_freq, T_frames)
        spectrogram magnitude maps using cosine and sine Gabor wavelet bases.
        Fully compatible with standard ONNX Runtime operators.
        """
        def __init__(self, n_fft: int = 8, hop_length: int = 2):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_freq = n_fft // 2 + 1

            # Construct discrete Fourier basis
            n = torch.arange(0, n_fft, dtype=torch.float32)
            k = torch.arange(0, self.n_freq, dtype=torch.float32).unsqueeze(1)
            
            # Hann window for spectral leakage reduction
            window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / (n_fft - 1)))
            
            # Cosine (real) and Sine (imag) filter weights
            cos_weights = window * torch.cos(2.0 * math.pi * k * n / n_fft)
            sin_weights = window * torch.sin(2.0 * math.pi * k * n / n_fft)
            
            # Pack into 1D convolution kernel of shape (n_freq * 2, 1, n_fft)
            filters = torch.cat([cos_weights, sin_weights], dim=0).unsqueeze(1)
            self.register_buffer("filters", filters)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Input x: (Batch, Channels=6, Time)
            Output: (Batch, Channels=6, N_freq, T_frames) magnitude spectrograms
            """
            B, C, T = x.shape
            # Flatten channels into batch: (B*C, 1, T)
            x_reshaped = x.reshape(B * C, 1, T)
            
            # Conv1d acts as STFT sliding window
            # Output: (B*C, 2 * n_freq, T_frames)
            out = F.conv1d(x_reshaped, self.filters, stride=self.hop_length, padding=self.n_fft // 2)
            
            real = out[:, :self.n_freq, :]
            imag = out[:, self.n_freq:, :]
            
            # Spectrogram magnitude: sqrt(real^2 + imag^2 + eps)
            mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-7)
            
            T_frames = mag.shape[-1]
            return mag.reshape(B, C, self.n_freq, T_frames)


    class DepthwiseSeparableConv2d(nn.Module):
        """
        Depthwise Separable 2D Convolution across time-frequency planes.
        Cuts computation by ~85-90% relative to standard 2D convolution.
        """
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size,
                padding=padding, groups=in_channels, bias=False
            )
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            return self.act(x)


    class ChannelSelfAttention(nn.Module):
        """
        Channel-wise Self-Attention for Sensor Cross-Axis Coupling.
        Cost is O(C^2 D) which is ultra-efficient for C=6 IMU axes.
        Models mechanical coupling between accelerometers and gyroscopes.
        """
        def __init__(self, channels: int = 6):
            super().__init__()
            self.channels = channels
            self.query = nn.Linear(channels, channels)
            self.key = nn.Linear(channels, channels)
            self.value = nn.Linear(channels, channels)
            self.scale = 1.0 / math.sqrt(channels)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x shape: (Batch, Channels=6, Features)
            """
            # Compute Q, K, V across channel dimension
            q = self.query(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, F)
            k = self.key(x.permute(0, 2, 1)).permute(0, 2, 1)
            v = self.value(x.permute(0, 2, 1)).permute(0, 2, 1)

            # Affinity: (B, C, C) = (B, C, F) @ (B, F, C)
            attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Attended values: (B, C, C) @ (B, C, F) -> (B, C, F)
            attended = torch.bmm(attn_weights, v)
            return x + self.gamma * attended


    class AttentionPooling(nn.Module):
        """
        Learned Attention Pooling over temporal hidden sequence:
        alpha_l = softmax(w^T h_l)
        c = sum_l alpha_l h_l
        Dynamically focuses on transient maneuvers (braking, acceleration, turns).
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.w = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            h: (Batch, Time, HiddenDim)
            Returns:
              context: (Batch, HiddenDim)
              weights: (Batch, Time)
            """
            scores = self.w(h).squeeze(-1)  # (Batch, Time)
            weights = F.softmax(scores, dim=-1)  # (Batch, Time)
            context = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (Batch, HiddenDim)
            return context, weights


    class SpectraNet(nn.Module):
        """
        Complete SPECTRA Architecture (Pillar 2):
        STFT -> Depthwise Separable 2D Convs -> Channel Self-Attention -> Bi-GRU -> Attention Pooling -> Multi-Head
        """
        def __init__(
            self,
            in_channels: int = 6,
            num_events: int = 5,
            n_fft: int = 8,
            hop_length: int = 2,
            gru_hidden: int = 64
        ):
            super().__init__()
            self.in_channels = in_channels
            self.num_events = num_events

            # Input normalization layer
            self.in_bn = nn.BatchNorm1d(in_channels)

            # 1. STFT Front-End
            self.stft = QuadratureSTFTFrontEnd(n_fft=n_fft, hop_length=hop_length)

            # 2. Depthwise Separable 2D Convolution Stages
            self.conv1 = DepthwiseSeparableConv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1)

            # 3. Channel Self-Attention over raw sensor channels
            self.channel_attn = ChannelSelfAttention(channels=in_channels)

            # Spatial projection to feature vectors per time step
            self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # pool frequency bins -> (B, 64, 1, T_spec)

            # 4. Bidirectional GRU for temporal dynamics
            # Feature size from conv2 pooled: 64
            self.gru = nn.GRU(
                input_size=64,
                hidden_size=gru_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            # Bi-GRU output size: gru_hidden * 2 = 128

            # 5. Learned Attention Pooling
            self.attn_pool = AttentionPooling(hidden_dim=gru_hidden * 2)

            # 6. Multi-Task Heads
            # Head 1: Forward displacement regression
            self.head_displacement = nn.Sequential(
                nn.Linear(gru_hidden * 2, 64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(64, 1)
            )
            nn.init.constant_(self.head_displacement[-1].bias, 15.0)

            # Head 2: Event Classifier
            self.head_event = nn.Sequential(
                nn.Linear(gru_hidden * 2, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(32, num_events)
            )

            # Head 3: Learned Uncertainty
            self.head_uncertainty = nn.Sequential(
                nn.Linear(gru_hidden * 2, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(32, 1),
                nn.Softplus()
            )
            nn.init.constant_(self.head_uncertainty[-2].bias, 1.0)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            x shape: (Batch, Channels=6, Time=20 or 40)
            Returns:
              - delta_d: (Batch, 1) in meters
              - event_logits: (Batch, 5)
              - sigma: (Batch, 1) standard deviation in meters
            """
            B, C, T = x.shape

            # Normalize raw sensor values
            x_norm = self.in_bn(x)

            # Apply Channel Self-Attention across the 6 IMU axes
            x_attn = self.channel_attn(x_norm)

            # Compute STFT Time-Frequency Spectrogram: (B, 6, N_freq, T_frames)
            spec = self.stft(x_attn)

            # Depthwise Separable 2D Convolutions
            feat2d = self.conv1(spec)
            feat2d = self.conv2(feat2d)  # (B, 64, N_freq, T_frames)

            # Frequency pooling -> (B, 64, 1, T_frames) -> (B, 64, T_frames)
            feat_seq = self.freq_pool(feat2d).squeeze(2)

            # Permute for GRU: (B, T_frames, 64)
            feat_seq = feat_seq.permute(0, 2, 1)

            gru_out, _ = self.gru(feat_seq)  # (B, T_frames, 128)

            # Learned Attention Pooling across time frames
            context, _ = self.attn_pool(gru_out)  # (B, 128)

            # Multi-Head Outputs
            delta_d = torch.clamp(self.head_displacement(context), min=0.0)
            event_logits = self.head_event(context)
            sigma = self.head_uncertainty(context) + 1e-3

            return delta_d, event_logits, sigma


class SpectraNetPredictor:
    """
    Inference Wrapper for SpectraNet (Pillar 2).
    Accepts NumPy IMU window of shape (20, 6) or (40, 6) and outputs (delta_d, event_class, sigma).
    """
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None

        if HAS_TORCH:
            self.model = SpectraNet().to(self.device)
            if model_path:
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
                    self.model.load_state_dict(state)
                    logger.info(f"Loaded trained SpectraNet weights from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load SpectraNet weights from {model_path}: {e}")
            self.model.eval()

    def predict(self, window_imu: np.ndarray) -> Tuple[float, int, float]:
        """
        Predict for a single window:
        window_imu: shape (Time, 6) -> acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        Returns:
          (delta_d_meters, event_class_id, uncertainty_sigma_meters)
        """
        if self.model is not None:
            with torch.no_grad():
                # Prepare tensor: (1, 6, Time)
                tensor_x = torch.from_numpy(np.ascontiguousarray(window_imu.T)).unsqueeze(0).float().to(self.device)
                d_d, ev_logits, sig = self.model(tensor_x)

                pred_d = float(d_d.cpu().item())
                pred_ev = int(torch.argmax(ev_logits, dim=1).cpu().item())
                pred_sig = float(sig.cpu().item())
                return pred_d, pred_ev, pred_sig
        else:
            acc_rms = np.sqrt(np.mean(window_imu[:, 0]**2 + window_imu[:, 1]**2))
            est_speed = np.clip(acc_rms * 1.5, 0.0, 30.0)
            pred_d = est_speed * 2.0
            pred_ev = 1 if est_speed > 1.0 else 0
            pred_sig = 0.5 + 0.1 * pred_d
            return float(pred_d), int(pred_ev), float(pred_sig)

