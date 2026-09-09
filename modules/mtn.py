"""
iNAV Motion Transformation Network (MTN) (Pillar 3)
Dedicated TCN-based neural pose alignment architecture:
Decouples vehicle frame orientation estimation (Euler angles: roll, pitch, yaw)
from forward speed regression, preventing coordinate rotation errors from
contaminating displacement integration.
"""

import logging
import math
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger("iNAV.mtn")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed; MTN defined conditionally.")


if HAS_TORCH:
    class Chomp1d(nn.Module):
        """Chomp module to ensure causality in temporal convolutions."""
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


    class TemporalBlock(nn.Module):
        """Residual Dilated Temporal Convolutional Block."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.chomp1 = Chomp1d(padding)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.act1 = nn.LeakyReLU(0.1, inplace=True)
            self.drop1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.chomp2 = Chomp1d(padding)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.act2 = nn.LeakyReLU(0.1, inplace=True)
            self.drop2 = nn.Dropout(dropout)

            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv1(x)
            out = self.chomp1(out)
            out = self.bn1(out)
            out = self.act1(out)
            out = self.drop1(out)

            out = self.conv2(out)
            out = self.chomp2(out)
            out = self.bn2(out)
            out = self.act2(out)
            out = self.drop2(out)

            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)


    class MotionTransformationNetwork(nn.Module):
        """
        Motion Transformation Network (MTN).
        Inputs: (Batch, 6, Time) IMU readings (accel + gyro)
        Outputs:
          - euler_angles: (Batch, 3) representing [roll (alpha), pitch (beta), yaw (gamma)] in radians
          - R_b_to_v: (Batch, 3, 3) SO(3) phone-body to vehicle rotation matrix
        """
        def __init__(
            self,
            in_channels: int = 6,
            num_channels: Tuple[int, ...] = (32, 64, 64),
            kernel_size: int = 3,
            dropout: float = 0.1
        ):
            super().__init__()
            self.in_bn = nn.BatchNorm1d(in_channels)
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_c = in_channels if i == 0 else num_channels[i - 1]
                out_c = num_channels[i]
                padding = (kernel_size - 1) * dilation_size
                layers.append(
                    TemporalBlock(in_c, out_c, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout)
                )

            self.tcn = nn.Sequential(*layers)

            # Global sequence aggregation
            self.pool = nn.AdaptiveAvgPool1d(1)

            # Angle regression head: produces 3 Euler angles
            # Roll (alpha), Pitch (beta), Yaw offset (gamma)
            self.fc = nn.Sequential(
                nn.Linear(num_channels[-1], 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(32, 3),
                nn.Tanh()  # Output in [-1, 1]
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x: (Batch, 6, Time)
            Returns:
              euler_angles: (Batch, 3) in [-pi, pi]
              R_mat: (Batch, 3, 3) rotation matrix
            """
            B = x.shape[0]
            x_norm = self.in_bn(x)
            feat = self.tcn(x_norm)
            feat = self.pool(feat).squeeze(-1)  # (Batch, 64)

            # Scale tanh outputs to [-pi, pi] for roll/yaw, [-pi/2, pi/2] for pitch
            raw = self.fc(feat)
            roll = raw[:, 0:1] * math.pi
            pitch = raw[:, 1:2] * (math.pi / 2.0)
            yaw = raw[:, 2:3] * math.pi
            euler_angles = torch.cat([roll, pitch, yaw], dim=1)  # (Batch, 3)

            # Construct 3x3 rotation matrix R_z(yaw) @ R_y(pitch) @ R_x(roll)
            cr, sr = torch.cos(roll), torch.sin(roll)
            cp, sp = torch.cos(pitch), torch.sin(pitch)
            cy, sy = torch.cos(yaw), torch.sin(yaw)

            # Analytical Euler to Rotation Matrix in batch
            r00 = cy * cp
            r01 = cy * sp * sr - sy * cr
            r02 = cy * sp * cr + sy * sr

            r10 = sy * cp
            r11 = sy * sp * sr + cy * cr
            r12 = sy * sp * cr - cy * sr

            r20 = -sp
            r21 = cp * sr
            r22 = cp * cr

            row0 = torch.cat([r00, r01, r02], dim=1).unsqueeze(1)  # (B, 1, 3)
            row1 = torch.cat([r10, r11, r12], dim=1).unsqueeze(1)  # (B, 1, 3)
            row2 = torch.cat([r20, r21, r22], dim=1).unsqueeze(1)  # (B, 1, 3)

            R_mat = torch.cat([row0, row1, row2], dim=1)  # (B, 3, 3)
            return euler_angles, R_mat
