"""
iNAV 3D Spatial Augmentation & Robustness Module (Pillar 1)
Implements DVSE 3D spatial rotation augmentation in SO(3) to prevent phone-mount
overfitting and simulate arbitrary dashboard, cradle, and pocket postures.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def random_rotation_matrix_so3() -> np.ndarray:
    """
    Sample a uniformly distributed 3D rotation matrix from SO(3)
    using the subgroup algorithm (Shoemake / Arvo uniform Haar measure).
    Returns:
        R: 3x3 orthogonal matrix with det(R) = +1.
    """
    # Sample 3 independent uniform random variables in [0, 1)
    u = np.random.uniform(0.0, 1.0, 3)
    
    # Random rotation around z axis
    theta = 2.0 * np.pi * u[0]
    phi = 2.0 * np.pi * u[1]
    z = u[2]
    
    # Reflector vector for Householder reflection
    r = np.sqrt(z)
    V = np.array([
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(1.0 - z)
    ])
    
    # 2D rotation matrix in plane
    R_z = np.array([
        [np.cos(theta), np.sin(theta), 0.0],
        [-np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Householder matrix H = I - 2 * v * v^T
    H = np.eye(3) - 2.0 * np.outer(V, V)
    
    # M = -H * R_z has determinant +1 and is uniformly distributed on SO(3)
    R = -H @ R_z
    return R.astype(np.float32)


def apply_3d_spatial_rotation(
    imu_seq: np.ndarray,
    R: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Rotate accelerometer and gyroscope 3-axis vectors synchronously.
    imu_seq: array of shape (Time, 6) or (6, Time)
             Channels 0..2: acc_x, acc_y, acc_z
             Channels 3..5: gyro_x, gyro_y, gyro_z
    """
    if R is None:
        R = random_rotation_matrix_so3()
        
    is_channels_first = (imu_seq.shape[0] == 6 and imu_seq.shape[1] != 6)
    if is_channels_first:
        # (6, Time) -> (Time, 6)
        data = imu_seq.T.copy()
    else:
        data = imu_seq.copy()
        
    acc = data[:, 0:3]   # (Time, 3)
    gyro = data[:, 3:6]  # (Time, 3)
    
    # alpha_in = R * alpha_p, omega_in = R * omega_p
    acc_rot = (R @ acc.T).T
    gyro_rot = (R @ gyro.T).T
    
    out = np.hstack([acc_rot, gyro_rot])
    if is_channels_first:
        return out.T
    return out


def apply_sensor_noise_and_scale(
    imu_seq: np.ndarray,
    acc_noise_std: float = 0.05,
    gyro_noise_std: float = 0.005,
    scale_jitter_std: float = 0.02
) -> np.ndarray:
    """
    Apply physical MEMS sensor imperfections:
    - Additive Gaussian noise (thermal white noise)
    - Multiplicative scale factor variation (temperature sensitivity drift)
    """
    out = imu_seq.copy()
    scale_acc = 1.0 + np.random.normal(0.0, scale_jitter_std)
    scale_gyro = 1.0 + np.random.normal(0.0, scale_jitter_std)
    
    if out.shape[0] == 6:
        # Channels first
        out[0:3, :] = out[0:3, :] * scale_acc + np.random.normal(0.0, acc_noise_std, out[0:3, :].shape)
        out[3:6, :] = out[3:6, :] * scale_gyro + np.random.normal(0.0, gyro_noise_std, out[3:6, :].shape)
    else:
        # Time first
        out[:, 0:3] = out[:, 0:3] * scale_acc + np.random.normal(0.0, acc_noise_std, out[:, 0:3].shape)
        out[:, 3:6] = out[:, 3:6] * scale_gyro + np.random.normal(0.0, gyro_noise_std, out[:, 3:6].shape)
        
    return out


if HAS_TORCH:
    def batch_random_rotation_so3(batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a batch of random 3D rotation matrices in SO(3) directly as PyTorch Tensors.
        Returns: (Batch, 3, 3)
        """
        matrices = [random_rotation_matrix_so3() for _ in range(batch_size)]
        arr = np.stack(matrices, axis=0)
        return torch.from_numpy(arr).to(device)

    def apply_batch_3d_spatial_rotation(
        x: torch.Tensor,
        prob: float = 0.5
    ) -> torch.Tensor:
        """
        x: Tensor of shape (Batch, Channels=6, Time)
        Synchronously rotates (ax, ay, az) and (gx, gy, gz) for each sample.
        """
        if prob <= 0.0:
            return x
            
        B, C, T = x.shape
        out = x.clone()
        
        # Decide which samples in batch to rotate
        mask = torch.rand(B, device=x.device) < prob
        n_rot = int(mask.sum().item())
        if n_rot == 0:
            return out
            
        # Sample rotation matrices
        R_batch = batch_random_rotation_so3(n_rot, x.device)  # (n_rot, 3, 3)
        
        acc = out[mask, 0:3, :]   # (n_rot, 3, T)
        gyro = out[mask, 3:6, :]  # (n_rot, 3, T)
        
        # (n_rot, 3, 3) @ (n_rot, 3, T) -> (n_rot, 3, T)
        acc_rot = torch.bmm(R_batch, acc)
        gyro_rot = torch.bmm(R_batch, gyro)
        
        out[mask, 0:3, :] = acc_rot
        out[mask, 3:6, :] = gyro_rot
        return out
