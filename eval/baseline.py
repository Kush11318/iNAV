"""
iNAV Baseline Dead-Reckoning Models
Implements classical non-AI dead reckoning baselines:
1. Raw Strapdown IMU Double-Integration (accelerometer integration -> t^2 error explosion)
2. Constant Velocity Model (freezes last GNSS speed and propagates along gyro heading)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from eval.score import latlon_to_local_xy_m, score_outage_segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.baseline")


def local_xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    lat0: float,
    lon0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse equirectangular projection from local Cartesian (x, y) in meters
    back to (lat, lon) in degrees.
    """
    R = 6371000.0
    phi0 = np.radians(lat0)
    lat = lat0 + np.degrees(y / R)
    lon = lon0 + np.degrees(x / (R * np.cos(phi0)))
    return lat, lon


def run_strapdown_baseline(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical strapdown dead reckoning:
    Integrate gyroscope yaw to update heading,
    integrate raw accelerometer to update speed,
    integrate speed to update position.
    Demonstrates quadratic drift accumulation (t^2).
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # Inputs
    acc_x = df_outage[config.COL_ACC_X].values
    gyro_z = df_outage[config.COL_GYRO_Z].values

    # Pre-allocate
    x = np.zeros(n)
    y = np.zeros(n)
    speed = np.zeros(n)
    heading_deg = np.zeros(n)

    speed[0] = max(init_speed_ms, 0.0)
    heading_deg[0] = init_heading_deg

    for k in range(1, n):
        # Heading integration (degrees)
        # Note: Gyro Z is in rad/s, convert to deg/s
        heading_deg[k] = (heading_deg[k - 1] + np.degrees(gyro_z[k] * dt)) % 360.0

        # Speed integration (m/s)
        speed[k] = max(speed[k - 1] + acc_x[k] * dt, 0.0)

        # Position step in navigation frame (heading: 0=N, 90=E)
        h_rad = np.radians(heading_deg[k])
        dx = speed[k] * np.sin(h_rad) * dt  # East
        dy = speed[k] * np.cos(h_rad) * dt  # North

        x[k] = x[k - 1] + dx
        y[k] = y[k - 1] + dy

    est_lat, est_lon = local_xy_to_latlon(x, y, init_lat, init_lon)
    return est_lat, est_lon, speed


def run_constant_velocity_baseline(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constant velocity dead reckoning baseline:
    Assumes speed remains constant throughout blackout,
    propagates along gyroscope-integrated heading.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    gyro_z = df_outage[config.COL_GYRO_Z].values

    x = np.zeros(n)
    y = np.zeros(n)
    heading_deg = np.zeros(n)
    speed = np.full(n, max(init_speed_ms, 0.0))
    heading_deg[0] = init_heading_deg

    for k in range(1, n):
        heading_deg[k] = (heading_deg[k - 1] + np.degrees(gyro_z[k] * dt)) % 360.0
        h_rad = np.radians(heading_deg[k])
        dx = speed[k] * np.sin(h_rad) * dt
        dy = speed[k] * np.cos(h_rad) * dt

        x[k] = x[k - 1] + dx
        y[k] = y[k - 1] + dy

    est_lat, est_lon = local_xy_to_latlon(x, y, init_lat, init_lon)
    return est_lat, est_lon, speed


if __name__ == "__main__":
    from eval.replay import run_benchmark_suite
    logger.info("Executing baseline dead-reckoning benchmark suite...")
    run_benchmark_suite(methods=["strapdown", "constant_velocity"], test_runs_limit=5)

