"""
iNAV Sensor Timeseries & Noise Profiler
Visualizes raw accelerometer, gyroscope, and magnetometer signals,
along with frequency spectrum (FFT) analysis of road vibration.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.viz.sensors")


def plot_sensor_diagnostics(
    sync_parquet_path: Path,
    out_png: Optional[Path] = None,
    time_window_sec: float = 60.0
) -> Path:
    """
    Generate comprehensive sensor diagnostics figure:
    1. 3-axis Accelerometer vs Gravity
    2. 3-axis Gyroscope
    3. Vibration Frequency Spectrum (FFT)
    4. Vehicle Speed & Yaw Rate
    """
    df = pd.read_parquet(sync_parquet_path)
    run_key = sync_parquet_path.stem.replace("sync_", "")

    # Pick an active driving window of duration time_window_sec
    if config.COL_TRUE_SPEED_MS in df.columns and (df[config.COL_TRUE_SPEED_MS] > 2.0).any():
        moving_mask = df[config.COL_TRUE_SPEED_MS] > 2.0
        start_t = float(df.loc[moving_mask, config.COL_TIME].iloc[0])
    else:
        start_t = float(df[config.COL_TIME].iloc[0])

    end_t = min(start_t + time_window_sec, float(df[config.COL_TIME].iloc[-1]))
    sub = df[(df[config.COL_TIME] >= start_t) & (df[config.COL_TIME] <= end_t)].copy()

    t = sub[config.COL_TIME].values - start_t

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#18191a")
    for ax in axes.flat:
        ax.set_facecolor("#242526")
        ax.tick_params(colors="#b0b3b8")
        ax.grid(True, linestyle="--", alpha=0.3, color="#484f58")

    # 1. Accelerometer
    ax_acc = axes[0, 0]
    ax_acc.plot(t, sub[config.COL_ACC_X], color="#29b6f6", label="Acc X (Forward)", lw=1.2)
    ax_acc.plot(t, sub[config.COL_ACC_Y], color="#ab47bc", label="Acc Y (Lateral)", lw=1.2)
    ax_acc.plot(t, sub[config.COL_ACC_Z], color="#26a69a", label="Acc Z (Vertical)", lw=1.2)
    ax_acc.set_title("Smartphone Accelerometer (m/s²)", color="white", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Time (s)", color="#b0b3b8")
    ax_acc.set_ylabel("Acceleration (m/s²)", color="#b0b3b8")
    ax_acc.legend(loc="upper right", facecolor="#18191a", edgecolor="#3a3b3c", labelcolor="white", fontsize=8)

    # 2. Gyroscope
    ax_gyro = axes[0, 1]
    ax_gyro.plot(t, np.degrees(sub[config.COL_GYRO_X]), color="#ef5350", label="Gyro X (Roll Rate)", lw=1.2)
    ax_gyro.plot(t, np.degrees(sub[config.COL_GYRO_Y]), color="#ffa726", label="Gyro Y (Pitch Rate)", lw=1.2)
    ax_gyro.plot(t, np.degrees(sub[config.COL_GYRO_Z]), color="#ffee58", label="Gyro Z (Yaw Rate)", lw=1.5)
    ax_gyro.set_title("Smartphone Gyroscope (deg/s)", color="white", fontsize=12, fontweight="bold")
    ax_gyro.set_xlabel("Time (s)", color="#b0b3b8")
    ax_gyro.set_ylabel("Angular Rate (deg/s)", color="#b0b3b8")
    ax_gyro.legend(loc="upper right", facecolor="#18191a", edgecolor="#3a3b3c", labelcolor="white", fontsize=8)

    # 3. Vibration FFT Spectrum (Vertical Accel)
    ax_fft = axes[1, 0]
    acc_z = sub[config.COL_ACC_Z].values
    fs = config.TARGET_SAMPLE_RATE_HZ
    n = len(acc_z)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(acc_z - np.mean(acc_z))) ** 2) / (n * fs)

    ax_fft.plot(freqs, psd, color="#26a69a", lw=1.5)
    ax_fft.axvspan(0, 1.5, color="#00e676", alpha=0.15, label="Vehicle Dynamics (< 1.5 Hz)")
    ax_fft.axvspan(1.5, 5.0, color="#ff5252", alpha=0.15, label="Road Vibration & Potholes (> 1.5 Hz)")
    ax_fft.set_title("Road Vibration Power Spectral Density (Acc Z)", color="white", fontsize=12, fontweight="bold")
    ax_fft.set_xlabel("Frequency (Hz)", color="#b0b3b8")
    ax_fft.set_ylabel("Power Spectrum", color="#b0b3b8")
    ax_fft.set_yscale("log")
    ax_fft.legend(loc="upper right", facecolor="#18191a", edgecolor="#3a3b3c", labelcolor="white", fontsize=8)

    # 4. CAN Ground Truth Speed & Phone GPS Speed
    ax_speed = axes[1, 1]
    gt_spd = sub[config.COL_TRUE_SPEED_MS].values * config.MS_TO_KMH
    phone_spd = sub[config.COL_GPS_SPEED_MS].values * config.MS_TO_KMH
    ax_speed.plot(t, gt_spd, color="#00e676", lw=2.0, label="CAN True Speed (km/h)")
    ax_speed.plot(t, phone_spd, color="#29b6f6", lw=1.2, linestyle="--", label="Phone GPS Speed (km/h)")
    ax_speed.set_title("Vehicle Dynamics (CAN vs Smartphone GPS)", color="white", fontsize=12, fontweight="bold")
    ax_speed.set_xlabel("Time (s)", color="#b0b3b8")
    ax_speed.set_ylabel("Speed (km/h)", color="#b0b3b8")
    ax_speed.legend(loc="lower right", facecolor="#18191a", edgecolor="#3a3b3c", labelcolor="white", fontsize=8)

    plt.suptitle(f"iNAV Sensor Diagnostics & Frequency Analysis (Run: {run_key})", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    if out_png is None:
        out_png = config.RESULTS_DIR / f"sensors_{run_key}.png"
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    logger.info(f"Sensor plot saved to {out_png}")
    return out_png


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV Sensor Profiler")
    parser.add_argument("--run", type=str, default="s1", help="Run key (e.g. 's1', 'vta10')")
    args = parser.parse_args()

    p = config.SYNC_PROCESSED_DIR / f"sync_{args.run.lower()}.parquet"
    if not p.exists():
        logger.error(f"File not found: {p}")
    else:
        plot_sensor_diagnostics(p)
