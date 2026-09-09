"""
iNAV Configuration Module
Central settings, paths, unit conversion constants, quarantine lists, and data split definitions.
"""

from pathlib import Path
import numpy as np

# ==============================================================================
# 1. Project and Dataset Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Raw IO-VNBD dataset paths
IOVNBD_RAW_DIR = PROJECT_ROOT / "IO-VNBD"
SYNC_DIR = IOVNBD_RAW_DIR / "Synchronised V abd S datasets" / "Uncategorised IOVNB Dataset"
RAW_S_DIR = SYNC_DIR / "S-Dataset"
RAW_V_DIR = SYNC_DIR / "V-Dataset"

# Processed output directories
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INGESTED_DIR = PROCESSED_DIR / "ingested"
SYNC_PROCESSED_DIR = PROCESSED_DIR / "synchronized"
WINDOWED_DIR = PROCESSED_DIR / "windowized"

# Models and results
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.csv"

# Ensure essential output directories exist
for p in [INGESTED_DIR, SYNC_PROCESSED_DIR, WINDOWED_DIR, MODELS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Physics & Unit Conversion Constants
# ==============================================================================
# Standard gravity constant in m/s^2
G_TO_MS2 = 9.80665

# Android getSpeed() returns m/s, despite the column header saying '(Kmh)'
# Multiply by 3.6 to convert m/s -> km/h, or multiply km/h by (1/3.6) -> m/s
MS_TO_KMH = 3.6
KMH_TO_MS = 1.0 / 3.6

# Angular conversions
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

# Sampling rate parameters
TARGET_SAMPLE_RATE_HZ = 10.0
TARGET_DT = 1.0 / TARGET_SAMPLE_RATE_HZ  # 0.1s

# Sliding window parameters
WINDOW_DURATION_SEC = 2.0  # 2.0 seconds (Standard)
WINDOW_SIZE = int(WINDOW_DURATION_SEC * TARGET_SAMPLE_RATE_HZ)  # 20 samples @ 10 Hz
STRIDE_SEC = 0.2  # 0.2s step
WINDOW_STRIDE = int(STRIDE_SEC * TARGET_SAMPLE_RATE_HZ)  # 2 samples @ 10 Hz

# Extended Context Window parameters (CarSpeedNet roadmap recommendation)
WINDOW_DURATION_EXTENDED_SEC = 4.0  # 4.0 seconds
WINDOW_SIZE_EXTENDED = int(WINDOW_DURATION_EXTENDED_SEC * TARGET_SAMPLE_RATE_HZ)  # 40 samples @ 10 Hz

# Outage simulation durations (in seconds)
OUTAGE_DURATIONS_SEC = [10, 30, 60, 120, 180]

# ==============================================================================
# 3. Canonical Column Naming & Mapping
# ==============================================================================
# Standard internal column names for processed data
COL_TIME = "time_s"
COL_GPS_LAT = "gps_lat"
COL_GPS_LON = "gps_lon"
COL_GPS_ALT = "gps_alt_m"
COL_GPS_SPEED_MS = "gps_speed_ms"
COL_GPS_BEARING = "gps_bearing_deg"
COL_GPS_ACCURACY = "gps_accuracy_m"
COL_GPS_SATS = "gps_satellites"

COL_ACC_X = "acc_x"  # m/s^2
COL_ACC_Y = "acc_y"
COL_ACC_Z = "acc_z"

COL_GRAV_X = "grav_x"  # m/s^2
COL_GRAV_Y = "grav_y"
COL_GRAV_Z = "grav_z"

COL_GYRO_X = "gyro_x"  # rad/s
COL_GYRO_Y = "gyro_y"
COL_GYRO_Z = "gyro_z"

COL_MAG_X = "mag_x"  # micro-Tesla
COL_MAG_Y = "mag_y"
COL_MAG_Z = "mag_z"

COL_ORIENT_YAW = "orient_yaw_deg"
COL_ORIENT_PITCH = "orient_pitch_deg"
COL_ORIENT_ROLL = "orient_roll_deg"

# Ground truth columns (from V-Dataset)
COL_TRUE_LAT = "gt_lat"
COL_TRUE_LON = "gt_lon"
COL_TRUE_SPEED_MS = "gt_speed_ms"
COL_TRUE_HEADING = "gt_heading_deg"
COL_TRUE_YAW_RATE = "gt_yaw_rate_degs"
COL_TRUE_WHEEL_FL = "wheel_speed_fl_rads"
COL_TRUE_WHEEL_FR = "wheel_speed_fr_rads"
COL_TRUE_WHEEL_RL = "wheel_speed_rl_rads"
COL_TRUE_WHEEL_RR = "wheel_speed_rr_rads"
COL_TRUE_ACCEL_LONG = "gt_accel_long_ms2"
COL_TRUE_ACCEL_LAT = "gt_accel_lat_ms2"
COL_TRUE_STEERING = "gt_steering_deg"
COL_TRUE_PEDAL = "gt_pedal_pct"
COL_TRUE_BRAKE = "gt_brake_flag"

# ==============================================================================
# 4. Quarantine List (Per IO-VNBD Dataset Audit)
# ==============================================================================
# Files with fatal corruptions, abnormal sampling rate, or dead sensors:
# - S-A4: Double-comma corruption shifting columns
# - 2Hz runs: S-A1 through S-A13 (insufficient rate for DR)
# - French burst-rate runs: S-T1, S-T4, S-T5, S-T6 (1ms irregular spikes)
# - Dead wheel speed runs: V-Vw1, V-Vw15 (all zero wheel speeds)
# - Major temporal gaps: S-Vtb1 (11-min gap)
QUARANTINE_FILES = {
    # Corrupted headers/columns
    "S-A4", "V-A4",
    # Low frequency 2Hz runs
    "S-A1", "V-A1", "S-A2", "V-A2", "S-A3", "V-A3",
    "S-A5", "V-A5", "S-A6", "V-A6", "S-A7", "V-A7",
    "S-A8", "V-A8", "S-A9", "V-A9", "S-A10", "V-A10",
    "S-A11", "V-A11", "S-A12", "V-A12", "S-A13", "V-A13",
    # Burst rate / missing sensors
    "S-T1", "V-T1", "S-T4", "V-T4", "S-T5", "V-T5", "S-T6", "V-T6",
    # Dead sensors / massive gaps
}

# ==============================================================================
# 5. 41-Group Leakage-Safe Campaign Partition (Aligned with iDead ADR 0004)
# ==============================================================================
# Disjoint Driver x Campaign partitioning to prevent paired/temporal leakage
# while preserving paired motorway ground truth in the Test set:
# - TRAIN: Driver A (S-*), Driver B (M-*), Driver C (St-*), Driver E (Vfa, Vfb, Vta, Vtb)
# - VAL: Driver D (Y-*) and S2
# - TEST: Driver E Vw Campaign (Vw1 through Vw20, West Midlands motorways, 20 paired runs)
TRAIN_CAMPAIGNS = ["s", "m", "st", "vfa", "vfb", "vta", "vtb"]
VAL_CAMPAIGNS = ["y", "s2"]
TEST_CAMPAIGNS = ["vw"]

# Backward-compatibility aliases
TRAIN_DRIVERS = TRAIN_CAMPAIGNS
VAL_DRIVERS = VAL_CAMPAIGNS
TEST_DRIVERS = TEST_CAMPAIGNS

# 7 Locked Leaderboard Columns (Standardized with iDead schema)
LEADERBOARD_COLUMNS = [
    "run_id",
    "timestamp",
    "config",
    "outage_s",
    "final_pos_error_m",
    "pct_of_distance",
    "cep50_m",
    "cep95_m",
    "along_track_m",
    "cross_track_m",
    "heading_error_deg",
    "outage_id",
    "split",
    "source_side",
    "has_ground_truth",
    "skip_reason",
]
LEADERBOARD_SUMMARY_PATH = RESULTS_DIR / "leaderboard_summary.csv"

