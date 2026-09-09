"""
iNAV Data Ingestion Pipeline
Parses, cleans, applies unit fixes, normalizes headers, and converts
raw IO-VNBD S-Dataset and V-Dataset CSVs into standardized Parquet files.
"""

import sys
import re
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.ingest")


def parse_satellite_count(val) -> float:
    """
    Robustly parse satellite count string or mangled Excel date representation.
    e.g. '18 / 19' -> 18.0, '12' -> 12.0, 'Aug-20' -> 20.0 (or NaN).
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if not val_str:
        return np.nan

    # Case 1: '18 / 19' -> used / in view
    if "/" in val_str:
        parts = val_str.split("/")
        try:
            return float(parts[0].strip())
        except ValueError:
            pass

    # Case 2: Clean numeric
    try:
        return float(val_str)
    except ValueError:
        pass

    # Case 3: Excel mangling like 'Aug-20' or '20-Aug'
    digits = re.findall(r"\d+", val_str)
    if digits:
        return float(digits[0])

    return np.nan


def clean_s_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean a Smartphone (S-Dataset) CSV file.
    Applies column stripping, unit corrections, and standard naming.
    """
    df_raw = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    # Strip column names
    col_map = {c: c.strip() for c in df_raw.columns}
    df = df_raw.rename(columns=col_map)

    # Identify date column (handles missing parenthesis variation)
    date_col = None
    for c in df.columns:
        if c.startswith("DATE (YYYY-MO-DD"):
            date_col = c
            break

    # Extract time in seconds
    if "TIME SINCE START (ms)" in df.columns:
        df[config.COL_TIME] = pd.to_numeric(df["TIME SINCE START (ms)"], errors="coerce") / 1000.0
    elif date_col is not None:
        try:
            dt = pd.to_datetime(df[date_col].str.replace("_", " "), errors="coerce")
            df[config.COL_TIME] = (dt - dt.iloc[0]).dt.total_seconds()
        except Exception:
            df[config.COL_TIME] = np.arange(len(df)) * config.TARGET_DT
    else:
        df[config.COL_TIME] = np.arange(len(df)) * config.TARGET_DT

    # Clean GPS columns
    gps_lat_col = next((c for c in df.columns if "LATITUDE" in c.upper()), None)
    gps_lon_col = next((c for c in df.columns if "LONGITUDE" in c.upper()), None)
    gps_alt_col = next((c for c in df.columns if "ALTITUDE" in c.upper()), None)
    gps_spd_col = next((c for c in df.columns if "GPS SPEED" in c.upper()), None)
    gps_acc_col = next((c for c in df.columns if "GPS ACCURACY" in c.upper()), None)
    gps_ori_col = next((c for c in df.columns if "GPS ORIENTATION" in c.upper() or "BEARING" in c.upper()), None)
    gps_sat_col = next((c for c in df.columns if "SATELLITES" in c.upper()), None)

    # Note: Android Location.getSpeed() is in m/s despite header saying 'Kmh'
    df[config.COL_GPS_LAT] = pd.to_numeric(df[gps_lat_col], errors="coerce") if gps_lat_col else np.nan
    df[config.COL_GPS_LON] = pd.to_numeric(df[gps_lon_col], errors="coerce") if gps_lon_col else np.nan
    df[config.COL_GPS_ALT] = pd.to_numeric(df[gps_alt_col], errors="coerce") if gps_alt_col else np.nan
    df[config.COL_GPS_SPEED_MS] = pd.to_numeric(df[gps_spd_col], errors="coerce") if gps_spd_col else np.nan
    df[config.COL_GPS_ACCURACY] = pd.to_numeric(df[gps_acc_col], errors="coerce") if gps_acc_col else np.nan
    df[config.COL_GPS_BEARING] = pd.to_numeric(df[gps_ori_col], errors="coerce") if gps_ori_col else np.nan

    if gps_sat_col and gps_sat_col in df.columns:
        df[config.COL_GPS_SATS] = df[gps_sat_col].apply(parse_satellite_count)
    else:
        df[config.COL_GPS_SATS] = np.nan

    # Sensor columns
    acc_x_col = next((c for c in df.columns if "ACCELEROMETER X" in c.upper()), None)
    acc_y_col = next((c for c in df.columns if "ACCELEROMETER Y" in c.upper()), None)
    acc_z_col = next((c for c in df.columns if "ACCELEROMETER Z" in c.upper()), None)
    df[config.COL_ACC_X] = pd.to_numeric(df[acc_x_col], errors="coerce") if acc_x_col else 0.0
    df[config.COL_ACC_Y] = pd.to_numeric(df[acc_y_col], errors="coerce") if acc_y_col else 0.0
    df[config.COL_ACC_Z] = pd.to_numeric(df[acc_z_col], errors="coerce") if acc_z_col else 0.0

    grav_x_col = next((c for c in df.columns if "GRAVITY X" in c.upper()), None)
    grav_y_col = next((c for c in df.columns if "GRAVITY Y" in c.upper()), None)
    grav_z_col = next((c for c in df.columns if "GRAVITY Z" in c.upper()), None)
    df[config.COL_GRAV_X] = pd.to_numeric(df[grav_x_col], errors="coerce") if grav_x_col else 0.0
    df[config.COL_GRAV_Y] = pd.to_numeric(df[grav_y_col], errors="coerce") if grav_y_col else 0.0
    df[config.COL_GRAV_Z] = pd.to_numeric(df[grav_z_col], errors="coerce") if grav_z_col else 0.0

    gyro_x_col = next((c for c in df.columns if "GYROSCOPE X" in c.upper() or "GYROSCOPE YAW" in c.upper()), None)
    gyro_y_col = next((c for c in df.columns if "GYROSCOPE Y" in c.upper()), None)
    gyro_z_col = next((c for c in df.columns if "GYROSCOPE Z" in c.upper()), None)
    df[config.COL_GYRO_X] = pd.to_numeric(df[gyro_x_col], errors="coerce") if gyro_x_col else 0.0
    df[config.COL_GYRO_Y] = pd.to_numeric(df[gyro_y_col], errors="coerce") if gyro_y_col else 0.0
    df[config.COL_GYRO_Z] = pd.to_numeric(df[gyro_z_col], errors="coerce") if gyro_z_col else 0.0

    mag_x_col = next((c for c in df.columns if "MAGNETIC FIELD X" in c.upper()), None)
    mag_y_col = next((c for c in df.columns if "MAGNETIC FIELD Y" in c.upper()), None)
    mag_z_col = next((c for c in df.columns if "MAGNETIC FIELD Z" in c.upper()), None)
    df[config.COL_MAG_X] = pd.to_numeric(df[mag_x_col], errors="coerce") if mag_x_col else np.nan
    df[config.COL_MAG_Y] = pd.to_numeric(df[mag_y_col], errors="coerce") if mag_y_col else np.nan
    df[config.COL_MAG_Z] = pd.to_numeric(df[mag_z_col], errors="coerce") if mag_z_col else np.nan

    yaw_col = next((c for c in df.columns if "ORIENTATION (AZIMUTH)" in c.upper() or "ORIENTATION YAW" in c.upper()), None)
    pitch_col = next((c for c in df.columns if "ORIENTATION (PITCH)" in c.upper()), None)
    roll_col = next((c for c in df.columns if "ORIENTATION (ROLL" in c.upper()), None)
    df[config.COL_ORIENT_YAW] = pd.to_numeric(df[yaw_col], errors="coerce") if yaw_col else np.nan
    df[config.COL_ORIENT_PITCH] = pd.to_numeric(df[pitch_col], errors="coerce") if pitch_col else np.nan
    df[config.COL_ORIENT_ROLL] = pd.to_numeric(df[roll_col], errors="coerce") if roll_col else np.nan

    # Keep only canonical columns
    canonical_cols = [
        config.COL_TIME,
        config.COL_GPS_LAT, config.COL_GPS_LON, config.COL_GPS_ALT,
        config.COL_GPS_SPEED_MS, config.COL_GPS_BEARING, config.COL_GPS_ACCURACY, config.COL_GPS_SATS,
        config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z,
        config.COL_GRAV_X, config.COL_GRAV_Y, config.COL_GRAV_Z,
        config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z,
        config.COL_MAG_X, config.COL_MAG_Y, config.COL_MAG_Z,
        config.COL_ORIENT_YAW, config.COL_ORIENT_PITCH, config.COL_ORIENT_ROLL
    ]

    clean_df = df[[c for c in canonical_cols if c in df.columns]].copy()

    # Drop rows where time is NaN
    clean_df = clean_df.dropna(subset=[config.COL_TIME])

    # Ensure strictly increasing time
    clean_df = clean_df.sort_values(by=config.COL_TIME).drop_duplicates(subset=[config.COL_TIME], keep="first")
    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def clean_v_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean a Vehicle OBD-II (V-Dataset) CSV file.
    Applies column stripping, unit conversions (Height km->m, Accel g->m/s^2),
    and sets standard ground truth names.
    """
    df_raw = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    col_map = {c: c.strip() for c in df_raw.columns}
    df = df_raw.rename(columns=col_map)

    # Time since start of day
    time_col = next((c for c in df.columns if "TIME SINCE" in c.upper()), None)
    if time_col:
        raw_time = pd.to_numeric(df[time_col], errors="coerce")
        df[config.COL_TIME] = raw_time - raw_time.dropna().iloc[0]
    else:
        df[config.COL_TIME] = np.arange(len(df)) * config.TARGET_DT

    # GPS Ground Truth
    lat_col = next((c for c in df.columns if "LATITUDE" in c.upper()), None)
    lon_col = next((c for c in df.columns if "LONGITUDE" in c.upper()), None)
    spd_kmh_col = next((c for c in df.columns if c.startswith("Velocity")), None)
    heading_col = next((c for c in df.columns if "HEADING" in c.upper()), None)
    yaw_rate_col = next((c for c in df.columns if "YAW RATE" in c.upper()), None)

    df[config.COL_TRUE_LAT] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df[config.COL_TRUE_LON] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan

    # Velocity (km/hr) -> m/s
    if spd_kmh_col:
        df[config.COL_TRUE_SPEED_MS] = pd.to_numeric(df[spd_kmh_col], errors="coerce") * config.KMH_TO_MS
    else:
        df[config.COL_TRUE_SPEED_MS] = np.nan

    df[config.COL_TRUE_HEADING] = pd.to_numeric(df[heading_col], errors="coerce") if heading_col else np.nan
    df[config.COL_TRUE_YAW_RATE] = pd.to_numeric(df[yaw_rate_col], errors="coerce") if yaw_rate_col else np.nan

    # Wheel speeds (rad/sec)
    w_fl = next((c for c in df.columns if "WHEEL SPEED FRONT LEFT" in c.upper()), None)
    w_fr = next((c for c in df.columns if "WHEEL SPEED FRONT RIGHT" in c.upper()), None)
    w_rl = next((c for c in df.columns if "WHEEL SPEED REAR LEFT" in c.upper()), None)
    w_rr = next((c for c in df.columns if "WHEEL SPEED REAR RIGHT" in c.upper()), None)
    df[config.COL_TRUE_WHEEL_FL] = pd.to_numeric(df[w_fl], errors="coerce") if w_fl else np.nan
    df[config.COL_TRUE_WHEEL_FR] = pd.to_numeric(df[w_fr], errors="coerce") if w_fr else np.nan
    df[config.COL_TRUE_WHEEL_RL] = pd.to_numeric(df[w_rl], errors="coerce") if w_rl else np.nan
    df[config.COL_TRUE_WHEEL_RR] = pd.to_numeric(df[w_rr], errors="coerce") if w_rr else np.nan

    # Accelerations (g -> m/s^2)
    acc_long = next((c for c in df.columns if "INDICATED LONGITUDINAL ACCELERATION" in c.upper()), None)
    acc_lat = next((c for c in df.columns if "INDICATED LATERAL ACCELERATION" in c.upper()), None)
    df[config.COL_TRUE_ACCEL_LONG] = (pd.to_numeric(df[acc_long], errors="coerce") * config.G_TO_MS2) if acc_long else np.nan
    df[config.COL_TRUE_ACCEL_LAT] = (pd.to_numeric(df[acc_lat], errors="coerce") * config.G_TO_MS2) if acc_lat else np.nan

    # Controls
    steer_col = next((c for c in df.columns if "STEERING ANGLE" in c.upper()), None)
    pedal_col = next((c for c in df.columns if "ACCELERATOR PEDAL" in c.upper()), None)
    brake_col = next((c for c in df.columns if "BRAKE POSITION" in c.upper()), None)
    df[config.COL_TRUE_STEERING] = pd.to_numeric(df[steer_col], errors="coerce") if steer_col else np.nan
    df[config.COL_TRUE_PEDAL] = pd.to_numeric(df[pedal_col], errors="coerce") if pedal_col else np.nan
    df[config.COL_TRUE_BRAKE] = pd.to_numeric(df[brake_col], errors="coerce") if brake_col else np.nan

    # Height: labeled in V-Dataset as 'Height (km)' but values are in meters!
    height_col = next((c for c in df.columns if "HEIGHT" in c.upper()), None)
    if height_col:
        df["gt_height_m"] = pd.to_numeric(df[height_col], errors="coerce")

    canonical_cols = [
        config.COL_TIME,
        config.COL_TRUE_LAT, config.COL_TRUE_LON,
        config.COL_TRUE_SPEED_MS, config.COL_TRUE_HEADING, config.COL_TRUE_YAW_RATE,
        config.COL_TRUE_WHEEL_FL, config.COL_TRUE_WHEEL_FR,
        config.COL_TRUE_WHEEL_RL, config.COL_TRUE_WHEEL_RR,
        config.COL_TRUE_ACCEL_LONG, config.COL_TRUE_ACCEL_LAT,
        config.COL_TRUE_STEERING, config.COL_TRUE_PEDAL, config.COL_TRUE_BRAKE,
        "gt_height_m"
    ]

    clean_df = df[[c for c in canonical_cols if c in df.columns]].copy()
    clean_df = clean_df.dropna(subset=[config.COL_TIME])
    clean_df = clean_df.sort_values(by=config.COL_TIME).drop_duplicates(subset=[config.COL_TIME], keep="first")
    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def normalize_run_key(filename: str, prefix: str) -> str:
    """
    Extract normalized run key from filename.
    e.g. 'S-Vta10.csv' with prefix 'S' -> 'vta10'
         'V-vta10.csv' with prefix 'V' -> 'vta10'
    """
    stem = Path(filename).stem
    if stem.upper().startswith(prefix.upper() + "-"):
        return stem[len(prefix) + 1:].lower()
    return stem.lower()


def get_available_pairs() -> List[Tuple[str, Path, Path]]:
    """
    Locate all matching S and V dataset pairs, excluding quarantined runs.
    Returns list of tuples: (normalized_run_key, s_path, v_path)
    """
    s_files = glob.glob(str(config.RAW_S_DIR / "*.csv"))
    v_files = glob.glob(str(config.RAW_V_DIR / "*.csv"))

    s_map = {normalize_run_key(f, "S"): Path(f) for f in s_files}
    v_map = {normalize_run_key(f, "V"): Path(f) for f in v_files}

    quarantine_lower = {x.replace("S-", "").replace("V-", "").lower() for x in config.QUARANTINE_FILES}

    matched_pairs = []
    for key in sorted(s_map.keys()):
        if key in v_map:
            if key in quarantine_lower:
                logger.info(f"Skipping quarantined run: {key}")
                continue
            matched_pairs.append((key, s_map[key], v_map[key]))

    return matched_pairs


def ingest_all(max_pairs: Optional[int] = None) -> None:
    """
    Ingest and convert all usable S and V dataset pairs into Parquet.
    """
    pairs = get_available_pairs()
    if max_pairs:
        pairs = pairs[:max_pairs]

    logger.info(f"Starting ingestion for {len(pairs)} pairs...")
    success_count = 0

    for idx, (key, s_path, v_path) in enumerate(pairs, 1):
        try:
            df_s = clean_s_dataset(s_path)
            df_v = clean_v_dataset(v_path)

            out_s = config.INGESTED_DIR / f"S_{key}.parquet"
            out_v = config.INGESTED_DIR / f"V_{key}.parquet"

            df_s.to_parquet(out_s, index=False)
            df_v.to_parquet(out_v, index=False)

            success_count += 1
            if idx % 10 == 0 or idx == len(pairs):
                logger.info(f"Ingested {idx}/{len(pairs)}: run '{key}' -> S:{len(df_s)} rows, V:{len(df_v)} rows")
        except Exception as e:
            logger.error(f"Failed to ingest pair '{key}': {e}", exc_info=True)

    logger.info(f"Ingestion complete: successfully processed {success_count}/{len(pairs)} pairs.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV Data Ingestion Pipeline")
    parser.add_argument("--max-pairs", type=int, default=None, help="Max pairs to ingest")
    parser.add_argument("--pair", type=str, default=None, help="Specific run key (e.g. 'm', 's1')")
    args = parser.parse_args()

    if args.pair:
        pairs = get_available_pairs()
        matched = [p for p in pairs if p[0] == args.pair.lower()]
        if not matched:
            logger.error(f"Pair '{args.pair}' not found or quarantined.")
        else:
            k, sp, vp = matched[0]
            logger.info(f"Ingesting single pair '{k}'...")
            clean_s_dataset(sp).to_parquet(config.INGESTED_DIR / f"S_{k}.parquet", index=False)
            clean_v_dataset(vp).to_parquet(config.INGESTED_DIR / f"V_{k}.parquet", index=False)
            logger.info(f"Saved to {config.INGESTED_DIR}")
    else:
        ingest_all(max_pairs=args.max_pairs)
