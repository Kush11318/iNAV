"""
iNAV Dataset Adapter: IO-VNBD / Parquet -> Canonical Sensor Samples.

Phase 2 Contract:
Adapts DataFrame rows / Parquet files from IO-VNBD into canonical ImuSample and GnssSample.
Does not change the underlying dataset.
Does not retrain models.
"""

from __future__ import annotations
import math
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple, Union
import numpy as np
import pandas as pd

from modules.sensor_types import (
    SensorFrame,
    Vec3,
    ImuSample,
    GnssSample,
    ImuValidity,
    GnssValidity,
    units
)


def _get_val(row: Mapping[str, Any] | pd.Series, keys: list[str], default: float = 0.0) -> float:
    """Helper to extract a numerical field from a row trying multiple fallback keys."""
    for k in keys:
        if k in row:
            v = row[k]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return default


def extract_timestamp_ns(row: Mapping[str, Any] | pd.Series) -> int:
    """
    Extracts canonical timestamp in nanoseconds from a row.
    Handles 'time_ns', 'timestamp_ns', 'time_s' (seconds), or 'epoch_ms'/'time_ms' (milliseconds).
    """
    if "time_ns" in row and not pd.isna(row["time_ns"]):
        return int(row["time_ns"])
    if "timestamp_ns" in row and not pd.isna(row["timestamp_ns"]):
        return int(row["timestamp_ns"])
    if "time_s" in row and not pd.isna(row["time_s"]):
        return units.sec_to_ns(float(row["time_s"]))
    if "timestamp_s" in row and not pd.isna(row["timestamp_s"]):
        return units.sec_to_ns(float(row["timestamp_s"]))
    if "epoch_ms" in row and not pd.isna(row["epoch_ms"]):
        return units.ms_to_ns(int(row["epoch_ms"]))
    if "time_ms" in row and not pd.isna(row["time_ms"]):
        return units.ms_to_ns(int(row["time_ms"]))
    return 0


def row_to_canonical_imu(
    row: Mapping[str, Any] | pd.Series,
    frame: SensorFrame = SensorFrame.PHONE_BODY
) -> ImuSample:
    """
    Converts a single IO-VNBD / Parquet / CSV row to a canonical ImuSample.

    Preserves:
      - Timestamp (ns)
      - Acceleration (m/s²)
      - Gyroscope (rad/s)
      - Validity flags
      - Stationary state if available
    """
    ts_ns = extract_timestamp_ns(row)

    ax = _get_val(row, ["acc_x", "accel_x", "ax", "acceleration_x"], 0.0)
    ay = _get_val(row, ["acc_y", "accel_y", "ay", "acceleration_y"], 0.0)
    az = _get_val(row, ["acc_z", "accel_z", "az", "acceleration_z"], 0.0)

    gx = _get_val(row, ["gyro_x", "gx", "rotation_x", "angular_velocity_x"], 0.0)
    gy = _get_val(row, ["gyro_y", "gy", "rotation_y", "angular_velocity_y"], 0.0)
    gz = _get_val(row, ["gyro_z", "gz", "rotation_z", "angular_velocity_z"], 0.0)

    validity = int(ImuValidity.NONE)
    if ts_ns > 0 or "time_s" in row or "time_ns" in row:
        validity |= int(ImuValidity.TIMESTAMP_VALID)

    # Validate numbers (not NaN / Inf)
    if not any(math.isnan(v) or math.isinf(v) for v in (ax, ay, az)):
        validity |= int(ImuValidity.ACCEL_VALID)
    if not any(math.isnan(v) or math.isinf(v) for v in (gx, gy, gz)):
        validity |= int(ImuValidity.GYRO_VALID)

    is_stat = False
    if "is_stationary" in row and not pd.isna(row["is_stationary"]):
        is_stat = bool(row["is_stationary"])
        validity |= int(ImuValidity.STATIONARY_VALID)
    elif "stationary" in row and not pd.isna(row["stationary"]):
        is_stat = bool(row["stationary"])
        validity |= int(ImuValidity.STATIONARY_VALID)

    return ImuSample(
        timestamp_ns=ts_ns,
        accel_mps2=Vec3(ax, ay, az),
        gyro_radps=Vec3(gx, gy, gz),
        frame=frame,
        validity=validity,
        is_stationary=is_stat
    )


def row_to_canonical_gnss(
    row: Mapping[str, Any] | pd.Series
) -> Optional[GnssSample]:
    """
    Converts a single IO-VNBD row to a canonical GnssSample if GNSS columns are present.
    Returns None if no valid GNSS coordinates are present.
    """
    lat = _get_val(row, ["gps_lat", "lat", "latitude"], float("nan"))
    lon = _get_val(row, ["gps_lon", "lon", "longitude"], float("nan"))

    if math.isnan(lat) or math.isnan(lon):
        return None

    ts_ns = extract_timestamp_ns(row)
    alt = _get_val(row, ["gps_alt_m", "gps_alt", "alt", "altitude"], 0.0)
    speed = _get_val(row, ["gps_speed_ms", "gps_speed", "speed_ms", "speed"], 0.0)
    bearing = _get_val(row, ["gps_bearing_deg", "gps_bearing", "bearing_deg", "heading_deg"], 0.0)
    accuracy = _get_val(row, ["gps_accuracy_m", "gps_accuracy", "accuracy_m"], 5.0)
    sats = int(_get_val(row, ["gps_satellites", "satellites", "sats"], -1.0))

    validity = int(GnssValidity.BASIC_FIX_VALID)
    if not math.isnan(alt):
        validity |= int(GnssValidity.ALTITUDE_VALID)
    if sats >= 0:
        validity |= int(GnssValidity.SAT_COUNT_VALID)

    hdop = -1.0
    if "hdop" in row and not pd.isna(row["hdop"]):
        hdop = float(row["hdop"])
        validity |= int(GnssValidity.HDOP_VALID)

    return GnssSample(
        timestamp_ns=ts_ns,
        latitude_deg=lat,
        longitude_deg=lon,
        altitude_m=alt,
        speed_mps=max(0.0, speed),
        bearing_deg=bearing % 360.0,
        horizontal_accuracy_m=accuracy,
        satellite_count=sats,
        hdop=hdop,
        validity=validity
    )


def dataframe_to_canonical_samples(
    df: pd.DataFrame,
    frame: SensorFrame = SensorFrame.PHONE_BODY
) -> Tuple[List[ImuSample], List[Optional[GnssSample]]]:
    """
    Converts a pandas DataFrame (e.g. read from IO-VNBD parquet/csv) into lists
    of canonical ImuSample and GnssSample.
    """
    imu_samples: List[ImuSample] = []
    gnss_samples: List[Optional[GnssSample]] = []

    for _, row in df.iterrows():
        imu_samples.append(row_to_canonical_imu(row, frame=frame))
        gnss_samples.append(row_to_canonical_gnss(row))

    return imu_samples, gnss_samples


def iter_canonical_samples(
    df: pd.DataFrame,
    frame: SensorFrame = SensorFrame.PHONE_BODY
) -> Generator[Tuple[ImuSample, Optional[GnssSample]], None, None]:
    """
    Generator yielding (ImuSample, Optional[GnssSample]) for streaming replay.
    """
    for _, row in df.iterrows():
        yield row_to_canonical_imu(row, frame=frame), row_to_canonical_gnss(row)
