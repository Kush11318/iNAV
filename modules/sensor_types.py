"""
iNAV Canonical Sensor Abstraction Layer.

Phase 2 Contract:
All sensor inputs (Android SensorEvent, Python dataset replay, external IMUs, CAN/GNSS)
must map into these canonical data classes without exception.

Coordinate Frames:
  PHONE_BODY:
    +X : Phone right edge
    +Y : Phone top edge (display up)
    +Z : Outward from display (screen normal)
  VEHICLE_FRD:
    +X : Vehicle Forward
    +Y : Vehicle Right
    +Z : Vehicle Down (Right-handed, positive clockwise yaw around +Z)
  WORLD_NED:
    +X : North
    +Y : East
    +Z : Down

Units:
  - Timestamps: int (nanoseconds)
  - Acceleration: float (m/s²)
  - Angular rate: float (rad/s)
  - Speed: float (m/s)
  - Coordinates: float (decimal degrees)
  - Bearing: float (degrees clockwise from True North [0, 360))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import NamedTuple, Optional, Union


class SensorFrame(IntEnum):
    UNKNOWN = 0
    PHONE_BODY = 1
    VEHICLE_FRD = 2
    WORLD_NED = 3


class ImuValidity(IntEnum):
    NONE = 0
    ACCEL_VALID = 1 << 0
    GYRO_VALID = 1 << 1
    TIMESTAMP_VALID = 1 << 2
    STATIONARY_VALID = 1 << 3
    ALL_VALID = ACCEL_VALID | GYRO_VALID | TIMESTAMP_VALID


class GnssValidity(IntEnum):
    NONE = 0
    LAT_LON_VALID = 1 << 0
    ALTITUDE_VALID = 1 << 1
    SPEED_VALID = 1 << 2
    BEARING_VALID = 1 << 3
    ACCURACY_VALID = 1 << 4
    SAT_COUNT_VALID = 1 << 5
    HDOP_VALID = 1 << 6
    BASIC_FIX_VALID = LAT_LON_VALID | SPEED_VALID | BEARING_VALID | ACCURACY_VALID


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


@dataclass
class ImuSample:
    """
    Canonical IMU measurement sample.

    Attributes:
        timestamp_ns: Monotonic/Epoch timestamp in nanoseconds (int64).
        accel_mps2: Linear acceleration + gravity vector in m/s².
        gyro_radps: Angular rate vector in rad/s (Right-handed rule).
        frame: Explicit SensorFrame enum (e.g. PHONE_BODY or VEHICLE_FRD).
        validity: Bitmask of ImuValidity flags.
        is_stationary: Optional flag if stationary condition is detected.
    """
    timestamp_ns: int
    accel_mps2: Vec3
    gyro_radps: Vec3
    frame: SensorFrame = SensorFrame.UNKNOWN
    validity: int = int(ImuValidity.ALL_VALID)
    is_stationary: bool = False

    @property
    def is_valid(self) -> bool:
        return (self.validity & int(ImuValidity.ALL_VALID)) == int(ImuValidity.ALL_VALID)

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns * 1e-9


@dataclass
class GnssSample:
    """
    Canonical GNSS position/velocity measurement sample.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds (int64).
        latitude_deg: WGS84 latitude in decimal degrees [-90.0, 90.0].
        longitude_deg: WGS84 longitude in decimal degrees [-180.0, 180.0].
        altitude_m: Ellipsoidal or MSL altitude in meters.
        speed_mps: Ground speed in m/s (must be >= 0.0).
        bearing_deg: Course over ground in degrees clockwise from True North [0.0, 360.0).
        horizontal_accuracy_m: 1-sigma estimated horizontal position accuracy in meters.
        satellite_count: Number of tracked/used SVs (-1 if unknown).
        hdop: Horizontal Dilution of Precision (-1.0 if not reported).
        validity: Bitmask of GnssValidity flags.
    """
    timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    speed_mps: float = 0.0
    bearing_deg: float = 0.0
    horizontal_accuracy_m: float = 0.0
    satellite_count: int = -1
    hdop: float = -1.0
    validity: int = int(GnssValidity.NONE)

    @property
    def has_basic_fix(self) -> bool:
        return (self.validity & int(GnssValidity.BASIC_FIX_VALID)) == int(GnssValidity.BASIC_FIX_VALID)

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns * 1e-9


# ==============================================================================
# Centralized Unit & Timestamp Conversions
# ==============================================================================

class units:
    """Canonical unit conversion utility functions."""

    @staticmethod
    def sec_to_ns(sec: float | int) -> int:
        return int(round(sec * 1e9))

    @staticmethod
    def ms_to_ns(ms: float | int) -> int:
        return int(round(ms * 1e6))

    @staticmethod
    def ns_to_sec(ns: int) -> float:
        return float(ns) * 1e-9

    @staticmethod
    def ns_to_ms(ns: int) -> float:
        return float(ns) * 1e-6

    @staticmethod
    def deg_to_rad(deg: float) -> float:
        return math.radians(deg)

    @staticmethod
    def rad_to_deg(rad: float) -> float:
        return math.degrees(rad)

    @staticmethod
    def kmh_to_mps(kmh: float) -> float:
        return kmh / 3.6

    @staticmethod
    def mps_to_kmh(mps: float) -> float:
        return mps * 3.6

    @staticmethod
    def g_to_mps2(g: float) -> float:
        return g * 9.80665


def make_android_imu_sample(
    sensor_event_timestamp_ns: int,
    ax_mps2: float, ay_mps2: float, az_mps2: float,
    gx_radps: float, gy_radps: float, gz_radps: float,
    is_stationary: bool = False
) -> ImuSample:
    """
    Constructs a canonical ImuSample from raw Android SensorEvent data.
    Android sensors operate strictly in PHONE_BODY frame.
    Alignment to VEHICLE_FRD is explicitly deferred to Phase 3.
    """
    return ImuSample(
        timestamp_ns=sensor_event_timestamp_ns,
        accel_mps2=Vec3(ax_mps2, ay_mps2, az_mps2),
        gyro_radps=Vec3(gx_radps, gy_radps, gz_radps),
        frame=SensorFrame.PHONE_BODY,
        validity=int(ImuValidity.ALL_VALID),
        is_stationary=is_stationary
    )
