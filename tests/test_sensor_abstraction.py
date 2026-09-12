"""
iNAV Phase 2 - Canonical Sensor Abstraction Test Suite.

Validates:
1. Unit conversions and contracts
2. Timestamp conversions
3. Frame enum contracts
4. ImuSample construction and validation
5. GnssSample construction and validation
6. Dataset adapter conversion
7. Android adapter boundary
8. 3-Source Equivalence (Dataset, Android, External IMU)
"""

import math
import sys
from pathlib import Path

# Ensure iNAV root is in Python path
INAV_ROOT = Path(__file__).resolve().parent.parent
if str(INAV_ROOT) not in sys.path:
    sys.path.insert(0, str(INAV_ROOT))

from modules.sensor_types import (
    SensorFrame,
    Vec3,
    ImuSample,
    GnssSample,
    ImuValidity,
    GnssValidity,
    units,
    make_android_imu_sample,
)
from modules.dataset_adapter import (
    row_to_canonical_imu,
    row_to_canonical_gnss,
    dataframe_to_canonical_samples,
    extract_timestamp_ns,
)
import pandas as pd


# ==============================================================================
# 1. Unit Conversion Tests
# ==============================================================================

def test_units_rad_per_sec_invariant():
    """Verify rad/s is preserved as rad/s without accidental conversion."""
    gyro_val_radps = 0.05235987755982988  # 3 deg/s in rad/s
    sample = ImuSample(
        timestamp_ns=1_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, 9.81),
        gyro_radps=Vec3(0.0, 0.0, gyro_val_radps),
        frame=SensorFrame.VEHICLE_FRD
    )
    # The value must remain EXACTLY what was passed in (no deg conversion applied)
    assert sample.gyro_radps.z == gyro_val_radps
    assert sample.gyro_radps.z != 3.0  # Not degrees


def test_units_mps2_invariant():
    """Verify m/s² is preserved as m/s² without accidental g-unit conversion."""
    accel_mps2 = 9.80665
    sample = ImuSample(
        timestamp_ns=1_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, accel_mps2),
        gyro_radps=Vec3(0.0, 0.0, 0.0),
        frame=SensorFrame.PHONE_BODY
    )
    assert sample.accel_mps2.z == accel_mps2
    assert sample.accel_mps2.z != 1.0  # Not converted to 1.0 g


def test_units_angular_conversions():
    """Verify explicit degree <-> radian conversions."""
    deg = 90.0
    rad = units.deg_to_rad(deg)
    assert math.isclose(rad, math.pi / 2.0, rel_tol=1e-9)
    assert math.isclose(units.rad_to_deg(rad), deg, rel_tol=1e-9)

    # 180 deg = pi rad
    assert math.isclose(units.deg_to_rad(180.0), math.pi, rel_tol=1e-9)
    assert math.isclose(units.rad_to_deg(math.pi), 180.0, rel_tol=1e-9)


def test_units_speed_conversions():
    """Verify explicit km/h <-> m/s conversions."""
    kmh = 36.0
    mps = units.kmh_to_mps(kmh)
    assert math.isclose(mps, 10.0, rel_tol=1e-9)
    assert math.isclose(units.mps_to_kmh(mps), kmh, rel_tol=1e-9)

    # 100 km/h = 27.777... m/s
    assert math.isclose(units.kmh_to_mps(100.0), 100.0 / 3.6, rel_tol=1e-9)


def test_units_gravity_conversion():
    """Verify standard Earth gravity conversion."""
    assert math.isclose(units.g_to_mps2(1.0), 9.80665, rel_tol=1e-9)
    assert math.isclose(units.g_to_mps2(2.0), 19.6133, rel_tol=1e-9)


# ==============================================================================
# 2. Timestamp Conversion Tests
# ==============================================================================

def test_timestamp_conversions():
    """Verify seconds, milliseconds, and nanoseconds conversions."""
    # 1.5 seconds -> 1,500,000,000 ns
    sec = 1.5
    ns = units.sec_to_ns(sec)
    assert ns == 1_500_000_000
    assert isinstance(ns, int)
    assert math.isclose(units.ns_to_sec(ns), sec, rel_tol=1e-9)

    # 250 milliseconds -> 250,000,000 ns
    ms = 250.0
    ns_from_ms = units.ms_to_ns(ms)
    assert ns_from_ms == 250_000_000
    assert isinstance(ns_from_ms, int)
    assert math.isclose(units.ns_to_ms(ns_from_ms), ms, rel_tol=1e-9)


# ==============================================================================
# 3. Frame Enum Tests
# ==============================================================================

def test_frame_enum_definitions():
    """Verify explicit frames exist, have unique values, and no magic integers."""
    assert SensorFrame.PHONE_BODY == 1
    assert SensorFrame.VEHICLE_FRD == 2
    assert SensorFrame.WORLD_NED == 3
    assert SensorFrame.UNKNOWN == 0

    # Ensure frame metadata is preserved
    s1 = ImuSample(
        timestamp_ns=100,
        accel_mps2=Vec3(0, 0, 9.8),
        gyro_radps=Vec3(0, 0, 0),
        frame=SensorFrame.PHONE_BODY
    )
    assert s1.frame == SensorFrame.PHONE_BODY

    s2 = ImuSample(
        timestamp_ns=100,
        accel_mps2=Vec3(0, 0, 9.8),
        gyro_radps=Vec3(0, 0, 0),
        frame=SensorFrame.VEHICLE_FRD
    )
    assert s2.frame == SensorFrame.VEHICLE_FRD
    assert s1.frame != s2.frame


# ==============================================================================
# 4. ImuSample Construction & Validation
# ==============================================================================

def test_imu_sample_validation():
    """Verify validity flags and is_valid check."""
    # Fully valid sample
    s_valid = ImuSample(
        timestamp_ns=1_000_000_000,
        accel_mps2=Vec3(0.1, 0.2, 9.81),
        gyro_radps=Vec3(0.01, -0.02, 0.05),
        frame=SensorFrame.PHONE_BODY,
        validity=int(ImuValidity.ALL_VALID),
        is_stationary=False
    )
    assert s_valid.is_valid is True
    assert s_valid.timestamp_sec == 1.0

    # Partially valid sample (missing gyro)
    s_partial = ImuSample(
        timestamp_ns=1_000_000_000,
        accel_mps2=Vec3(0.1, 0.2, 9.81),
        gyro_radps=Vec3(0.0, 0.0, 0.0),
        frame=SensorFrame.PHONE_BODY,
        validity=int(ImuValidity.ACCEL_VALID | ImuValidity.TIMESTAMP_VALID)
    )
    assert s_partial.is_valid is False
    assert (s_partial.validity & int(ImuValidity.ACCEL_VALID)) > 0


# ==============================================================================
# 5. GnssSample Construction & Validation
# ==============================================================================

def test_gnss_sample_validation():
    """Verify GnssSample contract, basic fix validity, and optional HDOP."""
    g = GnssSample(
        timestamp_ns=1_500_000_000,
        latitude_deg=12.9716,
        longitude_deg=77.5946,
        altitude_m=920.5,
        speed_mps=15.2,
        bearing_deg=45.0,
        horizontal_accuracy_m=2.3,
        satellite_count=14,
        validity=int(GnssValidity.BASIC_FIX_VALID | GnssValidity.ALTITUDE_VALID | GnssValidity.SAT_COUNT_VALID)
    )
    assert g.has_basic_fix is True
    assert g.latitude_deg == 12.9716
    assert g.longitude_deg == 77.5946
    assert g.speed_mps == 15.2
    assert g.bearing_deg == 45.0
    assert g.hdop == -1.0  # HDOP was NOT fabricated
    assert (g.validity & int(GnssValidity.HDOP_VALID)) == 0


# ==============================================================================
# 6. Dataset Adapter Tests
# ==============================================================================

def test_dataset_adapter_row_conversion():
    """Test IO-VNBD dataset row conversion preserving all fields."""
    row = {
        "time_s": 10.5,
        "acc_x": 0.12,
        "acc_y": -0.05,
        "acc_z": 9.78,
        "gyro_x": 0.001,
        "gyro_y": 0.002,
        "gyro_z": -0.035,
        "gps_lat": 28.6139,
        "gps_lon": 77.2090,
        "gps_speed_ms": 12.5,
        "gps_bearing_deg": 180.0,
        "gps_accuracy_m": 3.0,
        "gps_satellites": 8,
        "is_stationary": 0
    }

    imu_sample = row_to_canonical_imu(row, frame=SensorFrame.PHONE_BODY)
    assert imu_sample.timestamp_ns == 10_500_000_000
    assert math.isclose(imu_sample.accel_mps2.x, 0.12)
    assert math.isclose(imu_sample.accel_mps2.y, -0.05)
    assert math.isclose(imu_sample.accel_mps2.z, 9.78)
    assert math.isclose(imu_sample.gyro_radps.z, -0.035)
    assert imu_sample.frame == SensorFrame.PHONE_BODY
    assert imu_sample.is_valid is True
    assert imu_sample.is_stationary is False

    gnss_sample = row_to_canonical_gnss(row)
    assert gnss_sample is not None
    assert gnss_sample.timestamp_ns == 10_500_000_000
    assert math.isclose(gnss_sample.latitude_deg, 28.6139)
    assert math.isclose(gnss_sample.longitude_deg, 77.2090)
    assert math.isclose(gnss_sample.speed_mps, 12.5)
    assert math.isclose(gnss_sample.bearing_deg, 180.0)
    assert gnss_sample.satellite_count == 8
    assert gnss_sample.hdop == -1.0


def test_dataset_adapter_alternate_keys():
    """Test robustness to alternate column naming (e.g., raw S-dataset vs synced)."""
    row = {
        "time_ns": 123456789000,
        "ax": 1.0, "ay": 2.0, "az": 3.0,
        "gx": 0.1, "gy": 0.2, "gz": 0.3,
        "stationary": True
    }
    sample = row_to_canonical_imu(row)
    assert sample.timestamp_ns == 123456789000
    assert sample.accel_mps2.x == 1.0
    assert sample.gyro_radps.z == 0.3
    assert sample.is_stationary is True


# ==============================================================================
# 7. Android Adapter Tests
# ==============================================================================

def test_android_adapter_sample_creation():
    """Verify make_android_imu_sample strictly assigns PHONE_BODY and retains values."""
    t_ns = 543210987654
    ax, ay, az = 0.05, 0.10, 9.80
    gx, gy, gz = 0.001, -0.002, 0.015

    sample = make_android_imu_sample(t_ns, ax, ay, az, gx, gy, gz, is_stationary=False)
    assert sample.timestamp_ns == t_ns
    assert sample.accel_mps2.x == ax
    assert sample.accel_mps2.y == ay
    assert sample.accel_mps2.z == az
    assert sample.gyro_radps.x == gx
    assert sample.gyro_radps.y == gy
    assert sample.gyro_radps.z == gz
    assert sample.frame == SensorFrame.PHONE_BODY
    assert sample.is_valid is True


# ==============================================================================
# 8. 3-Source Equivalence Demonstration (User Requirement 14)
# ==============================================================================

def test_three_source_contract_equivalence():
    """
    Demonstrate that:
      A) Python dataset sample
      B) Android sensor sample
      C) External hypothetical sample
    All map into the identical semantic contract (ImuSample) and can be consumed generically.
    """
    # Source A: Python dataset sample (e.g. from IO-VNBD parquet row)
    dataset_row = {
        "time_s": 1.25,
        "acc_x": 0.15, "acc_y": -0.08, "acc_z": 9.79,
        "gyro_x": 0.002, "gyro_y": -0.001, "gyro_z": 0.045
    }
    sample_a = row_to_canonical_imu(dataset_row, frame=SensorFrame.PHONE_BODY)

    # Source B: Android SensorEvent sample
    sample_b = make_android_imu_sample(
        sensor_event_timestamp_ns=1_250_000_000,
        ax_mps2=0.15, ay_mps2=-0.08, az_mps2=9.79,
        gx_radps=0.002, gy_radps=-0.001, gz_radps=0.045
    )

    # Source C: External IMU sample (e.g. CAN bus / serial / Oxford-OxTS)
    # External automotive IMU is mounted in VEHICLE_FRD frame
    sample_c = ImuSample(
        timestamp_ns=1_250_000_000,
        accel_mps2=Vec3(0.15, -0.08, 9.79),
        gyro_radps=Vec3(0.002, -0.001, 0.045),
        frame=SensorFrame.VEHICLE_FRD,
        validity=int(ImuValidity.ALL_VALID),
        is_stationary=False
    )

    # Common generic consumer function: does NOT care where the sample originated!
    def generic_imu_consumer(sample: ImuSample) -> dict:
        assert isinstance(sample.timestamp_ns, int)
        assert isinstance(sample.frame, SensorFrame)
        assert sample.is_valid
        return {
            "t_sec": sample.timestamp_sec,
            "acc_norm": math.sqrt(sample.accel_mps2.x**2 + sample.accel_mps2.y**2 + sample.accel_mps2.z**2),
            "yaw_rate": sample.gyro_radps.z,
            "frame": sample.frame.name
        }

    res_a = generic_imu_consumer(sample_a)
    res_b = generic_imu_consumer(sample_b)
    res_c = generic_imu_consumer(sample_c)

    # Numerical equivalence between A and B (both PHONE_BODY)
    assert res_a["t_sec"] == res_b["t_sec"] == 1.25
    assert math.isclose(res_a["acc_norm"], res_b["acc_norm"], rel_tol=1e-6)
    assert math.isclose(res_a["yaw_rate"], res_b["yaw_rate"], rel_tol=1e-6)
    assert res_a["frame"] == res_b["frame"] == "PHONE_BODY"

    # Semantic equivalence with C (same units, explicit frame tag VEHICLE_FRD)
    assert res_c["t_sec"] == 1.25
    assert math.isclose(res_c["acc_norm"], res_a["acc_norm"], rel_tol=1e-6)
    assert math.isclose(res_c["yaw_rate"], res_a["yaw_rate"], rel_tol=1e-6)
    assert res_c["frame"] == "VEHICLE_FRD"


if __name__ == "__main__":
    test_functions = [
        test_units_rad_per_sec_invariant,
        test_units_mps2_invariant,
        test_units_angular_conversions,
        test_units_speed_conversions,
        test_units_gravity_conversion,
        test_timestamp_conversions,
        test_frame_enum_definitions,
        test_imu_sample_validation,
        test_gnss_sample_validation,
        test_dataset_adapter_row_conversion,
        test_dataset_adapter_alternate_keys,
        test_android_adapter_sample_creation,
        test_three_source_contract_equivalence,
    ]

    print("=" * 60)
    print("iNAV Phase 2 - Running Canonical Sensor Abstraction Tests")
    print("=" * 60)
    passed = 0
    for tf in test_functions:
        try:
            tf()
            print(f"[PASS] {tf.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {tf.__name__}: {e}")
            raise
    print("=" * 60)
    print(f"Results: {passed}/{len(test_functions)} tests passed successfully.")
    print("=" * 60)

