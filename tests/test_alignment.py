"""
iNAV Phase 3 - Comprehensive Alignment & Calibration Test Suite.

Validates all 19 required numerical test cases:
1. Flat phone leveling
2. +Pitch recovery
3. -Pitch recovery
4. +Roll recovery
5. -Roll recovery
6. Known yaw mounting offset
7. Gyro bias injection
8. Zero corrected gyro output
9. Body -> vehicle acceleration transform
10. Body -> vehicle gyro transform
11. Gravity detrending cancellation
12. Clockwise vehicle turn yaw sign parity
13. Counter-clockwise vehicle turn yaw sign parity
14. Insufficient PCA excitation rejection
15. Braking during PCA sign resolution
16. Remount detection
17. Remount debounce
18. Stationary gate threshold enforcement
19. Moving calibration rejection
"""

import math
import sys
from pathlib import Path

# Ensure iNAV root is in Python path
INAV_ROOT = Path(__file__).resolve().parent.parent
if str(INAV_ROOT) not in sys.path:
    sys.path.insert(0, str(INAV_ROOT))

import numpy as np

from modules.sensor_types import (
    SensorFrame,
    Vec3,
    ImuSample,
    GnssSample,
    ImuValidity,
    GnssValidity,
    units
)
from modules.alignment import (
    AlignmentEngine,
    AlignmentConfig,
    AlignmentState,
    AlignmentResult
)


G = 9.80665


# ==============================================================================
# 1. Flat Phone Leveling
# ==============================================================================
def test_1_flat_phone():
    """Flat face-up phone resting on table: specific force is [0, 0, +g] in PHONE_BODY."""
    engine = AlignmentEngine()
    acc_samples = [Vec3(0.0, 0.0, G) for _ in range(30)]
    gyro_samples = [Vec3(0.0, 0.0, 0.0) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_samples, gyro_samples)
    assert success is True
    assert engine.result.state == AlignmentState.STATIC_ALIGNED

    # Down unit vector in body frame must point along -Z_b
    assert np.allclose(engine.u_z_body, [0.0, 0.0, -1.0], atol=1e-4)

    # R_b_to_v * [0, 0, +g] must equal [0, 0, -g] in VEHICLE_FRD
    transformed_g = engine.result.R_b_to_v @ np.array([0.0, 0.0, G])
    assert math.isclose(transformed_g[2], -G, abs_tol=1e-4)
    assert math.isclose(transformed_g[0], 0.0, abs_tol=1e-4)
    assert math.isclose(transformed_g[1], 0.0, abs_tol=1e-4)

    # Pitch and roll relative to horizon must be ~0
    assert abs(engine.result.pitch_deg) < 1e-3
    assert abs(engine.result.roll_deg) < 1e-3


# ==============================================================================
# 2. +Pitch Recovery (Tilted 30 deg nose up)
# ==============================================================================
def test_2_positive_pitch():
    """Tilted +30 degrees nose up: specific force reaction points uphill (+X)."""
    engine = AlignmentEngine()
    pitch_rad = math.radians(30.0)
    ax = G * math.sin(pitch_rad)
    ay = 0.0
    az = G * math.cos(pitch_rad)

    acc_samples = [Vec3(ax, ay, az) for _ in range(30)]
    gyro_samples = [Vec3(0.0, 0.0, 0.0) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_samples, gyro_samples)
    assert success is True
    assert math.isclose(engine.result.pitch_deg, 30.0, abs_tol=0.1)
    assert math.isclose(engine.result.roll_deg, 0.0, abs_tol=0.1)


# ==============================================================================
# 3. -Pitch Recovery (Tilted 30 deg nose down)
# ==============================================================================
def test_3_negative_pitch():
    """Tilted -30 degrees nose down: specific force reaction points downhill (-X)."""
    engine = AlignmentEngine()
    pitch_rad = math.radians(-30.0)
    ax = G * math.sin(pitch_rad)
    ay = 0.0
    az = G * math.cos(pitch_rad)

    acc_samples = [Vec3(ax, ay, az) for _ in range(30)]
    gyro_samples = [Vec3(0.0, 0.0, 0.0) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_samples, gyro_samples)
    assert success is True
    assert math.isclose(engine.result.pitch_deg, -30.0, abs_tol=0.1)
    assert math.isclose(engine.result.roll_deg, 0.0, abs_tol=0.1)


# ==============================================================================
# 4. +Roll Recovery (Tilted 30 deg right side down)
# ==============================================================================
def test_4_positive_roll():
    """Tilted +30 degrees right side down: reaction force points uphill (-Y)."""
    engine = AlignmentEngine()
    roll_rad = math.radians(30.0)
    ax = 0.0
    ay = -G * math.sin(roll_rad)
    az = G * math.cos(roll_rad)

    acc_samples = [Vec3(ax, ay, az) for _ in range(30)]
    gyro_samples = [Vec3(0.0, 0.0, 0.0) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_samples, gyro_samples)
    assert success is True
    assert math.isclose(engine.result.pitch_deg, 0.0, abs_tol=0.1)
    assert math.isclose(engine.result.roll_deg, 30.0, abs_tol=0.1)


# ==============================================================================
# 5. -Roll Recovery (Tilted 30 deg left side down)
# ==============================================================================
def test_5_negative_roll():
    """Tilted -30 degrees left side down: reaction force points uphill (+Y)."""
    engine = AlignmentEngine()
    roll_rad = math.radians(-30.0)
    ax = 0.0
    ay = -G * math.sin(roll_rad)
    az = G * math.cos(roll_rad)

    acc_samples = [Vec3(ax, ay, az) for _ in range(30)]
    gyro_samples = [Vec3(0.0, 0.0, 0.0) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_samples, gyro_samples)
    assert success is True
    assert math.isclose(engine.result.pitch_deg, 0.0, abs_tol=0.1)
    assert math.isclose(engine.result.roll_deg, -30.0, abs_tol=0.1)


# ==============================================================================
# 6. Known Yaw Mounting Offset (Landscape Mount: +X Forward)
# ==============================================================================
def test_6_known_yaw_mounting():
    """Phone mounted landscape flat face-up, with phone +X pointing along vehicle forward."""
    engine = AlignmentEngine()
    # Step 1: Static leveling flat
    acc_static = [Vec3(0.0, 0.0, G) for _ in range(25)]
    gyro_static = [Vec3(0.0, 0.0, 0.0) for _ in range(25)]
    engine.calibrate_static_buffer(acc_static, gyro_static)

    # Step 2: Forward acceleration along phone +X (driving forward)
    # Accelerations with significant variability along phone X
    acc_driving = []
    gyro_driving = []
    speed_deltas = []
    for i in range(40):
        # Forward acceleration varying between 0.5 and 2.5 m/s^2
        a_fwd = 1.5 + 0.8 * math.sin(i * 0.2)
        acc_driving.append(Vec3(a_fwd, 0.0, G))
        gyro_driving.append(Vec3(0.0, 0.0, 0.0))
        speed_deltas.append(a_fwd * 0.1)

    success = engine.calibrate_dynamic_buffer(acc_driving, gyro_driving, speed_deltas)
    assert success is True
    assert engine.result.state == AlignmentState.FULL_ALIGNED

    # Forward row of R_b_to_v should align with body +X: [1, 0, 0]
    R = engine.result.R_b_to_v
    assert math.isclose(R[0, 0], 1.0, abs_tol=0.05)
    assert math.isclose(R[0, 1], 0.0, abs_tol=0.05)
    assert math.isclose(R[0, 2], 0.0, abs_tol=0.05)


# ==============================================================================
# 7. Gyro Bias Injection & Recovery
# ==============================================================================
def test_7_gyro_bias_injection():
    """Synthetic gyro bias injected during stationary calibration is accurately recovered."""
    engine = AlignmentEngine()
    injected_bias = Vec3(0.01, -0.02, 0.03)

    acc_samples = [Vec3(0.0, 0.0, G) for _ in range(30)]
    gyro_samples = [Vec3(injected_bias.x, injected_bias.y, injected_bias.z) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_samples, gyro_samples)
    assert success is True
    assert math.isclose(engine.result.gyro_bias_body.x, 0.01, abs_tol=1e-4)
    assert math.isclose(engine.result.gyro_bias_body.y, -0.02, abs_tol=1e-4)
    assert math.isclose(engine.result.gyro_bias_body.z, 0.03, abs_tol=1e-4)


# ==============================================================================
# 8. Zero Corrected Gyro Output
# ==============================================================================
def test_8_zero_corrected_gyro():
    """Stationary sensor with bias transforms to [0, 0, 0] in vehicle frame."""
    engine = AlignmentEngine()
    injected_bias = Vec3(0.01, -0.02, 0.03)

    acc_samples = [Vec3(0.0, 0.0, G) for _ in range(30)]
    gyro_samples = [Vec3(injected_bias.x, injected_bias.y, injected_bias.z) for _ in range(30)]
    engine.calibrate_static_buffer(acc_samples, gyro_samples)

    raw_sample = ImuSample(
        timestamp_ns=1_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, G),
        gyro_radps=Vec3(injected_bias.x, injected_bias.y, injected_bias.z),
        frame=SensorFrame.PHONE_BODY
    )
    transformed = engine.transform_imu(raw_sample)

    assert transformed.frame == SensorFrame.VEHICLE_FRD
    assert math.isclose(transformed.gyro_radps.x, 0.0, abs_tol=1e-5)
    assert math.isclose(transformed.gyro_radps.y, 0.0, abs_tol=1e-5)
    assert math.isclose(transformed.gyro_radps.z, 0.0, abs_tol=1e-5)


# ==============================================================================
# 9. Body -> Vehicle Acceleration Transform
# ==============================================================================
def test_9_body_to_vehicle_accel_transform():
    """Specific force vector properly rotates into vehicle frame."""
    engine = AlignmentEngine()
    # Portrait mount flat: +Y is forward, +X is right, +Z is up
    # Static leveling:
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    # Driving forward along body +Y
    acc_driving = [Vec3(0.0, 1.5 + 1.0 * math.sin(i), G) for i in range(40)]
    gyro_driving = [Vec3(0, 0, 0) for _ in range(40)]
    speed_deltas = [0.15 for _ in range(40)]
    engine.calibrate_dynamic_buffer(acc_driving, gyro_driving, speed_deltas)

    # In portrait mount, forward acceleration along +Y_b must map to +X_v (Vehicle Forward)
    test_sample = ImuSample(
        timestamp_ns=2_000_000_000,
        accel_mps2=Vec3(0.0, 2.0, G),
        gyro_radps=Vec3(0.0, 0.0, 0.0),
        frame=SensorFrame.PHONE_BODY
    )
    veh_sample = engine.transform_imu(test_sample)
    assert math.isclose(veh_sample.accel_mps2.x, 2.0, abs_tol=0.1)
    assert math.isclose(veh_sample.accel_mps2.z, -G, abs_tol=0.1)


# ==============================================================================
# 10. Body -> Vehicle Gyro Transform
# ==============================================================================
def test_10_body_to_vehicle_gyro_transform():
    """Body rotation rates rotate faithfully into vehicle frame."""
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    # Portrait flat (+Y fwd, +X right, +Z up). FRD (+X fwd, +Y right, +Z down).
    acc_driving = [Vec3(0.0, 1.5 + 1.0 * math.sin(i), G) for i in range(40)]
    engine.calibrate_dynamic_buffer(acc_driving, [Vec3(0, 0, 0) for _ in range(40)], [0.15]*40)

    # Roll about vehicle forward (+X_v) corresponds to roll about phone +Y_b
    test_sample = ImuSample(
        timestamp_ns=2_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, G),
        gyro_radps=Vec3(0.0, 0.10, 0.0), # Rate around body +Y
        frame=SensorFrame.PHONE_BODY
    )
    veh_sample = engine.transform_imu(test_sample)
    assert math.isclose(veh_sample.gyro_radps.x, 0.10, abs_tol=0.05)


# ==============================================================================
# 11. Gravity Detrending Cancellation
# ==============================================================================
def test_11_gravity_detrending():
    """At rest, linear acceleration detrends cleanly to [0, 0, 0] without doubling."""
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    sample_at_rest = ImuSample(
        timestamp_ns=1_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, G),
        gyro_radps=Vec3(0.0, 0.0, 0.0),
        frame=SensorFrame.PHONE_BODY
    )
    veh_sample = engine.transform_imu(sample_at_rest)
    lin_acc = engine.get_linear_accel(veh_sample)

    assert math.isclose(lin_acc.x, 0.0, abs_tol=1e-4)
    assert math.isclose(lin_acc.y, 0.0, abs_tol=1e-4)
    assert math.isclose(lin_acc.z, 0.0, abs_tol=1e-4)
    # Definitively verify that gravity is NOT doubled to -19.62
    assert not math.isclose(lin_acc.z, -2.0 * G, abs_tol=1.0)


# ==============================================================================
# 12. Clockwise Vehicle Turn Yaw Sign Parity (Critical Bug Check)
# ==============================================================================
def test_12_clockwise_yaw_parity():
    """
    CRITICAL CHECK:
    Vehicle turns RIGHT (clockwise looking down):
    In VEHICLE_FRD, omega_z_veh MUST BE POSITIVE (> 0).
    On flat phone (+Z pointing UP), clockwise looking down is NEGATIVE in right-hand rule.
    The alignment transformation MUST flip the sign so omega_z_veh > 0!
    """
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    # Driving forward
    acc_driving = [Vec3(0.0, 1.5 + 1.0 * math.sin(i), G) for i in range(40)]
    engine.calibrate_dynamic_buffer(acc_driving, [Vec3(0, 0, 0) for _ in range(40)], [0.15]*40)

    # Clockwise rotation: phone gyro reports negative around +Z_phone
    sample_right_turn = ImuSample(
        timestamp_ns=2_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, G),
        gyro_radps=Vec3(0.0, 0.0, -0.05), # Clockwise turn in phone frame
        frame=SensorFrame.PHONE_BODY
    )
    veh_sample = engine.transform_imu(sample_right_turn)

    # In vehicle frame (+Z down), turning right MUST be strictly positive!
    assert veh_sample.gyro_radps.z > 0.0
    assert math.isclose(veh_sample.gyro_radps.z, +0.05, abs_tol=1e-3)


# ==============================================================================
# 13. Counter-Clockwise Vehicle Turn Yaw Sign Parity
# ==============================================================================
def test_13_counter_clockwise_yaw_parity():
    """
    Vehicle turns LEFT (counter-clockwise looking down):
    In VEHICLE_FRD, omega_z_veh MUST BE NEGATIVE (< 0).
    On flat phone (+Z pointing UP), counter-clockwise is POSITIVE in right-hand rule.
    The alignment transformation MUST flip the sign so omega_z_veh < 0!
    """
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    acc_driving = [Vec3(0.0, 1.5 + 1.0 * math.sin(i), G) for i in range(40)]
    engine.calibrate_dynamic_buffer(acc_driving, [Vec3(0, 0, 0) for _ in range(40)], [0.15]*40)

    # Counter-clockwise rotation: phone gyro reports positive around +Z_phone
    sample_left_turn = ImuSample(
        timestamp_ns=2_000_000_000,
        accel_mps2=Vec3(0.0, 0.0, G),
        gyro_radps=Vec3(0.0, 0.0, +0.05), # Counter-clockwise turn in phone frame
        frame=SensorFrame.PHONE_BODY
    )
    veh_sample = engine.transform_imu(sample_left_turn)

    # In vehicle frame (+Z down), turning left MUST be strictly negative!
    assert veh_sample.gyro_radps.z < 0.0
    assert math.isclose(veh_sample.gyro_radps.z, -0.05, abs_tol=1e-3)


# ==============================================================================
# 14. Insufficient PCA Excitation Rejection
# ==============================================================================
def test_14_insufficient_pca_excitation():
    """PCA dynamic alignment rejects low-variability stationary/engine noise."""
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    # Standstill engine rumble noise: horizontal std < 0.1 m/s^2
    np.random.seed(42)
    acc_noise = []
    for _ in range(40):
        nx = np.random.normal(0.0, 0.05)
        ny = np.random.normal(0.0, 0.05)
        acc_noise.append(Vec3(nx, ny, G))

    success = engine.calibrate_dynamic_buffer(acc_noise, [Vec3(0, 0, 0) for _ in range(40)])
    # Must reject and remain in STATIC_ALIGNED / DYNAMIC_ALIGNING
    assert success is False
    assert engine.result.state != AlignmentState.FULL_ALIGNED


# ==============================================================================
# 15. Braking During PCA Sign Resolution
# ==============================================================================
def test_15_braking_during_pca():
    """
    Decelerating/braking while driving forward:
    speed delta is negative during positive forward braking.
    Sign resolution must NOT invert forward heading into reverse!
    """
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    # Phone portrait flat: forward is +Y_b.
    # Braking from 20 m/s down to 5 m/s: acceleration vector is negative along forward axis (-Y_b)
    acc_braking = []
    speed_deltas = []
    for i in range(40):
        a_brake = -1.8 - 0.8 * math.sin(i * 0.2) # Negative along body Y
        acc_braking.append(Vec3(0.0, a_brake, G))
        speed_deltas.append(a_brake * 0.1) # Negative speed delta

    success = engine.calibrate_dynamic_buffer(acc_braking, [Vec3(0, 0, 0) for _ in range(40)], speed_deltas)
    assert success is True

    # Forward direction (+X_v) must correctly identify body +Y as forward, NOT -Y!
    R = engine.result.R_b_to_v
    assert math.isclose(R[0, 1], 1.0, abs_tol=0.1) # Forward is +Y_b
    assert not math.isclose(R[0, 1], -1.0, abs_tol=0.1) # Not inverted into reverse!


# ==============================================================================
# 16. Remount Detection
# ==============================================================================
def test_16_remount_detection():
    """Angular tilt of 15 degrees trips remount detector after debounce."""
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])
    assert engine.result.is_aligned is True

    # Sudden mount shift: tilted by 20 degrees
    shift_rad = math.radians(20.0)
    shifted_acc = Vec3(0.0, G * math.sin(shift_rad), G * math.cos(shift_rad))

    remount_tripped = False
    # Feed shifted gravity repeatedly to satisfy low-pass filter and debounce
    for _ in range(25):
        if engine.check_remount(shifted_acc):
            remount_tripped = True
            break

    assert remount_tripped is True
    assert engine.result.state == AlignmentState.REMOUNT_DETECTED
    assert engine.result.remount_flag is True


# ==============================================================================
# 17. Remount Debounce
# ==============================================================================
def test_17_remount_debounce():
    """Transient shock/bump lasting only 1-2 epochs does not trip remount."""
    engine = AlignmentEngine()
    engine.calibrate_static_buffer([Vec3(0, 0, G) for _ in range(30)],
                                   [Vec3(0, 0, 0) for _ in range(30)])

    # Single pothole shock
    shock_acc = Vec3(0.0, 5.0, G + 8.0)
    for _ in range(2):
        engine.check_remount(shock_acc)

    # Returns to normal flat resting
    for _ in range(5):
        engine.check_remount(Vec3(0.0, 0.0, G))

    assert engine.result.state != AlignmentState.REMOUNT_DETECTED
    assert engine.result.remount_flag is False


# ==============================================================================
# 18. Stationary Gate Threshold Enforcement
# ==============================================================================
def test_18_stationary_gate():
    """Moving vehicle with high gyro or accel variance rejects static leveling."""
    engine = AlignmentEngine()
    # High variance due to driving vibration
    acc_moving = [Vec3(0.0, 0.0, G + 0.35 * math.sin(i)) for i in range(30)]
    gyro_moving = [Vec3(0.0, 0.0, 0.08 * math.sin(i)) for i in range(30)]

    success = engine.calibrate_static_buffer(acc_moving, gyro_moving)
    assert success is False
    assert engine.result.state == AlignmentState.UNINITIALIZED


# ==============================================================================
# 19. Moving Calibration Rejection (Centripetal / Dynamic Non-Gravity)
# ==============================================================================
def test_19_moving_calibration_rejection():
    """Acceleration norm differing significantly from 1g (e.g. cornering) is rejected."""
    engine = AlignmentEngine()
    # Vehicle cornering at high speed: total specific force norm = 11.5 m/s^2 > 1g
    acc_cornering = [Vec3(4.0, 0.0, G) for _ in range(30)] # norm ≈ 10.59
    gyro_static = [Vec3(0.0, 0.0, 0.0) for _ in range(30)]

    success = engine.calibrate_static_buffer(acc_cornering, gyro_static)
    assert success is False
    assert engine.result.state == AlignmentState.UNINITIALIZED


if __name__ == "__main__":
    tests = [
        test_1_flat_phone,
        test_2_positive_pitch,
        test_3_negative_pitch,
        test_4_positive_roll,
        test_5_negative_roll,
        test_6_known_yaw_mounting,
        test_7_gyro_bias_injection,
        test_8_zero_corrected_gyro,
        test_9_body_to_vehicle_accel_transform,
        test_10_body_to_vehicle_gyro_transform,
        test_11_gravity_detrending,
        test_12_clockwise_yaw_parity,
        test_13_counter_clockwise_yaw_parity,
        test_14_insufficient_pca_excitation,
        test_15_braking_during_pca,
        test_16_remount_detection,
        test_17_remount_debounce,
        test_18_stationary_gate,
        test_19_moving_calibration_rejection,
    ]

    print("=" * 65)
    print("iNAV Phase 3 - Running 19-Case Alignment & Calibration Test Suite")
    print("=" * 65)
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            raise

    print("=" * 65)
    print(f"Results: {passed}/{len(tests)} tests passed successfully.")
    print("=" * 65)
