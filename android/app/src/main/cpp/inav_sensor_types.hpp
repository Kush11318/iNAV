#pragma once

#ifndef INAV_NO_STDLIB
#include <cstdint>
#include <cmath>
#else
using uint8_t = unsigned char;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long;
#ifndef M_PI
static constexpr double M_PI = 3.14159265358979323846;
#endif
#endif

namespace inav {

/**
 * ==============================================================================
 * CANONICAL COORDINATE FRAMES
 * ==============================================================================
 * Explicit frame enumeration. Magic integers are forbidden.
 *
 * PHONE_BODY:
 *   Right-handed coordinate system affixed to the smartphone chassis (Android standard):
 *     +X : Points to the right edge of the device display.
 *     +Y : Points to the top edge of the device display (display 'up').
 *     +Z : Points perpendicularly outward from the front glass/screen.
 *
 * VEHICLE_FRD:
 *   Right-handed vehicle-body coordinate system standard in automotive/aerospace:
 *     +X : Vehicle Forward (longitudinal axis).
 *     +Y : Vehicle Right (lateral axis).
 *     +Z : Vehicle Down (normal axis, positive towards ground).
 *   Yaw is positive clockwise looking down from above (Euler Z-axis down).
 *
 * WORLD_NED:
 *   Right-handed local geodetic tangent frame:
 *     +X : True North.
 *     +Y : True East.
 *     +Z : Down (towards center of Earth).
 */
enum class SensorFrame : uint8_t {
    UNKNOWN = 0,
    PHONE_BODY = 1,
    VEHICLE_FRD = 2,
    WORLD_NED = 3
};

/**
 * 3-Dimensional Vector.
 * Frame must be specified by the enclosing sample or context;
 * raw Vec3 without frame context is strictly forbidden in filter updates.
 */
struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

/**
 * Validity bitflags for ImuSample.
 */
namespace ImuValidity {
    constexpr uint32_t NONE             = 0;
    constexpr uint32_t ACCEL_VALID      = (1 << 0);
    constexpr uint32_t GYRO_VALID       = (1 << 1);
    constexpr uint32_t TIMESTAMP_VALID  = (1 << 2);
    constexpr uint32_t STATIONARY_VALID = (1 << 3);
    constexpr uint32_t ALL_VALID        = (ACCEL_VALID | GYRO_VALID | TIMESTAMP_VALID);
}

/**
 * Validity bitflags for GnssSample.
 */
namespace GnssValidity {
    constexpr uint32_t NONE              = 0;
    constexpr uint32_t LAT_LON_VALID     = (1 << 0);
    constexpr uint32_t ALTITUDE_VALID    = (1 << 1);
    constexpr uint32_t SPEED_VALID       = (1 << 2);
    constexpr uint32_t BEARING_VALID     = (1 << 3);
    constexpr uint32_t ACCURACY_VALID    = (1 << 4);
    constexpr uint32_t SAT_COUNT_VALID   = (1 << 5);
    constexpr uint32_t HDOP_VALID        = (1 << 6);
    constexpr uint32_t BASIC_FIX_VALID   = (LAT_LON_VALID | SPEED_VALID | BEARING_VALID | ACCURACY_VALID);
}

/**
 * ==============================================================================
 * 1. CANONICAL IMU SAMPLE
 * ==============================================================================
 *
 * Explicit Unit & Sign Contracts:
 *   - timestamp_ns : int64_t nanoseconds since monotonic/epoch origin.
 *   - accel_mps2   : Linear acceleration + gravity in meters per second squared (m/s²).
 *   - gyro_radps   : Angular rate of rotation in radians per second (rad/s).
 *                    Right-handed sign convention around the frame's axes.
 *   - frame        : Explicit SensorFrame enum (PHONE_BODY, VEHICLE_FRD, etc.).
 *   - validity     : Bitmask from ImuValidity flags.
 *   - is_stationary: Boolean flag indicating zero-velocity / stationary detection if available.
 */
struct ImuSample {
    int64_t timestamp_ns{0};
    Vec3 accel_mps2{0.0, 0.0, 0.0};
    Vec3 gyro_radps{0.0, 0.0, 0.0};
    SensorFrame frame{SensorFrame::UNKNOWN};
    uint32_t validity{ImuValidity::NONE};
    bool is_stationary{false};

    constexpr ImuSample() = default;
    constexpr ImuSample(int64_t t_ns,
                        const Vec3& a,
                        const Vec3& g,
                        SensorFrame f,
                        uint32_t v = ImuValidity::ALL_VALID,
                        bool stat = false)
        : timestamp_ns(t_ns), accel_mps2(a), gyro_radps(g), frame(f), validity(v), is_stationary(stat) {}

    [[nodiscard]] bool isValid() const noexcept {
        return (validity & ImuValidity::ALL_VALID) == ImuValidity::ALL_VALID;
    }
};

/**
 * ==============================================================================
 * 2. CANONICAL GNSS SAMPLE
 * ==============================================================================
 *
 * Explicit Unit & Sign Contracts:
 *   - timestamp_ns          : int64_t nanoseconds.
 *   - latitude_deg          : WGS84 latitude in decimal degrees [-90.0, +90.0].
 *   - longitude_deg         : WGS84 longitude in decimal degrees [-180.0, +180.0].
 *   - altitude_m            : Height above WGS84 ellipsoid or MSL in meters (m).
 *   - speed_mps             : Horizontal ground speed in meters per second (m/s). Must be >= 0.
 *   - bearing_deg           : Ground course track in degrees [0.0, 360.0) clockwise from True North.
 *   - horizontal_accuracy_m : 1-sigma horizontal position accuracy estimate in meters (m).
 *   - satellite_count       : Number of satellites used in fix. (-1 if unknown).
 *   - hdop                  : Horizontal Dilution of Precision. Optional, only valid if HDOP_VALID set.
 *                             Do NOT fabricate HDOP if source does not provide it.
 *   - validity              : Bitmask from GnssValidity flags.
 */
struct GnssSample {
    int64_t timestamp_ns{0};
    double latitude_deg{0.0};
    double longitude_deg{0.0};
    double altitude_m{0.0};
    double speed_mps{0.0};
    double bearing_deg{0.0};
    double horizontal_accuracy_m{0.0};
    int32_t satellite_count{-1};
    double hdop{-1.0};
    uint32_t validity{GnssValidity::NONE};

    constexpr GnssSample() = default;

    [[nodiscard]] bool hasBasicFix() const noexcept {
        return (validity & GnssValidity::BASIC_FIX_VALID) == GnssValidity::BASIC_FIX_VALID;
    }
};

/**
 * ==============================================================================
 * 4 & 6. CENTRALIZED UNIT & TIMESTAMP CONVERSION HELPERS
 * ==============================================================================
 */
namespace units {
    // Timestamp Conversions (to canonical int64 nanoseconds)
    [[nodiscard]] constexpr int64_t sec_to_ns(double sec) noexcept {
        return static_cast<int64_t>(sec * 1e9);
    }

    [[nodiscard]] constexpr int64_t ms_to_ns(double ms) noexcept {
        return static_cast<int64_t>(ms * 1e6);
    }

    [[nodiscard]] constexpr int64_t ms_to_ns(int64_t ms) noexcept {
        return ms * 1000000LL;
    }

    [[nodiscard]] constexpr double ns_to_sec(int64_t ns) noexcept {
        return static_cast<double>(ns) * 1e-9;
    }

    [[nodiscard]] constexpr double ns_to_ms(int64_t ns) noexcept {
        return static_cast<double>(ns) * 1e-6;
    }

    // Kinematics / Angular Conversions
    [[nodiscard]] constexpr double deg_to_rad(double deg) noexcept {
        return deg * (M_PI / 180.0);
    }

    [[nodiscard]] constexpr double rad_to_deg(double rad) noexcept {
        return rad * (180.0 / M_PI);
    }

    [[nodiscard]] constexpr double kmh_to_mps(double kmh) noexcept {
        return kmh / 3.6;
    }

    [[nodiscard]] constexpr double mps_to_kmh(double mps) noexcept {
        return mps * 3.6;
    }

    [[nodiscard]] constexpr double g_to_mps2(double g) noexcept {
        return g * 9.80665;
    }
} // namespace units

/**
 * ==============================================================================
 * 9. ANDROID ADAPTER BOUNDARY
 * ==============================================================================
 * Standard Android SensorEvent:
 *   - timestamp: nanoseconds since system boot (uptime/elapsedRealtimeNano)
 *   - sensor values:
 *       TYPE_ACCELEROMETER: m/s² (+X right, +Y up, +Z out of screen)
 *       TYPE_GYROSCOPE    : rad/s (+X right, +Y up, +Z out of screen, RH rule)
 *
 * The Android sensor is inherently in PHONE_BODY frame.
 * Alignment to VEHICLE_FRD is explicitly deferred to Phase 3.
 */
inline ImuSample make_android_imu_sample(int64_t sensor_event_timestamp_ns,
                                        double ax_mps2, double ay_mps2, double az_mps2,
                                        double gx_radps, double gy_radps, double gz_radps) {
    return ImuSample(
        sensor_event_timestamp_ns,
        Vec3(ax_mps2, ay_mps2, az_mps2),
        Vec3(gx_radps, gy_radps, gz_radps),
        SensorFrame::PHONE_BODY,
        ImuValidity::ALL_VALID,
        false
    );
}

} // namespace inav
