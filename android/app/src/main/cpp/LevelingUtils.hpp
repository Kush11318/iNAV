#ifndef LEVELING_UTILS_HPP
#define LEVELING_UTILS_HPP

#include <cmath>
#include <array>

namespace inav {

/**
 * @brief Quaternion structure for 3D coordinate transformations
 */
struct Quaternion {
    double w{1.0}, x{0.0}, y{0.0}, z{0.0};

    // Rotates a 3D vector v by this unit quaternion: v' = q * v * q^-1
    std::array<double, 3> rotate(const std::array<double, 3>& v) const {
        double vx = v[0], vy = v[1], vz = v[2];
        
        // Quat vector product
        double tx = 2.0 * (y * vz - z * vy);
        double ty = 2.0 * (z * vx - x * vz);
        double tz = 2.0 * (x * vy - y * vx);

        double rx = vx + w * tx + (y * tz - z * ty);
        double ry = vy + w * ty + (z * tx - x * tz);
        double rz = vz + w * tz + (x * ty - y * tx);

        return {rx, ry, rz};
    }
};

class LevelingUtils {
public:
    /**
     * @brief Estimates static Pitch and Roll from stationary accelerometer readings.
     * @param acc_static 3-axis accelerometer vector during standstill [ax, ay, az] (m/s^2)
     * @param out_pitch Output Pitch angle in radians
     * @param out_roll Output Roll angle in radians
     */
    static void computeStaticPitchRoll(const std::array<double, 3>& acc_static,
                                      double& out_pitch,
                                      double& out_roll) {
        double ax = acc_static[0];
        double ay = acc_static[1];
        double az = acc_static[2];

        // Roll: rotation around X-axis
        out_roll = std::atan2(-ay, -az);
        
        // Pitch: rotation around Y-axis
        out_pitch = std::atan2(ax, std::sqrt(ay * ay + az * az));
    }

    /**
     * @brief Transforms raw phone accelerometer data to the vehicle frame and isolates vertical linear acceleration.
     * @param raw_acc 3-axis phone acceleration [ax, ay, az] (m/s^2)
     * @param q_phone_to_veh Unit quaternion representing phone-to-vehicle rotation matrix
     * @param gravity_constant Nominal gravity (default: 9.80665 m/s^2)
     * @return Formatted struct with vehicle frame accelerations and vertical linear force
     */
    struct LevelingResult {
        std::array<double, 3> acc_vehicle; // [ax_long, ay_lat, az_vert]
        double az_linear;                   // Vertical linear acceleration (az_vert - g)
    };

    static LevelingResult levelAndDetrend(const std::array<double, 3>& raw_acc,
                                          const Quaternion& q_phone_to_veh,
                                          double gravity_constant = 9.80665) {
        LevelingResult result;
        
        // 1. Rotate raw phone acceleration into the vehicle frame
        result.acc_vehicle = q_phone_to_veh.rotate(raw_acc);

        // 2. Extract vertical axis (Z) and compensate for Earth gravity
        // Positive shock is upward rebound, negative shock is crater drop
        result.az_linear = result.acc_vehicle[2] - gravity_constant;

        return result;
    }
};

} // namespace inav

#endif // LEVELING_UTILS_HPP
