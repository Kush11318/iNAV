#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "Butterworth2to8Hz.hpp"
#include "LevelingUtils.hpp"

namespace inav {

constexpr double EARTH_RADIUS = 6371000.0;
constexpr double PI = 3.14159265358979323846;

struct NavState {
    double lat;             // Latitude (degrees)
    double lon;             // Longitude (degrees)
    double speed_ms;        // Forward speed (m/s)
    double heading_deg;     // Heading clockwise from North (degrees)
    double gyro_bias;       // Gyroscope yaw bias (rad/s)
    double accel_bias;      // Accelerometer bias (m/s^2)
    double scale_factor_k;  // Learned scale factor
};

/**
 * @brief Road Anomaly Classification Result
 * type: 0 = None / Normal, 1 = Pothole (crater drop), 2 = Speed Breaker (upward bump)
 */
struct AnomalyResult {
    int type{0};
    double filtered_bump{0.0};
    bool is_anomaly{false};
};

class DeadReckoningFilter {
public:
    DeadReckoningFilter(double dt = 0.1)
        : dt_(dt), p_N_(0.0), p_E_(0.0), v_(0.0), psi_(0.0),
          bg_(0.0), ba_(0.0), k_scale_(1.0),
          ref_lat_(0.0), ref_lon_(0.0), is_initialized_(false),
          butterworth_(20.0),
          gnss_speed_(0.0), is_gnss_healthy_(false),
          has_velocity_net_input_(false), velocity_net_speed_(0.0),
          has_obd_speed_(false), obd_wheel_speed_(0.0) {}

    void initialize(double init_lat, double init_lon, double init_speed_ms, double init_heading_deg) {
        ref_lat_ = init_lat;
        ref_lon_ = init_lon;
        p_N_ = 0.0;
        p_E_ = 0.0;
        v_ = (init_speed_ms < 0.3) ? 0.0 : init_speed_ms;
        psi_ = init_heading_deg * (PI / 180.0);
        bg_ = 0.0;
        ba_ = 0.0;
        k_scale_ = 1.0;
        is_initialized_ = true;
        butterworth_.reset();
        anomaly_counter_ = 0;
        last_anomaly_type_ = 0;
        has_velocity_net_input_ = false;
        velocity_net_speed_ = 0.0;
        has_obd_speed_ = false;
        obd_wheel_speed_ = 0.0;
        gnss_speed_ = 0.0;
        is_gnss_healthy_ = false;
    }

    void set_gnss_state(double gnss_speed, bool is_gnss_healthy) {
        gnss_speed_ = std::max(0.0, gnss_speed);
        is_gnss_healthy_ = is_gnss_healthy;
    }

    void set_obd_speed(double speed_ms, bool is_connected) {
        obd_wheel_speed_ = std::max(0.0, speed_ms);
        has_obd_speed_ = is_connected;
    }

    /**
     * @brief 3-Step Road Anomaly Detection Pipeline:
     * 1. Coordinate leveling onto vehicle Z-axis (subtracting gravity)
     * 2. 2-8 Hz Butterworth bandpass filtering (suspension resonance)
     * 3. Vehicle speed-gated transient classification (Pothole vs Speed Breaker)
     */
    AnomalyResult process_road_anomaly(
        double ax, double ay, double az,
        const Quaternion& q_phone_to_veh,
        double speed_ms
    ) {
        AnomalyResult res;

        // Step 1: Coordinate Leveling & Earth Gravity Detrending
        auto leveled = LevelingUtils::levelAndDetrend({ax, ay, az}, q_phone_to_veh, 9.80665);

        // Step 2: 2-8 Hz Butterworth Bandpass Filter
        double filtered = butterworth_.process(leveled.az_linear);
        res.filtered_bump = filtered;
        last_filtered_bump_ = filtered;

        // Step 3: Speed-Gated Signature Classification (> 15 km/h = 4.16 m/s)
        if (speed_ms >= 4.16) {
            if (filtered < -1.8) {
                // Crater drop -> Pothole
                res.type = 1;
                res.is_anomaly = true;
                anomaly_counter_ = 12;
                last_anomaly_type_ = 1;
            } else if (filtered > +1.8) {
                // Compression bump -> Speed Breaker
                res.type = 2;
                res.is_anomaly = true;
                anomaly_counter_ = 12;
                last_anomaly_type_ = 2;
            }
        }

        if (anomaly_counter_ > 0) {
            anomaly_counter_--;
            res.is_anomaly = true;
            res.type = last_anomaly_type_;
        }

        return res;
    }

    /**
     * @brief Production-Grade Speed & Position Propagation
     * Replaces open-loop raw accelerometer integration with 3-tier gated speed propagation:
     * Tier 1: Stationary Desk Clamp (ZUPT)
     * Tier 2: GNSS Master Mode (Doppler Speed)
     * Tier 3: Dead-Reckoning Mode (VelocityNet AI / OBD Wheel Speed / Drag Coasting)
     */
    void predict(
        double step_dt,
        double gyro_yaw,
        bool is_stationary_classified,
        double imu_variance
    ) {
        if (!is_initialized_) return;

        // Heading propagation
        double omega_corr = gyro_yaw - bg_;
        if (std::abs(omega_corr) < 0.02) {
            omega_corr = 0.0;
        }
        psi_ += omega_corr * step_dt;
        psi_ = std::fmod(psi_, 2.0 * PI);
        if (psi_ < 0.0) psi_ += 2.0 * PI;

        // ------------------------------------------------------------------------
        // TIER 1: Stationary Desk Clamp / ZUPT (Zero-Velocity Update)
        // ------------------------------------------------------------------------
        if (is_stationary_classified || is_desk_fidget(imu_variance)) {
            v_ = 0.0;
            bg_ = 0.98 * bg_ + 0.02 * gyro_yaw; // Bias calibration during rest
            return; // Freeze position integration
        }

        // ------------------------------------------------------------------------
        // TIER 2: GNSS Master Mode (Open-Sky Driving)
        // ------------------------------------------------------------------------
        if (is_gnss_healthy_) {
            v_ = (gnss_speed_ < 0.3) ? 0.0 : gnss_speed_;
            // When GNSS is healthy, position is anchored to GNSS fixes
            p_N_ = 0.0;
            p_E_ = 0.0;
            return;
        }
        // ------------------------------------------------------------------------
        // TIER 3: Dead-Reckoning Mode (Tunnel / GNSS Blackout)
        // ------------------------------------------------------------------------
        else {
            if (has_velocity_net_input_ && velocity_net_speed_ > 0.2) {
                // Primary DR: VelocityNet AI displacement (v = delta_d / dt)
                v_ = velocity_net_speed_ * k_scale_;
            } else if (has_obd_speed_ && obd_wheel_speed_ > 0.0) {
                // Secondary DR: OBD-II wheel speed sensor from CAN bus
                v_ = obd_wheel_speed_;
            } else {
                // Fallback DR: Aerodynamic Drag Coasting
                // v(t) = max(0, v_prev - a_drag * dt) where a_drag ≈ 0.05 m/s^2
                constexpr double kAerodynamicDragCoeff = 0.05;
                v_ = std::max(0.0, v_ - kAerodynamicDragCoeff * step_dt);
            }
        }

        // ------------------------------------------------------------------------
        // POSITION PROPAGATION (Driven purely by bounded velocity in DR mode)
        // ------------------------------------------------------------------------
        double displacement = v_ * step_dt;
        p_N_ += displacement * std::cos(psi_);
        p_E_ += displacement * std::sin(psi_);
    }

    // VelocityNet learned displacement update
    void update_velocity_net(double delta_d, double sigma, int event_class, double window_dur = 2.0, bool is_shock = false) {
        if (!is_initialized_) return;
        // Never allow VelocityNet to override real GNSS Doppler speed when GNSS is healthy
        if (is_gnss_healthy_) return;

        if (event_class == 0 || delta_d < 0.2) {
            has_velocity_net_input_ = false;
            velocity_net_speed_ = 0.0;
            v_ = 0.0;
            return;
        }

        has_velocity_net_input_ = true;
        velocity_net_speed_ = delta_d / window_dur;

        double base_var = std::pow((sigma * k_scale_) / window_dur, 2);
        double R = base_var;
        if (is_shock || event_class == 2) {
            R *= 10.0;
        } else if (event_class == 3) {
            R *= 2.0;
        }
        R = std::max(R, 0.04);

        double P_vv = 0.25;
        double K = P_vv / (P_vv + R);
        double innov = (velocity_net_speed_ * k_scale_) - v_;
        v_ = std::max(0.0, v_ + K * innov);
    }

    // Zero-Velocity Update (ZUPT)
    void update_zupt(double current_gyro_yaw) {
        if (!is_initialized_) return;
        v_ = 0.0;
        bg_ = 0.95 * bg_ + 0.05 * current_gyro_yaw;
    }

    // Continuous GNSS anchor when GNSS is available and healthy
    void update_gnss(double lat, double lon, double speed_ms, double heading_deg) {
        if (!is_initialized_) {
            initialize(lat, lon, speed_ms, heading_deg);
            return;
        }
        ref_lat_ = lat;
        ref_lon_ = lon;
        p_N_ = 0.0;
        p_E_ = 0.0;
        is_gnss_healthy_ = true;
        gnss_speed_ = std::max(0.0, speed_ms);
        v_ = (gnss_speed_ < 0.3) ? 0.0 : gnss_speed_;
        if (heading_deg >= 0.0) {
            psi_ = heading_deg * (PI / 180.0);
        }
    }

    NavState get_state() const {
        NavState s;
        if (!is_initialized_) {
            s.lat = 0.0;
            s.lon = 0.0;
            s.speed_ms = 0.0;
            s.heading_deg = 0.0;
            s.gyro_bias = 0.0;
            s.accel_bias = 0.0;
            s.scale_factor_k = 1.0;
            return s;
        }
        double lat_rad = ref_lat_ * (PI / 180.0);
        s.lat = ref_lat_ + (p_N_ / EARTH_RADIUS) * (180.0 / PI);
        s.lon = ref_lon_ + (p_E_ / (EARTH_RADIUS * std::cos(lat_rad))) * (180.0 / PI);
        s.speed_ms = v_;
        s.heading_deg = std::fmod(psi_ * (180.0 / PI), 360.0);
        if (s.heading_deg < 0.0) s.heading_deg += 360.0;
        s.gyro_bias = bg_;
        s.accel_bias = ba_;
        s.scale_factor_k = k_scale_;
        return s;
    }

    void set_scale_factor(double k) { k_scale_ = std::clamp(k, 0.5, 3.5); }

private:
    /**
     * @brief Detects high-entropy hand fidgeting vs vehicle chassis rumble
     */
    bool is_desk_fidget(double imu_variance) {
        constexpr double kHandFidgetThreshold = 0.8; // rad^2/s^2
        return (v_ == 0.0 && imu_variance > kHandFidgetThreshold);
    }

    double dt_;
    double p_N_, p_E_;
    double v_;
    double psi_;
    double bg_, ba_;
    double k_scale_;
    double ref_lat_, ref_lon_;
    bool is_initialized_;

    Butterworth2to8Hz butterworth_;
    int anomaly_counter_{0};
    int last_anomaly_type_{0};
    double last_filtered_bump_{0.0};

    // Gated speed state
    double gnss_speed_;
    bool is_gnss_healthy_;
    bool has_velocity_net_input_;
    double velocity_net_speed_;
    bool has_obd_speed_;
    double obd_wheel_speed_;
};

} // namespace inav
