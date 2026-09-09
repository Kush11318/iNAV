#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

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

class DeadReckoningFilter {
public:
    DeadReckoningFilter(double dt = 0.1)
        : dt_(dt), p_N_(0.0), p_E_(0.0), v_(0.0), psi_(0.0),
          bg_(0.0), ba_(0.0), k_scale_(1.0),
          ref_lat_(28.6139), ref_lon_(77.2090), is_initialized_(true) {}

    void initialize(double init_lat, double init_lon, double init_speed_ms, double init_heading_deg) {
        ref_lat_ = init_lat;
        ref_lon_ = init_lon;
        p_N_ = 0.0;
        p_E_ = 0.0;
        v_ = std::max(0.0, init_speed_ms);
        psi_ = init_heading_deg * (PI / 180.0);
        bg_ = 0.0;
        ba_ = 0.0;
        k_scale_ = 1.0;
        is_initialized_ = true;
    }

    // High-rate IMU kinematic prediction step
    void predict(double acc_fwd, double gyro_yaw, double step_dt) {
        if (!is_initialized_) return;

        double omega_corr = gyro_yaw - bg_;
        double acc_corr = acc_fwd - ba_;

        psi_ += omega_corr * step_dt;
        psi_ = std::fmod(psi_, 2.0 * PI);
        if (psi_ < 0.0) psi_ += 2.0 * PI;

        v_ = std::max(0.0, v_ + acc_corr * step_dt);

        p_N_ += v_ * std::cos(psi_) * step_dt;
        p_E_ += v_ * std::sin(psi_) * step_dt;

        // Discretized velocity process noise scaling: Var(v) += (sigma_a * drive_inflation)^2 * dt^2
        // Smartphone MEMS (BMI160/MPU-6500): sigma_a = 0.06 m/s^2, driving inflation = 2.5 -> 0.15 m/s^2
        constexpr double sigma_a_eff = 0.15;
        P_vv_ += (sigma_a_eff * sigma_a_eff) * (step_dt * step_dt);
    }

    // VelocityNet / SpectraNet displacement update with NIS outlier gating & Joseph covariance
    void update_velocity_net(double delta_d, double sigma, int event_class, double window_dur = 2.0) {
        if (!is_initialized_) return;

        double v_net = (delta_d * k_scale_) / window_dur;
        double base_var = std::pow((sigma * k_scale_) / window_dur, 2);

        // Adaptive covariance scaling based on road event
        double R = base_var;
        if (event_class == 2) {
            R *= 4.0; // Rough road / potholes: high mechanical shock noise
        } else if (event_class == 3) {
            R *= 2.0; // Dynamic maneuver
        }
        R = std::max(R, 0.04);

        // Innovation
        double innov = v_net - v_;
        double S = P_vv_ + R; // Innovation variance

        // NIS Chi-Square Outlier Gating (chi2_0.99 for 1-DoF = 6.635)
        double nis = (innov * innov) / S;
        if (nis > 6.635) {
            // Reject anomalous speed jump / sensor glitch
            return;
        }

        // Kalman gain
        double K = P_vv_ / S;

        // State update
        v_ = std::max(0.0, v_ + K * innov);

        // Joseph-form covariance update: P = (1 - K*H)^2 * P + K^2 * R
        // Guarantees positive definiteness under single-precision / 32-bit ARM float
        double I_KH = 1.0 - K;
        P_vv_ = (I_KH * I_KH) * P_vv_ + (K * K) * R;
    }

    // Zero-Velocity Update (ZUPT)
    void update_zupt(double current_gyro_yaw) {
        if (!is_initialized_) return;
        v_ = 0.0;
        P_vv_ = 1e-4; // Strict standstill certainty
        // Bias tracking during standstill
        bg_ = 0.95 * bg_ + 0.05 * current_gyro_yaw;
    }

    // Road Curvature Observation Update (§3)
    void update_road_curvature(double gyro_yaw, double curvature_kappa) {
        if (!is_initialized_) return;
        double w_expected = v_ * curvature_kappa;
        double innov = gyro_yaw - w_expected - bg_;
        // Directly calibrate residual gyro bias from road curvature
        bg_ += 0.1 * innov;
        bg_ = std::clamp(bg_, -0.05, 0.05);
    }

    NavState get_state() const {
        NavState s;
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
    double dt_;
    double p_N_, p_E_;
    double v_;
    double psi_;
    double bg_, ba_;
    double k_scale_;
    double P_vv_{0.25}; // Prior speed variance
    double ref_lat_, ref_lon_;
    bool is_initialized_;
};

} // namespace inav
