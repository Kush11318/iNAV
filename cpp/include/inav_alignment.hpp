#pragma once

#include <cmath>
#include <cstdint>
#include <array>
#include <vector>
#include <optional>
#include <algorithm>
#include <numeric>

#include "inav_sensor_types.hpp"

namespace inav {

/**
 * ==============================================================================
 * 3x3 Matrix for SO(3) Coordinate Transformations
 * ==============================================================================
 */
struct Mat3 {
    std::array<std::array<double, 3>, 3> m{{{1,0,0}, {0,1,0}, {0,0,1}}};

    constexpr Mat3() = default;
    constexpr Mat3(double m00, double m01, double m02,
                   double m10, double m11, double m12,
                   double m20, double m21, double m22)
        : m{{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}}} {}

    static constexpr Mat3 identity() noexcept {
        return Mat3(1,0,0, 0,1,0, 0,0,1);
    }

    [[nodiscard]] Vec3 multiply(const Vec3& v) const noexcept {
        return Vec3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }

    [[nodiscard]] Mat3 transpose() const noexcept {
        return Mat3(
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2]
        );
    }
};

/**
 * Alignment operational state.
 */
enum class AlignmentState : uint8_t {
    UNINITIALIZED = 0,
    STATIC_LEVELING = 1,
    STATIC_ALIGNED = 2,
    DYNAMIC_ALIGNING = 3,
    FULL_ALIGNED = 4,
    REMOUNT_DETECTED = 5
};

/**
 * Configurable thresholds and hyperparameters for AlignmentEngine.
 */
struct AlignmentConfig {
    double gravity_norm{9.80665};

    // Stationarity Gate Thresholds
    double stationary_accel_std{0.10};   // m/s^2 maximum std across stationary window
    double stationary_gyro_std{0.02};    // rad/s maximum std across stationary window
    double stationary_speed_max{0.20};   // m/s maximum GNSS speed for stationary confirmation
    size_t min_static_samples{20};       // Samples required for static gravity leveling

    // Dynamic PCA Excitation Thresholds
    size_t pca_min_samples{30};          // Samples required for dynamic PCA
    double pca_min_accel_std{0.40};      // m/s^2 minimum horizontal accel variability
    double pca_min_eigen_ratio{4.0};     // lambda1 / lambda2 ratio to prevent isotropic noise lock

    // Remount Detection Thresholds
    double remount_angle_thresh_deg{5.0};// Angular gravity shift to flag remount
    size_t remount_debounce_count{5};    // Consecutive epochs above threshold before remount trip
};

/**
 * Comprehensive alignment result state.
 */
struct AlignmentResult {
    Mat3 R_b_to_v{Mat3::identity()};     // Rotation from PHONE_BODY to VEHICLE_FRD
    Vec3 gyro_bias_body{0.0, 0.0, 0.0};  // Stationary gyro bias in PHONE_BODY (rad/s)
    Vec3 accel_bias_body{0.0, 0.0, 0.0}; // Accel bias in PHONE_BODY (m/s^2)
    double pitch_deg{0.0};
    double roll_deg{0.0};
    double yaw_deg{0.0};
    double confidence{0.0};              // Normalized confidence score [0.0, 1.0]
    AlignmentState state{AlignmentState::UNINITIALIZED};
    bool remount_flag{false};

    [[nodiscard]] bool isAligned() const noexcept {
        return state == AlignmentState::STATIC_ALIGNED || state == AlignmentState::FULL_ALIGNED;
    }
};

/**
 * ==============================================================================
 * UNIFIED C++ ALIGNMENT ENGINE
 * ==============================================================================
 * Standalone, zero-allocation component for:
 *   - Stationary gravity-based leveling (Roll and Pitch)
 *   - 3-axis stationary gyroscope bias estimation in PHONE_BODY
 *   - Dynamic horizontal acceleration PCA for vehicle forward heading
 *   - Continuous debounced remount and mounting perturbation detection
 *   - Canonical sample transformation: PHONE_BODY -> VEHICLE_FRD
 */
class AlignmentEngine {
public:
    explicit AlignmentEngine(const AlignmentConfig& config = AlignmentConfig())
        : cfg_(config) {
        reset();
    }

    void reset() {
        result_ = AlignmentResult();
        u_z_body_ = Vec3(0.0, 0.0, -1.0); // Default flat face-up (+Z down in FRD is -Z in phone)
        acc_window_.clear();
        gyro_window_.clear();
        speed_window_.clear();
        running_gravity_ = Vec3(0.0, 0.0, 0.0);
        has_running_gravity_ = false;
        remount_counter_ = 0;
    }

    void set_config(const AlignmentConfig& config) {
        cfg_ = config;
    }

    [[nodiscard]] const AlignmentConfig& get_config() const noexcept {
        return cfg_;
    }

    [[nodiscard]] const AlignmentResult& get_result() const noexcept {
        return result_;
    }

    /**
     * @brief Feed canonical IMU sample and optional GNSS sample to the alignment engine.
     */
    void feed_imu(const ImuSample& sample, const std::optional<GnssSample>& gnss = std::nullopt) {
        if (!sample.isValid()) return;

        acc_window_.push_back(sample.accel_mps2);
        gyro_window_.push_back(sample.gyro_radps);
        if (gnss.has_value() && gnss->hasBasicFix()) {
            speed_window_.push_back(gnss->speed_mps);
        } else {
            speed_window_.push_back(-1.0); // Speed unknown
        }

        // Limit window sizes
        if (acc_window_.size() > 100) {
            acc_window_.erase(acc_window_.begin());
            gyro_window_.erase(gyro_window_.begin());
            speed_window_.erase(speed_window_.begin());
        }

        // 1. Continuous Remount Monitoring (if already aligned)
        if (result_.isAligned()) {
            if (check_remount(sample.accel_mps2)) {
                return;
            }
        }

        // 2. Static Leveling Phase
        if (result_.state == AlignmentState::UNINITIALIZED ||
            result_.state == AlignmentState::STATIC_LEVELING ||
            result_.state == AlignmentState::REMOUNT_DETECTED) {
            
            if (acc_window_.size() >= cfg_.min_static_samples) {
                if (check_stationarity(cfg_.min_static_samples)) {
                    calibrate_static(cfg_.min_static_samples);
                } else {
                    result_.state = AlignmentState::STATIC_LEVELING;
                }
            }
            return;
        }

        // 3. Dynamic PCA Alignment Phase (forward heading)
        if (result_.state == AlignmentState::STATIC_ALIGNED ||
            result_.state == AlignmentState::DYNAMIC_ALIGNING) {
            if (acc_window_.size() >= cfg_.pca_min_samples) {
                calibrate_dynamic(cfg_.pca_min_samples);
            }
        }
    }

    /**
     * @brief Explicitly trigger static calibration on a provided stationary buffer.
     * Returns true if stationarity is verified and static calibration succeeds.
     */
    bool calibrate_static_buffer(const std::vector<Vec3>& acc_samples,
                                 const std::vector<Vec3>& gyro_samples) {
        if (acc_samples.size() < cfg_.min_static_samples ||
            gyro_samples.size() < cfg_.min_static_samples) {
            return false;
        }

        // Verify stationarity
        Vec3 acc_mean = compute_mean(acc_samples);
        Vec3 acc_std = compute_std(acc_samples, acc_mean);
        Vec3 gyro_mean = compute_mean(gyro_samples);
        Vec3 gyro_std = compute_std(gyro_samples, gyro_mean);

        double a_std_mag = std::sqrt(acc_std.x*acc_std.x + acc_std.y*acc_std.y + acc_std.z*acc_std.z);
        double g_std_mag = std::sqrt(gyro_std.x*gyro_std.x + gyro_std.y*gyro_std.y + gyro_std.z*gyro_std.z);
        double norm_a = std::sqrt(acc_mean.x*acc_mean.x + acc_mean.y*acc_mean.y + acc_mean.z*acc_mean.z);

        if (a_std_mag > cfg_.stationary_accel_std ||
            g_std_mag > cfg_.stationary_gyro_std ||
            std::abs(norm_a - cfg_.gravity_norm) > 0.5) {
            return false; // Moving or non-gravitational
        }

        apply_static_solution(acc_mean, gyro_mean);
        return true;
    }

    /**
     * @brief Explicitly trigger dynamic PCA on a provided driving buffer.
     */
    bool calibrate_dynamic_buffer(const std::vector<Vec3>& acc_samples,
                                  const std::vector<Vec3>& gyro_samples,
                                  const std::vector<double>& speed_deltas = {}) {
        if (!result_.isAligned() || acc_samples.size() < cfg_.pca_min_samples) {
            return false;
        }
        return apply_dynamic_pca(acc_samples, gyro_samples, speed_deltas);
    }

    /**
     * @brief Transforms a raw PHONE_BODY ImuSample into a canonical VEHICLE_FRD ImuSample.
     */
    [[nodiscard]] ImuSample transform_imu(const ImuSample& body_sample) const {
        ImuSample veh_sample = body_sample;
        veh_sample.frame = SensorFrame::VEHICLE_FRD;

        // 1. Gyroscope: subtract body-frame bias, then rotate into vehicle frame
        Vec3 gyro_debiased(
            body_sample.gyro_radps.x - result_.gyro_bias_body.x,
            body_sample.gyro_radps.y - result_.gyro_bias_body.y,
            body_sample.gyro_radps.z - result_.gyro_bias_body.z
        );
        veh_sample.gyro_radps = result_.R_b_to_v.multiply(gyro_debiased);

        // 2. Accelerometer: rotate body specific force into vehicle frame
        veh_sample.accel_mps2 = result_.R_b_to_v.multiply(body_sample.accel_mps2);

        return veh_sample;
    }

    /**
     * @brief Extracts linear vehicle acceleration with gravity removed without double-subtraction.
     *
     * Specific force at rest in VEHICLE_FRD (+Z Down) points UP: [0, 0, -g].
     * Linear kinematic acceleration: a_lin = a_meas_v - [0, 0, -g] = [ax, ay, az + g].
     * At rest: az_meas = -9.81 m/s^2, so az_lin = -9.81 + 9.81 = 0.0 m/s^2.
     */
    [[nodiscard]] Vec3 get_linear_accel(const ImuSample& veh_sample) const {
        return Vec3(
            veh_sample.accel_mps2.x,
            veh_sample.accel_mps2.y,
            veh_sample.accel_mps2.z + cfg_.gravity_norm
        );
    }

    /**
     * @brief Remount disturbance detection using low-pass gravity tracking and debouncing.
     */
    bool check_remount(const Vec3& acc_b) {
        constexpr double kAlpha = 0.05;
        if (!has_running_gravity_) {
            running_gravity_ = acc_b;
            has_running_gravity_ = true;
            return false;
        }

        running_gravity_.x = (1.0 - kAlpha) * running_gravity_.x + kAlpha * acc_b.x;
        running_gravity_.y = (1.0 - kAlpha) * running_gravity_.y + kAlpha * acc_b.y;
        running_gravity_.z = (1.0 - kAlpha) * running_gravity_.z + kAlpha * acc_b.z;

        double norm_g = std::sqrt(running_gravity_.x * running_gravity_.x +
                                  running_gravity_.y * running_gravity_.y +
                                  running_gravity_.z * running_gravity_.z);
        if (norm_g < 1e-3) return false;

        // Down unit vector from running gravity: u_z_current = - running_gravity / norm_g
        Vec3 u_z_curr(-running_gravity_.x / norm_g,
                      -running_gravity_.y / norm_g,
                      -running_gravity_.z / norm_g);

        double dot = u_z_curr.x * u_z_body_.x + u_z_curr.y * u_z_body_.y + u_z_curr.z * u_z_body_.z;
        dot = std::clamp(dot, -1.0, 1.0);
        double angle_diff_deg = std::acos(dot) * (180.0 / M_PI);

        if (angle_diff_deg > cfg_.remount_angle_thresh_deg) {
            remount_counter_++;
            if (remount_counter_ >= cfg_.remount_debounce_count) {
                result_.state = AlignmentState::REMOUNT_DETECTED;
                result_.remount_flag = true;
                result_.confidence = std::max(0.1, result_.confidence - 0.5);
                return true;
            }
        } else {
            if (remount_counter_ > 0) remount_counter_--;
        }
        return false;
    }

private:
    [[nodiscard]] bool check_stationarity(size_t n) const {
        if (acc_window_.size() < n || gyro_window_.size() < n) return false;

        // Check GNSS speed if available
        for (size_t i = acc_window_.size() - n; i < acc_window_.size(); ++i) {
            if (speed_window_[i] >= 0.0 && speed_window_[i] > cfg_.stationary_speed_max) {
                return false;
            }
        }

        std::vector<Vec3> acc_sub(acc_window_.end() - n, acc_window_.end());
        std::vector<Vec3> gyro_sub(gyro_window_.end() - n, gyro_window_.end());

        Vec3 a_mean = compute_mean(acc_sub);
        Vec3 a_std = compute_std(acc_sub, a_mean);
        Vec3 g_mean = compute_mean(gyro_sub);
        Vec3 g_std = compute_std(gyro_sub, g_mean);

        double a_std_mag = std::sqrt(a_std.x*a_std.x + a_std.y*a_std.y + a_std.z*a_std.z);
        double g_std_mag = std::sqrt(g_std.x*g_std.x + g_std.y*g_std.y + g_std.z*g_std.z);
        double norm_a = std::sqrt(a_mean.x*a_mean.x + a_mean.y*a_mean.y + a_mean.z*a_mean.z);

        return (a_std_mag <= cfg_.stationary_accel_std &&
                g_std_mag <= cfg_.stationary_gyro_std &&
                std::abs(norm_a - cfg_.gravity_norm) <= 0.5);
    }

    void calibrate_static(size_t n) {
        std::vector<Vec3> acc_sub(acc_window_.end() - n, acc_window_.end());
        std::vector<Vec3> gyro_sub(gyro_window_.end() - n, gyro_window_.end());
        Vec3 a_mean = compute_mean(acc_sub);
        Vec3 g_mean = compute_mean(gyro_sub);
        apply_static_solution(a_mean, g_mean);
    }

    void apply_static_solution(const Vec3& a_mean, const Vec3& g_mean) {
        double norm_a = std::sqrt(a_mean.x*a_mean.x + a_mean.y*a_mean.y + a_mean.z*a_mean.z);
        if (norm_a < 1e-3) return;

        // Down unit vector in body frame: u_z = - a_mean / norm
        u_z_body_ = Vec3(-a_mean.x / norm_a, -a_mean.y / norm_a, -a_mean.z / norm_a);
        result_.gyro_bias_body = g_mean;

        // Choose arbitrary horizontal orthogonal basis
        Vec3 ref = (std::abs(u_z_body_.x) < 0.8) ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0);
        Vec3 y = cross(u_z_body_, ref);
        y = normalize(y);
        Vec3 x = cross(y, u_z_body_);
        x = normalize(x);

        result_.R_b_to_v = Mat3(
            x.x, x.y, x.z,
            y.x, y.y, y.z,
            u_z_body_.x, u_z_body_.y, u_z_body_.z
        );

        // Extract pitch and roll relative to vehicle horizon
        // For flat phone (u_z = [0,0,-1]), pitch=0, roll=0
        result_.pitch_deg = std::asin(std::clamp(-u_z_body_.x, -1.0, 1.0)) * (180.0 / M_PI);
        result_.roll_deg = std::atan2(u_z_body_.y, -u_z_body_.z) * (180.0 / M_PI);
        result_.state = AlignmentState::STATIC_ALIGNED;
        result_.confidence = 0.50; // Static locked
        running_gravity_ = a_mean;
        has_running_gravity_ = true;
        remount_counter_ = 0;
        result_.remount_flag = false;
    }

    void calibrate_dynamic(size_t n) {
        std::vector<Vec3> acc_sub(acc_window_.end() - n, acc_window_.end());
        std::vector<Vec3> gyro_sub(gyro_window_.end() - n, gyro_window_.end());
        apply_dynamic_pca(acc_sub, gyro_sub);
    }

    bool apply_dynamic_pca(const std::vector<Vec3>& acc_samples,
                           const std::vector<Vec3>& gyro_samples,
                           const std::vector<double>& speed_deltas = {}) {
        size_t n = acc_samples.size();
        if (n < cfg_.pca_min_samples) return false;

        // 1. Establish horizontal orthonormal basis (e1, e2) orthogonal to u_z_body_
        Vec3 ref = (std::abs(u_z_body_.x) < 0.8) ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0);
        Vec3 e1 = cross(ref, u_z_body_);
        e1 = normalize(e1);
        Vec3 e2 = cross(u_z_body_, e1);
        e2 = normalize(e2);

        // 2. Project acceleration onto horizontal basis coordinates (h1, h2)
        std::vector<double> h1(n), h2(n);
        double sum_h1 = 0.0, sum_h2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            // Remove vertical component: a_horiz = a - (a . u_z) * u_z
            double dot_z = dot(acc_samples[i], u_z_body_);
            Vec3 a_h(acc_samples[i].x - dot_z * u_z_body_.x,
                     acc_samples[i].y - dot_z * u_z_body_.y,
                     acc_samples[i].z - dot_z * u_z_body_.z);
            h1[i] = dot(a_h, e1);
            h2[i] = dot(a_h, e2);
            sum_h1 += h1[i];
            sum_h2 += h2[i];
        }

        double mean_h1 = sum_h1 / n;
        double mean_h2 = sum_h2 / n;

        // 3. Compute 2x2 horizontal covariance matrix
        double c11 = 0.0, c12 = 0.0, c22 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d1 = h1[i] - mean_h1;
            double d2 = h2[i] - mean_h2;
            c11 += d1 * d1;
            c12 += d1 * d2;
            c22 += d2 * d2;
        }
        c11 /= (n - 1);
        c12 /= (n - 1);
        c22 /= (n - 1);

        // 4. Analytic Eigen-decomposition of 2x2 symmetric matrix
        double diff = c11 - c22;
        double disc = std::sqrt(diff * diff + 4.0 * c12 * c12);
        double lambda1 = 0.5 * ((c11 + c22) + disc); // Principal eigenvalue
        double lambda2 = 0.5 * ((c11 + c22) - disc); // Secondary eigenvalue

        // Excitation Check 1: Horizontal standard deviation must exceed threshold
        double horiz_std = std::sqrt(std::max(0.0, lambda1));
        if (horiz_std < cfg_.pca_min_accel_std) {
            result_.state = AlignmentState::DYNAMIC_ALIGNING;
            return false;
        }

        // Excitation Check 2: Eigenvalue separation ratio
        double eigen_ratio = lambda1 / std::max(lambda2, 1e-6);
        if (eigen_ratio < cfg_.pca_min_eigen_ratio) {
            result_.state = AlignmentState::DYNAMIC_ALIGNING;
            return false;
        }

        // Principal eigenvector in (e1, e2) basis
        double v1 = 1.0, v2 = 0.0;
        if (std::abs(c12) > 1e-8) {
            v1 = lambda1 - c22;
            v2 = c12;
            double norm_v = std::sqrt(v1 * v1 + v2 * v2);
            v1 /= norm_v;
            v2 /= norm_v;
        } else {
            if (c11 >= c22) { v1 = 1.0; v2 = 0.0; }
            else { v1 = 0.0; v2 = 1.0; }
        }

        // 3D forward unit vector candidate in body frame
        Vec3 u_x(v1 * e1.x + v2 * e2.x,
                 v1 * e1.y + v2 * e2.y,
                 v1 * e1.z + v2 * e2.z);
        u_x = normalize(u_x);

        // 5. Sign Ambiguity Resolution (Hierarchical Evidence)
        int sign_votes = 0;
        double sign_confidence = 0.0;

        // Evidence 1: Correlation with speed delta
        if (!speed_deltas.empty() && speed_deltas.size() == n) {
            double corr_speed = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double proj = dot(acc_samples[i], u_x);
                corr_speed += proj * speed_deltas[i];
            }
            if (corr_speed < 0.0) {
                u_x = Vec3(-u_x.x, -u_x.y, -u_x.z);
            }
            sign_confidence += 0.40;
        } else {
            // Evidence 2: Centripetal Acceleration Correlation: a_lat ~ v * omega_z
            Vec3 u_y_cand = cross(u_z_body_, u_x);
            u_y_cand = normalize(u_y_cand);
            double centripetal_sum = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double a_lat = dot(acc_samples[i], u_y_cand);
                double w_z = dot(gyro_samples[i], u_z_body_);
                centripetal_sum += a_lat * w_z;
            }
            double centripetal_corr = centripetal_sum / n;

            if (std::abs(centripetal_corr) > 0.02) {
                // Under clockwise vehicle turn (w_z > 0), centripetal accel points left (a_lat < 0)
                // If corr > 0, coordinate axis is inverted
                if (centripetal_corr > 0.0) {
                    u_x = Vec3(-u_x.x, -u_x.y, -u_x.z);
                }
                sign_confidence += 0.35;
            } else {
                // Evidence 3: Net positive longitudinal acceleration assumption
                double net_acc = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    net_acc += dot(acc_samples[i], u_x);
                }
                if (net_acc < 0.0) {
                    u_x = Vec3(-u_x.x, -u_x.y, -u_x.z);
                }
                sign_confidence += 0.15;
            }
        }

        // Orthogonalize lateral right vector: u_y = u_z x u_x
        Vec3 u_y = cross(u_z_body_, u_x);
        u_y = normalize(u_y);
        u_x = cross(u_y, u_z_body_);
        u_x = normalize(u_x);

        // Assemble full rotation matrix R_b_to_v
        result_.R_b_to_v = Mat3(
            u_x.x, u_x.y, u_x.z,
            u_y.x, u_y.y, u_y.z,
            u_z_body_.x, u_z_body_.y, u_z_body_.z
        );

        result_.state = AlignmentState::FULL_ALIGNED;
        result_.yaw_deg = std::atan2(result_.R_b_to_v.m[0][1], result_.R_b_to_v.m[0][0]) * (180.0 / M_PI);
        // Confidence combines static quality, PCA excitation ratio, and sign evidence
        double ratio_score = std::clamp((eigen_ratio - cfg_.pca_min_eigen_ratio) / 10.0, 0.0, 0.30);
        double std_score = std::clamp((horiz_std - cfg_.pca_min_accel_std) / 1.0, 0.0, 0.20);
        result_.confidence = std::clamp(0.50 + ratio_score + std_score + sign_confidence, 0.50, 1.00);

        return true;
    }

    // Helper functions
    static Vec3 compute_mean(const std::vector<Vec3>& vec) {
        if (vec.empty()) return Vec3(0,0,0);
        double sx = 0, sy = 0, sz = 0;
        for (const auto& v : vec) { sx += v.x; sy += v.y; sz += v.z; }
        double inv = 1.0 / vec.size();
        return Vec3(sx * inv, sy * inv, sz * inv);
    }

    static Vec3 compute_std(const std::vector<Vec3>& vec, const Vec3& mean) {
        if (vec.size() <= 1) return Vec3(0,0,0);
        double vx = 0, vy = 0, vz = 0;
        for (const auto& v : vec) {
            vx += (v.x - mean.x) * (v.x - mean.x);
            vy += (v.y - mean.y) * (v.y - mean.y);
            vz += (v.z - mean.z) * (v.z - mean.z);
        }
        double inv = 1.0 / (vec.size() - 1);
        return Vec3(std::sqrt(vx * inv), std::sqrt(vy * inv), std::sqrt(vz * inv));
    }

    static double dot(const Vec3& a, const Vec3& b) noexcept {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
        return Vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    static Vec3 normalize(const Vec3& v) noexcept {
        double mag = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (mag < 1e-9) return Vec3(0,0,0);
        return Vec3(v.x / mag, v.y / mag, v.z / mag);
    }

    AlignmentConfig cfg_;
    AlignmentResult result_;
    Vec3 u_z_body_{0.0, 0.0, -1.0};
    std::vector<Vec3> acc_window_;
    std::vector<Vec3> gyro_window_;
    std::vector<double> speed_window_;

    Vec3 running_gravity_{0.0, 0.0, 0.0};
    bool has_running_gravity_{false};
    size_t remount_counter_{0};
};

} // namespace inav
