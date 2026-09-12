#pragma once
#ifndef INAV_NO_STDLIB
#include <cmath>
#include <cstdint>
#include <array>
#include <algorithm>
#else
extern "C" {
    __declspec(dllimport) double sin(double);
    __declspec(dllimport) double cos(double);
    __declspec(dllimport) double sqrt(double);
    __declspec(dllimport) double atan2(double, double);
    __declspec(dllimport) double fmod(double, double);
    __declspec(dllimport) double pow(double, double);
}
namespace std {
    using ::sin;
    using ::cos;
    using ::sqrt;
    using ::atan2;
    using ::fmod;
    using ::pow;
    template <typename T> constexpr const T& max(const T& a, const T& b) { return (a < b) ? b : a; }
    template <typename T> constexpr const T& min(const T& a, const T& b) { return (b < a) ? b : a; }
    template <typename T> constexpr const T& clamp(const T& v, const T& lo, const T& hi) { return (v < lo) ? lo : (hi < v) ? hi : v; }
    template <typename T, unsigned long long N>
    struct array {
        T elems[N];
        constexpr unsigned long long size() const noexcept { return N; }
        constexpr T& operator[](unsigned long long i) noexcept { return elems[i]; }
        constexpr const T& operator[](unsigned long long i) const noexcept { return elems[i]; }
        void fill(const T& val) noexcept { for (unsigned long long i = 0; i < N; ++i) elems[i] = val; }
    };
}
using size_t = unsigned long long;
#endif

#include "inav_sensor_types.hpp"
#include "inav_gnss_health.hpp"

namespace inav {

constexpr double UKF_EARTH_RADIUS = 6371000.0;
constexpr double UKF_PI = 3.14159265358979323846;

// Phase 5 GNSS Fusion & Gating Constants
constexpr double UKF_NIS_GATE_2D = 9.21;                 // Chi-Square 2-DOF 99% threshold
constexpr double UKF_NIS_GATE_1D = 6.635;                // Chi-Square 1-DOF 99% threshold
constexpr double UKF_MIN_HEADING_SPEED = 2.5;            // Minimum speed (m/s) for course/heading update
constexpr double UKF_DEFAULT_SPEED_SIGMA = 0.5;          // Baseline 1-sigma speed noise (m/s)
constexpr double UKF_DEFAULT_HEADING_SIGMA_DEG = 3.0;    // Baseline 1-sigma course noise (deg)

struct UKFNavState {
    double lat{0.0};             // Latitude (degrees)
    double lon{0.0};             // Longitude (degrees)
    double speed_ms{0.0};        // Forward speed (m/s)
    double heading_deg{0.0};     // Heading clockwise from North (degrees)
    double gyro_bias{0.0};       // Gyroscope yaw bias (rad/s)
    double accel_bias{0.0};      // Accelerometer forward bias (m/s^2)
    double scale_factor_k{1.0};  // VelocityNet scale factor (unitless)
};

/**
 * @brief 7-State Unscented Kalman Filter for 2D Vehicle Dead Reckoning.
 *
 * State Vector:
 *   x = [p_N, p_E, v_fwd, psi, b_g, b_a, k]^T
 *     p_N   : North position (m)
 *     p_E   : East position (m)
 *     v_fwd : Forward vehicle speed (m/s)
 *     psi   : Vehicle heading angle (rad, clockwise from North)
 *     b_g   : Gyroscope yaw rate bias (rad/s)
 *     b_a   : Accelerometer forward bias (m/s^2)
 *     k     : VelocityNet scale factor (unitless)
 */
class UKFNavigationFilter {
public:
    static constexpr size_t DIM_X = 7;
    static constexpr size_t NUM_SIGMA = 2 * DIM_X + 1; // 15

    explicit UKFNavigationFilter(
        double dt = 0.1,
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0
    ) : dt_(dt), alpha_(alpha), beta_(beta), kappa_(kappa) {
        init_weights();
        reset_covariance();
    }

    void initialize(
        double init_lat,
        double init_lon,
        double init_speed_ms,
        double init_heading_rad,
        double init_gyro_bias = 0.0,
        double init_accel_bias = 0.0
    ) {
        ref_lat_ = init_lat;
        ref_lon_ = init_lon;

        x_.fill(0.0);
        x_[0] = 0.0; // p_N
        x_[1] = 0.0; // p_E
        x_[2] = std::max(init_speed_ms, 0.0); // v_fwd
        x_[3] = std::fmod(init_heading_rad, 2.0 * UKF_PI);
        if (x_[3] < 0.0) x_[3] += 2.0 * UKF_PI;
        x_[4] = init_gyro_bias; // b_g
        x_[5] = init_accel_bias; // b_a
        x_[6] = 1.0; // k

        reset_covariance();
        is_initialized_ = true;
    }

    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }

    [[nodiscard]] const std::array<double, DIM_X>& get_x() const noexcept { return x_; }
    [[nodiscard]] const std::array<std::array<double, DIM_X>, DIM_X>& get_P() const noexcept { return P_; }

    [[nodiscard]] UKFNavState get_nav_state() const noexcept {
        UKFNavState s;
        if (!is_initialized_) return s;

        double lat_rad = ref_lat_ * (UKF_PI / 180.0);
        s.lat = ref_lat_ + (x_[0] / UKF_EARTH_RADIUS) * (180.0 / UKF_PI);
        s.lon = ref_lon_ + (x_[1] / (UKF_EARTH_RADIUS * std::cos(lat_rad))) * (180.0 / UKF_PI);
        s.speed_ms = x_[2];
        s.heading_deg = std::fmod(x_[3] * (180.0 / UKF_PI), 360.0);
        if (s.heading_deg < 0.0) s.heading_deg += 360.0;
        s.gyro_bias = x_[4];
        s.accel_bias = x_[5];
        s.scale_factor_k = x_[6];
        return s;
    }

    /**
     * @brief UKF Prediction Step: Propagates sigma points through kinematic motion model.
     * @param acc_fwd Forward vehicle acceleration in VEHICLE_FRD (+X_v, m/s^2)
     * @param gyro_yaw Vehicle yaw rate in VEHICLE_FRD (+Z_v, rad/s)
     * @param dt Elapsed timestep in seconds
     */
    void predict(double acc_fwd, double gyro_yaw, double dt = -1.0, bool is_stationary = false) {
        if (!is_initialized_) return;
        if (dt <= 0.0) dt = dt_;

        if (is_stationary) {
            x_[2] = 0.0; // Clamp forward velocity to zero
            // Update heading with debiased gyro
            double omega_corr = gyro_yaw - x_[4];
            x_[3] = std::fmod(x_[3] + omega_corr * dt, 2.0 * UKF_PI);
            if (x_[3] < 0.0) x_[3] += 2.0 * UKF_PI;
            x_[4] = 0.98 * x_[4] + 0.02 * gyro_yaw; // Smooth gyro bias calibration at rest
            return; // Freeze position integration
        }

        auto sigma = generate_sigma_points();
        std::array<std::array<double, DIM_X>, NUM_SIGMA> sigma_f;

        // Propagate each sigma point
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            const auto& s = sigma[i];
            double pN = s[0];
            double pE = s[1];
            double v = s[2];
            double psi = s[3];
            double bg = s[4];
            double ba = s[5];
            double k = s[6];

            // De-bias inputs
            double omega_corr = gyro_yaw - bg;
            double acc_corr = acc_fwd - ba;

            // Heading propagation
            double new_psi = std::fmod(psi + omega_corr * dt, 2.0 * UKF_PI);
            if (new_psi < 0.0) new_psi += 2.0 * UKF_PI;
            double mid_psi = psi + 0.5 * omega_corr * dt;

            // Speed propagation
            double new_v = std::max(v + acc_corr * dt, 0.0);
            double mid_v = 0.5 * (v + new_v);

            // Position propagation
            double new_pN = pN + mid_v * std::cos(mid_psi) * dt;
            double new_pE = pE + mid_v * std::sin(mid_psi) * dt;

            sigma_f[i] = {new_pN, new_pE, new_v, new_psi, bg, ba, k};
        }

        // Predicted state mean
        std::array<double, DIM_X> x_pred{};
        for (size_t idx = 0; idx < DIM_X; ++idx) {
            if (idx == 3) {
                // Circular mean for heading angle
                double sin_sum = 0.0;
                double cos_sum = 0.0;
                for (size_t i = 0; i < NUM_SIGMA; ++i) {
                    sin_sum += w_m_[i] * std::sin(sigma_f[i][3]);
                    cos_sum += w_m_[i] * std::cos(sigma_f[i][3]);
                }
                double mean_psi = std::fmod(std::atan2(sin_sum, cos_sum), 2.0 * UKF_PI);
                if (mean_psi < 0.0) mean_psi += 2.0 * UKF_PI;
                x_pred[3] = mean_psi;
            } else {
                double sum = 0.0;
                for (size_t i = 0; i < NUM_SIGMA; ++i) {
                    sum += w_m_[i] * sigma_f[i][idx];
                }
                x_pred[idx] = sum;
            }
        }

        // Predicted covariance
        std::array<std::array<double, DIM_X>, DIM_X> P_pred{};
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            std::array<double, DIM_X> diff;
            for (size_t j = 0; j < DIM_X; ++j) {
                diff[j] = sigma_f[i][j] - x_pred[j];
            }
            // Wrap heading error into [-pi, +pi)
            diff[3] = std::fmod(diff[3] + UKF_PI, 2.0 * UKF_PI);
            if (diff[3] < 0.0) diff[3] += 2.0 * UKF_PI;
            diff[3] -= UKF_PI;

            for (size_t r = 0; r < DIM_X; ++r) {
                for (size_t c = 0; c < DIM_X; ++c) {
                    P_pred[r][c] += w_c_[i] * diff[r] * diff[c];
                }
            }
        }

        // Add process noise Q * dt
        for (size_t r = 0; r < DIM_X; ++r) {
            for (size_t c = 0; c < DIM_X; ++c) {
                P_pred[r][c] += Q_[r][c] * dt;
            }
        }

        x_ = x_pred;
        P_ = P_pred;
        symmetrize_P();
    }

    /**
     * @brief VelocityNet Learned Forward Displacement Update.
     */
    void update_velocity_net(
        double delta_d_pred,
        double sigma_pred,
        int event_class = 1,
        double window_dur = 2.0
    ) {
        if (!is_initialized_) return;

        double v_net = delta_d_pred / window_dur;
        double base_var = std::pow(sigma_pred / window_dur, 2);

        double variance = base_var;
        if (event_class == 2) {
            variance *= 4.0;
        } else if (event_class == 3) {
            variance *= 2.0;
        }
        variance = std::max(variance, 0.04);

        // h(s) observes forward speed x[2]
        update_scalar_measurement(
            v_net,
            [](const std::array<double, DIM_X>& s) { return s[2]; },
            variance,
            false
        );
    }

    /**
     * @brief Zero-Velocity Update (ZUPT).
     */
    void update_zupt(double gyro_reading) {
        if (!is_initialized_) return;

        // Measurement 1: Speed = 0.0 m/s
        update_scalar_measurement(
            0.0,
            [](const std::array<double, DIM_X>& s) { return s[2]; },
            1e-4,
            false
        );

        // Measurement 2: Gyro bias matches current stationary rate
        update_scalar_measurement(
            gyro_reading,
            [](const std::array<double, DIM_X>& s) { return s[4]; },
            1e-4,
            false
        );
    }

    [[nodiscard]] double get_ref_lat() const noexcept { return ref_lat_; }
    [[nodiscard]] double get_ref_lon() const noexcept { return ref_lon_; }

    /**
     * @brief Computes 2D position Normalized Innovation Squared (NIS) without updating the state.
     * Used for candidate validation in quarantine and outlier gating.
     */
    [[nodiscard]] double compute_pos_nis(
        double p_N,
        double p_E,
        double accuracy_m,
        double r_scale = 1.0
    ) const {
        if (!is_initialized_) return 999999.0;

        double sigma_pos = std::max(accuracy_m, 1.0);
        double R_diag = (sigma_pos * sigma_pos) * std::max(r_scale, 1.0);

        double S00 = P_[0][0] + R_diag;
        double S01 = P_[0][1];
        double S10 = P_[1][0];
        double S11 = P_[1][1] + R_diag;

        double det = S00 * S11 - S01 * S10;
        if (det < 1e-12) return 999999.0;
        double inv_det = 1.0 / det;

        double S_inv00 =  S11 * inv_det;
        double S_inv01 = -S01 * inv_det;
        double S_inv10 = -S10 * inv_det;
        double S_inv11 =  S00 * inv_det;

        double y0 = p_N - x_[0];
        double y1 = p_E - x_[1];

        return y0 * (S_inv00 * y0 + S_inv01 * y1) +
               y1 * (S_inv10 * y0 + S_inv11 * y1);
    }

    /**
     * @brief 2D GNSS Position Measurement Update with Innovation Gating (NIS).
     *
     * Measurement model:
     *   z_pos = [p_N, p_E]^T
     *   h(x)  = [x[0], x[1]]^T
     *
     * Covariance:
     *   sigma_pos = max(accuracy_m, 1.0)
     *   R_pos = (sigma_pos^2 * r_scale) * I_2
     *
     * Innovation Gate:
     *   NIS = y^T S^-1 y <= 9.21 (Chi-square 2-DOF 99%)
     */
    bool update_gnss_pos(
        double p_N,
        double p_E,
        double accuracy_m,
        double r_scale = 1.0,
        double* out_nis = nullptr
    ) {
        if (!is_initialized_) return false;

        double sigma_pos = std::max(accuracy_m, 1.0);
        double R_diag = (sigma_pos * sigma_pos) * std::max(r_scale, 1.0);

        auto sigma = generate_sigma_points();

        std::array<std::array<double, 2>, NUM_SIGMA> gamma_pts;
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            gamma_pts[i][0] = sigma[i][0]; // p_N
            gamma_pts[i][1] = sigma[i][1]; // p_E
        }

        std::array<double, 2> z_pred{0.0, 0.0};
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            z_pred[0] += w_m_[i] * gamma_pts[i][0];
            z_pred[1] += w_m_[i] * gamma_pts[i][1];
        }

        double P_zz[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
        double P_xz[DIM_X][2];
        for (size_t r = 0; r < DIM_X; ++r) {
            P_xz[r][0] = 0.0;
            P_xz[r][1] = 0.0;
        }

        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            double dz0 = gamma_pts[i][0] - z_pred[0];
            double dz1 = gamma_pts[i][1] - z_pred[1];

            std::array<double, DIM_X> dx;
            for (size_t j = 0; j < DIM_X; ++j) {
                dx[j] = sigma[i][j] - x_[j];
            }
            dx[3] = std::fmod(dx[3] + UKF_PI, 2.0 * UKF_PI);
            if (dx[3] < 0.0) dx[3] += 2.0 * UKF_PI;
            dx[3] -= UKF_PI;

            P_zz[0][0] += w_c_[i] * dz0 * dz0;
            P_zz[0][1] += w_c_[i] * dz0 * dz1;
            P_zz[1][0] += w_c_[i] * dz1 * dz0;
            P_zz[1][1] += w_c_[i] * dz1 * dz1;

            for (size_t j = 0; j < DIM_X; ++j) {
                P_xz[j][0] += w_c_[i] * dx[j] * dz0;
                P_xz[j][1] += w_c_[i] * dx[j] * dz1;
            }
        }

        P_zz[0][0] += R_diag;
        P_zz[1][1] += R_diag;

        double det = P_zz[0][0] * P_zz[1][1] - P_zz[0][1] * P_zz[1][0];
        if (det < 1e-12) return false;
        double inv_det = 1.0 / det;

        double P_zz_inv[2][2];
        P_zz_inv[0][0] =  P_zz[1][1] * inv_det;
        P_zz_inv[0][1] = -P_zz[0][1] * inv_det;
        P_zz_inv[1][0] = -P_zz[1][0] * inv_det;
        P_zz_inv[1][1] =  P_zz[0][0] * inv_det;

        double y0 = p_N - z_pred[0];
        double y1 = p_E - z_pred[1];

        double nis = y0 * (P_zz_inv[0][0] * y0 + P_zz_inv[0][1] * y1) +
                     y1 * (P_zz_inv[1][0] * y0 + P_zz_inv[1][1] * y1);

        if (out_nis != nullptr) {
            *out_nis = nis;
        }

        if (nis > UKF_NIS_GATE_2D) {
            return false;
        }

        double K[DIM_X][2];
        for (size_t r = 0; r < DIM_X; ++r) {
            K[r][0] = P_xz[r][0] * P_zz_inv[0][0] + P_xz[r][1] * P_zz_inv[1][0];
            K[r][1] = P_xz[r][0] * P_zz_inv[0][1] + P_xz[r][1] * P_zz_inv[1][1];
        }

        for (size_t r = 0; r < DIM_X; ++r) {
            x_[r] += K[r][0] * y0 + K[r][1] * y1;
        }
        x_[3] = std::fmod(x_[3], 2.0 * UKF_PI);
        if (x_[3] < 0.0) x_[3] += 2.0 * UKF_PI;
        x_[2] = std::max(x_[2], 0.0);
        x_[6] = std::clamp(x_[6], 0.7, 1.3);

        double M[2][DIM_X];
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < DIM_X; ++c) {
                M[r][c] = P_zz[r][0] * K[c][0] + P_zz[r][1] * K[c][1];
            }
        }
        for (size_t r = 0; r < DIM_X; ++r) {
            for (size_t c = 0; c < DIM_X; ++c) {
                P_[r][c] -= (K[r][0] * M[0][c] + K[r][1] * M[1][c]);
            }
        }

        symmetrize_P();
        return true;
    }

    /**
     * @brief GNSS Forward Ground Speed Update with Innovation Gating.
     *
     * Measurement model:
     *   z_speed = v_gnss
     *   h(x)    = x[2]
     */
    bool update_gnss_speed(
        double v_gnss,
        double sigma_v = UKF_DEFAULT_SPEED_SIGMA,
        double r_scale = 1.0,
        double* out_nis = nullptr
    ) {
        if (!is_initialized_) return false;
        double R = (sigma_v * sigma_v) * std::max(r_scale, 1.0);

        auto sigma = generate_sigma_points();
        std::array<double, NUM_SIGMA> gamma_pts;
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            gamma_pts[i] = sigma[i][2]; // v_fwd
        }

        double z_pred = 0.0;
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            z_pred += w_m_[i] * gamma_pts[i];
        }

        double P_zz = 0.0;
        std::array<double, DIM_X> P_xz{};
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            double dz = gamma_pts[i] - z_pred;
            std::array<double, DIM_X> dx;
            for (size_t j = 0; j < DIM_X; ++j) {
                dx[j] = sigma[i][j] - x_[j];
            }
            dx[3] = std::fmod(dx[3] + UKF_PI, 2.0 * UKF_PI);
            if (dx[3] < 0.0) dx[3] += 2.0 * UKF_PI;
            dx[3] -= UKF_PI;

            P_zz += w_c_[i] * dz * dz;
            for (size_t j = 0; j < DIM_X; ++j) {
                P_xz[j] += w_c_[i] * dx[j] * dz;
            }
        }
        P_zz += R;
        if (P_zz < 1e-12) return false;

        double y = v_gnss - z_pred;
        double nis = (y * y) / P_zz;
        if (out_nis != nullptr) {
            *out_nis = nis;
        }

        if (nis > UKF_NIS_GATE_1D) {
            return false;
        }

        std::array<double, DIM_X> K;
        for (size_t j = 0; j < DIM_X; ++j) {
            K[j] = P_xz[j] / P_zz;
        }

        for (size_t j = 0; j < DIM_X; ++j) {
            x_[j] += K[j] * y;
        }
        x_[3] = std::fmod(x_[3], 2.0 * UKF_PI);
        if (x_[3] < 0.0) x_[3] += 2.0 * UKF_PI;
        x_[2] = std::max(x_[2], 0.0);
        x_[6] = std::clamp(x_[6], 0.7, 1.3);

        for (size_t r = 0; r < DIM_X; ++r) {
            for (size_t c = 0; c < DIM_X; ++c) {
                P_[r][c] -= P_zz * K[r] * K[c];
            }
        }
        symmetrize_P();
        return true;
    }

    /**
     * @brief GNSS Course / Heading Update with Speed Gating & Innovation Gating.
     *
     * Measurement model:
     *   z_psi = psi_gnss (rad, clockwise from North)
     *   h(x)  = x[3]
     */
    bool update_gnss_course(
        double psi_gnss_rad,
        double v_gnss,
        double sigma_psi_rad = (UKF_DEFAULT_HEADING_SIGMA_DEG * UKF_PI / 180.0),
        double r_scale = 1.0,
        double* out_nis = nullptr
    ) {
        if (!is_initialized_) return false;
        // Speed gate: vehicle must be moving sufficiently fast for GNSS track to represent vehicle heading
        if (v_gnss <= UKF_MIN_HEADING_SPEED) {
            return false;
        }

        double R = (sigma_psi_rad * sigma_psi_rad) * std::max(r_scale, 1.0);

        auto sigma = generate_sigma_points();
        std::array<double, NUM_SIGMA> gamma_pts;
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            gamma_pts[i] = sigma[i][3]; // psi
        }

        double sin_s = 0.0;
        double cos_s = 0.0;
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            sin_s += w_m_[i] * std::sin(gamma_pts[i]);
            cos_s += w_m_[i] * std::cos(gamma_pts[i]);
        }
        double z_pred = std::fmod(std::atan2(sin_s, cos_s), 2.0 * UKF_PI);
        if (z_pred < 0.0) z_pred += 2.0 * UKF_PI;

        double P_zz = 0.0;
        std::array<double, DIM_X> P_xz{};
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            double dz = gamma_pts[i] - z_pred;
            dz = std::fmod(dz + UKF_PI, 2.0 * UKF_PI);
            if (dz < 0.0) dz += 2.0 * UKF_PI;
            dz -= UKF_PI;

            std::array<double, DIM_X> dx;
            for (size_t j = 0; j < DIM_X; ++j) {
                dx[j] = sigma[i][j] - x_[j];
            }
            dx[3] = std::fmod(dx[3] + UKF_PI, 2.0 * UKF_PI);
            if (dx[3] < 0.0) dx[3] += 2.0 * UKF_PI;
            dx[3] -= UKF_PI;

            P_zz += w_c_[i] * dz * dz;
            for (size_t j = 0; j < DIM_X; ++j) {
                P_xz[j] += w_c_[i] * dx[j] * dz;
            }
        }
        P_zz += R;
        if (P_zz < 1e-12) return false;

        double y = psi_gnss_rad - z_pred;
        y = std::fmod(y + UKF_PI, 2.0 * UKF_PI);
        if (y < 0.0) y += 2.0 * UKF_PI;
        y -= UKF_PI;

        double nis = (y * y) / P_zz;
        if (out_nis != nullptr) {
            *out_nis = nis;
        }

        if (nis > UKF_NIS_GATE_1D) {
            return false;
        }

        std::array<double, DIM_X> K;
        for (size_t j = 0; j < DIM_X; ++j) {
            K[j] = P_xz[j] / P_zz;
        }

        for (size_t j = 0; j < DIM_X; ++j) {
            x_[j] += K[j] * y;
        }
        x_[3] = std::fmod(x_[3], 2.0 * UKF_PI);
        if (x_[3] < 0.0) x_[3] += 2.0 * UKF_PI;
        x_[2] = std::max(x_[2], 0.0);
        x_[6] = std::clamp(x_[6], 0.7, 1.3);

        for (size_t r = 0; r < DIM_X; ++r) {
            for (size_t c = 0; c < DIM_X; ++c) {
                P_[r][c] -= P_zz * K[r] * K[c];
            }
        }
        symmetrize_P();
        return true;
    }

    /**
     * @brief High-level canonical GNSS sample processing integrated with GNSS Health Manager.
     */
    bool update_gnss_sample(
        const GnssSample& sample,
        GnssHealthManager& health_mgr,
        bool* pos_accepted = nullptr,
        bool* speed_accepted = nullptr,
        bool* course_accepted = nullptr
    ) {
        if (pos_accepted) *pos_accepted = false;
        if (speed_accepted) *speed_accepted = false;
        if (course_accepted) *course_accepted = false;

        if (!is_initialized_) return false;

        if (!health_mgr.validate_sample(sample)) {
            health_mgr.on_measurement_rejected();
            return false;
        }

        double d_lat = (sample.latitude_deg - ref_lat_) * (UKF_PI / 180.0);
        double d_lon = (sample.longitude_deg - ref_lon_) * (UKF_PI / 180.0);
        double ref_lat_rad = ref_lat_ * (UKF_PI / 180.0);
        double p_N = UKF_EARTH_RADIUS * d_lat;
        double p_E = UKF_EARTH_RADIUS * d_lon * std::cos(ref_lat_rad);

        if (health_mgr.get_state() == GnssHealthState::PURE_DR) {
            health_mgr.start_quarantine();
        }

        double candidate_nis = compute_pos_nis(p_N, p_E, sample.horizontal_accuracy_m);
        if (candidate_nis > health_mgr.config().nis_pos_2d_threshold) {
            health_mgr.on_measurement_rejected();
            return false;
        }

        double r_scale = 1.0;
        bool can_update = health_mgr.process_candidate(sample, r_scale);
        if (!can_update) {
            return false;
        }

        double pos_nis = 0.0;
        bool p_ok = update_gnss_pos(p_N, p_E, sample.horizontal_accuracy_m, r_scale, &pos_nis);
        if (pos_accepted) *pos_accepted = p_ok;
        if (!p_ok) {
            health_mgr.on_measurement_rejected();
            return false;
        }

        if (sample.validity & GnssValidity::SPEED_VALID) {
            double spd_nis = 0.0;
            bool s_ok = update_gnss_speed(sample.speed_mps, UKF_DEFAULT_SPEED_SIGMA, r_scale, &spd_nis);
            if (speed_accepted) *speed_accepted = s_ok;
        }

        if (sample.validity & GnssValidity::BEARING_VALID) {
            double bearing_rad = sample.bearing_deg * (UKF_PI / 180.0);
            double crs_nis = 0.0;
            bool c_ok = update_gnss_course(bearing_rad, sample.speed_mps, UKF_DEFAULT_HEADING_SIGMA_DEG * (UKF_PI / 180.0), r_scale, &crs_nis);
            if (course_accepted) *course_accepted = c_ok;
        }

        return true;
    }

    /**
     * @brief Probabilistic Map Measurement Update (Pillar 5).
     *
     * Fuses snapped road coordinate constraint and road heading into the 7-state UKF.
     * Features:
     * - Confidence-scaled measurement covariance: R_pos = diag(sigma_map^2, sigma_map^2)
     * - 2-DOF Chi-Square NIS innovation gating (threshold 9.21)
     * - Optional road-aligned heading constraint when speed > 2.5 m/s and confidence > 0.5
     *   with 1-DOF Chi-Square NIS gating (threshold 6.635)
     */
    bool update_map_match(
        double p_N_match,
        double p_E_match,
        double psi_road,
        double confidence,
        bool is_heading_valid = false,
        double* out_pos_nis = nullptr,
        double* out_hdg_nis = nullptr
    ) {
        if (!is_initialized_) return false;
        if (confidence < 0.25) return false;

        // Unit normal vector perpendicular to road: n = [-sin(psi_road), cos(psi_road)]
        double n_N = -std::sin(psi_road);
        double n_E =  std::cos(psi_road);

        // Cross-track innovation along normal
        double y_ct = n_N * (p_N_match - x_[0]) + n_E * (p_E_match - x_[1]);

        double sigma_base = 2.5; // Base map road lane accuracy (m)
        double sigma_map = sigma_base / std::max(confidence, 0.25);
        double R_ct = sigma_map * sigma_map;

        // Innovation variance (1-DOF)
        double S = n_N * n_N * P_[0][0] + 2.0 * n_N * n_E * P_[0][1] + n_E * n_E * P_[1][1] + R_ct;
        if (S < 1e-12) return false;

        double nis = (y_ct * y_ct) / S;
        if (out_pos_nis != nullptr) {
            *out_pos_nis = nis;
        }

        // 1-DOF Chi-Square NIS gating (threshold 6.635 at p=0.01)
        if (nis > UKF_NIS_GATE_1D) {
            return false;
        }

        // Kalman gain strictly along normal (updates position without corrupting along-track velocity)
        double K_N = (P_[0][0] * n_N + P_[0][1] * n_E) / S;
        double K_E = (P_[1][0] * n_N + P_[1][1] * n_E) / S;

        x_[0] += K_N * y_ct;
        x_[1] += K_E * y_ct;

        // Covariance reduction strictly perpendicular to road
        P_[0][0] -= S * K_N * K_N;
        P_[0][1] -= S * K_N * K_E;
        P_[1][0] = P_[0][1];
        P_[1][1] -= S * K_E * K_E;
        symmetrize_P();

        if (is_heading_valid && x_[2] > UKF_MIN_HEADING_SPEED && confidence > 0.7) {
            double hdg_diff = std::fmod(x_[3] - psi_road + UKF_PI, 2.0 * UKF_PI) - UKF_PI;
            double d_hdg = hdg_diff < 0.0 ? -hdg_diff : hdg_diff;
            if (d_hdg < 0.26) { // < 15 deg
                double hdg_sigma = (UKF_DEFAULT_HEADING_SIGMA_DEG * UKF_PI / 180.0) / std::max(confidence, 0.5);
                update_gnss_course(psi_road, x_[2], hdg_sigma, 1.0, out_hdg_nis);
            }
        }

        return true;
    }

    /**
     * @brief Generic 1D Scalar Measurement Update.
     */
    template <typename HFunc>
    void update_scalar_measurement(
        double z,
        HFunc&& h_func,
        double R,
        bool is_angle = false
    ) {
        auto sigma = generate_sigma_points();
        std::array<double, NUM_SIGMA> gamma_pts;
        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            gamma_pts[i] = h_func(sigma[i]);
        }

        // Predicted measurement mean
        double z_pred = 0.0;
        if (is_angle) {
            double sin_s = 0.0;
            double cos_s = 0.0;
            for (size_t i = 0; i < NUM_SIGMA; ++i) {
                sin_s += w_m_[i] * std::sin(gamma_pts[i]);
                cos_s += w_m_[i] * std::cos(gamma_pts[i]);
            }
            z_pred = std::fmod(std::atan2(sin_s, cos_s), 2.0 * UKF_PI);
            if (z_pred < 0.0) z_pred += 2.0 * UKF_PI;
        } else {
            for (size_t i = 0; i < NUM_SIGMA; ++i) {
                z_pred += w_m_[i] * gamma_pts[i];
            }
        }

        // Innovation variance P_zz and cross-covariance P_xz
        double P_zz = 0.0;
        std::array<double, DIM_X> P_xz{};

        for (size_t i = 0; i < NUM_SIGMA; ++i) {
            double dz = gamma_pts[i] - z_pred;
            if (is_angle) {
                dz = std::fmod(dz + UKF_PI, 2.0 * UKF_PI);
                if (dz < 0.0) dz += 2.0 * UKF_PI;
                dz -= UKF_PI;
            }

            std::array<double, DIM_X> dx;
            for (size_t j = 0; j < DIM_X; ++j) {
                dx[j] = sigma[i][j] - x_[j];
            }
            dx[3] = std::fmod(dx[3] + UKF_PI, 2.0 * UKF_PI);
            if (dx[3] < 0.0) dx[3] += 2.0 * UKF_PI;
            dx[3] -= UKF_PI;

            P_zz += w_c_[i] * dz * dz;
            for (size_t j = 0; j < DIM_X; ++j) {
                P_xz[j] += w_c_[i] * dx[j] * dz;
            }
        }

        P_zz += R;
        if (P_zz < 1e-12) P_zz = 1e-12;

        // Kalman gain K = P_xz / P_zz
        std::array<double, DIM_X> K;
        for (size_t j = 0; j < DIM_X; ++j) {
            K[j] = P_xz[j] / P_zz;
        }

        // Innovation residual y = z - z_pred
        double y = z - z_pred;
        if (is_angle) {
            y = std::fmod(y + UKF_PI, 2.0 * UKF_PI);
            if (y < 0.0) y += 2.0 * UKF_PI;
            y -= UKF_PI;
        }

        // State update
        for (size_t j = 0; j < DIM_X; ++j) {
            x_[j] += K[j] * y;
        }
        x_[3] = std::fmod(x_[3], 2.0 * UKF_PI);
        if (x_[3] < 0.0) x_[3] += 2.0 * UKF_PI;
        x_[2] = std::max(x_[2], 0.0); // Speed non-negative
        x_[6] = std::clamp(x_[6], 0.7, 1.3); // Scale factor clamp

        // Covariance update: P = P - P_zz * (K * K^T)
        for (size_t r = 0; r < DIM_X; ++r) {
            for (size_t c = 0; c < DIM_X; ++c) {
                P_[r][c] -= P_zz * K[r] * K[c];
            }
        }

        symmetrize_P();
    }

private:
    void init_weights() {
        lam_ = alpha_ * alpha_ * (static_cast<double>(DIM_X) + kappa_) - static_cast<double>(DIM_X);
        gamma_ = std::sqrt(static_cast<double>(DIM_X) + lam_);

        double common_w = 1.0 / (2.0 * (static_cast<double>(DIM_X) + lam_));
        w_m_.fill(common_w);
        w_c_.fill(common_w);

        w_m_[0] = lam_ / (static_cast<double>(DIM_X) + lam_);
        w_c_[0] = w_m_[0] + (1.0 - alpha_ * alpha_ + beta_);
    }

    void reset_covariance() {
        for (size_t r = 0; r < DIM_X; ++r) {
            P_[r].fill(0.0);
            Q_[r].fill(0.0);
        }

        // Initial covariance P:
        // [25.0, 25.0, 1.0, (5 deg)^2, (0.001 rad/s)^2, (0.05 m/s^2)^2, (0.05)^2]
        P_[0][0] = 25.0; // p_N (5 m std)
        P_[1][1] = 25.0; // p_E (5 m std)
        P_[2][2] = 1.0;  // v_fwd (1 m/s std)
        P_[3][3] = std::pow(5.0 * (UKF_PI / 180.0), 2); // psi (5 deg std)
        P_[4][4] = std::pow(1e-3, 2); // b_g (0.001 rad/s std)
        P_[5][5] = std::pow(0.05, 2); // b_a (0.05 m/s^2 std)
        P_[6][6] = std::pow(0.05, 2); // k (5% uncertainty)

        // Process noise covariance Q:
        // [0.01, 0.01, 0.04, (0.2 deg)^2, (1e-6)^2, (1e-4)^2, (1e-5)^2]
        Q_[0][0] = 0.01;
        Q_[1][1] = 0.01;
        Q_[2][2] = 0.04;
        Q_[3][3] = std::pow(0.2 * (UKF_PI / 180.0), 2);
        Q_[4][4] = std::pow(1e-6, 2);
        Q_[5][5] = std::pow(1e-4, 2);
        Q_[6][6] = std::pow(1e-5, 2);
    }

    void symmetrize_P() {
        for (size_t r = 0; r < DIM_X; ++r) {
            for (size_t c = r + 1; c < DIM_X; ++c) {
                double avg = 0.5 * (P_[r][c] + P_[c][r]);
                P_[r][c] = avg;
                P_[c][r] = avg;
            }
        }
    }

    /**
     * @brief Generates 2L + 1 sigma points using lower triangular Cholesky decomposition L of P.
     */
    std::array<std::array<double, DIM_X>, NUM_SIGMA> generate_sigma_points() {
        symmetrize_P();

        // Regularize diagonal to protect positive definiteness (+1e-9)
        for (size_t i = 0; i < DIM_X; ++i) {
            P_[i][i] += 1e-9;
        }
        std::array<std::array<double, DIM_X>, DIM_X> A = P_;

        // Cholesky decomposition: A = L * L^T
        std::array<std::array<double, DIM_X>, DIM_X> L{};
        bool cholesky_ok = true;

        for (size_t j = 0; j < DIM_X; ++j) {
            double sum_sq = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum_sq += L[j][k] * L[j][k];
            }
            double diag_val = A[j][j] - sum_sq;
            if (diag_val <= 0.0) {
                cholesky_ok = false;
                break;
            }
            L[j][j] = std::sqrt(diag_val);

            for (size_t i = j + 1; i < DIM_X; ++i) {
                double sum_cross = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum_cross += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum_cross) / L[j][j];
            }
        }

        // Fallback if numerical conditioning degrades: diagonal floor
        if (!cholesky_ok) {
            for (size_t i = 0; i < DIM_X; ++i) {
                for (size_t j = 0; j < DIM_X; ++j) L[i][j] = 0.0;
                L[i][i] = std::sqrt(std::max(A[i][i], 1e-8));
            }
        }

        std::array<std::array<double, DIM_X>, NUM_SIGMA> sigma;
        sigma[0] = x_;

        for (size_t i = 0; i < DIM_X; ++i) {
            for (size_t j = 0; j < DIM_X; ++j) {
                sigma[i + 1][j] = x_[j] + gamma_ * L[j][i];
                sigma[i + 1 + DIM_X][j] = x_[j] - gamma_ * L[j][i];
            }
        }

        return sigma;
    }

    double dt_{0.1};
    double alpha_{1e-3};
    double beta_{2.0};
    double kappa_{0.0};
    double lam_{0.0};
    double gamma_{0.0};

    std::array<double, NUM_SIGMA> w_m_{};
    std::array<double, NUM_SIGMA> w_c_{};

    std::array<double, DIM_X> x_{};
    std::array<std::array<double, DIM_X>, DIM_X> P_{};
    std::array<std::array<double, DIM_X>, DIM_X> Q_{};

    double ref_lat_{0.0};
    double ref_lon_{0.0};
    bool is_initialized_{false};
};

} // namespace inav
