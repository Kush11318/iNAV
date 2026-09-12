#pragma once

#ifndef INAV_NO_STDLIB
#include <cstdint>
#include <cmath>
#include <algorithm>
#else
using uint8_t = unsigned char;
using int64_t = long long;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
#endif

#include "inav_sensor_types.hpp"

namespace inav {

/**
 * @brief Authoritative GNSS Health State Machine for Phase 5.
 * States:
 *   AIDED      : Fresh, high-quality GNSS updates actively fused into UKF.
 *   DEGRADED   : Marginal fix quality (accuracy >= 15m or high HDOP); fused with inflated covariance.
 *   PURE_DR    : GNSS outage (timeout > 1.5s or fix invalid); pure dead reckoning.
 *   QUARANTINE : GNSS reappeared after PURE_DR; holding candidate fixes to reject multipath spikes.
 */
enum class GnssHealthState : uint8_t {
    PURE_DR = 0,
    QUARANTINE = 1,
    AIDED = 2,
    DEGRADED = 3
};

struct GnssHealthConfig {
    int64_t outage_timeout_ns{1500000000LL};       // 1.5 seconds missing timeout
    int quarantine_required_epochs{3};             // 3 consecutive consistent valid fixes
    int64_t trust_ramp_duration_ns{5000000000LL};  // 5.0 seconds covariance ramp
    double initial_ramp_scale{9.0};                // R_eff starts at (1.0 + 9.0) = 10x R_base
    double degraded_accuracy_m{15.0};              // Acc >= 15m enters DEGRADED
    double unusable_accuracy_m{50.0};              // Acc >= 50m rejected as invalid
    double speed_heading_min_mps{2.5};             // Heading course updates active only when v > 2.5 m/s
    double nis_pos_2d_threshold{9.21};             // 2-DOF Chi-Square 99% threshold
    double nis_scalar_1d_threshold{6.635};         // 1-DOF Chi-Square 99% threshold
};

class GnssHealthManager {
public:
    explicit GnssHealthManager(const GnssHealthConfig& cfg = GnssHealthConfig{})
        : config_(cfg) {}

    void reset() {
        state_ = GnssHealthState::PURE_DR;
        last_valid_gnss_time_ns_ = 0;
        quarantine_count_ = 0;
        ramp_start_time_ns_ = 0;
        is_ramp_active_ = false;
        total_accepted_fixes_ = 0;
        total_rejected_fixes_ = 0;
    }

    [[nodiscard]] GnssHealthState get_state() const noexcept { return state_; }
    [[nodiscard]] int get_quarantine_count() const noexcept { return quarantine_count_; }
    [[nodiscard]] bool is_ramp_active() const noexcept { return is_ramp_active_; }
    [[nodiscard]] uint64_t get_total_accepted() const noexcept { return total_accepted_fixes_; }
    [[nodiscard]] uint64_t get_total_rejected() const noexcept { return total_rejected_fixes_; }

    [[nodiscard]] bool validate_sample(const GnssSample& sample) const noexcept {
        if (!sample.hasBasicFix()) return false;
        if (sample.horizontal_accuracy_m <= 0.0 || sample.horizontal_accuracy_m >= config_.unusable_accuracy_m) {
            return false;
        }
        return true;
    }

    void start_quarantine() noexcept {
        state_ = GnssHealthState::QUARANTINE;
        quarantine_count_ = 0;
    }

    [[nodiscard]] double get_effective_r_scale(int64_t current_time_ns) const noexcept {
        double ramp_factor = 1.0;
        if (is_ramp_active_) {
            int64_t elapsed_ns = current_time_ns - ramp_start_time_ns_;
            if (elapsed_ns >= config_.trust_ramp_duration_ns || elapsed_ns < 0) {
                ramp_factor = 1.0;
            } else {
                double progress = static_cast<double>(elapsed_ns) / static_cast<double>(config_.trust_ramp_duration_ns);
                ramp_factor = 1.0 + config_.initial_ramp_scale * (1.0 - progress);
            }
        }
        double quality_inflation = (state_ == GnssHealthState::DEGRADED) ? 4.0 : 1.0;
        return ramp_factor * quality_inflation;
    }

    /**
     * @brief Evaluates whether a new GNSS sample should be accepted as a candidate,
     * updates health transitions, and computes the dynamic covariance scale factor.
     */
    bool process_candidate(const GnssSample& sample, double& out_r_scale) {
        out_r_scale = 1.0;

        // 1. Basic fix and sanity checks
        if (!validate_sample(sample)) {
            total_rejected_fixes_++;
            return false;
        }

        // 2. Check if coming from PURE_DR (Outage recovery)
        if (state_ == GnssHealthState::PURE_DR) {
            start_quarantine();
        }

        // 3. Quarantine handling
        if (state_ == GnssHealthState::QUARANTINE) {
            // Fix candidate must be quarantined until N consecutive valid epochs pass
            quarantine_count_++;
            if (quarantine_count_ < config_.quarantine_required_epochs) {
                // In quarantine: do NOT update UKF yet; hold candidate to verify stability
                last_valid_gnss_time_ns_ = sample.timestamp_ns;
                return false;
            }

            // Quarantine completed successfully! Transition to AIDED / DEGRADED and start trust ramp
            if (sample.horizontal_accuracy_m >= config_.degraded_accuracy_m) {
                state_ = GnssHealthState::DEGRADED;
            } else {
                state_ = GnssHealthState::AIDED;
            }
            ramp_start_time_ns_ = sample.timestamp_ns;
            is_ramp_active_ = true;
            quarantine_count_ = 0;
        }

        // 4. Normal AIDED / DEGRADED state updates
        if (state_ == GnssHealthState::AIDED || state_ == GnssHealthState::DEGRADED) {
            if (sample.horizontal_accuracy_m >= config_.degraded_accuracy_m) {
                state_ = GnssHealthState::DEGRADED;
            } else {
                state_ = GnssHealthState::AIDED;
            }
        }

        // 5. Compute effective covariance scaling (Trust ramp + Degradation inflation)
        double ramp_factor = 1.0;
        if (is_ramp_active_) {
            int64_t elapsed_ns = sample.timestamp_ns - ramp_start_time_ns_;
            if (elapsed_ns >= config_.trust_ramp_duration_ns || elapsed_ns < 0) {
                is_ramp_active_ = false;
                ramp_factor = 1.0;
            } else {
                double progress = static_cast<double>(elapsed_ns) / static_cast<double>(config_.trust_ramp_duration_ns);
                // Linear trust ramp: starts at (1.0 + initial_ramp_scale) and decreases to 1.0
                ramp_factor = 1.0 + config_.initial_ramp_scale * (1.0 - progress);
            }
        }

        double quality_inflation = (state_ == GnssHealthState::DEGRADED) ? 4.0 : 1.0;
        out_r_scale = ramp_factor * quality_inflation;

        last_valid_gnss_time_ns_ = sample.timestamp_ns;
        total_accepted_fixes_++;
        return true;
    }

    /**
     * @brief Checks timestamp against outage timeout. Call during IMU predict ticks.
     */
    void check_timeout(int64_t current_time_ns) {
        if (last_valid_gnss_time_ns_ == 0) return;
        int64_t elapsed_ns = current_time_ns - last_valid_gnss_time_ns_;
        if (elapsed_ns > config_.outage_timeout_ns) {
            if (state_ != GnssHealthState::PURE_DR) {
                state_ = GnssHealthState::PURE_DR;
                quarantine_count_ = 0;
                is_ramp_active_ = false;
            }
        }
    }

    /**
     * @brief Called when NIS gate rejects a measurement in the UKF.
     */
    void on_measurement_rejected() {
        total_rejected_fixes_++;
        if (state_ == GnssHealthState::QUARANTINE) {
            // Quarantine fix failed consistency check; reset counter
            quarantine_count_ = 0;
        }
    }

    [[nodiscard]] const GnssHealthConfig& config() const noexcept { return config_; }

private:
    GnssHealthConfig config_{};
    GnssHealthState state_{GnssHealthState::PURE_DR};
    int64_t last_valid_gnss_time_ns_{0};
    int quarantine_count_{0};
    int64_t ramp_start_time_ns_{0};
    bool is_ramp_active_{false};
    uint64_t total_accepted_fixes_{0};
    uint64_t total_rejected_fixes_{0};
};

} // namespace inav
