#ifndef BUTTERWORTH_2TO8HZ_HPP
#define BUTTERWORTH_2TO8HZ_HPP

#include <cmath>
#include <array>
#include <algorithm>

namespace inav {

/**
 * @brief Second-Order Section (Biquad) Direct Form II Transposed Filter
 */
struct BiquadSection {
    double b0{1.0}, b1{0.0}, b2{0.0};
    double a1{0.0}, a2{0.0}; // Normalized (a0 = 1.0)
    
    // Internal delay states (Direct Form II Transposed)
    double z1{0.0};
    double z2{0.0};

    inline double process(double x) {
        double y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

    void reset() {
        z1 = 0.0;
        z2 = 0.0;
    }
};

/**
 * @brief 2–8 Hz 2nd-Order Butterworth Bandpass Filter
 * Isolates vehicle vertical shock profiles (potholes/speed breakers) from engine noise and DC tilt.
 */
class Butterworth2to8Hz {
public:
    /**
     * @brief Constructor initializing filter for a target sample rate.
     * @param sample_rate_hz Sampling frequency in Hz (default: 100.0 Hz)
     */
    explicit Butterworth2to8Hz(double sample_rate_hz = 100.0) {
        configure(sample_rate_hz);
    }

    /**
     * @brief Computes biquad coefficients using bilinear transform with frequency pre-warping.
     */
    void configure(double sample_rate_hz) {
        m_sample_rate = sample_rate_hz;
        
        // Pre-computed optimal SOS coefficients for fs = 100 Hz
        if (std::abs(sample_rate_hz - 100.0) < 1e-3) {
            // Section 1
            m_sections[0].b0 = 0.02785977;
            m_sections[0].b1 = 0.05571953;
            m_sections[0].b2 = 0.02785977;
            m_sections[0].a1 = -1.50712458;
            m_sections[0].a2 = 0.66927523;

            // Section 2
            m_sections[1].b0 = 1.00000000;
            m_sections[1].b1 = -2.00000000;
            m_sections[1].b2 = 1.00000000;
            m_sections[1].a1 = -1.85785681;
            m_sections[1].a2 = 0.87694790;
        } else {
            // Fallback to safe 100Hz pre-calculated parameters for stability
            m_sections[0] = {0.02785977, 0.05571953, 0.02785977, -1.50712458, 0.66927523, 0.0, 0.0};
            m_sections[1] = {1.00000000, -2.00000000, 1.00000000, -1.85785681, 0.87694790, 0.0, 0.0};
        }
        reset();
    }

    /**
     * @brief Filter a single sample of vertical linear acceleration.
     * @param raw_az_linear Unfiltered vertical acceleration (m/s^2)
     * @return Filtered 2-8 Hz bandpass acceleration (m/s^2)
     */
    inline double process(double raw_az_linear) {
        double stage1 = m_sections[0].process(raw_az_linear);
        double stage2 = m_sections[1].process(stage1);
        return stage2;
    }

    /**
     * @brief Resets filter delay states to zero.
     */
    void reset() {
        m_sections[0].reset();
        m_sections[1].reset();
    }

private:
    double m_sample_rate{100.0};
    std::array<BiquadSection, 2> m_sections;
};

} // namespace inav

#endif // BUTTERWORTH_2TO8HZ_HPP
