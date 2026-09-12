#include <cassert>
#include <cmath>
#include <iostream>

#include "inav_sensor_types.hpp"
#include "inav_filter.hpp"

int main() {
    std::cout << "[INFO] Running C++ Sensor Types & Filter Coexistence Test..." << std::endl;

    // 1. Frame Enum Contract
    assert(static_cast<uint8_t>(inav::SensorFrame::PHONE_BODY) == 1);
    assert(static_cast<uint8_t>(inav::SensorFrame::VEHICLE_FRD) == 2);
    assert(static_cast<uint8_t>(inav::SensorFrame::WORLD_NED) == 3);

    // 2. Unit Conversions
    assert(inav::units::sec_to_ns(1.5) == 1500000000LL);
    assert(inav::units::ms_to_ns(250.0) == 250000000LL);
    assert(std::abs(inav::units::kmh_to_mps(36.0) - 10.0) < 1e-6);
    assert(std::abs(inav::units::deg_to_rad(180.0) - M_PI) < 1e-6);

    // 3. ImuSample Construction
    inav::ImuSample imu_sample(
        1000000000LL,
        inav::Vec3(0.0, 0.0, 9.80665),
        inav::Vec3(0.01, 0.02, 0.05),
        inav::SensorFrame::PHONE_BODY,
        inav::ImuValidity::ALL_VALID,
        false
    );
    assert(imu_sample.isValid());
    assert(imu_sample.frame == inav::SensorFrame::PHONE_BODY);

    // 4. GnssSample Construction
    inav::GnssSample gnss_sample;
    gnss_sample.timestamp_ns = 1000000000LL;
    gnss_sample.latitude_deg = 12.9716;
    gnss_sample.longitude_deg = 77.5946;
    gnss_sample.speed_mps = 15.0;
    gnss_sample.bearing_deg = 90.0;
    gnss_sample.horizontal_accuracy_m = 3.0;
    gnss_sample.satellite_count = 12;
    gnss_sample.validity = inav::GnssValidity::BASIC_FIX_VALID;
    assert(gnss_sample.hasBasicFix());
    assert(gnss_sample.hdop == -1.0); // Not fabricated

    // 5. Filter Coexistence Test
    inav::DeadReckoningFilter filter(0.1);
    filter.initialize(12.9716, 77.5946, 10.0, 90.0);

    // Predict using canonical ImuSample overload
    inav::ImuSample sample_vehicle(
        1100000000LL,
        inav::Vec3(1.0, 0.0, 9.81),
        inav::Vec3(0.0, 0.0, 0.02),
        inav::SensorFrame::VEHICLE_FRD,
        inav::ImuValidity::ALL_VALID,
        false
    );
    filter.predict(sample_vehicle, 0.1);

    // Update using canonical GnssSample overload
    filter.update_gnss(gnss_sample);

    auto state = filter.get_state();
    assert(std::abs(state.lat - 12.9716) < 1e-4);
    assert(std::abs(state.lon - 77.5946) < 1e-4);

    // 6. Android Adapter Helper Test
    auto android_sample = inav::make_android_imu_sample(
        1200000000LL,
        0.1, 0.2, 9.8,
        0.01, -0.02, 0.03
    );
    assert(android_sample.frame == inav::SensorFrame::PHONE_BODY);
    assert(android_sample.isValid());

    std::cout << "[SUCCESS] All C++ canonical sensor abstraction tests PASSED!" << std::endl;
    return 0;
}
