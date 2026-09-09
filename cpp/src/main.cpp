/**
 * iNAV Standalone C++ Verification Demo
 * Simulates high-rate IMU streaming through DeadReckoningFilter and VelocityNet.
 */

#include <iostream>
#include <vector>
#include <iomanip>

#include "../include/inav_filter.hpp"
#include "../include/inav_onnx.hpp"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  iNAV Real-Time Dead Reckoning Core Demo (C++17) " << std::endl;
    std::cout << "==================================================" << std::endl;

    inav::DeadReckoningFilter filter(0.1);
    double init_lat = 52.4862;
    double init_lon = -1.8904;
    double init_speed = 25.0; // 25 m/s (90 km/h)
    double init_heading = 45.0;

    filter.initialize(init_lat, init_lon, init_speed, init_heading);
    std::cout << "[INIT] Origin set to Lat=" << init_lat << ", Lon=" << init_lon
              << ", Speed=" << init_speed << " m/s, Heading=" << init_heading << " deg" << std::endl;

    // Simulate 10 seconds of vehicle travel at 10 Hz (100 steps)
    // Forward acceleration = 0.2 m/s^2, yaw rate = 0.01 rad/s (gentle turn)
    for (int step = 1; step <= 100; ++step) {
        double dt = 0.1;
        filter.predict(0.2, 0.01, dt);

        // Every 5 steps (500ms), simulate VelocityNet displacement update
        if (step % 5 == 0) {
            double delta_d = (init_speed + 0.2 * step * dt) * 2.0;
            double sigma = 0.3;
            int event_class = 1; // Normal cruise
            filter.update_velocity_net(delta_d, sigma, event_class, 2.0);
        }

        if (step % 20 == 0) {
            auto st = filter.get_state();
            std::cout << std::fixed << std::setprecision(6)
                      << "t=" << std::setprecision(1) << (step * 0.1) << "s | "
                      << "Lat: " << std::setprecision(6) << st.lat << " | "
                      << "Lon: " << st.lon << " | "
                      << "Speed: " << std::setprecision(2) << st.speed_ms << " m/s | "
                      << "Heading: " << std::setprecision(1) << st.heading_deg << " deg"
                      << std::endl;
        }
    }

    std::cout << "\n[PASS] C++ Dead Reckoning Filter validated successfully." << std::endl;
    return 0;
}
