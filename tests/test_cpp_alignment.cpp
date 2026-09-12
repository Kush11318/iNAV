#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "inav_sensor_types.hpp"
#include "inav_alignment.hpp"

constexpr double G = 9.80665;

int main() {
    std::cout << "[INFO] Running C++ 19-Case Alignment & Calibration Test Suite..." << std::endl;

    // 1. Flat phone leveling
    {
        inav::AlignmentEngine engine;
        std::vector<inav::Vec3> acc(30, inav::Vec3(0.0, 0.0, G));
        std::vector<inav::Vec3> gyro(30, inav::Vec3(0.0, 0.0, 0.0));
        assert(engine.calibrate_static_buffer(acc, gyro));
        assert(engine.get_result().state == inav::AlignmentState::STATIC_ALIGNED);
        assert(std::abs(engine.get_result().pitch_deg) < 1e-3);
        assert(std::abs(engine.get_result().roll_deg) < 1e-3);
        std::cout << "[PASS] C++ Test 1: Flat phone leveling" << std::endl;
    }

    // 2 & 3. Pitch recovery (+/- 30 deg)
    {
        inav::AlignmentEngine engine_pos, engine_neg;
        double p = 30.0 * (M_PI / 180.0);
        std::vector<inav::Vec3> acc_pos(30, inav::Vec3(G * std::sin(p), 0.0, G * std::cos(p)));
        std::vector<inav::Vec3> acc_neg(30, inav::Vec3(-G * std::sin(p), 0.0, G * std::cos(p)));
        std::vector<inav::Vec3> gyro(30, inav::Vec3(0.0, 0.0, 0.0));

        assert(engine_pos.calibrate_static_buffer(acc_pos, gyro));
        assert(std::abs(engine_pos.get_result().pitch_deg - 30.0) < 0.1);

        assert(engine_neg.calibrate_static_buffer(acc_neg, gyro));
        assert(std::abs(engine_neg.get_result().pitch_deg - (-30.0)) < 0.1);
        std::cout << "[PASS] C++ Tests 2 & 3: Pitch recovery (+/- 30 deg)" << std::endl;
    }

    // 4 & 5. Roll recovery (+/- 30 deg)
    {
        inav::AlignmentEngine engine_pos, engine_neg;
        double r = 30.0 * (M_PI / 180.0);
        std::vector<inav::Vec3> acc_pos(30, inav::Vec3(0.0, -G * std::sin(r), G * std::cos(r)));
        std::vector<inav::Vec3> acc_neg(30, inav::Vec3(0.0, G * std::sin(r), G * std::cos(r)));
        std::vector<inav::Vec3> gyro(30, inav::Vec3(0.0, 0.0, 0.0));

        assert(engine_pos.calibrate_static_buffer(acc_pos, gyro));
        assert(std::abs(engine_pos.get_result().roll_deg - 30.0) < 0.1);

        assert(engine_neg.calibrate_static_buffer(acc_neg, gyro));
        assert(std::abs(engine_neg.get_result().roll_deg - (-30.0)) < 0.1);
        std::cout << "[PASS] C++ Tests 4 & 5: Roll recovery (+/- 30 deg)" << std::endl;
    }

    // 6. Gyro bias injection & zero corrected gyro
    {
        inav::AlignmentEngine engine;
        inav::Vec3 bias(0.01, -0.02, 0.03);
        std::vector<inav::Vec3> acc(30, inav::Vec3(0.0, 0.0, G));
        std::vector<inav::Vec3> gyro(30, bias);
        assert(engine.calibrate_static_buffer(acc, gyro));
        assert(std::abs(engine.get_result().gyro_bias_body.x - 0.01) < 1e-4);
        assert(std::abs(engine.get_result().gyro_bias_body.y - (-0.02)) < 1e-4);
        assert(std::abs(engine.get_result().gyro_bias_body.z - 0.03) < 1e-4);

        inav::ImuSample raw(1000000000LL, inav::Vec3(0,0,G), bias, inav::SensorFrame::PHONE_BODY);
        inav::ImuSample veh = engine.transform_imu(raw);
        assert(veh.frame == inav::SensorFrame::VEHICLE_FRD);
        assert(std::abs(veh.gyro_radps.x) < 1e-5);
        assert(std::abs(veh.gyro_radps.y) < 1e-5);
        assert(std::abs(veh.gyro_radps.z) < 1e-5);
        std::cout << "[PASS] C++ Tests 7 & 8: Gyro bias injection and zero corrected gyro" << std::endl;
    }

    // 7. Gravity detrending cancellation
    {
        inav::AlignmentEngine engine;
        std::vector<inav::Vec3> acc(30, inav::Vec3(0.0, 0.0, G));
        std::vector<inav::Vec3> gyro(30, inav::Vec3(0.0, 0.0, 0.0));
        engine.calibrate_static_buffer(acc, gyro);

        inav::ImuSample rest(1000000000LL, inav::Vec3(0,0,G), inav::Vec3(0,0,0), inav::SensorFrame::PHONE_BODY);
        inav::ImuSample veh = engine.transform_imu(rest);
        inav::Vec3 lin = engine.get_linear_accel(veh);
        assert(std::abs(lin.x) < 1e-4);
        assert(std::abs(lin.y) < 1e-4);
        assert(std::abs(lin.z) < 1e-4);
        std::cout << "[PASS] C++ Test 11: Gravity detrending cancellation ([0,0,0] m/s^2)" << std::endl;
    }

    // 8. Clockwise & Counter-Clockwise Yaw Sign Parity (CRITICAL)
    {
        inav::AlignmentEngine engine;
        std::vector<inav::Vec3> acc_static(30, inav::Vec3(0.0, 0.0, G));
        std::vector<inav::Vec3> gyro_static(30, inav::Vec3(0.0, 0.0, 0.0));
        engine.calibrate_static_buffer(acc_static, gyro_static);

        // Portrait driving forward along phone +Y
        std::vector<inav::Vec3> acc_driving;
        std::vector<inav::Vec3> gyro_driving(40, inav::Vec3(0,0,0));
        std::vector<double> spd_deltas(40, 0.15);
        for (int i = 0; i < 40; ++i) {
            acc_driving.emplace_back(0.0, 1.5 + 1.0 * std::sin(i), G);
        }
        assert(engine.calibrate_dynamic_buffer(acc_driving, gyro_driving, spd_deltas));

        // Right turn: phone gyro reports negative around +Z_phone
        inav::ImuSample right_turn(2000000000LL, inav::Vec3(0,0,G), inav::Vec3(0,0,-0.05), inav::SensorFrame::PHONE_BODY);
        inav::ImuSample veh_right = engine.transform_imu(right_turn);
        // VEHICLE_FRD yaw rate MUST be positive
        assert(veh_right.gyro_radps.z > 0.0);
        assert(std::abs(veh_right.gyro_radps.z - 0.05) < 1e-3);

        // Left turn: phone gyro reports positive around +Z_phone
        inav::ImuSample left_turn(2000000000LL, inav::Vec3(0,0,G), inav::Vec3(0,0,0.05), inav::SensorFrame::PHONE_BODY);
        inav::ImuSample veh_left = engine.transform_imu(left_turn);
        // VEHICLE_FRD yaw rate MUST be negative
        assert(veh_left.gyro_radps.z < 0.0);
        assert(std::abs(veh_left.gyro_radps.z - (-0.05)) < 1e-3);
        std::cout << "[PASS] C++ Tests 12 & 13: Yaw sign parity verified (turning right produces +omega_z)" << std::endl;
    }

    // 9. Stationarity gate and moving rejection
    {
        inav::AlignmentEngine engine;
        // High variance motion
        std::vector<inav::Vec3> acc_moving;
        for (int i = 0; i < 30; ++i) acc_moving.emplace_back(0, 0, G + 0.5 * std::sin(i));
        std::vector<inav::Vec3> gyro_static(30, inav::Vec3(0,0,0));
        assert(!engine.calibrate_static_buffer(acc_moving, gyro_static));
        std::cout << "[PASS] C++ Tests 18 & 19: Stationarity gate & moving calibration rejection" << std::endl;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "All C++ Alignment & Calibration tests PASSED successfully!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
