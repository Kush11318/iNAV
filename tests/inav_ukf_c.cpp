// C Interface DLL for Python-C++ Numerical Parity Verification of UKF
#define INAV_NO_STDLIB 1

extern "C" {
    __declspec(dllexport) int DllMainCRTStartup(void* hinstDLL, unsigned int fdwReason, void* lpReserved) {
        return 1;
    }
}

#include "../cpp/include/inav_ukf.hpp"

static inav::UKFNavigationFilter g_ukf_instances[8];
static inav::GnssHealthManager g_health_instances[8];
static bool g_used[8] = {false};

extern "C" {

__declspec(dllexport) int ukf_create(double dt, double alpha, double beta, double kappa) {
    for (int i = 0; i < 8; ++i) {
        if (!g_used[i]) {
            g_used[i] = true;
            g_ukf_instances[i] = inav::UKFNavigationFilter(dt, alpha, beta, kappa);
            g_health_instances[i].reset();
            return i;
        }
    }
    return -1;
}

__declspec(dllexport) void ukf_destroy(int handle) {
    if (handle >= 0 && handle < 8) {
        g_used[handle] = false;
    }
}

__declspec(dllexport) void ukf_initialize(
    int handle,
    double lat,
    double lon,
    double speed,
    double heading,
    double gyro_bias,
    double accel_bias
) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        g_ukf_instances[handle].initialize(lat, lon, speed, heading, gyro_bias, accel_bias);
        g_health_instances[handle].reset();
    }
}

__declspec(dllexport) void ukf_predict(int handle, double acc_fwd, double gyro_yaw, double dt) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        g_ukf_instances[handle].predict(acc_fwd, gyro_yaw, dt);
    }
}

__declspec(dllexport) void ukf_update_velocity_net(
    int handle,
    double delta_d,
    double sigma,
    int event_class,
    double window_dur
) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        g_ukf_instances[handle].update_velocity_net(delta_d, sigma, event_class, window_dur);
    }
}

__declspec(dllexport) void ukf_update_zupt(int handle, double gyro_reading) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        g_ukf_instances[handle].update_zupt(gyro_reading);
    }
}

__declspec(dllexport) double ukf_compute_pos_nis(int handle, double p_N, double p_E, double accuracy_m, double r_scale) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return g_ukf_instances[handle].compute_pos_nis(p_N, p_E, accuracy_m, r_scale);
    }
    return 999999.0;
}

__declspec(dllexport) int ukf_update_gnss_pos(int handle, double p_N, double p_E, double accuracy_m, double r_scale, double* out_nis) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return g_ukf_instances[handle].update_gnss_pos(p_N, p_E, accuracy_m, r_scale, out_nis) ? 1 : 0;
    }
    return 0;
}

__declspec(dllexport) int ukf_update_gnss_speed(int handle, double v_gnss, double sigma_v, double r_scale, double* out_nis) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return g_ukf_instances[handle].update_gnss_speed(v_gnss, sigma_v, r_scale, out_nis) ? 1 : 0;
    }
    return 0;
}

__declspec(dllexport) int ukf_update_gnss_course(int handle, double psi_gnss, double v_gnss, double sigma_psi, double r_scale, double* out_nis) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return g_ukf_instances[handle].update_gnss_course(psi_gnss, v_gnss, sigma_psi, r_scale, out_nis) ? 1 : 0;
    }
    return 0;
}

__declspec(dllexport) int ukf_health_get_state(int handle) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return static_cast<int>(g_health_instances[handle].get_state());
    }
    return 0;
}

__declspec(dllexport) int ukf_health_get_quarantine_count(int handle) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return g_health_instances[handle].get_quarantine_count();
    }
    return 0;
}

__declspec(dllexport) void ukf_health_check_timeout(int handle, long long current_time_ns) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        g_health_instances[handle].check_timeout(current_time_ns);
    }
}

__declspec(dllexport) void ukf_health_reset(int handle) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        g_health_instances[handle].reset();
    }
}

__declspec(dllexport) int ukf_update_gnss_sample(
    int handle,
    long long timestamp_ns,
    double lat,
    double lon,
    double alt,
    double speed,
    double bearing,
    double accuracy,
    unsigned int validity
) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        inav::GnssSample s;
        s.timestamp_ns = timestamp_ns;
        s.latitude_deg = lat;
        s.longitude_deg = lon;
        s.altitude_m = alt;
        s.speed_mps = speed;
        s.bearing_deg = bearing;
        s.horizontal_accuracy_m = accuracy;
        s.validity = validity;
        return g_ukf_instances[handle].update_gnss_sample(s, g_health_instances[handle]) ? 1 : 0;
    }
    return 0;
}

__declspec(dllexport) int ukf_update_map_match(
    int handle,
    double p_N,
    double p_E,
    double psi_road,
    double confidence,
    int is_heading_valid,
    double* out_pos_nis,
    double* out_hdg_nis
) {
    if (handle >= 0 && handle < 8 && g_used[handle]) {
        return g_ukf_instances[handle].update_map_match(
            p_N, p_E, psi_road, confidence, (is_heading_valid != 0), out_pos_nis, out_hdg_nis
        ) ? 1 : 0;
    }
    return 0;
}

__declspec(dllexport) void ukf_get_x(int handle, double* out_x) {
    if (handle >= 0 && handle < 8 && g_used[handle] && out_x) {
        const auto& x = g_ukf_instances[handle].get_x();
        for (int i = 0; i < 7; ++i) {
            out_x[i] = x[i];
        }
    }
}

__declspec(dllexport) void ukf_get_P(int handle, double* out_P) {
    if (handle >= 0 && handle < 8 && g_used[handle] && out_P) {
        const auto& P = g_ukf_instances[handle].get_P();
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                out_P[r * 7 + c] = P[r][c];
            }
        }
    }
}

} // extern "C"
