/**
 * iNAV Android Native Interface (JNI) Bridge
 * Connects the Android Kotlin background sensor service with the C++17
 * dead reckoning engine and ONNX Runtime VelocityNet inference session.
 */

#include <jni.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <android/log.h>

#include "inav_filter.hpp"
#include "inav_onnx.hpp"
#include "Butterworth2to8Hz.hpp"
#include "LevelingUtils.hpp"

#define LOG_TAG "iNAV_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::unique_ptr<inav::VelocityNetSession> g_velocity_net;
static std::unique_ptr<inav::DeadReckoningFilter> g_filter;
static std::vector<float> g_window_buffer;
static int g_last_event_class = 1;
static float g_last_sigma = 0.5f;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_inav_navigation_NativeBridge_nativeInit(
    JNIEnv* env,
    jobject /* this */,
    jstring model_path_jstr
) {
    const char* path_chars = env->GetStringUTFChars(model_path_jstr, nullptr);
    std::string model_path(path_chars);
    env->ReleaseStringUTFChars(model_path_jstr, path_chars);

    g_velocity_net = std::make_unique<inav::VelocityNetSession>(model_path);
    g_filter = std::make_unique<inav::DeadReckoningFilter>(0.1);
    g_window_buffer.clear();
    g_window_buffer.reserve(120);

    LOGI("Native iNAV engine initialized with model: %s", model_path.c_str());
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_inav_navigation_NativeBridge_nativeReset(
    JNIEnv* /* env */,
    jobject /* this */,
    jdouble init_lat,
    jdouble init_lon,
    jdouble init_speed_ms,
    jdouble init_heading_deg
) {
    if (g_filter) {
        g_filter->initialize(init_lat, init_lon, init_speed_ms, init_heading_deg);
        LOGI("Native filter reset origin: Lat=%.6f, Lon=%.6f, Spd=%.1f m/s, Hdg=%.1f deg",
             init_lat, init_lon, init_speed_ms, init_heading_deg);
    }
    g_window_buffer.clear();
}

JNIEXPORT void JNICALL
Java_com_inav_navigation_NativeBridge_nativeUpdateGnss(
    JNIEnv* /* env */,
    jobject /* this */,
    jdouble lat,
    jdouble lon,
    jdouble speed_ms,
    jdouble heading_deg
) {
    if (g_filter) {
        g_filter->update_gnss(lat, lon, speed_ms, heading_deg);
    }
}

JNIEXPORT void JNICALL
Java_com_inav_navigation_NativeBridge_nativeSetScaleFactor(
    JNIEnv* /* env */,
    jobject /* this */,
    jdouble scale_k
) {
    if (g_filter) {
        g_filter->set_scale_factor(scale_k);
        LOGI("Scale factor updated: k=%.3f", scale_k);
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_com_inav_navigation_NativeBridge_nativeProcessImu(
    JNIEnv* env,
    jobject /* this */,
    jfloat ax, jfloat ay, jfloat az,
    jfloat gx, jfloat gy, jfloat gz,
    jfloat qw, jfloat qx, jfloat qy, jfloat qz,
    jdouble dt,
    jboolean is_stationary,
    jdouble gnss_speed,
    jboolean is_gnss_healthy,
    jdouble obd_speed,
    jboolean is_obd_connected
) {
    if (!g_filter) {
        return nullptr;
    }

    inav::Quaternion q{
        static_cast<double>(qw),
        static_cast<double>(qx),
        static_cast<double>(qy),
        static_cast<double>(qz)
    };

    // Fallback: If quaternion is uninitialized, compute pitch & roll from gravity
    if (std::abs(q.w) < 1e-4 && std::abs(q.x) < 1e-4 && std::abs(q.y) < 1e-4 && std::abs(q.z) < 1e-4) {
        double pitch = 0.0, roll = 0.0;
        inav::LevelingUtils::computeStaticPitchRoll({ax, ay, az}, pitch, roll);
        double cr = std::cos(roll * 0.5);
        double sr = std::sin(roll * 0.5);
        double cp = std::cos(pitch * 0.5);
        double sp = std::sin(pitch * 0.5);
        q.w = cr * cp;
        q.x = sr * cp;
        q.y = cr * sp;
        q.z = -sr * sp;
    }

    auto st = g_filter->get_state();

    // 3-Step Road Anomaly Detection (Leveling -> Butterworth 2-8Hz -> Speed-Gated Classification)
    auto anomaly = g_filter->process_road_anomaly(ax, ay, az, q, st.speed_ms);

    // Compute rotational IMU variance (rad^2/s^2) to detect desk handling / fidgeting
    double imu_variance = static_cast<double>(gx * gx + gy * gy + gz * gz);

    // Set GNSS and OBD state
    g_filter->set_gnss_state(gnss_speed, is_gnss_healthy);
    g_filter->set_obd_speed(obd_speed, is_obd_connected);

    // 3-Tier Gated Kinematic Prediction (No unconstrained raw accel integration)
    g_filter->predict(dt, static_cast<double>(gz), is_stationary, imu_variance);

    // Buffer sample for rolling window: 6 channels [ax, ay, az, gx, gy, gz]
    g_window_buffer.push_back(ax);
    g_window_buffer.push_back(ay);
    g_window_buffer.push_back(az);
    g_window_buffer.push_back(gx);
    g_window_buffer.push_back(gy);
    g_window_buffer.push_back(gz);

    // When 20 samples (120 floats = 2.0s) accumulated, run inference every 500ms
    if (g_window_buffer.size() >= 120) {
        if (g_velocity_net && g_velocity_net->is_initialized()) {
            std::vector<float> win(g_window_buffer.end() - 120, g_window_buffer.end());
            auto out = g_velocity_net->predict(win);

            g_last_event_class = out.event_class;
            g_last_sigma = out.sigma;

            if (out.event_class == 0 || out.delta_d < 0.1f) {
                g_filter->update_zupt(gz);
            } else {
                g_filter->update_velocity_net(out.delta_d, out.sigma, out.event_class, 2.0, anomaly.is_anomaly);
            }
        }
        // Retain rolling window buffer
        if (g_window_buffer.size() > 240) {
            g_window_buffer.erase(g_window_buffer.begin(), g_window_buffer.begin() + 120);
        }
    }

    auto updated_st = g_filter->get_state();

    // Return [lat, lon, speed_ms, heading_deg, event_class, sigma, filtered_bump, anomaly_type]
    jdoubleArray result = env->NewDoubleArray(8);
    jdouble buf[8] = {
        updated_st.lat,
        updated_st.lon,
        updated_st.speed_ms,
        updated_st.heading_deg,
        static_cast<double>(anomaly.is_anomaly ? 2 : g_last_event_class),
        static_cast<double>(g_last_sigma),
        anomaly.filtered_bump,
        static_cast<double>(anomaly.type) // 0=none, 1=pothole, 2=speed breaker
    };
    env->SetDoubleArrayRegion(result, 0, 8, buf);
    return result;
}

} // extern "C"
