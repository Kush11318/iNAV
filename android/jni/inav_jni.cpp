/**
 * iNAV Android Native Interface (JNI) Bridge
 * Connects the Android Kotlin background sensor service with the C++17
 * dead reckoning engine and ONNX Runtime VelocityNet inference session.
 */

#include <jni.h>
#include <memory>
#include <vector>
#include <string>

#include "../../cpp/include/inav_filter.hpp"
#include "../../cpp/include/inav_onnx.hpp"

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
    }
    g_window_buffer.clear();
}

JNIEXPORT jdoubleArray JNICALL
Java_com_inav_navigation_NativeBridge_nativeProcessImu(
    JNIEnv* env,
    jobject /* this */,
    jfloat ax, jfloat ay, jfloat az,
    jfloat gx, jfloat gy, jfloat gz,
    jdouble dt
) {
    if (!g_filter) {
        return nullptr;
    }

    // Kinematic propagation (assuming forward acc along ax and yaw along gz)
    g_filter->predict(static_cast<double>(ax), static_cast<double>(gz), dt);

    // Buffer sample for rolling window: 6 channels
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
                g_filter->update_velocity_net(out.delta_d, out.sigma, out.event_class, 2.0);
            }
        }
        // Retain rolling window buffer
        if (g_window_buffer.size() > 240) {
            g_window_buffer.erase(g_window_buffer.begin(), g_window_buffer.begin() + 120);
        }
    }

    auto st = g_filter->get_state();

    // Return [lat, lon, speed_ms, heading_deg, event_class, sigma]
    jdoubleArray result = env->NewDoubleArray(6);
    jdouble buf[6] = {
        st.lat,
        st.lon,
        st.speed_ms,
        st.heading_deg,
        static_cast<double>(g_last_event_class),
        static_cast<double>(g_last_sigma)
    };
    env->SetDoubleArrayRegion(result, 0, 6, buf);
    return result;
}

} // extern "C"
