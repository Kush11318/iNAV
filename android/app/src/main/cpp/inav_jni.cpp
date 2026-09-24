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
#include "inav_ukf.hpp"
#include "inav_onnx.hpp"
#include "Butterworth2to8Hz.hpp"
#include "LevelingUtils.hpp"
#include "inav_alignment.hpp"
#include "inav_road_graph.hpp"
#include "inav_map_matcher.hpp"

#define LOG_TAG "iNAV_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include <time.h>

static std::unique_ptr<inav::VelocityNetSession> g_velocity_net;
static std::unique_ptr<inav::DeadReckoningFilter> g_filter;
static std::unique_ptr<inav::UKFNavigationFilter> g_ukf;
static inav::GnssHealthManager g_gnss_health;
static std::unique_ptr<inav::AlignmentEngine> g_alignment;
static std::unique_ptr<inav::RoadGraph> g_road_graph;
static std::unique_ptr<inav::FixedLagHMMMapMatcher> g_map_matcher;
static std::vector<float> g_window_buffer;
static int g_last_event_class = 1;
static float g_last_sigma = 0.5f;
static int g_map_epoch_counter = 0;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_inav_navigation_NativeBridge_jniInit(
    JNIEnv* env,
    jobject /* this */,
    jstring model_path_jstr
) {
    const char* path_chars = env->GetStringUTFChars(model_path_jstr, nullptr);
    std::string model_path(path_chars);
    env->ReleaseStringUTFChars(model_path_jstr, path_chars);

    g_velocity_net = std::make_unique<inav::VelocityNetSession>(model_path);
    g_filter = std::make_unique<inav::DeadReckoningFilter>(0.1);
    g_ukf = std::make_unique<inav::UKFNavigationFilter>(0.1);
    g_gnss_health.reset();
    g_alignment = std::make_unique<inav::AlignmentEngine>();
    g_road_graph = std::make_unique<inav::RoadGraph>();
    g_map_matcher = std::make_unique<inav::FixedLagHMMMapMatcher>(g_road_graph.get());
    g_window_buffer.clear();
    g_window_buffer.reserve(120);
    g_map_epoch_counter = 0;

    LOGI("Native iNAV engine initialized with model: %s (C++ 7-State UKF + HMM Map Matcher active)", model_path.c_str());
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_inav_navigation_NativeBridge_jniReset(
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
    if (g_ukf) {
        g_ukf->initialize(init_lat, init_lon, init_speed_ms, init_heading_deg * (inav::UKF_PI / 180.0));
        g_gnss_health.reset();
        LOGI("Native UKF reset origin: Lat=%.6f, Lon=%.6f, Spd=%.1f m/s, Hdg=%.1f deg",
             init_lat, init_lon, init_speed_ms, init_heading_deg);
    }
    if (g_alignment) {
        g_alignment->reset();
    }
    if (g_map_matcher) {
        g_map_matcher->reset();
    }
    if (g_road_graph) {
        g_road_graph->set_reference_origin(init_lat, init_lon);
    }
    g_map_epoch_counter = 0;
    g_window_buffer.clear();
}

JNIEXPORT void JNICALL
Java_com_inav_navigation_NativeBridge_jniApplyStaticCalibration(
    JNIEnv* /* env */,
    jobject /* this */,
    jdouble ax_mean,
    jdouble ay_mean,
    jdouble az_mean,
    jdouble gx_bias,
    jdouble gy_bias,
    jdouble gz_bias
) {
    if (g_alignment) {
        inav::Vec3 a_mean(ax_mean, ay_mean, az_mean);
        inav::Vec3 g_mean(gx_bias, gy_bias, gz_bias);
        g_alignment->apply_static_solution(a_mean, g_mean);
        LOGI("Applied 15s Startup Calibration to AlignmentEngine: gravity=[%.3f, %.3f, %.3f], gyro_bias=[%.5f, %.5f, %.5f] rad/s, Pitch=%.2f deg, Roll=%.2f deg",
             ax_mean, ay_mean, az_mean, gx_bias, gy_bias, gz_bias,
             g_alignment->get_result().pitch_deg, g_alignment->get_result().roll_deg);
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_com_inav_navigation_NativeBridge_jniGetAlignmentStatus(
    JNIEnv* env,
    jobject /* this */
) {
    // Returns [pitch_deg, roll_deg, yaw_deg, confidence, state, gyro_bias_x, gyro_bias_y, gyro_bias_z]
    jdoubleArray result = env->NewDoubleArray(8);
    if (!g_alignment) {
        jdouble zeros[8] = {0,0,0,0,0,0,0,0};
        env->SetDoubleArrayRegion(result, 0, 8, zeros);
        return result;
    }
    const auto& res = g_alignment->get_result();
    jdouble buf[8] = {
        res.pitch_deg,
        res.roll_deg,
        res.yaw_deg,
        res.confidence,
        static_cast<double>(res.state),
        res.gyro_bias_body.x,
        res.gyro_bias_body.y,
        res.gyro_bias_body.z
    };
    env->SetDoubleArrayRegion(result, 0, 8, buf);
    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_inav_navigation_NativeBridge_jniUpdateGnss(
    JNIEnv* /* env */,
    jobject /* this */,
    jdouble lat,
    jdouble lon,
    jdouble speed_ms,
    jdouble heading_deg,
    jdouble accuracy_m,
    jlong timestamp_ns
) {
    if (g_ukf && g_ukf->is_initialized()) {
        inav::GnssSample sample;
        sample.timestamp_ns = static_cast<int64_t>(timestamp_ns);
        sample.latitude_deg = lat;
        sample.longitude_deg = lon;
        sample.speed_mps = (speed_ms >= 0.0) ? speed_ms : 0.0;
        sample.bearing_deg = (heading_deg >= 0.0) ? heading_deg : 0.0;
        sample.horizontal_accuracy_m = (accuracy_m > 0.0) ? accuracy_m : 5.0;

        sample.validity = inav::GnssValidity::LAT_LON_VALID | inav::GnssValidity::ACCURACY_VALID;
        if (speed_ms >= 0.0) sample.validity |= inav::GnssValidity::SPEED_VALID;
        if (heading_deg >= 0.0) sample.validity |= inav::GnssValidity::BEARING_VALID;

        bool pos_ok = false, spd_ok = false, crs_ok = false;
        bool ok = g_ukf->update_gnss_sample(sample, g_gnss_health, &pos_ok, &spd_ok, &crs_ok);
        LOGI("GNSS update: Lat=%.6f, Lon=%.6f, Acc=%.1fm, Spd=%.1fm/s, Hdg=%.1fdeg | State=%d, PosOk=%d, SpdOk=%d, CrsOk=%d",
             lat, lon, accuracy_m, speed_ms, heading_deg, static_cast<int>(g_gnss_health.get_state()), pos_ok, spd_ok, crs_ok);
        return ok ? JNI_TRUE : JNI_FALSE;
    }
    return JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_com_inav_navigation_NativeBridge_jniGetGnssHealthState(
    JNIEnv* /* env */,
    jobject /* this */
) {
    return static_cast<jint>(g_gnss_health.get_state());
}

JNIEXPORT void JNICALL
Java_com_inav_navigation_NativeBridge_jniSetScaleFactor(
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
Java_com_inav_navigation_NativeBridge_jniProcessImu(
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
    if (!g_filter && !g_ukf) {
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

    auto st = g_filter ? g_filter->get_state() : inav::NavState{};

    // 3-Step Road Anomaly Detection (Leveling -> Butterworth 2-8Hz -> Speed-Gated Classification)
    auto anomaly = g_filter ? g_filter->process_road_anomaly(ax, ay, az, q, st.speed_ms) : inav::AnomalyResult{};

    // Compute rotational IMU variance (rad^2/s^2) to detect desk handling / fidgeting
    double imu_variance = static_cast<double>(gx * gx + gy * gy + gz * gz);

    // Feed into Unified Alignment Engine (PHONE_BODY -> VEHICLE_FRD)
    inav::ImuSample body_sample = inav::make_android_imu_sample(
        0,
        static_cast<double>(ax), static_cast<double>(ay), static_cast<double>(az),
        static_cast<double>(gx), static_cast<double>(gy), static_cast<double>(gz)
    );
    body_sample.is_stationary = is_stationary;

    std::optional<inav::GnssSample> gnss_opt = std::nullopt;
    if (is_gnss_healthy && gnss_speed >= 0.0) {
        inav::GnssSample g;
        g.speed_mps = gnss_speed;
        g.validity = inav::GnssValidity::SPEED_VALID;
        gnss_opt = g;
    }

    if (g_alignment) {
        g_alignment->feed_imu(body_sample, gnss_opt);
    }
    inav::ImuSample veh_sample = g_alignment ? g_alignment->transform_imu(body_sample) : body_sample;

    // Set GNSS and OBD state on legacy filter
    if (g_filter) {
        g_filter->set_gnss_state(gnss_speed, is_gnss_healthy);
        g_filter->set_obd_speed(obd_speed, is_obd_connected);
        g_filter->predict(dt, veh_sample.gyro_radps.z, is_stationary, imu_variance);
    }

    // Check GNSS timeout using monotonic clock
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    int64_t now_ns = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    g_gnss_health.check_timeout(now_ns);

    // C++ 7-State UKF Prediction & ZUPT
    if (g_ukf && g_ukf->is_initialized()) {
        g_ukf->predict(veh_sample.accel_mps2.x, veh_sample.gyro_radps.z, dt, is_stationary);
        if (is_stationary) {
            g_ukf->update_zupt(veh_sample.gyro_radps.z);
        }

        // 2 Hz Map Matching Update (every 5 IMU epochs at 10 Hz)
        g_map_epoch_counter++;
        if (g_road_graph && !g_road_graph->edges().empty() && g_map_matcher && (g_map_epoch_counter % 5 == 0)) {
            double p_N = g_ukf->get_x()[0];
            double p_E = g_ukf->get_x()[1];
            double heading_rad = g_ukf->get_x()[3];
            double speed_mps = g_ukf->get_x()[2];
            double sigma_pos = std::sqrt(g_ukf->get_P()[0][0] + g_ukf->get_P()[1][1]);

            auto match_res = g_map_matcher->match(p_N, p_E, heading_rad, speed_mps, sigma_pos, 0.5);
            if (!match_res.is_off_road && match_res.confidence >= 0.25) {
                double pos_nis = 0.0, hdg_nis = 0.0;
                g_ukf->update_map_match(
                    match_res.snapped_point.p_N,
                    match_res.snapped_point.p_E,
                    match_res.road_heading_rad,
                    match_res.confidence,
                    match_res.confidence > 0.5,
                    &pos_nis,
                    &hdg_nis
                );
            }
        }
    }

    // Buffer sample in vehicle frame: 6 channels [ax_v, ay_v, az_v, gx_v, gy_v, gz_v]
    g_window_buffer.push_back(static_cast<float>(veh_sample.accel_mps2.x));
    g_window_buffer.push_back(static_cast<float>(veh_sample.accel_mps2.y));
    g_window_buffer.push_back(static_cast<float>(veh_sample.accel_mps2.z));
    g_window_buffer.push_back(static_cast<float>(veh_sample.gyro_radps.x));
    g_window_buffer.push_back(static_cast<float>(veh_sample.gyro_radps.y));
    g_window_buffer.push_back(static_cast<float>(veh_sample.gyro_radps.z));

    // When 20 samples (120 floats = 2.0s) accumulated, run inference every 500ms
    if (g_window_buffer.size() >= 120) {
        if (g_velocity_net && g_velocity_net->is_initialized()) {
            std::vector<float> win(g_window_buffer.end() - 120, g_window_buffer.end());
            auto out = g_velocity_net->predict(win);

            g_last_event_class = out.event_class;
            g_last_sigma = out.sigma;

            if (out.event_class == 0 || out.delta_d < 0.1f) {
                if (g_filter) g_filter->update_zupt(veh_sample.gyro_radps.z);
                if (g_ukf && g_ukf->is_initialized()) {
                    g_ukf->update_zupt(veh_sample.gyro_radps.z);
                }
            } else {
                if (g_filter) g_filter->update_velocity_net(out.delta_d, out.sigma, out.event_class, 2.0, anomaly.is_anomaly);
                if (g_ukf && g_ukf->is_initialized()) {
                    g_ukf->update_velocity_net(out.delta_d, out.sigma, out.event_class, 2.0);
                }
            }
        }
        // Retain rolling window buffer
        if (g_window_buffer.size() > 240) {
            g_window_buffer.erase(g_window_buffer.begin(), g_window_buffer.begin() + 120);
        }
    }

    // Active state source: C++ 7-State UKF
    double out_lat{0.0}, out_lon{0.0}, out_speed_ms{0.0}, out_heading_deg{0.0};
    if (g_ukf && g_ukf->is_initialized()) {
        auto ukf_st = g_ukf->get_nav_state();
        out_lat = ukf_st.lat;
        out_lon = ukf_st.lon;
        out_speed_ms = ukf_st.speed_ms;
        out_heading_deg = ukf_st.heading_deg;
    } else if (g_filter) {
        auto legacy_st = g_filter->get_state();
        out_lat = legacy_st.lat;
        out_lon = legacy_st.lon;
        out_speed_ms = legacy_st.speed_ms;
        out_heading_deg = legacy_st.heading_deg;
    }

    // Return [lat, lon, speed_ms, heading_deg, event_class, sigma, filtered_bump, anomaly_type]
    jdoubleArray result = env->NewDoubleArray(8);
    jdouble buf[8] = {
        out_lat,
        out_lon,
        out_speed_ms,
        out_heading_deg,
        static_cast<double>(anomaly.is_anomaly ? 2 : g_last_event_class),
        static_cast<double>(g_last_sigma),
        anomaly.filtered_bump,
        static_cast<double>(anomaly.type) // 0=none, 1=pothole, 2=speed breaker
    };
    env->SetDoubleArrayRegion(result, 0, 8, buf);
    return result;
}

} // extern "C"
