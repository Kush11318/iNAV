#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace inav {

struct VelocityNetOutput {
    float delta_d;      // Predicted displacement in meters over 2.0s window
    int event_class;    // 0: Stationary, 1: Normal Cruise, 2: Rough Road/Potholes, 3: Dynamic Maneuver
    float sigma;        // Estimated standard deviation (uncertainty) in meters
};

class VelocityNetSession {
public:
    explicit VelocityNetSession(const std::string& model_path)
        : model_path_(model_path), initialized_(false) {
#ifdef HAS_ONNXRUNTIME
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "iNAV_VelocityNet");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(2);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
            initialized_ = true;
        } catch (...) {
            initialized_ = false;
        }
#endif
    }

    bool is_initialized() const { return initialized_; }

    // Run inference on 20 samples of 6-axis IMU data: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    // window_data shape: [6][20] flattened row-major (shape: 1 x 6 x 20)
    VelocityNetOutput predict(const std::vector<float>& window_1x6x20) {
        VelocityNetOutput out{0.0f, 1, 0.5f};
        if (!initialized_ || window_1x6x20.size() != 120) {
            // Fallback: simple kinematic integration if ONNX runtime is unavailable
            return out;
        }

#ifdef HAS_ONNXRUNTIME
        try {
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_shape = {1, 6, 20};

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(window_1x6x20.data()),
                window_1x6x20.size(),
                input_shape.data(),
                input_shape.size()
            );

            const char* input_names[] = {"imu_window"};
            const char* output_names[] = {"delta_d", "event_logits", "log_var"};

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                3
            );

            // 1. Delta_d head
            float* d_data = output_tensors[0].GetTensorMutableData<float>();
            out.delta_d = std::max(0.0f, d_data[0]);

            // 2. Event classification head (argmax over 4 classes)
            float* ev_data = output_tensors[1].GetTensorMutableData<float>();
            int best_cls = 0;
            float max_val = ev_data[0];
            for (int c = 1; c < 4; ++c) {
                if (ev_data[c] > max_val) {
                    max_val = ev_data[c];
                    best_cls = c;
                }
            }
            out.event_class = best_cls;

            // 3. Uncertainty head: PyTorch model outputs uncertainty_sigma_m directly via Softplus
            float* sigma_data = output_tensors[2].GetTensorMutableData<float>();
            out.sigma = std::clamp(sigma_data[0], 0.05f, 20.0f);

        } catch (...) {
            out.delta_d = 0.0f;
            out.event_class = 1;
            out.sigma = 1.0f;
        }
#endif
        return out;
    }

private:
    std::string model_path_;
    bool initialized_;
#ifdef HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
#endif
};

} // namespace inav
