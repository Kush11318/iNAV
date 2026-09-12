// Native Windows C++ Alignment & Calibration Verification Suite
// Compiles and links natively on Windows without MSVC/MinGW runtime installations,
// linking directly against msvcrt.dll and kernel32.dll.

extern "C" {
    typedef void* HANDLE;
    #define STD_OUTPUT_HANDLE ((unsigned int)-11)
    __declspec(dllimport) HANDLE __stdcall GetStdHandle(unsigned int nStdHandle);
    __declspec(dllimport) int __stdcall WriteFile(HANDLE hFile, const void* lpBuffer, unsigned int nNumberOfBytesToWrite, unsigned int* lpNumberOfBytesWritten, void* lpOverlapped);
    __declspec(dllimport) void __stdcall ExitProcess(unsigned int uExitCode);
    __declspec(dllimport) double sin(double);
    __declspec(dllimport) double cos(double);
    __declspec(dllimport) double sqrt(double);
    __declspec(dllimport) double atan2(double, double);
    __declspec(dllimport) double asin(double);
    __declspec(dllimport) double acos(double);
    __declspec(dllimport) int sprintf(char* str, const char* format, ...);
}

void print(const char* s) {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    unsigned int len = 0;
    while (s[len]) len++;
    unsigned int written = 0;
    WriteFile(hOut, s, len, &written, 0);
}

static constexpr double G = 9.80665;
static constexpr double PI = 3.14159265358979323846;

static double dabs(double v) { return v < 0.0 ? -v : v; }
static double dclamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double norm() const { return sqrt(x*x + y*y + z*z); }
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(double s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(double s) const { return Vec3(x/s, y/s, z/s); }
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const {
        return Vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);
    }
};

struct Mat3 {
    double m[3][3]{{1,0,0},{0,1,0},{0,0,1}};
    constexpr Mat3() = default;
    constexpr Mat3(double m00, double m01, double m02,
                   double m10, double m11, double m12,
                   double m20, double m21, double m22) {
        m[0][0]=m00; m[0][1]=m01; m[0][2]=m02;
        m[1][0]=m10; m[1][1]=m11; m[1][2]=m12;
        m[2][0]=m20; m[2][1]=m21; m[2][2]=m22;
    }

    Vec3 multiply(const Vec3& v) const {
        return Vec3(
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        );
    }

    Mat3 transpose() const {
        return Mat3(m[0][0], m[1][0], m[2][0],
                    m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2]);
    }

    Mat3 matmul(const Mat3& b) const {
        Mat3 r;
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                r.m[i][j] = m[i][0]*b.m[0][j] + m[i][1]*b.m[1][j] + m[i][2]*b.m[2][j];
            }
        }
        return r;
    }

    double det() const {
        return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
             - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
             + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    }
};

enum class AlignmentState : unsigned char {
    UNINITIALIZED = 0,
    STATIC_LEVELING = 1,
    STATIC_ALIGNED = 2,
    DYNAMIC_ALIGNING = 3,
    FULL_ALIGNED = 4,
    REMOUNT_DETECTED = 5
};

struct AlignmentConfig {
    double gravity_norm{9.80665};
    double stationary_accel_std{0.10};
    double stationary_gyro_std{0.02};
    double stationary_speed_max{0.20};
    unsigned int min_static_samples{20};
    unsigned int pca_min_samples{30};
    double pca_min_accel_std{0.40};
    double pca_min_eigen_ratio{4.0};
    double remount_angle_thresh_deg{5.0};
    unsigned int remount_debounce_count{5};
};

struct AlignmentResult {
    Mat3 R_b_to_v;
    Vec3 gyro_bias_body{0.0, 0.0, 0.0};
    Vec3 accel_bias_body{0.0, 0.0, 0.0};
    double pitch_deg{0.0};
    double roll_deg{0.0};
    double yaw_deg{0.0};
    double confidence{0.0};
    AlignmentState state{AlignmentState::UNINITIALIZED};
    bool remount_flag{false};
};

struct ImuSample {
    long long timestamp_ns{0};
    Vec3 accel_mps2;
    Vec3 gyro_radps;
    int frame{1}; // 1 = PHONE_BODY, 2 = VEHICLE_FRD
    ImuSample(long long t, const Vec3& a, const Vec3& g, int f)
        : timestamp_ns(t), accel_mps2(a), gyro_radps(g), frame(f) {}
};

class AlignmentEngine {
public:
    explicit AlignmentEngine(AlignmentConfig cfg = AlignmentConfig()) : config_(cfg) {}

    bool calibrate_static_buffer(const Vec3* acc, const Vec3* gyro, unsigned int count) {
        if (count < config_.min_static_samples) return false;
        Vec3 mean_a(0,0,0), mean_w(0,0,0);
        for (unsigned int i = 0; i < count; ++i) {
            mean_a = mean_a + acc[i];
            mean_w = mean_w + gyro[i];
        }
        mean_a = mean_a / count;
        mean_w = mean_w / count;

        double var_a = 0.0, var_w = 0.0;
        for (unsigned int i = 0; i < count; ++i) {
            Vec3 da = acc[i] - mean_a;
            Vec3 dw = gyro[i] - mean_w;
            var_a += da.dot(da);
            var_w += dw.dot(dw);
        }
        double std_a = sqrt(var_a / (3.0 * count));
        double std_w = sqrt(var_w / (3.0 * count));

        if (std_a > config_.stationary_accel_std || std_w > config_.stationary_gyro_std) {
            return false;
        }

        double a_norm = mean_a.norm();
        if (a_norm < 1.0) return false;
        Vec3 u_z = (mean_a * -1.0) / a_norm;

        double pitch = asin(-dclamp(u_z.x, -1.0, 1.0));
        double roll = atan2(u_z.y, -u_z.z);

        Vec3 u_x(cos(pitch), 0.0, -sin(pitch));
        Vec3 u_y = u_z.cross(u_x);
        double norm_y = u_y.norm();
        if (norm_y < 1e-6) return false;
        u_y = u_y / norm_y;
        u_x = u_y.cross(u_z);
        double norm_x = u_x.norm();
        if (norm_x < 1e-6) return false;
        u_x = u_x / norm_x;

        R_static_level_ = Mat3(
            u_x.x, u_x.y, u_x.z,
            u_y.x, u_y.y, u_y.z,
            u_z.x, u_z.y, u_z.z
        );

        result_.R_b_to_v = R_static_level_;
        result_.gyro_bias_body = mean_w;
        result_.accel_bias_body = Vec3(0,0,0);
        result_.pitch_deg = pitch * (180.0 / PI);
        result_.roll_deg = roll * (180.0 / PI);
        result_.yaw_deg = 0.0;
        result_.confidence = 0.50;
        result_.state = AlignmentState::STATIC_ALIGNED;
        result_.remount_flag = false;
        u_z_nominal_ = u_z;
        remount_counter_ = 0;
        return true;
    }

    bool calibrate_dynamic_buffer(const Vec3* acc, const Vec3* gyro, const double* spd_deltas, unsigned int count) {
        if (result_.state != AlignmentState::STATIC_ALIGNED &&
            result_.state != AlignmentState::DYNAMIC_ALIGNING &&
            result_.state != AlignmentState::FULL_ALIGNED) {
            return false;
        }
        if (count < config_.pca_min_samples) return false;

        double mean_ax = 0.0, mean_ay = 0.0;
        for (unsigned int i = 0; i < count; ++i) {
            Vec3 a_veh = R_static_level_.multiply(acc[i]);
            mean_ax += a_veh.x;
            mean_ay += a_veh.y;
        }
        mean_ax /= count;
        mean_ay /= count;

        double c_xx = 0.0, c_xy = 0.0, c_yy = 0.0;
        for (unsigned int i = 0; i < count; ++i) {
            Vec3 a_veh = R_static_level_.multiply(acc[i]);
            double dx = a_veh.x - mean_ax;
            double dy = a_veh.y - mean_ay;
            c_xx += dx * dx;
            c_xy += dx * dy;
            c_yy += dy * dy;
        }
        c_xx /= (count - 1);
        c_xy /= (count - 1);
        c_yy /= (count - 1);

        double trace = c_xx + c_yy;
        double diff = c_xx - c_yy;
        double disc = sqrt(diff * diff + 4.0 * c_xy * c_xy);
        double lambda1 = 0.5 * (trace + disc);
        double lambda2 = 0.5 * (trace - disc);

        if (sqrt(lambda1) < config_.pca_min_accel_std) return false;
        if (lambda2 > 1e-9 && (lambda1 / lambda2) < config_.pca_min_eigen_ratio) return false;

        double theta = 0.5 * atan2(2.0 * c_xy, diff);
        Vec3 e1(cos(theta), sin(theta), 0.0);

        double forward_evidence = 0.0;
        for (unsigned int i = 0; i < count; ++i) {
            Vec3 a_veh = R_static_level_.multiply(acc[i]);
            double proj = a_veh.x * e1.x + a_veh.y * e1.y;
            forward_evidence += proj * spd_deltas[i];
        }

        if (forward_evidence < 0.0) {
            theta += PI;
            if (theta > PI) theta -= 2.0 * PI;
        }

        double c_psi = cos(theta);
        double s_psi = sin(theta);
        Mat3 R_yaw(
            c_psi,  s_psi, 0.0,
           -s_psi,  c_psi, 0.0,
            0.0,    0.0,   1.0
        );

        result_.R_b_to_v = R_yaw.matmul(R_static_level_);
        result_.yaw_deg = theta * (180.0 / PI);
        result_.confidence = 0.95;
        result_.state = AlignmentState::FULL_ALIGNED;
        return true;
    }

    ImuSample transform_imu(const ImuSample& raw) const {
        Vec3 gyro_corr = raw.gyro_radps - result_.gyro_bias_body;
        Vec3 gyro_veh = result_.R_b_to_v.multiply(gyro_corr);
        Vec3 accel_veh = result_.R_b_to_v.multiply(raw.accel_mps2);
        return ImuSample(raw.timestamp_ns, accel_veh, gyro_veh, 2);
    }

    Vec3 get_linear_accel(const ImuSample& veh) const {
        return Vec3(veh.accel_mps2.x, veh.accel_mps2.y, veh.accel_mps2.z - (-config_.gravity_norm));
    }

    bool check_remount(const Vec3& a_stat) {
        if (result_.state != AlignmentState::STATIC_ALIGNED &&
            result_.state != AlignmentState::FULL_ALIGNED) {
            return false;
        }
        double norm = a_stat.norm();
        if (norm < 1.0) return false;
        Vec3 u_z_current = (a_stat * -1.0) / norm;
        double cos_ang = dclamp(u_z_current.dot(u_z_nominal_), -1.0, 1.0);
        double ang_deg = acos(cos_ang) * (180.0 / PI);

        if (ang_deg > config_.remount_angle_thresh_deg) {
            remount_counter_++;
            if (remount_counter_ >= config_.remount_debounce_count) {
                result_.state = AlignmentState::REMOUNT_DETECTED;
                result_.remount_flag = true;
                result_.confidence = 0.0;
                return true;
            }
        } else {
            if (remount_counter_ > 0) remount_counter_--;
        }
        return false;
    }

    const AlignmentResult& get_result() const { return result_; }

private:
    AlignmentConfig config_;
    AlignmentResult result_;
    Mat3 R_static_level_{Mat3()};
    Vec3 u_z_nominal_{0,0,0};
    unsigned int remount_counter_{0};
};

#define ASSERT_TRUE(expr, msg) \
    if (!(expr)) { \
        print("[FAIL] Assertion failed: " msg "\n"); \
        ExitProcess(1); \
    }

extern "C" int mainCRTStartup(void) {
    print("=================================================================\n");
    print("iNAV Phase 3 - Native Windows C++ Alignment Verification Suite\n");
    print("=================================================================\n");

    // 1. Flat Phone
    {
        AlignmentEngine engine;
        Vec3 acc[30], gyro[30];
        for (int i=0; i<30; ++i) { acc[i] = Vec3(0, 0, G); gyro[i] = Vec3(0,0,0); }
        ASSERT_TRUE(engine.calibrate_static_buffer(acc, gyro, 30), "flat static calibration");
        ASSERT_TRUE(engine.get_result().state == AlignmentState::STATIC_ALIGNED, "state == STATIC_ALIGNED");
        ASSERT_TRUE(dabs(engine.get_result().pitch_deg) < 1e-3, "pitch == 0");
        ASSERT_TRUE(dabs(engine.get_result().roll_deg) < 1e-3, "roll == 0");
        print("[PASS] C++ Test 1: Flat phone leveling (pitch=0, roll=0)\n");
    }

    // 2 & 3. Pitch recovery (+/- 30 deg)
    {
        AlignmentEngine e_pos, e_neg;
        double p = 30.0 * (PI / 180.0);
        Vec3 acc_p[30], acc_n[30], gyro[30];
        for (int i=0; i<30; ++i) {
            acc_p[i] = Vec3(G * sin(p), 0.0, G * cos(p));
            acc_n[i] = Vec3(-G * sin(p), 0.0, G * cos(p));
            gyro[i] = Vec3(0,0,0);
        }
        ASSERT_TRUE(e_pos.calibrate_static_buffer(acc_p, gyro, 30), "pitch +30 calib");
        ASSERT_TRUE(dabs(e_pos.get_result().pitch_deg - 30.0) < 0.1, "pitch == +30");
        ASSERT_TRUE(e_neg.calibrate_static_buffer(acc_n, gyro, 30), "pitch -30 calib");
        ASSERT_TRUE(dabs(e_neg.get_result().pitch_deg - (-30.0)) < 0.1, "pitch == -30");
        print("[PASS] C++ Tests 2 & 3: Pitch recovery (+30 deg and -30 deg)\n");
    }

    // 4 & 5. Roll recovery (+/- 30 deg)
    {
        AlignmentEngine e_pos, e_neg;
        double r = 30.0 * (PI / 180.0);
        Vec3 acc_p[30], acc_n[30], gyro[30];
        for (int i=0; i<30; ++i) {
            acc_p[i] = Vec3(0.0, -G * sin(r), G * cos(r));
            acc_n[i] = Vec3(0.0, G * sin(r), G * cos(r));
            gyro[i] = Vec3(0,0,0);
        }
        ASSERT_TRUE(e_pos.calibrate_static_buffer(acc_p, gyro, 30), "roll +30 calib");
        ASSERT_TRUE(dabs(e_pos.get_result().roll_deg - 30.0) < 0.1, "roll == +30");
        ASSERT_TRUE(e_neg.calibrate_static_buffer(acc_n, gyro, 30), "roll -30 calib");
        ASSERT_TRUE(dabs(e_neg.get_result().roll_deg - (-30.0)) < 0.1, "roll == -30");
        print("[PASS] C++ Tests 4 & 5: Roll recovery (+30 deg and -30 deg)\n");
    }

    // 6. Gyro bias injection & zero corrected gyro
    {
        AlignmentEngine engine;
        Vec3 bias(0.01, -0.02, 0.03);
        Vec3 acc[30], gyro[30];
        for (int i=0; i<30; ++i) { acc[i] = Vec3(0,0,G); gyro[i] = bias; }
        ASSERT_TRUE(engine.calibrate_static_buffer(acc, gyro, 30), "bias calib");
        ASSERT_TRUE(dabs(engine.get_result().gyro_bias_body.x - 0.01) < 1e-4, "bias x");
        ASSERT_TRUE(dabs(engine.get_result().gyro_bias_body.y - (-0.02)) < 1e-4, "bias y");
        ASSERT_TRUE(dabs(engine.get_result().gyro_bias_body.z - 0.03) < 1e-4, "bias z");

        ImuSample raw(1000000000LL, Vec3(0,0,G), bias, 1);
        ImuSample veh = engine.transform_imu(raw);
        ASSERT_TRUE(veh.frame == 2, "frame is VEHICLE_FRD");
        ASSERT_TRUE(dabs(veh.gyro_radps.x) < 1e-5, "veh gyro x == 0");
        ASSERT_TRUE(dabs(veh.gyro_radps.y) < 1e-5, "veh gyro y == 0");
        ASSERT_TRUE(dabs(veh.gyro_radps.z) < 1e-5, "veh gyro z == 0");
        print("[PASS] C++ Tests 7 & 8: Gyro bias injection and zero corrected gyro\n");
    }

    // 7. Gravity cancellation
    {
        AlignmentEngine engine;
        Vec3 acc[30], gyro[30];
        for (int i=0; i<30; ++i) { acc[i] = Vec3(0,0,G); gyro[i] = Vec3(0,0,0); }
        engine.calibrate_static_buffer(acc, gyro, 30);

        ImuSample rest(1000000000LL, Vec3(0,0,G), Vec3(0,0,0), 1);
        ImuSample veh = engine.transform_imu(rest);
        Vec3 lin = engine.get_linear_accel(veh);
        ASSERT_TRUE(dabs(lin.x) < 1e-4, "lin ax == 0");
        ASSERT_TRUE(dabs(lin.y) < 1e-4, "lin ay == 0");
        ASSERT_TRUE(dabs(lin.z) < 1e-4, "lin az == 0");
        print("[PASS] C++ Test 11: Gravity detrending cancellation ([0,0,0] m/s^2)\n");
    }

    // 8. Clockwise & Counter-Clockwise Yaw Sign Parity (CRITICAL)
    {
        AlignmentEngine engine;
        Vec3 acc_s[30], gyro_s[30];
        for (int i=0; i<30; ++i) { acc_s[i] = Vec3(0,0,G); gyro_s[i] = Vec3(0,0,0); }
        engine.calibrate_static_buffer(acc_s, gyro_s, 30);

        // Portrait driving forward along phone +Y
        Vec3 acc_d[40], gyro_d[40];
        double spd_d[40];
        for (int i=0; i<40; ++i) {
            acc_d[i] = Vec3(0.0, 1.5 + 1.0 * sin(i), G);
            gyro_d[i] = Vec3(0,0,0);
            spd_d[i] = 0.15;
        }
        ASSERT_TRUE(engine.calibrate_dynamic_buffer(acc_d, gyro_d, spd_d, 40), "pca calib");

        // Right turn: phone gyro reports negative around +Z_phone
        ImuSample right_turn(2000000000LL, Vec3(0,0,G), Vec3(0,0,-0.05), 1);
        ImuSample veh_right = engine.transform_imu(right_turn);
        // VEHICLE_FRD yaw rate MUST be positive
        ASSERT_TRUE(veh_right.gyro_radps.z > 0.0, "veh right turn omega_z > 0");
        ASSERT_TRUE(dabs(veh_right.gyro_radps.z - 0.05) < 1e-3, "veh right turn omega_z == +0.05");

        // Left turn: phone gyro reports positive around +Z_phone
        ImuSample left_turn(2000000000LL, Vec3(0,0,G), Vec3(0,0,0.05), 1);
        ImuSample veh_left = engine.transform_imu(left_turn);
        // VEHICLE_FRD yaw rate MUST be negative
        ASSERT_TRUE(veh_left.gyro_radps.z < 0.0, "veh left turn omega_z < 0");
        ASSERT_TRUE(dabs(veh_left.gyro_radps.z - (-0.05)) < 1e-3, "veh left turn omega_z == -0.05");
        print("[PASS] C++ Tests 12 & 13: Yaw sign parity (CW turn -> +omega_z, CCW turn -> -omega_z)\n");
    }

    // 9. Stationarity gate and moving rejection
    {
        AlignmentEngine engine;
        Vec3 acc_m[30], gyro_s[30];
        for (int i=0; i<30; ++i) {
            acc_m[i] = Vec3(0, 0, G + 0.5 * sin(i));
            gyro_s[i] = Vec3(0,0,0);
        }
        ASSERT_TRUE(!engine.calibrate_static_buffer(acc_m, gyro_s, 30), "reject moving calib");
        print("[PASS] C++ Tests 18 & 19: Stationarity gate & moving calibration rejection\n");
    }

    // 10. SO(3) Orthogonality & Determinant Check
    {
        AlignmentEngine engine;
        Vec3 acc_s[30], gyro_s[30];
        for (int i=0; i<30; ++i) { acc_s[i] = Vec3(0,0,G); gyro_s[i] = Vec3(0,0,0); }
        engine.calibrate_static_buffer(acc_s, gyro_s, 30);
        Mat3 R = engine.get_result().R_b_to_v;
        Mat3 Rt = R.transpose();
        Mat3 I = Rt.matmul(R);

        double max_ortho_err = 0.0;
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                double target = (i == j) ? 1.0 : 0.0;
                double err = dabs(I.m[i][j] - target);
                if (err > max_ortho_err) max_ortho_err = err;
            }
        }
        double det = R.det();
        double det_err = dabs(det - 1.0);
        ASSERT_TRUE(max_ortho_err < 1e-5, "R^T R == I");
        ASSERT_TRUE(det_err < 1e-5, "det(R) == +1.0");
        print("[PASS] C++ SO(3) Check: Orthogonality err < 1e-5, Det err < 1e-5, det(R) == +1.000000\n");
    }

    // 11. Remount detection state machine
    {
        AlignmentEngine engine;
        Vec3 acc_s[30], gyro_s[30];
        for (int i=0; i<30; ++i) { acc_s[i] = Vec3(0,0,G); gyro_s[i] = Vec3(0,0,0); }
        engine.calibrate_static_buffer(acc_s, gyro_s, 30);

        // Transient perturbation (< 5 epochs)
        Vec3 perturbed(0.0, G * sin(10.0 * PI / 180.0), G * cos(10.0 * PI / 180.0));
        bool tripped = false;
        for (int i=0; i<3; ++i) {
            if (engine.check_remount(perturbed)) tripped = true;
        }
        ASSERT_TRUE(!tripped, "transient disturbance does not trip remount");

        // Back to normal debounces counter
        for (int i=0; i<3; ++i) engine.check_remount(Vec3(0,0,G));

        // Persistent disturbance (>= 5 epochs)
        for (int i=0; i<5; ++i) {
            tripped = engine.check_remount(perturbed);
        }
        ASSERT_TRUE(tripped, "persistent disturbance trips remount");
        ASSERT_TRUE(engine.get_result().state == AlignmentState::REMOUNT_DETECTED, "state == REMOUNT_DETECTED");
        ASSERT_TRUE(engine.get_result().confidence == 0.0, "confidence dropped to 0");
        print("[PASS] C++ Remount State Machine: Transient debounced, Persistent disturbance tripped\n");
    }

    print("\n=================================================================\n");
    print("ALL NATIVE WINDOWS C++ ALIGNMENT TESTS PASSED SUCCESSFULLY!\n");
    print("=================================================================\n");
    ExitProcess(0);
}
