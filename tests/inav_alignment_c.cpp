// C interface DLL for Python-C++ Numerical Parity Verification
extern "C" {
    __declspec(dllexport) int DllMainCRTStartup(void* hinstDLL, unsigned int fdwReason, void* lpReserved) {
        return 1;
    }
    __declspec(dllimport) double sin(double);
    __declspec(dllimport) double cos(double);
    __declspec(dllimport) double sqrt(double);
    __declspec(dllimport) double atan2(double, double);
    __declspec(dllimport) double asin(double);
    __declspec(dllimport) double acos(double);
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
};

extern "C" {

struct CAlignmentResult {
    double R_b_to_v[9]; // row-major 3x3
    double gyro_bias[3];
    double pitch_deg;
    double roll_deg;
    double yaw_deg;
    double confidence;
    int state;
};

__declspec(dllexport) int cpp_calibrate_static(
    const double* acc_xyz,
    const double* gyro_xyz,
    int count,
    CAlignmentResult* out_res
) {
    if (count < 20) return 0;
    Vec3 mean_a(0,0,0), mean_w(0,0,0);
    for (int i = 0; i < count; ++i) {
        mean_a = mean_a + Vec3(acc_xyz[3*i], acc_xyz[3*i+1], acc_xyz[3*i+2]);
        mean_w = mean_w + Vec3(gyro_xyz[3*i], gyro_xyz[3*i+1], gyro_xyz[3*i+2]);
    }
    mean_a = mean_a / count;
    mean_w = mean_w / count;

    double var_a = 0.0, var_w = 0.0;
    for (int i = 0; i < count; ++i) {
        Vec3 da = Vec3(acc_xyz[3*i], acc_xyz[3*i+1], acc_xyz[3*i+2]) - mean_a;
        Vec3 dw = Vec3(gyro_xyz[3*i], gyro_xyz[3*i+1], gyro_xyz[3*i+2]) - mean_w;
        var_a += da.dot(da);
        var_w += dw.dot(dw);
    }
    double std_a = sqrt(var_a / (3.0 * count));
    double std_w = sqrt(var_w / (3.0 * count));

    if (std_a > 0.10 || std_w > 0.02) return 0;

    double a_norm = mean_a.norm();
    if (a_norm < 1.0) return 0;
    Vec3 u_z = (mean_a * -1.0) / a_norm;

    double pitch = asin(-dclamp(u_z.x, -1.0, 1.0));
    double roll = atan2(u_z.y, -u_z.z);

    Vec3 u_x(cos(pitch), 0.0, -sin(pitch));
    Vec3 u_y = u_z.cross(u_x);
    double norm_y = u_y.norm();
    if (norm_y < 1e-6) return 0;
    u_y = u_y / norm_y;
    u_x = u_y.cross(u_z);
    double norm_x = u_x.norm();
    if (norm_x < 1e-6) return 0;
    u_x = u_x / norm_x;

    Mat3 R(
        u_x.x, u_x.y, u_x.z,
        u_y.x, u_y.y, u_y.z,
        u_z.x, u_z.y, u_z.z
    );

    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            out_res->R_b_to_v[i*3 + j] = R.m[i][j];

    out_res->gyro_bias[0] = mean_w.x;
    out_res->gyro_bias[1] = mean_w.y;
    out_res->gyro_bias[2] = mean_w.z;
    out_res->pitch_deg = pitch * (180.0 / PI);
    out_res->roll_deg = roll * (180.0 / PI);
    out_res->yaw_deg = 0.0;
    out_res->confidence = 0.50;
    out_res->state = 2; // STATIC_ALIGNED
    return 1;
}

__declspec(dllexport) int cpp_calibrate_dynamic(
    const double* R_static_9,
    const double* acc_xyz,
    const double* gyro_xyz,
    const double* spd_deltas,
    int count,
    CAlignmentResult* out_res
) {
    if (count < 30) return 0;
    Mat3 R_stat(
        R_static_9[0], R_static_9[1], R_static_9[2],
        R_static_9[3], R_static_9[4], R_static_9[5],
        R_static_9[6], R_static_9[7], R_static_9[8]
    );

    double mean_ax = 0.0, mean_ay = 0.0;
    for (int i = 0; i < count; ++i) {
        Vec3 a_veh = R_stat.multiply(Vec3(acc_xyz[3*i], acc_xyz[3*i+1], acc_xyz[3*i+2]));
        mean_ax += a_veh.x;
        mean_ay += a_veh.y;
    }
    mean_ax /= count;
    mean_ay /= count;

    double c_xx = 0.0, c_xy = 0.0, c_yy = 0.0;
    for (int i = 0; i < count; ++i) {
        Vec3 a_veh = R_stat.multiply(Vec3(acc_xyz[3*i], acc_xyz[3*i+1], acc_xyz[3*i+2]));
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

    if (sqrt(lambda1) < 0.40) return 0;
    if (lambda2 > 1e-9 && (lambda1 / lambda2) < 4.0) return 0;

    double theta = 0.5 * atan2(2.0 * c_xy, diff);
    Vec3 e1(cos(theta), sin(theta), 0.0);

    double forward_evidence = 0.0;
    for (int i = 0; i < count; ++i) {
        Vec3 a_veh = R_stat.multiply(Vec3(acc_xyz[3*i], acc_xyz[3*i+1], acc_xyz[3*i+2]));
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

    Mat3 R_full = R_yaw.matmul(R_stat);

    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            out_res->R_b_to_v[i*3 + j] = R_full.m[i][j];

    out_res->yaw_deg = atan2(out_res->R_b_to_v[1], out_res->R_b_to_v[0]) * (180.0 / PI);
    out_res->confidence = 0.95;
    out_res->state = 4; // FULL_ALIGNED
    return 1;
}

__declspec(dllexport) void cpp_transform_imu(
    const double* R_9,
    const double* gyro_bias_3,
    const double* raw_acc_3,
    const double* raw_gyro_3,
    double* out_veh_acc_3,
    double* out_veh_gyro_3,
    double* out_lin_acc_3
) {
    Mat3 R(
        R_9[0], R_9[1], R_9[2],
        R_9[3], R_9[4], R_9[5],
        R_9[6], R_9[7], R_9[8]
    );
    Vec3 raw_a(raw_acc_3[0], raw_acc_3[1], raw_acc_3[2]);
    Vec3 raw_w(raw_gyro_3[0], raw_gyro_3[1], raw_gyro_3[2]);
    Vec3 bias(gyro_bias_3[0], gyro_bias_3[1], gyro_bias_3[2]);

    Vec3 w_corr = raw_w - bias;
    Vec3 w_veh = R.multiply(w_corr);
    Vec3 a_veh = R.multiply(raw_a);

    out_veh_acc_3[0] = a_veh.x;
    out_veh_acc_3[1] = a_veh.y;
    out_veh_acc_3[2] = a_veh.z;

    out_veh_gyro_3[0] = w_veh.x;
    out_veh_gyro_3[1] = w_veh.y;
    out_veh_gyro_3[2] = w_veh.z;

    // Linear accel: veh.accel - [0, 0, -G]
    out_lin_acc_3[0] = a_veh.x;
    out_lin_acc_3[1] = a_veh.y;
    out_lin_acc_3[2] = a_veh.z - (-G);
}

} // extern "C"
