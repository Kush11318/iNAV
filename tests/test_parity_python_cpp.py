import ctypes
import math
import numpy as np
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.alignment import AlignmentEngine as PyAlignmentEngine, AlignmentConfig, AlignmentState
from modules.sensor_types import ImuSample as PyImuSample, Vec3 as PyVec3, SensorFrame

G = 9.80665

# Load C++ DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'inav_alignment_c.dll'))
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"DLL not found at {dll_path}")

cpp_lib = ctypes.CDLL(dll_path)

class CAlignmentResult(ctypes.Structure):
    _fields_ = [
        ("R_b_to_v", ctypes.c_double * 9),
        ("gyro_bias", ctypes.c_double * 3),
        ("pitch_deg", ctypes.c_double),
        ("roll_deg", ctypes.c_double),
        ("yaw_deg", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("state", ctypes.c_int)
    ]

cpp_lib.cpp_calibrate_static.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(CAlignmentResult)
]
cpp_lib.cpp_calibrate_static.restype = ctypes.c_int

cpp_lib.cpp_calibrate_dynamic.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(CAlignmentResult)
]
cpp_lib.cpp_calibrate_dynamic.restype = ctypes.c_int

cpp_lib.cpp_transform_imu.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
]
cpp_lib.cpp_transform_imu.restype = None

def run_parity_tests():
    print("=" * 80)
    print("PYTHON <-> C++ NUMERICAL PARITY VERIFICATION (CASES A - J)")
    print("=" * 80)
    
    cases = [
        ("A. Flat orientation", 0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, G], [0.0, 0.0, 0.0]),
        ("B. +30 deg pitch", 30.0, 0.0, 0.0, [0.0, 0.0, 0.0], [G * math.sin(math.radians(30)), 0.0, G * math.cos(math.radians(30))], [0.0, 0.0, 0.0]),
        ("C. -30 deg pitch", -30.0, 0.0, 0.0, [0.0, 0.0, 0.0], [-G * math.sin(math.radians(30)), 0.0, G * math.cos(math.radians(30))], [0.0, 0.0, 0.0]),
        ("D. +30 deg roll", 0.0, 30.0, 0.0, [0.0, 0.0, 0.0], [0.0, -G * math.sin(math.radians(30)), G * math.cos(math.radians(30))], [0.0, 0.0, 0.0]),
        ("E. -30 deg roll", 0.0, -30.0, 0.0, [0.0, 0.0, 0.0], [0.0, G * math.sin(math.radians(30)), G * math.cos(math.radians(30))], [0.0, 0.0, 0.0]),
        ("F. Known yaw mounting offset (+30 deg)", 0.0, 0.0, 30.0, [0.0, 0.0, 0.0], [0.0, 0.0, G], [0.0, 0.0, 0.0]),
        ("G. Gyro bias injection", 0.0, 0.0, 0.0, [0.015, -0.025, 0.035], [0.0, 0.0, G], [0.015, -0.025, 0.035]),
        ("H. Gravity cancellation", 0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, G], [0.0, 0.0, 0.0]),
        ("I. Positive vehicle yaw rate (+0.05 rad/s CW)", 0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, G], [0.0, 0.0, -0.05]),
        ("J. Negative vehicle yaw rate (-0.05 rad/s CCW)", 0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, G], [0.0, 0.0, 0.05]),
    ]

    all_max_diffs = []
    
    header = f"{'Case':<42} | {'Max |dR|':<10} | {'|dpitch|':<9} | {'|droll|':<8} | {'|dyaw|':<8} | {'|dbias|':<8} | {'|dveh_w|':<8} | {'|dveh_a|':<8} | {'|dlin_a|':<8}"
    print(header)
    print("-" * len(header))

    for case_name, p_nom, r_nom, y_nom, bias, sample_a, sample_w in cases:
        # Generate static buffer
        count = 30
        acc_buf = [sample_a] * count
        gyro_buf = [bias] * count
        
        # Python Alignment
        py_engine = PyAlignmentEngine()
        py_acc_vecs = [PyVec3(a[0], a[1], a[2]) for a in acc_buf]
        py_gyro_vecs = [PyVec3(w[0], w[1], w[2]) for w in gyro_buf]
        py_ok = py_engine.calibrate_static_buffer(py_acc_vecs, py_gyro_vecs)
        assert py_ok
        
        # C++ Alignment
        c_res = CAlignmentResult()
        c_acc_arr = (ctypes.c_double * (count * 3))(*[v for vec in acc_buf for v in vec])
        c_gyro_arr = (ctypes.c_double * (count * 3))(*[v for vec in gyro_buf for v in vec])
        c_ok = cpp_lib.cpp_calibrate_static(c_acc_arr, c_gyro_arr, count, ctypes.byref(c_res))
        assert c_ok == 1

        # Dynamic Alignment for Case F
        if y_nom != 0.0:
            dyn_count = 40
            # Rotate excitation by known yaw angle
            psi = math.radians(y_nom)
            dyn_acc = []
            dyn_spd = [0.15] * dyn_count
            for i in range(dyn_count):
                a_fwd = 1.5 + 0.8 * math.sin(i)
                # phone x, y
                ax = -a_fwd * math.sin(psi)
                ay = a_fwd * math.cos(psi)
                dyn_acc.append([ax, ay, G])
            
            # Python dynamic
            py_dyn_vecs = [PyVec3(a[0], a[1], a[2]) for a in dyn_acc]
            py_dyn_gyro = [PyVec3(0, 0, 0)] * dyn_count
            py_engine.calibrate_dynamic_buffer(py_dyn_vecs, py_dyn_gyro, dyn_spd)

            # C++ dynamic
            c_dyn_acc_arr = (ctypes.c_double * (dyn_count * 3))(*[v for vec in dyn_acc for v in vec])
            c_dyn_gyro_arr = (ctypes.c_double * (dyn_count * 3))(*[0.0] * (dyn_count * 3))
            c_dyn_spd_arr = (ctypes.c_double * dyn_count)(*dyn_spd)
            cpp_lib.cpp_calibrate_dynamic(c_res.R_b_to_v, c_dyn_acc_arr, c_dyn_gyro_arr, c_dyn_spd_arr, dyn_count, ctypes.byref(c_res))

        # Compare Static / Full Results
        py_res = py_engine.result
        py_R = py_res.R_b_to_v
        cpp_R = np.array(c_res.R_b_to_v).reshape(3, 3)

        diff_R = np.max(np.abs(py_R - cpp_R))
        diff_pitch = abs(py_res.pitch_deg - c_res.pitch_deg)
        diff_roll = abs(py_res.roll_deg - c_res.roll_deg)
        diff_yaw = abs(py_res.yaw_deg - c_res.yaw_deg)
        diff_bias = np.max(np.abs(np.array([py_res.gyro_bias_body.x, py_res.gyro_bias_body.y, py_res.gyro_bias_body.z]) - np.array(c_res.gyro_bias)))

        # Transform Sample in Python
        raw_py = PyImuSample(1000000000, PyVec3(*sample_a), PyVec3(*sample_w), SensorFrame.PHONE_BODY)
        veh_py = py_engine.transform_imu(raw_py)
        lin_py = py_engine.get_linear_accel(veh_py)

        # Transform Sample in C++
        c_raw_a = (ctypes.c_double * 3)(*sample_a)
        c_raw_w = (ctypes.c_double * 3)(*sample_w)
        c_veh_a = (ctypes.c_double * 3)()
        c_veh_w = (ctypes.c_double * 3)()
        c_lin_a = (ctypes.c_double * 3)()
        cpp_lib.cpp_transform_imu(
            c_res.R_b_to_v,
            c_res.gyro_bias,
            c_raw_a,
            c_raw_w,
            c_veh_a,
            c_veh_w,
            c_lin_a
        )

        diff_veh_a = np.max(np.abs(np.array([veh_py.accel_mps2.x, veh_py.accel_mps2.y, veh_py.accel_mps2.z]) - np.array(c_veh_a)))
        diff_veh_w = np.max(np.abs(np.array([veh_py.gyro_radps.x, veh_py.gyro_radps.y, veh_py.gyro_radps.z]) - np.array(c_veh_w)))
        diff_lin_a = np.max(np.abs(np.array([lin_py.x, lin_py.y, lin_py.z]) - np.array(c_lin_a)))

        max_case_diff = max(diff_R, diff_pitch, diff_roll, diff_yaw, diff_bias, diff_veh_a, diff_veh_w, diff_lin_a)
        all_max_diffs.append(max_case_diff)

        print(f"{case_name:<42} | {diff_R:<10.2e} | {diff_pitch:<9.2e} | {diff_roll:<8.2e} | {diff_yaw:<8.2e} | {diff_bias:<8.2e} | {diff_veh_w:<8.2e} | {diff_veh_a:<8.2e} | {diff_lin_a:<8.2e}")

        # Assert parity within 1e-5 tolerance
        assert max_case_diff < 1e-5, f"Parity failure in {case_name}: max diff {max_case_diff}"

    overall_max = max(all_max_diffs)
    print("-" * len(header))
    print(f"[OVERALL PARITY] Maximum absolute difference across all cases: {overall_max:.2e} (Tolerance: 1.00e-05)")
    print(f"[OVERALL PARITY] RESULT: PASS (All {len(cases)} cases mathematically identical)")
    print("=" * 80)
    return True

if __name__ == "__main__":
    run_parity_tests()
