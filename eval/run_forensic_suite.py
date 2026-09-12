"""
Forensic Evaluation Harness for Root Cause Verification (Experiments 1 - 7)
Does NOT modify any production code.
Parametrically evaluates all proposed MINS/estimator changes on the exact 180s motorway test set.
"""

import sys
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from eval.outage_sim import generate_outage_schedule
from eval.score import score_outage_segment
from eval.replay import (
    estimate_scale_factor_rls,
    estimate_pre_outage_gyro_bias,
    local_xy_to_latlon,
)
from modules.alignment import AlignmentEngine
from modules.velocity_net import VelocityNetPredictor
from modules.esekf import (
    ESEKFNavigationFilter,
    quat_to_rot_matrix,
    quat_multiply,
    skew_symmetric,
)


def run_parametric_esekf(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    predictor: VelocityNetPredictor,
    alignment: AlignmentEngine,
    k_scale: float = 1.0,
    init_gyro_bias_z: float = 0.0,
    deadband_rads: float = 0.0,
    apply_nhc: bool = False,
    coupled_jacobian: bool = False,
    use_fej: bool = False,
    use_road_heading: bool = False,
    dt: float = config.TARGET_DT,
    log_diagnostic: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Parametric ES-EKF replay for controlled ablation experiments.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), None

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
    gt_hdg = df_outage[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_outage.columns else np.zeros(n)
    gt_lat = df_outage[config.COL_TRUE_LAT].values if config.COL_TRUE_LAT in df_outage.columns else np.zeros(n)
    gt_lon = df_outage[config.COL_TRUE_LON].values if config.COL_TRUE_LON in df_outage.columns else np.zeros(n)
    time_s = df_outage[config.COL_TIME].values if config.COL_TIME in df_outage.columns else np.arange(n) * dt

    # Initialize 15-state ES-EKF filter
    esekf = ESEKFNavigationFilter(dt=dt)
    esekf.initialize(
        init_lat_ned=0.0,
        init_lon_ned=0.0,
        init_speed_ms=init_speed_ms,
        init_heading_rad=np.radians(init_heading_deg)
    )
    
    # Inject initial gyro bias
    esekf.w_b[2] = init_gyro_bias_z
    esekf.freeze_gyro_bias = True

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)
    est_hdg_deg = np.zeros(n)

    # FEJ variables
    R_fej = quat_to_rot_matrix(esekf.q)
    v_fej = esekf.v.copy()
    p_fej = esekf.p.copy()

    diag_rows = []
    window_buffer = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]

        # 1. Transform body to vehicle frame
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # Apply deadband if configured
        fwd_corr = a_v[0] - esekf.a_b[0]
        yaw_corr = w_v[2] - esekf.w_b[2]
        
        yaw_rate_for_integ = yaw_corr
        if deadband_rads > 0.0 and abs(yaw_rate_for_integ) < deadband_rads:
            yaw_rate_for_integ = 0.0

        # Current heading from quaternion
        R = quat_to_rot_matrix(esekf.q)
        psi = np.arctan2(R[1, 0], R[0, 0])

        # Propagate heading
        mid_psi = psi + 0.5 * yaw_rate_for_integ * dt
        new_psi = (psi + yaw_rate_for_integ * dt) % (2.0 * np.pi)

        # Update quaternion with pure yaw rotation
        cy = np.cos(new_psi * 0.5)
        sy = np.sin(new_psi * 0.5)
        esekf.q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)

        # Propagate forward speed (clamped non-negative)
        v_curr = float(np.linalg.norm(esekf.v[:2]))
        v_next = max(v_curr + fwd_corr * dt, 0.0)
        v_mid = 0.5 * (v_curr + v_next)

        # Update nominal velocity and position in horizontal plane
        esekf.v[0] = v_next * np.cos(new_psi)
        esekf.v[1] = v_next * np.sin(new_psi)
        esekf.v[2] = 0.0  # zero vertical velocity

        esekf.p[0] += v_mid * np.cos(mid_psi) * dt
        esekf.p[1] += v_mid * np.sin(mid_psi) * dt
        esekf.p[2] = 0.0

        # Error state covariance transition
        R_mat = R_fej if use_fej else quat_to_rot_matrix(esekf.q)
        f_b_vec = np.array([fwd_corr, 0.0, 0.0])
        w_b_vec = np.array([0.0, 0.0, yaw_rate_for_integ])

        F_x = np.eye(esekf.dim_error, dtype=np.float64)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R_mat @ skew_symmetric(f_b_vec) * dt
        F_x[3:6, 9:12] = -R_mat * dt
        F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(w_b_vec) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        Q_d = esekf.compute_discrete_Q(dt)
        esekf.P = F_x @ esekf.P @ F_x.T + Q_d
        esekf.P = 0.5 * (esekf.P + esekf.P.T)

        # Optional NHC update
        if apply_nhc:
            R_curr = quat_to_rot_matrix(esekf.q)
            v_b = R_curr.T @ esekf.v
            y_nhc = -v_b[1:3]
            H_nhc = np.zeros((2, esekf.dim_error), dtype=np.float64)
            H_nhc[0, 3:6] = R_curr[:, 1]
            H_nhc[1, 3:6] = R_curr[:, 2]
            if coupled_jacobian:
                H_nhc[0, 6:9] = np.array([v_b[2], 0.0, -v_b[0]])
                H_nhc[1, 6:9] = np.array([-v_b[1], v_b[0], 0.0])
            R_cov_nhc = np.diag([0.1, 0.05])
            esekf.update_error_state(y_nhc, H_nhc, R_cov_nhc, chi2_gate=False)

        # Rolling window for AI displacement inference
        imu_sample = np.concatenate([a_b, w_b])
        window_buffer.append(imu_sample)

        if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
            win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
            pred_d, ev_class, sigma = predictor.predict(win_arr)

            if ev_class == 0 or pred_d < 0.1:
                esekf.update_zupt()
            else:
                scaled_d = pred_d * k_scale
                v_fwd = scaled_d / 2.0
                var = max((sigma * k_scale / 2.0) ** 2, 0.04)
                if ev_class == 2:
                    var *= 4.0
                elif ev_class == 3:
                    var *= 2.0
                
                # Forward speed update
                R_up = quat_to_rot_matrix(esekf.q)
                v_current = float(np.linalg.norm(esekf.v[:2]))
                y_spd = np.array([v_fwd - v_current])
                H_spd = np.zeros((1, esekf.dim_error))
                H_spd[0, 3] = R_up[0, 0]
                H_spd[0, 4] = R_up[1, 0]
                if coupled_jacobian:
                    v_b_meas = R_up.T @ esekf.v
                    H_spd[0, 6:9] = np.array([0.0, -v_b_meas[2], v_b_meas[1]])

                R_cov_spd = np.array([[var]])
                accepted = esekf.update_error_state(y_spd, H_spd, R_cov_spd, chi2_gate=True)
                if accepted:
                    new_v = max(float(np.linalg.norm(esekf.v[:2])), 0.0)
                    psi_curr = np.arctan2(R_up[1, 0], R_up[0, 0])
                    esekf.v[0] = new_v * np.cos(psi_curr)
                    esekf.v[1] = new_v * np.sin(psi_curr)
                    esekf.v[2] = 0.0

        if len(window_buffer) > 40:
            window_buffer.pop(0)

        est_pN[k] = esekf.p[0]
        est_pE[k] = esekf.p[1]
        est_speed[k] = esekf.get_speed_ms()
        curr_hdg = np.degrees(new_psi) % 360.0
        est_hdg_deg[k] = curr_hdg

        if log_diagnostic:
            # Error metrics at step k
            # compute lat/lon for point k
            lat_k, lon_k = local_xy_to_latlon(np.array([esekf.p[1]]), np.array([esekf.p[0]]), init_lat, init_lon)
            from eval.outage_sim import haversine_distance_m
            pos_err = haversine_distance_m(lat_k[0], lon_k[0], gt_lat[k], gt_lon[k])
            
            # Heading error wrapped to [-180, 180]
            h_err = (curr_hdg - gt_hdg[k] + 180.0) % 360.0 - 180.0
            
            # Approximate along-track / cross-track
            psi_gt_rad = np.radians(gt_hdg[k])
            # displacement vector in meters: dN = pN, dE = pE
            # true displacement vector from start
            # we can approximate cross-track error as pos_err * sin(h_err)
            xtrack = pos_err * np.sin(np.radians(h_err))
            atrack = pos_err * np.cos(np.radians(h_err))

            diag_rows.append({
                "timestamp": float(time_s[k]),
                "ground_truth_heading": float(gt_hdg[k]),
                "estimated_heading": float(curr_hdg),
                "heading_error": float(h_err),
                "gyro_z_raw": float(gyro_raw[k, 2]),
                "gyro_bias": float(esekf.w_b[2]),
                "gyro_z_corrected": float(yaw_rate_for_integ),
                "angular_rate": float(w_v[2]),
                "position_error": float(pos_err),
                "cross_track_error": float(xtrack),
                "along_track_error": float(atrack),
            })

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    df_diag = pd.DataFrame(diag_rows) if log_diagnostic else None
    return est_lat, est_lon, est_speed, df_diag


def evaluate_test_set(
    config_name: str,
    inject_bias: bool = False,
    deadband_rads: float = 0.0,
    apply_nhc: bool = False,
    coupled_jacobian: bool = False,
    use_fej: bool = False,
    use_road_heading: bool = False,
    return_first_diag: bool = False
):
    """
    Evaluates the 4 paired 180s motorway test outages across vw14b, vw14c, vw2, vw4.
    """
    test_runs = [
        ("vw14b", Path("data/processed/synchronized/sync_vw14b.parquet"), 5),
        ("vw14c", Path("data/processed/synchronized/sync_vw14c.parquet"), 4),
        ("vw2",   Path("data/processed/synchronized/sync_vw2.parquet"),   1),
        ("vw4",   Path("data/processed/synchronized/sync_vw4.parquet"),   5),
    ]

    best_pt = config.MODELS_DIR / "velocity_net_best.pt"
    predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    scores = []
    saved_diag = None

    for run_name, p_file, target_oid in test_runs:
        if not p_file.exists():
            continue
        df = pd.read_parquet(p_file)
        dur = float(df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0])
        sched = generate_outage_schedule(dur, run_id=f"sync_{run_name}")
        
        # Match the 180s outage
        target_outage = None
        for s in sched:
            if s[2] == 180 and s[3] == target_oid if len(s) > 3 else (s[2] == 180):
                target_outage = s
                break
        if target_outage is None:
            # Find any 180s outage
            for s in sched:
                if s[2] == 180:
                    target_outage = s
                    break

        t_start, t_end, o_dur = target_outage[0], target_outage[1], target_outage[2]
        mask = (df[config.COL_TIME] >= t_start) & (df[config.COL_TIME] <= t_end)
        df_sub = df.loc[mask].reset_index(drop=True)

        if len(df_sub) < 5:
            continue

        outage_start_idx = df.index[df[config.COL_TIME] >= t_start][0]
        pre_idx_start = max(0, outage_start_idx - 10)
        pre_df = df.iloc[pre_idx_start:outage_start_idx]

        init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
        init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
        init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
        h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
        init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)

        # Pre-calibrate run alignment using initial warmup window (t < 45s) matching replay.py
        align = AlignmentEngine()
        warmup = df[df[config.COL_TIME] < 45.0]
        if len(warmup) >= 25:
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            gyro_w = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            norm_a = np.linalg.norm(mean_a)
            if norm_a > 1e-6:
                align.u_z_body = -mean_a / norm_a
                z = align.u_z_body
                ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
                y = np.cross(z, ref)
                y /= np.linalg.norm(y)
                x = np.cross(y, z)
                align.R_b_to_v = np.vstack([x, y, z])
                align.static_calibrated = True
            align.calibrate_dynamic(acc_w, gyro_w)
        else:
            acc_w = df[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values[max(0, outage_start_idx - 300):outage_start_idx]
            gyro_w = df[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values[max(0, outage_start_idx - 300):outage_start_idx]
            align.calibrate_dynamic(acc_w, gyro_w)

        # Pre-outage scale & bias
        pre_calib_window = df.iloc[max(0, outage_start_idx - 300):outage_start_idx]
        active_k = estimate_scale_factor_rls(pre_calib_window, predictor=predictor) if len(pre_calib_window) >= 30 else 1.0
        active_bg = estimate_pre_outage_gyro_bias(pre_calib_window, alignment=align) if len(pre_calib_window) >= 20 else 0.0

        bg_to_inject = active_bg if inject_bias else 0.0
        do_diag = return_first_diag and (saved_diag is None)

        p_lat, p_lon, _, df_diag = run_parametric_esekf(
            df_sub, init_lat, init_lon, init_spd, init_hdg,
            predictor=predictor, alignment=align, k_scale=active_k,
            init_gyro_bias_z=bg_to_inject,
            deadband_rads=deadband_rads,
            apply_nhc=apply_nhc,
            coupled_jacobian=coupled_jacobian,
            use_fej=use_fej,
            use_road_heading=use_road_heading,
            log_diagnostic=do_diag
        )

        if do_diag and df_diag is not None:
            saved_diag = df_diag

        gt_lats = df_sub[config.COL_TRUE_LAT].values
        gt_lons = df_sub[config.COL_TRUE_LON].values
        gt_hdgs = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

        from eval.outage_sim import haversine_distance_m
        dist_travelled = 0.0
        for i in range(1, len(gt_lats)):
            dist_travelled += haversine_distance_m(gt_lats[i-1], gt_lons[i-1], gt_lats[i], gt_lons[i])

        sc = score_outage_segment(
            p_lat, p_lon, gt_lats, gt_lons,
            distance_travelled_m=dist_travelled,
            duration_s=o_dur,
            gt_heading_deg=gt_hdgs
        )
        sc["run_id"] = run_name
        sc["config"] = config_name
        sc["active_bg"] = active_bg
        scores.append(sc)

    df_res = pd.DataFrame(scores)
    return df_res, saved_diag


if __name__ == "__main__":
    print("Forensic runner ready.")
