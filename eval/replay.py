"""
iNAV Trajectory Replay Engine
Replays synchronized IMU sensor streams with simulated GNSS blackouts
through baseline models or AI displacement filters, evaluates drift,
and updates the leaderboard.
"""

import sys
import glob
import logging
from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment, record_instance_leaderboard
from eval.baseline import run_strapdown_baseline, run_constant_velocity_baseline, local_xy_to_latlon
from modules.alignment import AlignmentEngine
from modules.ukf import UKFNavigationFilter
from modules.gnss_health import GnssHealthManager, GnssHealthState
from modules.sensor_types import GnssSample, GnssValidity
from modules.esekf import ESEKFNavigationFilter
from modules.velocity_net import VelocityNetPredictor
from modules.map_matcher import FixedLagHMMMapMatcher, RoadGraph, RoadSegment, load_road_graph_from_osm_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.replay")


def run_inav_esekf_pipeline(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    predictor: Optional[VelocityNetPredictor] = None,
    alignment: Optional[AlignmentEngine] = None,
    k_scale: float = 1.0,
    init_gyro_bias_z: float = 0.0,
    freeze_gyro_bias: bool = True,
    apply_nhc: bool = True,
    dt: float = config.TARGET_DT
):
    """
    Upgraded iNAV Pipeline (Pillars 1-5):
    15-State ES-EKF + Joseph Covariance + NIS Outlier Gating + VelocityNet AI Velocity Prediction + NHC.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    if predictor is None:
        best_pt = config.MODELS_DIR / "velocity_net_best.pt"
        predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    # Initialize Alignment Engine if not pre-calibrated
    if alignment is None:
        alignment = AlignmentEngine()
        calib_samples = min(50, n)
        alignment.calibrate_static(acc_raw[:calib_samples], gyro_raw[:calib_samples])
        alignment.calibrate_dynamic(acc_raw[:calib_samples], gyro_raw[:calib_samples])

    # Initialize 15-state ES-EKF filter
    esekf = ESEKFNavigationFilter(dt=dt)
    esekf.initialize(
        init_lat_ned=0.0,
        init_lon_ned=0.0,
        init_speed_ms=init_speed_ms,
        init_heading_rad=np.radians(init_heading_deg)
    )
    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    window_buffer = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]

        # 1. Transform body to vehicle frame
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 2. ES-EKF High-rate strapdown propagation & Joseph covariance with NHC
        esekf.predict_vehicular(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt)

        # 3. Rolling window for AI displacement inference
        imu_sample = np.concatenate([a_b, w_b])
        window_buffer.append(imu_sample)

        if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
            win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
            pred_d, ev_class, sigma = predictor.predict(win_arr)

            # ZUPT or NIS-gated forward speed update
            if ev_class == 0 or pred_d < 0.1:
                esekf.update_zupt()
            else:
                scaled_d = pred_d * k_scale
                v_fwd = scaled_d / 2.0
                var = max((sigma * k_scale / 2.0) ** 2, 0.04)
                if ev_class == 2:
                    var *= 4.0  # rough road
                elif ev_class == 3:
                    var *= 2.0  # dynamic maneuver
                esekf.update_forward_speed(speed_ms=v_fwd, variance=var)

        if len(window_buffer) > 40:
            window_buffer.pop(0)

        est_pN[k] = esekf.p[0]
        est_pE[k] = esekf.p[1]
        est_speed[k] = esekf.get_speed_ms()

    # Convert local NED to Lat/Lon
    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed


def run_inav_ukf_pipeline(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    predictor: Optional[VelocityNetPredictor] = None,
    alignment: Optional[AlignmentEngine] = None,
    k_scale: float = 1.0,
    dt: float = config.TARGET_DT,
    feed_gnss: bool = True,
    map_matcher: Optional[FixedLagHMMMapMatcher] = None
):
    """
    iNAV Hybrid Navigation Filter:
    Auto-alignment -> VelocityNet Displacement -> UKF Fusion with NHC & ZUPT + GNSS Health Fusion.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    if predictor is None:
        best_pt = config.MODELS_DIR / "velocity_net_best.pt"
        predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    # Initialize Alignment Engine if not pre-calibrated
    if alignment is None:
        alignment = AlignmentEngine()
        calib_samples = min(50, n)
        alignment.calibrate_static(acc_raw[:calib_samples], gyro_raw[:calib_samples])
        alignment.calibrate_dynamic(acc_raw[:calib_samples], gyro_raw[:calib_samples])

    # Initialize UKF filter and GNSS Health Manager
    ukf = UKFNavigationFilter(dt=dt)
    ukf.initialize(
        init_lat=init_lat,
        init_lon=init_lon,
        init_speed_ms=init_speed_ms,
        init_heading_rad=np.radians(init_heading_deg)
    )
    health_mgr = GnssHealthManager()

    has_gnss_cols = (
        config.COL_GPS_LAT in df_outage.columns and
        config.COL_GPS_LON in df_outage.columns
    )

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    # Rolling window buffer for VelocityNet (20 samples = 2.0s)
    window_buffer = []

    for k in range(n):
        # 1. Transform IMU from phone frame to vehicle frame
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # Dynamic dt calculation with validation
        if "timestamp_ns" in df_outage.columns:
            if k > 0:
                dt_k = (df_outage["timestamp_ns"].iloc[k] - df_outage["timestamp_ns"].iloc[k - 1]) * 1e-9
            else:
                dt_k = dt
        elif config.COL_TIME in df_outage.columns:
            if k > 0:
                dt_k = float(df_outage[config.COL_TIME].iloc[k] - df_outage[config.COL_TIME].iloc[k - 1])
            else:
                dt_k = dt
        else:
            dt_k = dt

        # Reject negative, duplicate, or unreasonable gaps (> 0.5s)
        if dt_k <= 0.0 or dt_k > 0.5 or not math.isfinite(dt_k):
            dt_k = dt

        # 2. Filter prediction step
        ukf.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=dt_k)

        # Monotonic timestamp for GNSS timeout & health checks
        if "timestamp_ns" in df_outage.columns:
            t_ns = int(df_outage["timestamp_ns"].iloc[k])
        elif config.COL_TIME in df_outage.columns:
            t_ns = int(float(df_outage[config.COL_TIME].iloc[k]) * 1e9)
        else:
            t_ns = int(k * dt * 1e9)

        health_mgr.check_timeout(t_ns)

        # GNSS Measurement update (active outside outages)
        if feed_gnss and has_gnss_cols:
            gnss_avail = True
            if "gnss_available" in df_outage.columns:
                gnss_avail = bool(df_outage["gnss_available"].iloc[k])
            elif "is_outage" in df_outage.columns:
                gnss_avail = not bool(df_outage["is_outage"].iloc[k])

            if gnss_avail:
                g_lat = df_outage[config.COL_GPS_LAT].iloc[k]
                g_lon = df_outage[config.COL_GPS_LON].iloc[k]
                if math.isfinite(g_lat) and math.isfinite(g_lon) and g_lat != 0.0:
                    g_spd = float(df_outage[config.COL_GPS_SPEED_MS].iloc[k]) if config.COL_GPS_SPEED_MS in df_outage.columns else 0.0
                    g_brg = float(df_outage[config.COL_GPS_BEARING].iloc[k]) if config.COL_GPS_BEARING in df_outage.columns else 0.0
                    g_acc = float(df_outage[config.COL_GPS_ACCURACY].iloc[k]) if config.COL_GPS_ACCURACY in df_outage.columns else 3.0
                    if not math.isfinite(g_acc) or g_acc <= 0.0:
                        g_acc = 3.0

                    sample = GnssSample(
                        timestamp_ns=t_ns,
                        latitude_deg=float(g_lat),
                        longitude_deg=float(g_lon),
                        speed_mps=float(g_spd) if math.isfinite(g_spd) else 0.0,
                        bearing_deg=float(g_brg) if math.isfinite(g_brg) else 0.0,
                        horizontal_accuracy_m=float(g_acc),
                        validity=GnssValidity.BASIC_FIX_VALID
                    )
                    ukf.update_gnss_sample(sample, health_mgr)

        # Channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        imu_sample = np.concatenate([a_b, w_b])
        window_buffer.append(imu_sample)

        # 3. Apply VelocityNet measurement every 5 steps (0.5s) using rolling 20-sample window
        if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
            win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
            pred_d, ev_class, sigma = predictor.predict(win_arr)

            # If stationary event (class 0) or near zero speed, execute ZUPT
            if ev_class == 0 or pred_d < 0.1:
                ukf.update_zupt(gyro_reading=w_v[2])
            else:
                scaled_d = pred_d * k_scale
                scaled_sigma = max(sigma * k_scale, 0.2)
                ukf.update_velocity_net(delta_d_pred=scaled_d, sigma_pred=scaled_sigma, event_class=ev_class, window_dur=2.0)

        # 4. Map Matching measurement update (Pillar 5)
        if map_matcher is not None and (k % 5 == 0):
            p_N_curr = float(ukf.x[0])
            p_E_curr = float(ukf.x[1])
            hdg_curr_deg = math.degrees(ukf.x[3]) % 360.0
            spd_curr = float(ukf.x[2])
            sigma_p = math.sqrt(max(0.01, float(max(ukf.P[0, 0], ukf.P[1, 1]))))

            if map_matcher.graph.is_ref_set:
                curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
                curr_lat, curr_lon = float(curr_lat), float(curr_lon)
                if map_matcher.graph.is_in_bounds(curr_lat, curr_lon):
                    p_graph = map_matcher.graph.latlon_to_local(curr_lat, curr_lon)
                    match_res = map_matcher.match(
                        point_xy=p_graph,
                        heading_deg=hdg_curr_deg,
                        speed_ms=spd_curr,
                        travel_dist_m=max(spd_curr * 0.5, 0.05),
                        sigma_pos_m=sigma_p,
                        dt=0.5
                    )
                    if not match_res.is_off_road and match_res.confidence >= 0.25:
                        snap_lat, snap_lon = map_matcher.graph.local_to_latlon(
                            float(match_res.snapped_point[0]), float(match_res.snapped_point[1])
                        )
                        d_lat = math.radians(snap_lat - init_lat)
                        d_lon = math.radians(snap_lon - init_lon)
                        p_N_match = 6371000.0 * d_lat
                        p_E_match = 6371000.0 * d_lon * math.cos(math.radians(init_lat))

                        ukf.update_map_match(
                            p_N_match=float(p_N_match),
                            p_E_match=float(p_E_match),
                            psi_road=float(match_res.road_heading_rad),
                            confidence=float(match_res.confidence),
                            is_heading_valid=bool(match_res.is_heading_valid)
                        )
            else:
                match_res = map_matcher.match(
                    point_xy=np.array([p_N_curr, p_E_curr]),
                    heading_deg=hdg_curr_deg,
                    speed_ms=spd_curr,
                    travel_dist_m=max(spd_curr * 0.5, 0.05),
                    sigma_pos_m=sigma_p,
                    dt=0.5
                )
                if not match_res.is_off_road and match_res.confidence >= 0.25:
                    ukf.update_map_match(
                        p_N_match=float(match_res.snapped_point[0]),
                        p_E_match=float(match_res.snapped_point[1]),
                        psi_road=float(match_res.road_heading_rad),
                        confidence=float(match_res.confidence),
                        is_heading_valid=bool(match_res.is_heading_valid)
                    )

        # Trim buffer to max 40 samples
        if len(window_buffer) > 40:
            window_buffer.pop(0)

        est_pN[k] = ukf.x[0]
        est_pE[k] = ukf.x[1]
        est_speed[k] = ukf.x[2]

    # Convert local North-East coordinates back to Lat/Lon
    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed


def run_inav_cpp_kinematic_pipeline(
    df_outage: pd.DataFrame,
    init_lat: float,
    init_lon: float,
    init_speed_ms: float,
    init_heading_deg: float,
    predictor: Optional[VelocityNetPredictor] = None,
    alignment: Optional[AlignmentEngine] = None,
    k_scale: float = 1.0,
    gyro_deadband: float = 0.0,
    dt: float = config.TARGET_DT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    C++ / Android Production Kinematic Filter Replay Engine (inav::DeadReckoningFilter).
    Direct Python equivalent of cpp/include/inav_filter.hpp.
    """
    n = len(df_outage)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    if predictor is None:
        best_pt = config.MODELS_DIR / "velocity_net_best.pt"
        predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    acc_raw = df_outage[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
    gyro_raw = df_outage[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

    if alignment is None:
        alignment = AlignmentEngine()
        calib_samples = min(50, n)
        alignment.calibrate_static(acc_raw[:calib_samples], gyro_raw[:calib_samples])
        alignment.calibrate_dynamic(acc_raw[:calib_samples], gyro_raw[:calib_samples])

    p_N = 0.0
    p_E = 0.0
    v = max(init_speed_ms, 0.0) if init_speed_ms >= 0.3 else 0.0
    psi = math.radians(init_heading_deg)
    bg = 0.0

    est_pN = np.zeros(n)
    est_pE = np.zeros(n)
    est_speed = np.zeros(n)

    window_buffer = []

    for k in range(n):
        a_b = acc_raw[k]
        w_b = gyro_raw[k]
        a_v, w_v = alignment.transform_imu(a_b, w_b)

        # 1. Heading propagation
        omega_corr = float(w_v[2] - bg)
        if gyro_deadband > 0.0 and abs(omega_corr) < gyro_deadband:
            omega_corr = 0.0
        psi = (psi + omega_corr * dt) % (2.0 * math.pi)

        # 2. Window inference
        imu_sample = np.concatenate([a_b, w_b])
        window_buffer.append(imu_sample)

        if len(window_buffer) >= config.WINDOW_SIZE and (k % 5 == 0):
            win_arr = np.ascontiguousarray(np.array(window_buffer[-config.WINDOW_SIZE:], dtype=np.float32))
            pred_d, ev_class, sigma = predictor.predict(win_arr)

            if ev_class == 0 or pred_d < 0.2:
                v = 0.0
                bg = 0.98 * bg + 0.02 * float(w_v[2])
            else:
                v_net = (pred_d * k_scale) / 2.0
                base_var = max(((sigma * k_scale) / 2.0) ** 2, 0.04)
                R_meas = base_var * 4.0 if ev_class == 2 else (base_var * 2.0 if ev_class == 3 else base_var)
                P_vv = 0.25
                K = P_vv / (P_vv + R_meas)
                v = v + K * (v_net - v)
                v = max(v, 0.0)

        if len(window_buffer) > 40:
            window_buffer.pop(0)

        # Position step
        displacement = v * dt
        p_N += displacement * math.cos(psi)
        p_E += displacement * math.sin(psi)

        est_pN[k] = p_N
        est_pE[k] = p_E
        est_speed[k] = v

    est_lat, est_lon = local_xy_to_latlon(est_pE, est_pN, init_lat, init_lon)
    return est_lat, est_lon, est_speed


def estimate_scale_factor_rls(df_history: pd.DataFrame, predictor: object, lam: float = 0.98) -> float:
    """
    Sliding-Window Recursive Least Squares (RLS) estimation of vehicle scale factor k
    as formulated in DVSE / CarSpeedNet literature:
      min_k sum lambda^(t-i) (v_gnss,i - k * v_ai,i)^2
    """
    k = 1.0
    P = 0.04
    if len(df_history) < 25 or predictor is None:
        return 1.0

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    for i in range(0, len(df_history) - 20, 2):
        v_gps = float(df_history[config.COL_TRUE_SPEED_MS].iloc[i:i+20].mean())
        if v_gps < 2.0:  # Moving gate: only calibrate when vehicle is actively driving
            continue
        w = np.ascontiguousarray(df_history[cols].iloc[i:i+20].values, dtype=np.float32)
        v_ai = float(predictor.predict(w)[0]) / 2.0
        if v_ai < 0.5:
            continue
        G = (P * v_ai) / (lam + v_ai**2 * P)
        k = k + G * (v_gps - k * v_ai)
        P = (1.0 / lam) * (1.0 - G * v_ai) * P

    # Physical clamping: typical tyre/chassis variance across cars is within [0.80, 1.35]
    return float(np.clip(k, 0.80, 1.35))


def estimate_pre_outage_gyro_bias(
    df_history: pd.DataFrame,
    alignment: Optional[AlignmentEngine] = None,
    dt: float = config.TARGET_DT
) -> float:
    """
    Pre-Outage Gyro Yaw Bias Estimation (§1):
    During healthy GNSS lock prior to tunnel/blackout, compares vehicle-frame
    gyro yaw reading against GNSS course-over-ground angular rate:
      delta_w = w_gyro_z - d(psi_GNSS)/dt
    Moving median isolates static/thermal gyro bias bg_z.
    """
    if len(df_history) < 20 or config.COL_TRUE_HEADING not in df_history.columns or alignment is None:
        return 0.0

    hdg = np.radians(df_history[config.COL_TRUE_HEADING].values)
    hdg_unwrapped = np.unwrap(hdg)
    yaw_rate_gnss = np.diff(hdg_unwrapped) / dt

    cols_gyro = [config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]
    cols_acc = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]
    gyro_raw = df_history[cols_gyro].values[:-1]
    acc_raw = df_history[cols_acc].values[:-1]

    _, w_v = alignment.transform_imu(acc_raw, gyro_raw)
    gyro_yaw = w_v[:, 2]

    diff = gyro_yaw - yaw_rate_gnss
    mask = np.abs(yaw_rate_gnss) < 0.15  # filter steering transients
    if np.sum(mask) > 10:
        bg_est = float(np.median(diff[mask]))
    else:
        bg_est = float(np.median(diff))
    return float(np.clip(bg_est, -0.05, 0.05))


def evaluate_run_outages(
    sync_parquet_path: Path,
    method: str = "inav_esekf",
    durations: List[int] = config.OUTAGE_DURATIONS_SEC,
    predictor: Optional[object] = None,
    map_matcher: Optional[FixedLagHMMMapMatcher] = None
) -> List[Dict]:
    """
    Simulate blackouts on a synchronized run and evaluate dead-reckoning performance.
    """
    df = pd.read_parquet(sync_parquet_path)
    total_dur = df[config.COL_TIME].iloc[-1] - df[config.COL_TIME].iloc[0]
    run_key = sync_parquet_path.stem.replace("sync_", "")
    schedule = generate_outage_schedule(total_dur, durations=durations, run_id=run_key)
    if not schedule:
        return []

    df_sim, df_outages = inject_outages(df, schedule)

    outage_scores = []
    k_calib = 1.0
    run_align = None

    ai_methods = [
        "inav_ukf", "inav_ai_ukf_v1", "inav_esekf", "inav_esekf_snapped", "inav_snapped",
        "inav_cpp", "inav_cpp_kinematic", "inav_cpp_core"
    ]
    if predictor is None and method in ai_methods:
        best_pt = config.MODELS_DIR / "velocity_net_best.pt"
        predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    if predictor is not None and method in ai_methods:
        warmup = df[df[config.COL_TIME] < 45.0]
        if len(warmup) >= 25:
            k_calib = estimate_scale_factor_rls(warmup, predictor=predictor)
            logger.info(f"RLS Calibrated scale factor k={k_calib:.3f} for {sync_parquet_path.stem}")

            # Pre-calibrate run alignment using warmup acceleration & gyroscope
            from modules.alignment import AlignmentState
            run_align = AlignmentEngine()
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            gyro_w = warmup[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref)
            y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

    for _, row in df_outages.iterrows():
        oid = int(row["outage_id"])
        dur = int(row["duration_s"])
        mask = (df_sim["outage_id"] == oid)
        df_sub = df.loc[mask]

        if len(df_sub) < 5:
            continue

        # Extract pre-outage state strictly from 1.0s window before outage onset
        outage_start_idx = mask.idxmax()
        pre_idx_start = max(0, outage_start_idx - 10)
        pre_df = df.iloc[pre_idx_start:outage_start_idx]

        if len(pre_df) > 0:
            init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
            init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
            init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
            h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
            init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
        else:
            init_idx = max(0, outage_start_idx - 1)
            init_lat = float(df.loc[init_idx, config.COL_TRUE_LAT])
            init_lon = float(df.loc[init_idx, config.COL_TRUE_LON])
            init_spd = float(df.loc[init_idx, config.COL_TRUE_SPEED_MS])
            init_hdg = float(df.loc[init_idx, config.COL_TRUE_HEADING])

        # Dynamically estimate scale factor k using RLS and pre-outage gyro yaw bias
        pre_calib_window = df.iloc[max(0, outage_start_idx - 300):outage_start_idx]
        active_k = estimate_scale_factor_rls(pre_calib_window, predictor=predictor) if len(pre_calib_window) >= 30 else k_calib
        active_bg = estimate_pre_outage_gyro_bias(pre_calib_window, alignment=run_align) if len(pre_calib_window) >= 20 else 0.0

        # Run chosen dead reckoning method with normalized config naming
        if method in ["strapdown", "baseline_strapdown_v1", "baseline_cpp_strapdown_v1"]:
            cfg_name = "baseline_strapdown_v1"
            p_lat, p_lon, _ = run_strapdown_baseline(
                df_sub, init_lat, init_lon, init_spd, init_hdg
            )
        elif method in ["constant_velocity", "baseline_cv_heading_v1"]:
            cfg_name = "baseline_cv_heading_v1"
            p_lat, p_lon, _ = run_constant_velocity_baseline(
                df_sub, init_lat, init_lon, init_spd, init_hdg
            )
        elif method in ["inav_ukf", "inav_ai_ukf_v1"]:
            cfg_name = "inav_ai_ukf_v1"
            p_lat, p_lon, _ = run_inav_ukf_pipeline(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                predictor=predictor, alignment=run_align, k_scale=active_k,
                map_matcher=None
            )
        elif method in ["inav_ukf_map", "inav_map_aided"]:
            cfg_name = "inav_ukf_map"
            if map_matcher is not None:
                map_matcher.reset()
            p_lat, p_lon, _ = run_inav_ukf_pipeline(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                predictor=predictor, alignment=run_align, k_scale=active_k,
                map_matcher=map_matcher
            )
        elif method in ["inav_esekf", "inav_spectra_esekf", "inav_spectra_esekf_v2"]:
            cfg_name = "inav_esekf"
            p_lat, p_lon, _ = run_inav_esekf_pipeline(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                predictor=predictor, alignment=run_align, k_scale=active_k
            )
        elif method in ["inav_esekf_snapped", "inav_snapped"]:
            cfg_name = "inav_esekf_snapped"
            p_lat, p_lon, _ = run_inav_esekf_pipeline(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                predictor=predictor, alignment=run_align, k_scale=active_k
            )
        elif method in ["inav_cpp", "inav_cpp_kinematic", "inav_cpp_core"]:
            cfg_name = "inav_cpp_kinematic"
            p_lat, p_lon, _ = run_inav_cpp_kinematic_pipeline(
                df_sub, init_lat, init_lon, init_spd, init_hdg,
                predictor=predictor, alignment=run_align, k_scale=active_k
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        gt_lat = df_sub[config.COL_TRUE_LAT].values
        gt_lon = df_sub[config.COL_TRUE_LON].values
        gt_hdg = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

        score = score_outage_segment(
            p_lat, p_lon, gt_lat, gt_lon,
            distance_travelled_m=row["distance_travelled_m"],
            duration_s=dur,
            gt_heading_deg=gt_hdg
        )
        score["outage_id"] = oid
        score["run"] = sync_parquet_path.stem.replace("sync_", "")
        score["method"] = cfg_name
        score["config"] = cfg_name
        outage_scores.append(score)

    return outage_scores


def run_benchmark_suite(
    methods: List[str] = ["inav_ukf"],
    test_runs_limit: Optional[int] = 10,
    predictor: Optional[VelocityNetPredictor] = None
) -> pd.DataFrame:
    """
    Run evaluation harness across test set runs and update leaderboard.
    """
    test_files = []
    sync_files = glob.glob(str(config.SYNC_PROCESSED_DIR / "sync_*.parquet"))

    for f in sorted(sync_files):
        k = Path(f).stem.replace("sync_", "").lower()
        if any(k.startswith(p) for p in config.TEST_DRIVERS):
            test_files.append(Path(f))

    if test_runs_limit and test_runs_limit > 0:
        test_files = test_files[:test_runs_limit]

    logger.info(f"Benchmarking {len(test_files)} test runs across methods: {methods}")

    if any(m in ["inav_ukf", "inav_ai_ukf_v1", "inav_ukf_map"] for m in methods) and predictor is None:
        best_pt = config.MODELS_DIR / "velocity_net_best.pt"
        predictor = VelocityNetPredictor(model_path=str(best_pt) if best_pt.exists() else None)

    all_scores = []
    for method in methods:
        method_scores = []
        map_matcher = None
        if method in ["inav_ukf_map", "inav_map_aided"]:
            uk_osm = config.DATA_DIR / "osm_uk_test_roads.json"
            assets_osm = config.BASE_DIR / "android" / "app" / "src" / "main" / "assets" / "osm_roads_cache.json"
            if uk_osm.exists():
                graph = load_road_graph_from_osm_json(str(uk_osm), ref_lat=52.20, ref_lon=-2.19)
                map_matcher = FixedLagHMMMapMatcher(graph=graph)
            elif assets_osm.exists():
                graph = load_road_graph_from_osm_json(str(assets_osm), ref_lat=22.72, ref_lon=75.83)
                map_matcher = FixedLagHMMMapMatcher(graph=graph)

        for f in test_files:
            sc = evaluate_run_outages(f, method=method, predictor=predictor, map_matcher=map_matcher)
            method_scores.extend(sc)

        if not method_scores:
            continue

        df_m = pd.DataFrame(method_scores)
        all_scores.append(df_m)

        overall_drift = float(df_m["pct_of_distance"].median()) if "pct_of_distance" in df_m.columns else float(df_m["drift_percent"].median())
        record_instance_leaderboard(method_scores)
        logger.info(f"Method '{method}' evaluated across {len(test_files)} runs: Overall Median Drift = {overall_drift:.2f}%")

    if all_scores:
        return pd.concat(all_scores, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV Trajectory Replay & Benchmark Harness")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "strapdown", "constant_velocity", "inav_ukf", "inav_ukf_map", "inav_esekf", "inav_cpp_kinematic"],
        help="Specific method to evaluate or 'all'"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Max test runs to evaluate (0 for all test runs)"
    )
    args = parser.parse_args()

    eval_methods = ["constant_velocity", "strapdown", "inav_ukf", "inav_ukf_map", "inav_esekf", "inav_cpp_kinematic"] if args.method == "all" else [args.method]
    limit = args.max_runs if args.max_runs > 0 else None
    run_benchmark_suite(methods=eval_methods, test_runs_limit=limit)
