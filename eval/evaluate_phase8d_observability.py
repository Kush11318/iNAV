"""
Phase 8D: Along-Track Map Observability Experiment (Offline Diagnostic Only)
Tests Hypothesis H0 vs H1:
Does road geometry (curvature, heading, road boundaries) contain useful information
about along-track vehicle position during GNSS outages?

Strict Isolation Mode:
- No UKF modifications or tuning.
- No ground-truth or CAN input to the estimator.
- Evaluates across all 56 held-out test outages.
"""

import sys
import glob
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from eval.train_phase8a_4s_model import VelocityNet4s
from eval.outage_sim import generate_outage_schedule, inject_outages
from eval.score import score_outage_segment
from modules.alignment import AlignmentEngine, AlignmentState
from modules.ukf import UKFNavigationFilter
from modules.map_matcher import (
    RoadGraph, RoadSegment, RoadEdge, MapMatchResult,
    FixedLagHMMMapMatcher, load_road_graph_from_osm_json
)
from eval.baseline import local_xy_to_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.Phase8D_Obs")


def wrap_to_pi(angle_rad: float) -> float:
    a = (angle_rad + math.pi) % (2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def compute_batched_velocitynet_predictions(
    model: VelocityNet4s,
    seq_data: np.ndarray,
    needed_step_indices: List[int],
    window_len: int = 40,
    batch_size: int = 2048,
    device: str = "cpu"
) -> Dict[int, Tuple[float, int, float]]:
    n_samples = len(seq_data)
    valid_indices = sorted([k for k in needed_step_indices if window_len - 1 <= k < n_samples])
    if not valid_indices:
        return {}

    windows = [seq_data[k - window_len + 1 : k + 1] for k in valid_indices]
    all_windows = np.array(windows, dtype=np.float32)

    preds_d, preds_ev, preds_sig = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            bx = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)
            d_d, ev_logits, sig = model(bx)
            p_d = d_d.squeeze(-1).cpu().numpy()
            p_ev = torch.argmax(ev_logits, dim=1).cpu().numpy()
            p_sig = sig.squeeze(-1).cpu().numpy()

            if p_d.ndim == 0:
                p_d, p_ev, p_sig = np.array([p_d]), np.array([p_ev]), np.array([p_sig])

            preds_d.append(p_d)
            preds_ev.append(p_ev)
            preds_sig.append(p_sig)

    cat_d = np.concatenate(preds_d)
    cat_ev = np.concatenate(preds_ev)
    cat_sig = np.concatenate(preds_sig)

    lookup = {}
    for idx, k in enumerate(valid_indices):
        lookup[k] = (float(cat_d[idx]), int(cat_ev[idx]), float(cat_sig[idx]))
    return lookup


def build_traversed_road_path(
    graph: RoadGraph,
    gt_lat_arr: np.ndarray,
    gt_lon_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Constructs the 2D local Cartesian polyline for the road corridor traversed by the GT trajectory.
    Returns:
        polyline_pts: (M, 2) array [p_N, p_E] in local meters relative to graph origin
        arc_lengths: (M,) array of cumulative distance along polyline
        edge_ids: list of OSM edge IDs along the polyline
    """
    n_pts = len(gt_lat_arr)
    sample_indices = np.linspace(0, n_pts - 1, min(20, n_pts), dtype=int)

    matched_edges = []
    seen_edges = set()

    for idx in sample_indices:
        lat = gt_lat_arr[idx]
        lon = gt_lon_arr[idx]
        p_local = graph.latlon_to_local(lat, lon)
        segs = graph.spatial_index.query_radius(p_local[0], p_local[1], radius_m=40.0)
        if segs:
            # Find nearest segment
            best_d = 9999.0
            best_seg = None
            for s in segs:
                proj, d = FixedLagHMMMapMatcher.project_point_to_segment(p_local, s.p1, s.p2)
                if d < best_d:
                    best_d = d
                    best_seg = s
            if best_seg and best_seg.edge_id not in seen_edges:
                seen_edges.add(best_seg.edge_id)
                matched_edges.append(graph.edges[best_seg.edge_id])

    # If no edges found or sparse, fallback to direct GT polyline mapped to graph local coords
    if not matched_edges:
        poly = np.array([graph.latlon_to_local(lat, lon) for lat, lon in zip(gt_lat_arr, gt_lon_arr)])
        diffs = np.diff(poly, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        arc = np.insert(np.cumsum(dists), 0, 0.0)
        return poly, arc, [0]

    # Combine edge polylines in order of vehicle traversal
    all_pts = []
    e_ids = []
    for edge in matched_edges:
        e_ids.append(edge.id)
        for pt in edge.polyline:
            if not all_pts or np.linalg.norm(pt - all_pts[-1]) > 0.1:
                all_pts.append(pt)

    poly = np.array(all_pts, dtype=np.float64)
    diffs = np.diff(poly, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.insert(np.cumsum(dists), 0, 0.0)
    return poly, arc, e_ids


def project_point_onto_path(
    point: np.ndarray,
    path_pts: np.ndarray,
    path_arc: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Orthogonally projects 2D point [p_N, p_E] onto polyline path.
    Returns:
        (arc_length_s, cross_track_dist, tangent_heading_deg, curvature_kappa)
    """
    best_dist = float("inf")
    best_s = 0.0
    best_hdg = 0.0
    best_seg_idx = 0

    n_segs = len(path_pts) - 1
    for i in range(n_segs):
        p1 = path_pts[i]
        p2 = path_pts[i + 1]
        proj, dist = FixedLagHMMMapMatcher.project_point_to_segment(point, p1, p2)
        if dist < best_dist:
            best_dist = dist
            best_seg_idx = i
            seg_len = path_arc[i + 1] - path_arc[i]
            t = float(np.linalg.norm(proj - p1) / max(seg_len, 1e-6))
            best_s = float(path_arc[i] + t * seg_len)
            d_N = p2[0] - p1[0]
            d_E = p2[1] - p1[1]
            best_hdg = math.degrees(math.atan2(d_E, d_N)) % 360.0

    # Local curvature approximation: d(psi)/ds between adjacent segments
    kappa = 0.0
    if 0 < best_seg_idx < n_segs - 1:
        prev_N = path_pts[best_seg_idx][0] - path_pts[best_seg_idx - 1][0]
        prev_E = path_pts[best_seg_idx][1] - path_pts[best_seg_idx - 1][1]
        prev_hdg = math.degrees(math.atan2(prev_E, prev_N)) % 360.0

        next_N = path_pts[best_seg_idx + 1][0] - path_pts[best_seg_idx][0]
        next_E = path_pts[best_seg_idx + 1][1] - path_pts[best_seg_idx][1]
        next_hdg = math.degrees(math.atan2(next_E, next_N)) % 360.0

        d_psi = wrap_to_pi(math.radians(next_hdg - prev_hdg))
        d_s = max(float(path_arc[best_seg_idx + 1] - path_arc[best_seg_idx - 1]), 1.0)
        kappa = float(d_psi / d_s)

    return best_s, best_dist, best_hdg, kappa


def classify_road_geometry(
    kappa: float,
    s_curr: float,
    path_arc: np.ndarray,
    all_kappas: np.ndarray,
    is_junction: bool,
    is_roundabout: bool,
    is_parallel: bool
) -> Tuple[str, str]:
    """
    Classifies local road geometry:
    STRAIGHT, CURVE, CURVE ENTRY, CURVE EXIT, INTERSECTION, ROUNDABOUT, OVERPASS/PARALLEL, OTHER
    """
    if is_roundabout:
        return "ROUNDABOUT", "OSM tagged circular loop / roundabout"
    if is_junction:
        return "INTERSECTION", "Node degree >= 3 junction within 25m"
    if is_parallel:
        return "OVERPASS/PARALLEL", "Multi-lane parallel carriageway within 15m"

    abs_kap = abs(kappa)

    # Find where curves occur along the path
    curve_mask = np.abs(all_kappas) >= 0.002  # R <= 500m
    if abs_kap >= 0.002:
        return "CURVE", f"High curvature |kappa|={abs_kap:.4f} rad/m (R={1.0/abs_kap:.0f}m)"

    if len(curve_mask) > 0 and np.any(curve_mask):
        curve_s_coords = path_arc[:-1][curve_mask]
        dist_to_curves = curve_s_coords - s_curr
        # Upcoming curve within 30m: CURVE ENTRY
        if np.any((dist_to_curves > 0) & (dist_to_curves <= 30.0)):
            return "CURVE ENTRY", "Transition into curve within 30m"
        # Just exited curve within 30m: CURVE EXIT
        if np.any((dist_to_curves < 0) & (dist_to_curves >= -30.0)):
            return "CURVE EXIT", "Transition out of curve within 30m"

    if abs_kap < 0.001:
        return "STRAIGHT", f"Tangential straight |kappa|={abs_kap:.4f} rad/m (R > 1000m)"

    return "OTHER", "Moderate curvature transition"


def run_phase8d_experiment():
    print("="*80, flush=True)
    print("PHASE 8D: ALONG-TRACK MAP OBSERVABILITY EXPERIMENT", flush=True)
    print("Evaluating Along-Track Observability & Road Geometry Information", flush=True)
    print("="*80, flush=True)

    # 1. Load Road Graph from OSM
    osm_path = config.BASE_DIR / "data" / "osm_uk_all_test_roads.json"
    if not osm_path.exists():
        osm_path = config.BASE_DIR / "data" / "osm_uk_test_roads.json"
    print(f"Loading canonical RoadGraph from {osm_path}...", flush=True)
    graph = load_road_graph_from_osm_json(str(osm_path), ref_lat=52.202, ref_lon=-2.192)
    print(f"Loaded {len(graph.edges)} road edges across {len(graph.nodes)} nodes.", flush=True)

    # 2. Load 4s VelocityNet Model
    model_path = config.MODELS_DIR / "phase8a_4s_velocity_net.pt"
    model = VelocityNet4s(in_channels=6, num_events=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    test_files = [
        config.BASE_DIR / "data" / "sample_test_trajectory.parquet",
        config.BASE_DIR / "data" / "sample_test_trajectory_motorway.parquet",
    ]
    for r in ["vw10", "vw11", "vw12", "vw13", "vw14a", "vw14b", "vw14c", "vw16a", "vw16b", "vw17", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8", "vw9"]:
        p = config.SYNC_PROCESSED_DIR / f"sync_{r}.parquet"
        if p.exists():
            test_files.append(p)

    cols = [config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z, config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]

    window_records = []
    outage_summary_records = []

    print(f"\nBeginning evaluation across {len(test_files)} trajectories...", flush=True)

    for tf in test_files:
        run_name = tf.stem.replace("sync_", "")
        df = pd.read_parquet(tf)
        n = len(df)
        total_dur = n * config.TARGET_DT
        if total_dur < 30.0:
            continue

        schedule = generate_outage_schedule(total_dur, run_id=run_name)
        if not schedule:
            continue

        df_sim, df_outages = inject_outages(df, schedule)

        # Alignment
        run_align = AlignmentEngine()
        warmup = df[df[config.COL_TIME] < min(45.0, total_dur * 0.2)]
        if len(warmup) >= 25:
            acc_w = warmup[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            mean_a = np.mean(acc_w, axis=0)
            u_z = -mean_a / np.linalg.norm(mean_a)
            ref = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            y = np.cross(u_z, ref)
            y /= np.linalg.norm(y)
            x = np.cross(y, u_z)
            run_align.result.R_b_to_v = np.vstack([x, y, u_z])
            run_align.result.state = AlignmentState.FULL_ALIGNED
            run_align.result.confidence = 1.0

        for c in cols:
            df[c] = df[c].interpolate().bfill().ffill().fillna(0.0)

        seq_data = df[cols].values.astype(np.float32)

        needed_indices = set(k for k in range(0, n, 5) if k >= 39)
        for _, row in df_outages.iterrows():
            mask = (df_sim["outage_id"] == int(row["outage_id"]))
            s_idx = mask.idxmax()
            sub_len = len(df.loc[mask])
            for k in range(0, sub_len, 5):
                if s_idx + k >= 39:
                    needed_indices.add(s_idx + k)

        lookup_preds = compute_batched_velocitynet_predictions(
            model=model,
            seq_data=seq_data,
            needed_step_indices=list(needed_indices),
            window_len=40,
            batch_size=2048,
            device="cpu"
        )

        for _, row in df_outages.iterrows():
            oid = int(row["outage_id"])
            dur = int(row["duration_s"])
            mask = (df_sim["outage_id"] == oid)
            df_sub = df.loc[mask]
            if len(df_sub) < 5:
                continue

            outage_start_idx = mask.idxmax()
            n_out = len(df_sub)

            # Build reference traversed road polyline path from GT
            gt_lats = df_sub[config.COL_TRUE_LAT].values
            gt_lons = df_sub[config.COL_TRUE_LON].values
            gt_speeds = df_sub[config.COL_TRUE_SPEED_MS].values
            gt_hdgs = df_sub[config.COL_TRUE_HEADING].values if config.COL_TRUE_HEADING in df_sub.columns else None

            path_pts, path_arc, road_e_ids = build_traversed_road_path(graph, gt_lats, gt_lons)

            # Precalculate path segment curvatures
            path_kappas = []
            for si in range(len(path_pts) - 1):
                _, _, _, kap = project_point_onto_path(path_pts[si], path_pts, path_arc)
                path_kappas.append(kap)
            path_kappas = np.array(path_kappas)

            # Pre-outage filter state
            pre_idx_start = max(0, outage_start_idx - 10)
            pre_df = df.iloc[pre_idx_start:outage_start_idx]
            if len(pre_df) > 0:
                init_lat = float(pre_df[config.COL_TRUE_LAT].iloc[-1])
                init_lon = float(pre_df[config.COL_TRUE_LON].iloc[-1])
                init_spd = float(pre_df[config.COL_TRUE_SPEED_MS].mean())
                h_rads = np.radians(pre_df[config.COL_TRUE_HEADING].values)
                init_hdg = float(np.degrees(np.arctan2(np.mean(np.sin(h_rads)), np.mean(np.cos(h_rads)))) % 360.0)
            else:
                init_lat = float(gt_lats[0])
                init_lon = float(gt_lons[0])
                init_spd = float(gt_speeds[0])
                init_hdg = float(gt_hdgs[0]) if gt_hdgs is not None else 0.0

            # Onset origin: project initial point onto path to find s_0
            p_onset_local = graph.latlon_to_local(init_lat, init_lon)
            s_0, _, _, _ = project_point_onto_path(p_onset_local, path_pts, path_arc)

            # -------------------------------------------------------------
            # CONDITION A: UKF Only (No Map Matching)
            # -------------------------------------------------------------
            ukf_a = UKFNavigationFilter()
            ukf_a.initialize(init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_spd, init_heading_rad=math.radians(init_hdg))

            # -------------------------------------------------------------
            # CONDITION B: Existing UKF + Existing Cross-Track / Heading Map Matcher
            # -------------------------------------------------------------
            ukf_b = UKFNavigationFilter()
            ukf_b.initialize(init_lat=init_lat, init_lon=init_lon, init_speed_ms=init_spd, init_heading_rad=math.radians(init_hdg))
            matcher_b = FixedLagHMMMapMatcher(graph=graph)

            acc_raw = df_sub[[config.COL_ACC_X, config.COL_ACC_Y, config.COL_ACC_Z]].values
            gyro_raw = df_sub[[config.COL_GYRO_X, config.COL_GYRO_Y, config.COL_GYRO_Z]].values

            lat_a_arr = np.zeros(n_out)
            lon_a_arr = np.zeros(n_out)
            lat_b_arr = np.zeros(n_out)
            lon_b_arr = np.zeros(n_out)

            outage_windows = []

            for k in range(n_out):
                t_rel = k * config.TARGET_DT
                a_b = acc_raw[k]
                w_b = gyro_raw[k]
                a_v, w_v = run_align.transform_imu(a_b, w_b)

                # Predict
                ukf_a.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=config.TARGET_DT)
                ukf_b.predict(acc_fwd=a_v[0], gyro_yaw=w_v[2], dt=config.TARGET_DT)

                global_k = outage_start_idx + k

                # Apply VelocityNet 4s measurement every 5 steps
                if global_k in lookup_preds and (k % 5 == 0):
                    raw_d, ev_class, raw_sig = lookup_preds[global_k]
                    cal_d = raw_d / 2.0
                    cal_sig = max(raw_sig / 2.0, 0.2)

                    if ev_class == 0 or cal_d < 0.1:
                        ukf_a.update_zupt(gyro_reading=w_v[2])
                        ukf_b.update_zupt(gyro_reading=w_v[2])
                    else:
                        ukf_a.update_velocity_net(delta_d_pred=cal_d, sigma_pred=cal_sig, event_class=ev_class, window_dur=2.0)
                        ukf_b.update_velocity_net(delta_d_pred=cal_d, sigma_pred=cal_sig, event_class=ev_class, window_dur=2.0)

                    # Map matching update for Condition B
                    p_N_curr = float(ukf_b.x[0])
                    p_E_curr = float(ukf_b.x[1])
                    curr_lat, curr_lon = local_xy_to_latlon(p_E_curr, p_N_curr, init_lat, init_lon)
                    p_graph = graph.latlon_to_local(float(curr_lat), float(curr_lon))
                    hdg_b_deg = math.degrees(ukf_b.x[3]) % 360.0

                    match_res = matcher_b.match(
                        point_xy=p_graph,
                        heading_deg=hdg_b_deg,
                        speed_ms=float(ukf_b.x[2]),
                        travel_dist_m=max(float(ukf_b.x[2]) * 0.5, 0.05),
                        sigma_pos_m=float(math.sqrt(max(0.01, max(ukf_b.P[0, 0], ukf_b.P[1, 1])))),
                        dt=0.5
                    )

                    if not match_res.is_off_road and match_res.confidence >= 0.25:
                        snap_lat, snap_lon = graph.local_to_latlon(match_res.snapped_point[0], match_res.snapped_point[1])
                        d_lat = math.radians(snap_lat - init_lat)
                        d_lon = math.radians(snap_lon - init_lon)
                        p_N_match = float(6371000.0 * d_lat)
                        p_E_match = float(6371000.0 * d_lon * math.cos(math.radians(init_lat)))

                        ukf_b.update_map_match(
                            p_N_match=p_N_match,
                            p_E_match=p_E_match,
                            psi_road=float(match_res.road_heading_rad),
                            confidence=float(match_res.confidence),
                            is_heading_valid=bool(match_res.is_heading_valid)
                        )

                    # ---------------------------------------------------------
                    # DIAGNOSTIC C: ALONG-TRACK ARC-LENGTH OBSERVATIONS
                    # ---------------------------------------------------------
                    # 1. Ground truth arc length s_GT
                    p_gt_local = graph.latlon_to_local(gt_lats[k], gt_lons[k])
                    s_gt_abs, _, _, _ = project_point_onto_path(p_gt_local, path_pts, path_arc)
                    s_GT = max(0.0, s_gt_abs - s_0)

                    # 2. UKF arc length s_UKF (from Condition B)
                    p_ukf_b_local = graph.latlon_to_local(curr_lat, curr_lon)
                    s_ukf_abs, _, _, _ = project_point_onto_path(p_ukf_b_local, path_pts, path_arc)
                    s_UKF = max(0.0, s_ukf_abs - s_0)

                    # 3. Map-snapped arc length s_map
                    p_snap_local = match_res.snapped_point
                    s_map_abs, cross_track_m, psi_map, kappa = project_point_onto_path(p_snap_local, path_pts, path_arc)
                    s_map = max(0.0, s_map_abs - s_0)

                    e_map = s_map - s_GT
                    e_UKF = s_UKF - s_GT
                    psi_UKF = hdg_b_deg
                    hdg_err = math.degrees(wrap_to_pi(math.radians(psi_UKF - psi_map)))

                    # Road classification
                    is_junc = len(match_res.matched_segment.name) > 0 and ("Junction" in match_res.matched_segment.name or "Roundabout" in match_res.matched_segment.name)
                    is_round = "roundabout" in match_res.matched_segment.name.lower()
                    is_par = "M5" in match_res.matched_segment.name or "M42" in match_res.matched_segment.name or "M40" in match_res.matched_segment.name

                    geom_class, geom_reason = classify_road_geometry(
                        kappa=kappa,
                        s_curr=s_map_abs,
                        path_arc=path_arc,
                        all_kappas=path_kappas,
                        is_junction=is_junc,
                        is_roundabout=is_round,
                        is_parallel=is_par
                    )

                    win_rec = {
                        "run_id": run_name,
                        "outage_id": oid,
                        "duration_s": dur,
                        "timestamp": round(t_rel, 2),
                        "geometry_class": geom_class,
                        "road_id": match_res.matched_segment.edge_id,
                        "s_map": round(s_map, 2),
                        "s_GT": round(s_GT, 2),
                        "s_UKF": round(s_UKF, 2),
                        "e_map": round(e_map, 2),
                        "e_UKF": round(e_UKF, 2),
                        "psi_map": round(psi_map, 2),
                        "psi_UKF": round(psi_UKF, 2),
                        "heading_error": round(hdg_err, 2),
                        "curvature": round(kappa, 5),
                        "gyro_yaw_rate": round(float(w_v[2]), 4),
                        "HMM_confidence": round(float(match_res.confidence), 3),
                        "cross_track": round(float(cross_track_m), 2)
                    }
                    window_records.append(win_rec)
                    outage_windows.append(win_rec)

                # Track coordinates for trajectory scoring
                curr_a_lat, curr_a_lon = local_xy_to_latlon(float(ukf_a.x[1]), float(ukf_a.x[0]), init_lat, init_lon)
                curr_b_lat, curr_b_lon = local_xy_to_latlon(float(ukf_b.x[1]), float(ukf_b.x[0]), init_lat, init_lon)
                lat_a_arr[k] = curr_a_lat
                lon_a_arr[k] = curr_a_lon
                lat_b_arr[k] = curr_b_lat
                lon_b_arr[k] = curr_b_lon

            # Score Condition A vs Condition B
            sc_a = score_outage_segment(lat_a_arr, lon_a_arr, gt_lats, gt_lons, distance_travelled_m=row["distance_travelled_m"], duration_s=dur, gt_heading_deg=gt_hdgs)
            sc_b = score_outage_segment(lat_b_arr, lon_b_arr, gt_lats, gt_lons, distance_travelled_m=row["distance_travelled_m"], duration_s=dur, gt_heading_deg=gt_hdgs)

            # Map along-track error stats for this outage
            if outage_windows:
                df_ow = pd.DataFrame(outage_windows)
                map_mae = float(df_ow["e_map"].abs().mean())
                ukf_mae = float(df_ow["e_UKF"].abs().mean())
                corr_s = float(df_ow[["s_map", "s_GT"]].corr().iloc[0, 1]) if len(df_ow) > 2 else 1.0
            else:
                map_mae = sc_b["final_pos_error_m"]
                ukf_mae = sc_a["final_pos_error_m"]
                corr_s = 1.0

            outage_summary_records.append({
                "run": run_name,
                "outage_id": oid,
                "duration_s": dur,
                "dist_gt_m": round(float(row["distance_travelled_m"]), 2),
                # Condition A: UKF Only
                "fpe_a_m": round(float(sc_a["final_pos_error_m"]), 2),
                "drift_a_pct": round(float(sc_a["pct_of_distance"]), 2),
                "along_a_m": round(float(sc_a["along_track_m"]), 2),
                "cross_a_m": round(float(sc_a["cross_track_m"]), 2),
                "hdg_a_deg": round(float(sc_a["heading_error_deg"]), 2),
                # Condition B: UKF + Existing Map Matching
                "fpe_b_m": round(float(sc_b["final_pos_error_m"]), 2),
                "drift_b_pct": round(float(sc_b["pct_of_distance"]), 2),
                "along_b_m": round(float(sc_b["along_track_m"]), 2),
                "cross_b_m": round(float(sc_b["cross_track_m"]), 2),
                "hdg_b_deg": round(float(sc_b["heading_error_deg"]), 2),
                # Diagnostic C: Along-Track Map Diagnostic
                "map_mae_m": round(map_mae, 2),
                "ukf_mae_m": round(ukf_mae, 2),
                "corr_s": round(corr_s, 4)
            })

        print(f"Finished {run_name} (Windows logged so far: {len(window_records)})", flush=True)

    # Convert to DataFrames
    df_windows = pd.DataFrame(window_records)
    df_outages = pd.DataFrame(outage_summary_records)

    # Save CSV artifact
    csv_path = config.BASE_DIR / "eval" / "phase8d_window_observability.csv"
    df_windows.to_csv(csv_path, index=False)
    print(f"\nSaved per-window observability dataset ({len(df_windows)} rows) to {csv_path}", flush=True)

    # =========================================================================
    # PLOTS GENERATION
    # =========================================================================
    viz_dir = config.BASE_DIR / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating 7 required Phase 8D observability plots in {viz_dir}...", flush=True)

    # Plot A: s_GT vs s_UKF
    plt.figure(figsize=(7, 6))
    plt.scatter(df_windows["s_GT"], df_windows["s_UKF"], alpha=0.3, s=10, c="tab:blue", label="UKF Along-Track")
    max_s = max(df_windows["s_GT"].max(), df_windows["s_UKF"].max())
    plt.plot([0, max_s], [0, max_s], "k--", label="Ideal 1:1 Ground Truth")
    plt.xlabel("Ground Truth Arc-Length $s_{GT}$ (m)")
    plt.ylabel("UKF Estimated Arc-Length $s_{UKF}$ (m)")
    plt.title("Plot A: $s_{GT}$ vs $s_{UKF}$ (Along-Track Progression)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_s_gt_vs_s_ukf.png", dpi=150)
    plt.close()

    # Plot B: s_GT vs s_map
    plt.figure(figsize=(7, 6))
    plt.scatter(df_windows["s_GT"], df_windows["s_map"], alpha=0.3, s=10, c="tab:green", label="Map-Matched Along-Track")
    max_sm = max(df_windows["s_GT"].max(), df_windows["s_map"].max())
    plt.plot([0, max_sm], [0, max_sm], "k--", label="Ideal 1:1 Ground Truth")
    plt.xlabel("Ground Truth Arc-Length $s_{GT}$ (m)")
    plt.ylabel("Map Snapped Arc-Length $s_{map}$ (m)")
    plt.title("Plot B: $s_{GT}$ vs $s_{map}$ (Map-Derived Along-Track Progression)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_s_gt_vs_s_map.png", dpi=150)
    plt.close()

    # Plot C: Along-Track Error vs Time
    plt.figure(figsize=(8, 5))
    plt.scatter(df_windows["timestamp"], df_windows["e_UKF"], alpha=0.2, s=8, c="tab:blue", label="UKF Error $e_{UKF}$")
    plt.scatter(df_windows["timestamp"], df_windows["e_map"], alpha=0.2, s=8, c="tab:red", label="Map Error $e_{map}$")
    plt.axhline(0, color="k", linestyle="--", alpha=0.6)
    plt.xlabel("Elapsed Outage Time (s)")
    plt.ylabel("Along-Track Error (m)")
    plt.title("Plot C: Along-Track Position Error ($e = s - s_{GT}$) vs Outage Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_along_track_error_vs_time.png", dpi=150)
    plt.close()

    # Plot D: Map along-track error by geometry class
    plt.figure(figsize=(9, 5))
    geom_order = ["STRAIGHT", "CURVE", "CURVE ENTRY", "CURVE EXIT", "OVERPASS/PARALLEL", "OTHER"]
    data_d = [df_windows[df_windows["geometry_class"] == g]["e_map"].abs().dropna().values for g in geom_order if g in df_windows["geometry_class"].values]
    labels_d = [g for g in geom_order if g in df_windows["geometry_class"].values]
    plt.boxplot(data_d, tick_labels=labels_d, showfliers=False)
    plt.ylabel("Absolute Map Error $|e_{map}|$ (m)")
    plt.title("Plot D: Map Along-Track Absolute Error by Road Geometry Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_map_along_track_error_by_geometry.png", dpi=150)
    plt.close()

    # Plot E: UKF along-track error by geometry class
    plt.figure(figsize=(9, 5))
    data_e = [df_windows[df_windows["geometry_class"] == g]["e_UKF"].abs().dropna().values for g in geom_order if g in df_windows["geometry_class"].values]
    plt.boxplot(data_e, tick_labels=labels_d, showfliers=False)
    plt.ylabel("Absolute UKF Error $|e_{UKF}|$ (m)")
    plt.title("Plot E: UKF Along-Track Absolute Error by Road Geometry Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_ukf_along_track_error_by_geometry.png", dpi=150)
    plt.close()

    # Plot F: Heading profile vs road heading profile on representative curve
    curve_df = df_windows[df_windows["geometry_class"] == "CURVE"]
    if not curve_df.empty:
        rep_run = curve_df["run_id"].iloc[0]
        rep_oid = curve_df["outage_id"].iloc[0]
        rep_sub = df_windows[(df_windows["run_id"] == rep_run) & (df_windows["outage_id"] == rep_oid)]
    else:
        rep_sub = df_windows.head(50)

    plt.figure(figsize=(8, 4))
    plt.plot(rep_sub["timestamp"], rep_sub["psi_map"], "g-", label="Road Heading $\psi_{map}(s)$")
    plt.plot(rep_sub["timestamp"], rep_sub["psi_UKF"], "b--", label="UKF Heading $\psi_{UKF}(t)$")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Heading (deg)")
    plt.title(f"Plot F: Heading Profile Comparison on Representative Outage ({rep_sub['run_id'].iloc[0]})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_heading_profile_representative_curves.png", dpi=150)
    plt.close()

    # Plot G: Curvature profile vs observed yaw-rate
    plt.figure(figsize=(8, 4))
    plt.plot(rep_sub["timestamp"], rep_sub["curvature"], "r-", label="Road Curvature $\kappa(s)$ (rad/m)")
    # Normalize yaw rate by dividing by speed approx ~15 m/s to get vehicle curvature
    plt.plot(rep_sub["timestamp"], rep_sub["gyro_yaw_rate"] / 15.0, "m--", label="Observed Gyro Curvature $\omega_z / v$ (rad/m)")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Curvature (rad/m)")
    plt.title(f"Plot G: Road Curvature vs Gyro Curvature on Representative Outage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "phase8d_curvature_profile_representative_curves.png", dpi=150)
    plt.close()

    print("All 7 plots generated successfully.", flush=True)

    # =========================================================================
    # SUMMARY TABLES & REPORT GENERATION
    # =========================================================================
    print("\n" + "="*80, flush=True)
    print("1. SUMMARY BY ROAD GEOMETRY CLASS", flush=True)
    print("="*80, flush=True)
    geom_stats = []
    for g in geom_order:
        sub = df_windows[df_windows["geometry_class"] == g]
        if len(sub) == 0:
            continue
        e_m = sub["e_map"].abs()
        e_u = sub["e_UKF"].abs()
        geom_stats.append({
            "Geometry Class": g,
            "N": len(sub),
            "Map MAE (m)": round(float(e_m.mean()), 2),
            "Map Med (m)": round(float(e_m.median()), 2),
            "Map P75 (m)": round(float(np.percentile(e_m, 75)), 2),
            "Map P95 (m)": round(float(np.percentile(e_m, 95)), 2),
            "UKF MAE (m)": round(float(e_u.mean()), 2),
            "UKF Med (m)": round(float(e_u.median()), 2),
            "UKF P75 (m)": round(float(np.percentile(e_u, 75)), 2),
            "UKF P95 (m)": round(float(np.percentile(e_u, 95)), 2),
            "Improvement (m)": round(float(e_u.mean() - e_m.mean()), 2),
            "Heading Res (deg)": round(float(sub["heading_error"].abs().mean()), 2),
            "HMM Conf": round(float(sub["HMM_confidence"].mean()), 3),
        })
    df_geom_summary = pd.DataFrame(geom_stats)
    print(df_geom_summary.to_string(index=False), flush=True)

    print("\n" + "="*80, flush=True)
    print("2. SUMMARY BY OUTAGE DURATION", flush=True)
    print("="*80, flush=True)
    dur_stats = []
    for d in [10, 30, 60, 120, 180]:
        sub = df_windows[df_windows["duration_s"] == d]
        if len(sub) == 0:
            continue
        e_m = sub["e_map"].abs()
        e_u = sub["e_UKF"].abs()
        dur_stats.append({
            "Duration (s)": d,
            "N Windows": len(sub),
            "Map Median Error (m)": round(float(e_m.median()), 2),
            "Map P95 Error (m)": round(float(np.percentile(e_m, 95)), 2),
            "UKF Median Error (m)": round(float(e_u.median()), 2),
            "UKF P95 Error (m)": round(float(np.percentile(e_u, 95)), 2),
            "Correlation s_map, s_GT": round(float(sub[["s_map", "s_GT"]].corr().iloc[0, 1]), 4),
            "Correlation s_UKF, s_GT": round(float(sub[["s_UKF", "s_GT"]].corr().iloc[0, 1]), 4),
        })
    df_dur_summary = pd.DataFrame(dur_stats)
    print(df_dur_summary.to_string(index=False), flush=True)

    print("\n" + "="*80, flush=True)
    print("3. CONTROL EXPERIMENT: UKF ONLY vs UKF + EXISTING MAP vs MAP DIAGNOSTIC", flush=True)
    print("="*80, flush=True)
    ctrl_summary = df_outages.groupby("duration_s").agg({
        "fpe_a_m": "median",
        "fpe_b_m": "median",
        "drift_a_pct": "median",
        "drift_b_pct": "median",
        "cross_a_m": "median",
        "cross_b_m": "median",
        "hdg_a_deg": "median",
        "hdg_b_deg": "median",
        "map_mae_m": "median",
        "ukf_mae_m": "median",
    })
    print(ctrl_summary.to_string(), flush=True)

    # Write Markdown Report
    rep_path = config.BASE_DIR / "eval" / "phase8d_observability_report.md"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# PHASE 8D: ALONG-TRACK MAP OBSERVABILITY EXPERIMENT REPORT\n\n")
        f.write("## 1. Executive Summary & Final Decision\n\n")
        f.write("### Research Hypothesis Evaluation:\n")
        f.write("- **H0**: Map road geometry does not provide sufficiently accurate along-track position information beyond UKF + VelocityNet.\n")
        f.write("- **H1**: Distinctive road geometry, especially curves and verified transitions, provides additional along-track observability.\n\n")
        f.write("## 2. Road Geometry Classification & Performance Table\n\n")
        f.write(df_geom_summary.to_markdown(index=False))
        f.write("\n\n## 3. Performance by Outage Duration\n\n")
        f.write(df_dur_summary.to_markdown(index=False))
        f.write("\n\n## 4. Control Experiment (Condition A vs B vs Diagnostic C)\n\n")
        f.write(ctrl_summary.to_markdown())
        f.write("\n")
    print(f"\nWrote markdown report to {rep_path}", flush=True)


if __name__ == "__main__":
    run_phase8d_experiment()
