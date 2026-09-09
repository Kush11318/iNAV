"""
iNAV Evaluation Scoring Module
Calculates standard navigation benchmark metrics:
- ATE (Absolute Trajectory Error)
- FPE (Final Position Error)
- Drift % (FPE / distance travelled * 100%)
- CEP50 & CEP95 (Circular Error Probable 50th and 95th percentiles)
- Along-track vs Cross-track error decomposition
- Heading error
- Updates results/leaderboard.csv
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.score")

EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equirectangular projection from (lat, lon) in degrees to local Cartesian (x_east, y_north) in meters.
    """
    R = 6371000.0
    phi0 = np.radians(lat0)
    x = R * np.radians(lon - lon0) * np.cos(phi0)
    y = R * np.radians(lat - lat0)
    return x, y


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute great-circle haversine distance between coordinates on spherical Earth in meters."""
    R = 6371000.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)

    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R * c


def wrap_heading_error(pred_heading_deg: float, gt_heading_deg: float) -> float:
    """Compute shortest angular difference between two headings in degrees, wrapped to [0.0, 180.0]."""
    if pd.isna(pred_heading_deg) or pd.isna(gt_heading_deg):
        return np.nan
    diff = (pred_heading_deg - gt_heading_deg + 180.0) % 360.0 - 180.0
    return float(abs(diff))


def decompose_along_cross_track(
    pred_lat: float,
    pred_lon: float,
    gt_lat: float,
    gt_lon: float,
    track_heading_deg: float,
) -> Tuple[float, float]:
    """Decompose endpoint position error into along-track and cross-track components in local ENU frame."""
    if pd.isna(pred_lat) or pd.isna(pred_lon) or pd.isna(gt_lat) or pd.isna(gt_lon) or pd.isna(track_heading_deg):
        return np.nan, np.nan

    R = 6371000.0
    e_E = (pred_lon - gt_lon) * (np.pi / 180.0) * R * np.cos(np.deg2rad(gt_lat))
    e_N = (pred_lat - gt_lat) * (np.pi / 180.0) * R

    psi = np.deg2rad(track_heading_deg)
    along_track = e_E * np.sin(psi) + e_N * np.cos(psi)
    cross_track = e_E * np.cos(psi) - e_N * np.sin(psi)
    return float(along_track), float(cross_track)


def compute_trajectory_errors(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
    gt_heading_deg: Optional[np.ndarray] = None,
    pred_heading_deg: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute 7 locked navigation metrics between predicted and ground truth tracks.
    """
    mask = ~(np.isnan(pred_lat) | np.isnan(pred_lon) | np.isnan(gt_lat) | np.isnan(gt_lon))
    if np.sum(mask) == 0:
        return {
            "final_pos_error_m": np.nan, "fpe_m": np.nan, "ate_m": np.nan,
            "cep50_m": np.nan, "cep95_m": np.nan,
            "along_track_m": np.nan, "cross_track_m": np.nan,
            "heading_error_deg": np.nan, "valid_samples": 0
        }

    p_lat, p_lon = pred_lat[mask], pred_lon[mask]
    g_lat, g_lon = gt_lat[mask], gt_lon[mask]

    # Radial point-wise errors via Haversine
    radial_errors = haversine_distance(p_lat, p_lon, g_lat, g_lon)

    ate = float(np.mean(radial_errors))
    fpe = float(radial_errors[-1])
    cep50 = float(np.percentile(radial_errors, 50))
    cep95 = float(np.percentile(radial_errors, 95))

    along_track = np.nan
    cross_track = np.nan
    hdg_err = np.nan

    if gt_heading_deg is not None:
        last_hdg = float(gt_heading_deg[mask][-1])
        along_track, cross_track = decompose_along_cross_track(p_lat[-1], p_lon[-1], g_lat[-1], g_lon[-1], last_hdg)
        if pred_heading_deg is not None:
            hdg_err = wrap_heading_error(float(pred_heading_deg[mask][-1]), last_hdg)
        elif len(p_lat) >= 2:
            dE = (p_lon[-1] - p_lon[-2]) * (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(g_lat[-1]))
            dN = (p_lat[-1] - p_lat[-2]) * (np.pi / 180.0) * EARTH_RADIUS_M
            if np.hypot(dE, dN) > 0.05:
                est_h = float(np.degrees(np.arctan2(dE, dN)) % 360.0)
                hdg_err = wrap_heading_error(est_h, last_hdg)

    return {
        "final_pos_error_m": fpe,
        "fpe_m": fpe,
        "ate_m": ate,
        "cep50_m": cep50,
        "cep95_m": cep95,
        "along_track_m": along_track,
        "cross_track_m": cross_track,
        "heading_error_deg": hdg_err,
        "valid_samples": int(np.sum(mask))
    }


def score_outage_segment(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
    distance_travelled_m: float,
    duration_s: float,
    gt_heading_deg: Optional[np.ndarray] = None,
    pred_heading_deg: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Score dead-reckoning performance during a single blackout segment.
    """
    metrics = compute_trajectory_errors(pred_lat, pred_lon, gt_lat, gt_lon, gt_heading_deg, pred_heading_deg)
    fpe = metrics["final_pos_error_m"]

    # Drift % of true path distance
    pct_dist = (fpe / max(distance_travelled_m, 1.0)) * 100.0 if not np.isnan(fpe) else np.nan

    metrics["duration_s"] = duration_s
    metrics["outage_s"] = int(duration_s)
    metrics["distance_m"] = distance_travelled_m
    metrics["true_distance_m"] = distance_travelled_m
    metrics["pct_of_distance"] = float(pct_dist)
    metrics["drift_percent"] = float(pct_dist)

    return metrics


def record_instance_leaderboard(
    instance_scores: List[Dict],
    csv_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Record individual instance scores to 16-column leaderboard.csv matching iDead schema.
    """
    if csv_path is None:
        csv_path = config.LEADERBOARD_PATH

    records = []
    for sc in instance_scores:
        records.append({
            "run_id": sc.get("run", sc.get("run_id", "unknown")),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": sc.get("method", sc.get("config", "inav_ukf")),
            "outage_s": int(sc.get("outage_s", sc.get("duration_s", 0))),
            "final_pos_error_m": round(sc.get("final_pos_error_m", sc.get("fpe_m", np.nan)), 4),
            "pct_of_distance": round(sc.get("pct_of_distance", sc.get("drift_percent", np.nan)), 4),
            "cep50_m": round(sc.get("cep50_m", np.nan), 4),
            "cep95_m": round(sc.get("cep95_m", np.nan), 4),
            "along_track_m": round(sc.get("along_track_m", np.nan), 4),
            "cross_track_m": round(sc.get("cross_track_m", np.nan), 4),
            "heading_error_deg": round(sc.get("heading_error_deg", np.nan), 4),
            "outage_id": sc.get("outage_id", ""),
            "split": sc.get("split", "test"),
            "source_side": sc.get("source_side", "phone"),
            "has_ground_truth": True,
            "skip_reason": sc.get("skip_reason", "")
        })

    df_new = pd.DataFrame(records)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            df_old = pd.read_csv(csv_path)
            # Match schema
            common_cols = [c for c in config.LEADERBOARD_COLUMNS if c in df_old.columns and c in df_new.columns]
            if len(common_cols) >= 10:
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    # Enforce column order
    ordered_cols = [c for c in config.LEADERBOARD_COLUMNS if c in df_all.columns]
    df_all = df_all[ordered_cols]
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Recorded {len(df_new)} instances to leaderboard at {csv_path}")

    # Generate grouped summary table
    update_leaderboard_summary(df_all)
    return df_all


def update_leaderboard_summary(
    df_leaderboard: pd.DataFrame,
    summary_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate grouped summary statistics (median & p95) per (config, outage_s)
    matching iDead's results/leaderboard_summary.csv.
    """
    if summary_path is None:
        summary_path = config.LEADERBOARD_SUMMARY_PATH

    summary_rows = []
    for (cfg, dur), grp in df_leaderboard.groupby(["config", "outage_s"]):
        valid = grp.dropna(subset=["final_pos_error_m"])
        if len(valid) == 0:
            continue

        summary_rows.append({
            "config": cfg,
            "outage_s": int(dur),
            "n_paired": len(valid),
            "median_final_pos_error_m": round(float(valid["final_pos_error_m"].median()), 4),
            "p95_final_pos_error_m": round(float(valid["final_pos_error_m"].quantile(0.95)), 4),
            "median_pct_of_distance": round(float(valid["pct_of_distance"].median()), 4),
            "p95_pct_of_distance": round(float(valid["pct_of_distance"].quantile(0.95)), 4),
            "median_heading_error_deg": round(float(valid["heading_error_deg"].median()), 4) if "heading_error_deg" in valid.columns else np.nan
        })

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum.sort_values(["config", "outage_s"], inplace=True)
        df_sum.to_csv(summary_path, index=False)
        logger.info(f"Saved leaderboard summary to {summary_path}")
    return df_sum


# Backward compatibility alias
update_leaderboard = record_instance_leaderboard

