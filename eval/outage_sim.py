"""
iNAV GNSS Outage Simulation Module
Simulates GPS blackout scenarios (10s, 30s, 60s, 120s, 180s) on synchronized trajectories
following the WhONet benchmarking protocol.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
logger = logging.getLogger("iNAV.outage_sim")

GNSS_COLUMNS = [
    config.COL_GPS_LAT,
    config.COL_GPS_LON,
    config.COL_GPS_ALT,
    config.COL_GPS_SPEED_MS,
    config.COL_GPS_BEARING,
    config.COL_GPS_ACCURACY,
    config.COL_GPS_SATS
]


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two (lat, lon) pairs in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


import hashlib


def generate_outage_schedule(
    total_duration_sec: float,
    durations: List[int] = config.OUTAGE_DURATIONS_SEC,
    warmup_sec: float = 10.0,
    recovery_sec: float = 10.0,
    run_id: str = "default_run",
    seed: int = 42,
    avoid_real_gaps: Optional[List[Tuple[float, float]]] = None
) -> List[Tuple[float, float, int]]:
    """
    Generate non-overlapping outage time windows: [(start_time_s, end_time_s, duration_s), ...]
    Uses deterministic SHA-256 hash seeding per (seed, run_id, duration, index)
    to match the benchmark standard from iDead / IO-VNBD.
    """
    schedule = []
    margin = warmup_sec
    min_ts = 0.0
    max_ts = total_duration_sec

    # Start with valid span
    earliest_start = min_ts + margin
    occupied_intervals: List[Tuple[float, float]] = []
    if avoid_real_gaps:
        occupied_intervals.extend(avoid_real_gaps)

    for dur in durations:
        dur_f = float(dur)
        latest_start = max_ts - margin - dur_f
        if latest_start < earliest_start:
            continue

        # Available valid domain
        valid_intervals = [(earliest_start, latest_start)]
        for o_start, o_end in occupied_intervals:
            new_valid = []
            for v_a, v_b in valid_intervals:
                c_start = o_start - dur_f
                c_end = o_end
                if c_end <= v_a or c_start >= v_b:
                    new_valid.append((v_a, v_b))
                else:
                    if v_a < c_start:
                        new_valid.append((v_a, c_start))
                    if c_end < v_b:
                        new_valid.append((c_end, v_b))
            valid_intervals = new_valid

        total_valid = sum(b - a for a, b in valid_intervals if b > a)
        if total_valid <= 0.0:
            continue

        # Deterministic SHA-256 RNG
        hasher = hashlib.sha256(f"{seed}_{run_id}_{dur_f}_0".encode("utf-8"))
        inst_seed = int(hasher.hexdigest()[:8], 16)
        rng = np.random.default_rng(inst_seed)

        u = float(rng.uniform(0.0, total_valid))
        accum = 0.0
        chosen_start = valid_intervals[0][0]
        for a, b in valid_intervals:
            seg_len = b - a
            if seg_len <= 0:
                continue
            if accum + seg_len >= u:
                chosen_start = a + (u - accum)
                break
            accum += seg_len

        chosen_start = round(float(np.clip(chosen_start, earliest_start, latest_start)), 1)
        chosen_end = round(chosen_start + dur_f, 1)

        schedule.append((chosen_start, chosen_end, dur))
        # Add buffer around chosen window to occupied intervals
        occupied_intervals.append((max(0.0, chosen_start - 5.0), chosen_end + 5.0))

    schedule.sort(key=lambda x: x[0])
    return schedule


def inject_outages(
    df: pd.DataFrame,
    outage_schedule: List[Tuple[float, float, int]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply outage schedule to synchronized dataframe.
    Blanks GNSS columns during outage periods and adds:
      - 'gnss_available': bool
      - 'outage_id': int (0 for normal GNSS, 1..K for active outage)
    Returns:
      - df_sim: modified DataFrame with masked GNSS
      - df_outages: metadata DataFrame describing each outage
    """
    df_sim = df.copy()
    df_sim["gnss_available"] = True
    df_sim["outage_id"] = 0

    outage_records = []

    for oid, (start_t, end_t, dur) in enumerate(outage_schedule, 1):
        mask = (df_sim[config.COL_TIME] >= start_t) & (df_sim[config.COL_TIME] < end_t)
        if not np.any(mask):
            continue

        df_sim.loc[mask, "gnss_available"] = False
        df_sim.loc[mask, "outage_id"] = oid

        # Blank GNSS readings
        for col in GNSS_COLUMNS:
            if col in df_sim.columns:
                df_sim.loc[mask, col] = np.nan

        # Record outage segment metadata
        sub = df.loc[mask]
        start_lat = float(sub[config.COL_TRUE_LAT].iloc[0])
        start_lon = float(sub[config.COL_TRUE_LON].iloc[0])
        end_lat = float(sub[config.COL_TRUE_LAT].iloc[-1])
        end_lon = float(sub[config.COL_TRUE_LON].iloc[-1])

        straight_line_dist = haversine_distance_m(start_lat, start_lon, end_lat, end_lon)
        cum_dist = float(sub["gt_delta_d_m"].sum()) if "gt_delta_d_m" in sub.columns else straight_line_dist
        mean_speed = float(sub[config.COL_TRUE_SPEED_MS].mean()) if config.COL_TRUE_SPEED_MS in sub.columns else 0.0

        outage_records.append({
            "outage_id": oid,
            "duration_s": dur,
            "start_time_s": start_t,
            "end_time_s": end_t,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon,
            "distance_travelled_m": cum_dist,
            "straight_line_dist_m": straight_line_dist,
            "mean_speed_ms": mean_speed,
            "samples_count": int(np.sum(mask))
        })

    df_outages = pd.DataFrame(outage_records)
    return df_sim, df_outages
