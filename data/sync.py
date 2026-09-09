"""
iNAV Synchronization Module
Time-aligns smartphone sensor stream (S-Dataset) with vehicle CAN/OBD-II
ground truth (V-Dataset) using GPS position matching and speed cross-correlation,
then resamples both onto a uniform 10 Hz grid.
"""

import sys
import glob
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.sync")


def find_coarse_time_offset(df_s: pd.DataFrame, df_v: pd.DataFrame) -> float:
    """
    Find initial coarse time offset between S and V by finding where
    the vehicle position is closest to the first valid GPS point of the phone.
    Returns offset in seconds (t_v - t_s).
    """
    valid_s = df_s.dropna(subset=[config.COL_GPS_LAT, config.COL_GPS_LON, config.COL_TIME])
    valid_v = df_v.dropna(subset=[config.COL_TRUE_LAT, config.COL_TRUE_LON, config.COL_TIME])

    if len(valid_s) == 0 or len(valid_v) == 0:
        return 0.0

    s_lat0 = valid_s[config.COL_GPS_LAT].iloc[0]
    s_lon0 = valid_s[config.COL_GPS_LON].iloc[0]
    s_t0 = valid_s[config.COL_TIME].iloc[0]

    # Euclidean approximation in degrees
    d_lat = valid_v[config.COL_TRUE_LAT] - s_lat0
    d_lon = (valid_v[config.COL_TRUE_LON] - s_lon0) * np.cos(np.radians(s_lat0))
    dist_sq = d_lat**2 + d_lon**2

    closest_idx = dist_sq.idxmin()
    v_t_match = valid_v.loc[closest_idx, config.COL_TIME]

    coarse_offset = float(v_t_match - s_t0)
    return coarse_offset


def find_fine_lag_offset(
    df_s: pd.DataFrame,
    df_v: pd.DataFrame,
    coarse_offset: float,
    search_window_sec: float = 12.0,
    step_sec: float = 0.05
) -> Tuple[float, float]:
    """
    Refine time lag offset by maximizing Pearson correlation between
    smartphone GPS speed and vehicle ground-truth speed around coarse_offset.
    Returns (optimal_offset_sec, max_correlation_r).
    """
    s_time = df_s[config.COL_TIME].values
    v_time = df_v[config.COL_TIME].values

    s_spd = df_s[config.COL_GPS_SPEED_MS].interpolate().bfill().fillna(0).values
    v_spd = df_v[config.COL_TRUE_SPEED_MS].interpolate().bfill().fillna(0).values

    # Determine overlapping evaluation segment (up to 2000s)
    eval_start = max(s_time[0] + coarse_offset, v_time[0]) + 10.0
    eval_end = min(s_time[-1] + coarse_offset, v_time[-1]) - 10.0

    if eval_end - eval_start < 20.0:
        return coarse_offset, 0.0

    eval_end = min(eval_end, eval_start + 2000.0)
    dt = 0.1
    t_eval = np.arange(eval_start, eval_end, dt)

    v_eval_spd = np.interp(t_eval, v_time, v_spd)
    v_std = np.std(v_eval_spd)
    if v_std < 1e-4:
        return coarse_offset, 0.0

    shifts = np.arange(coarse_offset - search_window_sec, coarse_offset + search_window_sec + step_sec, step_sec)
    best_offset = coarse_offset
    best_corr = -1.0

    for shift in shifts:
        s_eval_spd = np.interp(t_eval, s_time + shift, s_spd)
        s_std = np.std(s_eval_spd)
        if s_std < 1e-4:
            continue
        r = float(np.corrcoef(s_eval_spd, v_eval_spd)[0, 1])
        if r > best_corr:
            best_corr = r
            best_offset = float(shift)

    return best_offset, best_corr


def synchronize_pair(run_key: str) -> Optional[pd.DataFrame]:
    """
    Load ingested Parquet files for run_key, align timestamps, resample to 10 Hz,
    and save synchronized Parquet.
    """
    s_path = config.INGESTED_DIR / f"S_{run_key}.parquet"
    v_path = config.INGESTED_DIR / f"V_{run_key}.parquet"

    if not s_path.exists() or not v_path.exists():
        logger.warning(f"Ingested files not found for pair '{run_key}'")
        return None

    df_s = pd.read_parquet(s_path)
    df_v = pd.read_parquet(v_path)

    coarse_offset = find_coarse_time_offset(df_s, df_v)
    fine_offset, corr_r = find_fine_lag_offset(df_s, df_v, coarse_offset)

    chosen_offset = fine_offset if corr_r > 0.3 else coarse_offset
    logger.info(
        f"Pair '{run_key}': Coarse={coarse_offset:.2f}s, Fine={fine_offset:.2f}s (r={corr_r:.3f}), "
        f"Using={chosen_offset:.2f}s"
    )

    # Shift S-time to V timeline
    s_aligned_time = df_s[config.COL_TIME].values + chosen_offset
    v_time = df_v[config.COL_TIME].values

    # Determine overlapping time bounds
    t_start = max(s_aligned_time[0], v_time[0])
    t_end = min(s_aligned_time[-1], v_time[-1])

    if t_end <= t_start + 5.0:
        logger.warning(f"Insufficient overlap for pair '{run_key}': {t_end - t_start:.2f}s")
        return None

    # Resample onto uniform 10 Hz grid
    t_grid = np.arange(t_start, t_end, config.TARGET_DT)
    sync_df = pd.DataFrame({config.COL_TIME: np.round(t_grid - t_start, 3)})

    # Interpolate S columns
    s_cols = [c for c in df_s.columns if c != config.COL_TIME]
    for col in s_cols:
        col_vals = df_s[col].interpolate(method="linear").bfill().ffill().values
        sync_df[col] = np.interp(t_grid, s_aligned_time, col_vals)

    # Interpolate V columns
    v_cols = [c for c in df_v.columns if c != config.COL_TIME]
    for col in v_cols:
        col_vals = df_v[col].interpolate(method="linear").bfill().ffill().values
        sync_df[col] = np.interp(t_grid, v_time, col_vals)

    # Ground truth displacement per epoch: Δd = gt_speed_ms * dt
    sync_df["gt_delta_d_m"] = sync_df[config.COL_TRUE_SPEED_MS].clip(lower=0.0) * config.TARGET_DT
    sync_df["gt_cum_dist_m"] = sync_df["gt_delta_d_m"].cumsum()

    # Save to synchronized Parquet
    out_file = config.SYNC_PROCESSED_DIR / f"sync_{run_key}.parquet"
    sync_df.to_parquet(out_file, index=False)
    logger.info(f"Synchronized '{run_key}' -> {len(sync_df)} rows ({sync_df[config.COL_TIME].iloc[-1]:.1f}s) saved to {out_file.name}")

    return sync_df


def sync_all(max_pairs: Optional[int] = None) -> None:
    """
    Synchronize all available ingested pairs.
    """
    s_files = glob.glob(str(config.INGESTED_DIR / "S_*.parquet"))
    run_keys = [Path(f).stem[2:] for f in sorted(s_files)]

    if max_pairs:
        run_keys = run_keys[:max_pairs]

    logger.info(f"Synchronizing {len(run_keys)} ingested pairs...")
    success = 0
    for idx, k in enumerate(run_keys, 1):
        try:
            res = synchronize_pair(k)
            if res is not None:
                success += 1
        except Exception as e:
            logger.error(f"Error synchronizing '{k}': {e}", exc_info=True)

    logger.info(f"Synchronization complete: {success}/{len(run_keys)} pairs processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iNAV Sensor Synchronization Pipeline")
    parser.add_argument("--pair", type=str, default=None, help="Specific run key (e.g. 's1')")
    parser.add_argument("--max-pairs", type=int, default=None, help="Max pairs to synchronize")
    args = parser.parse_args()

    if args.pair:
        synchronize_pair(args.pair.lower())
    else:
        sync_all(max_pairs=args.max_pairs)
