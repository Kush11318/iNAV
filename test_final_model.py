"""
iNAV Final Model Evaluation & Verification Script
================================================
Quickly test and verify the final VelocityNet Dead Reckoning model on real-world
synchronized driving trajectories (Parquet) with simulated GNSS blackouts.

Usage:
    python test_final_model.py
    python test_final_model.py --parquet data/sample_test_trajectory_motorway.parquet
    python test_final_model.py --method inav_ukf
    python test_final_model.py --method inav_esekf
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from eval.replay import evaluate_run_outages
from modules.velocity_net import VelocityNetPredictor
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.test_final_model")


def run_benchmark(
    parquet_path: str = "data/sample_test_trajectory.parquet",
    method: str = "inav_ukf",
    model_path: str = "models/velocity_net_best.pt"
):
    print("=" * 80)
    print("       🧭 iNAV: FINAL DEAD RECKONING MODEL BENCHMARK (PARQUET REPLAY)")
    print("=" * 80)

    p_file = Path(parquet_path)
    if not p_file.exists():
        raise FileNotFoundError(f"Parquet trajectory file not found: {parquet_path}")

    m_file = Path(model_path)
    predictor = None
    if m_file.exists():
        logger.info(f"Loading VelocityNet model: {m_file.name}...")
        predictor = VelocityNetPredictor(model_path=str(m_file))
    else:
        logger.warning(f"Model checkpoint not found at {model_path}, using filter baselines.")

    logger.info(f"Loading test trajectory: {p_file.name} ({p_file.stat().st_size / 1024:.1f} KB)...")
    raw_df = pd.read_parquet(p_file)
    duration_s = float(raw_df["time_s"].iloc[-1] - raw_df["time_s"].iloc[0])
    logger.info(f"Loaded {len(raw_df)} samples ({duration_s:.1f} seconds of driving at 10Hz)")

    logger.info(f"Running simulated GNSS blackouts with method: '{method}'...")
    scores = evaluate_run_outages(p_file, method=method, predictor=predictor)

    if not scores:
        print("No valid outage segments could be evaluated for this run.")
        return

    df_scores = pd.DataFrame(scores)

    print("\n" + "-" * 80)
    print(f"  🏁 OUTAGE PERFORMANCE RESULTS FOR: {p_file.name}")
    print("-" * 80)
    header = f"{'Outage (s)':<12} | {'Distance (m)':<14} | {'Error (m)':<12} | {'Drift (%)':<12} | {'Heading Err (°)':<16} | {'CEP50 (m)':<10}"
    print(header)
    print("-" * 80)

    for _, row in df_scores.iterrows():
        outage_s = f"{row.get('outage_s', 0):.0f}s"
        dist_m = f"{row.get('distance_m', 0):.1f} m"
        err_m = f"{row.get('final_pos_error_m', 0):.2f} m"
        drift = f"{row.get('pct_of_distance', 0):.2f}%"
        hdg_val = row.get('heading_error_deg', float('nan'))
        hdg = f"{hdg_val:.1f}°" if not pd.isna(hdg_val) else "N/A"
        cep = f"{row.get('cep50_m', 0):.1f} m"

        print(f"{outage_s:<12} | {dist_m:<14} | {err_m:<12} | {drift:<12} | {hdg:<16} | {cep:<10}")

    print("-" * 80)
    median_drift = df_scores['pct_of_distance'].median()
    print(f"  >>> Overall Median Drift: {median_drift:.2f}% (Target: < 10.0%)")
    if median_drift <= 10.0:
        print(f"  >>> Evaluation Status   : ✅ PASSED (Strictly meets ISRO < 10.0% target)")
    else:
        print(f"  >>> Evaluation Status   : ⚠️ Target Exceeded ({median_drift:.1f}% vs < 10.0% ISRO target)")
        print(f"  >>> Analysis Note       : Pure phone sensor dead-reckoning exhibits drift during extended outages.")
        print(f"                            Test with '--method inav_esekf_snapped' to evaluate HMM road graph snapping.")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iNAV Final Model Trajectory Replay Harness")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/sample_test_trajectory.parquet",
        help="Path to synchronized driving Parquet file (e.g. data/sample_test_trajectory.parquet or data/sample_test_trajectory_motorway.parquet)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="inav_esekf",
        choices=["inav_esekf", "inav_esekf_snapped", "inav_ukf", "constant_velocity", "strapdown"],
        help="Dead reckoning navigation filter to execute"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/velocity_net_best.pt",
        help="Path to VelocityNet model checkpoint"
    )
    args = parser.parse_args()

    run_benchmark(
        parquet_path=args.parquet,
        method=args.method,
        model_path=args.model
    )
