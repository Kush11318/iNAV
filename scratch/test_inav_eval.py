import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd()))
import config
from eval.replay import evaluate_run_outages
from modules.velocity_net import VelocityNetPredictor

model_path = config.MODELS_DIR / "velocity_net_best.pt"
predictor = VelocityNetPredictor(str(model_path))

sync_file = config.SYNC_PROCESSED_DIR / "sync_vta10.parquet"
print(f"Testing on {sync_file.name}...")

scores = evaluate_run_outages(sync_file, method="inav_ukf", durations=[10, 30, 60], predictor=predictor)
for s in scores:
    print(f"Duration: {s['duration_s']:3d}s | Dist: {s['distance_m']:6.1f}m | ATE: {s['ate_m']:5.2f}m | FPE: {s['fpe_m']:5.2f}m | Drift: {s['drift_percent']:5.2f}% | Along: {s['along_track_rms_m']:5.2f}m | Cross: {s['cross_track_rms_m']:5.2f}m")

