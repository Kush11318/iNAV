"""
Generate Graph 6: Real-Time Multi-Rate Pipeline & Android Compute Latency
Classic Google Colab / Scientific Matplotlib Style
"""

import sys
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "prototype_video_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("C:/Users/Dell/.gemini/antigravity-ide/brain/ba7187f2-a337-4478-8c8c-ca4ae6cb1a16")

# Set standard, clean Python scientific plot parameters (Google Colab style)
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def generate_graph6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})

    # --- SUBPLOT 1: Multi-Rate Frequency Hierarchy ---
    modules = ['IMU Ingestion\n(Acc & Gyro)', '7-State UKF\nCore Estimator', 'CAN / OBD-II\nWheel Speeds', 'UI Display\nMap Interpolator', 'OSM Road Map\nMatcher (HMM)']
    frequencies = [100.0, 10.0, 10.0, 10.0, 2.0]
    periods_ms = [10.0, 100.0, 100.0, 100.0, 500.0]
    colors = ['#0284c7', '#2563eb', '#f59e0b', '#10b981', '#8b5cf6']

    bars1 = ax1.barh(modules[::-1], frequencies[::-1], color=colors[::-1], height=0.55, edgecolor='#333333', linewidth=1.0)
    ax1.set_xscale('log')
    ax1.set_xlabel("Update Frequency (Hz) — Log Scale", fontweight='bold')
    ax1.set_title("Multi-Rate Pipeline Architecture Hierarchy", fontweight='bold', pad=12)
    ax1.set_xlim(0.8, 250)
    ax1.grid(axis='x', linestyle='--', alpha=0.55)

    for bar, freq, ms in zip(bars1, frequencies[::-1], periods_ms[::-1]):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax1.text(w * 1.15, y, f"{freq:g} Hz ({ms:.0f} ms loop)", va='center', ha='left', fontsize=10, fontweight='bold', color='#1e293b')

    # --- SUBPLOT 2: Smartphone Execution Latency vs 100 ms Budget ---
    stages = ['IMU + UKF\nPredict', 'CAN Speed\nUpdate', 'Map Matching\n(Amortized)', 'Total Frame\nLatency', 'SIH 10 Hz\nMax Budget']
    latencies = [0.82, 0.41, 10.55, 11.78, 100.00]
    bar_colors = ['#93c5fd', '#bfdbfe', '#c4b5fd', '#10b981', '#e2e8f0']

    bars2 = ax2.bar(stages, latencies, color=bar_colors, width=0.52, edgecolor='#333333', linewidth=1.0)
    ax2.set_ylabel("Execution Time per Cycle (ms)", fontweight='bold')
    ax2.set_title("Smartphone Latency vs 10 Hz Budget", fontweight='bold', pad=12)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle='--', alpha=0.55)

    # Annotate budget limit line
    ax2.axhline(100.0, color='#dc2626', linestyle='--', linewidth=1.5, alpha=0.85)
    ax2.text(2.0, 102.5, "SIH 10 Hz Deadline (100 ms)", color='#dc2626', fontweight='bold', fontsize=9.5, ha='center')

    for i, bar in enumerate(bars2):
        h = bar.get_height()
        fw = 'bold' if i >= 3 else 'normal'
        col = '#047857' if i == 3 else ('#b91c1c' if i == 4 else '#334155')
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f} ms", ha='center', va='bottom', fontsize=9.5, fontweight=fw, color=col)

    # Highlight 88.2% Headroom badge
    ax2.text(0.5, 0.76, "88.2% Compute Headroom\n(11.8 ms actual vs 100 ms budget)", transform=ax2.transAxes,
             ha='center', va='center', fontsize=9.5, color='#047857', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecfdf5', edgecolor='#10b981', linewidth=1.2))

    plt.tight_layout()
    out_file = OUTPUT_DIR / "graph6_multirate_pipeline.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    shutil.copy2(out_file, ARTIFACTS_DIR / out_file.name)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    generate_graph6()
