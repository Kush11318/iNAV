"""
Forensic Barometer Provenance Audit Script
Exhaustively audits all original IO-VNBD data files, schemas, columns, numeric ranges,
and ingestion pipelines across C:/Projects/SIH 2026.
"""

import os
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

ROOT = Path("C:/Projects/SIH 2026")
IOVNBD_DIR = ROOT / "IO-VNBD"
INAV_DIR = ROOT / "iNAV"

print("=" * 80)
print("FORENSIC AUDIT OF RAW DATA PROVENANCE (IO-VNBD)")
print("=" * 80)

# -------------------------------------------------------------
# 1. Complete File Inventory
# -------------------------------------------------------------
print("\n[STEP 1] ENUMERATING ALL FILES IN IO-VNBD DIRECTORY...")
iovnbd_files = []
ext_counts = Counter()

if IOVNBD_DIR.exists():
    for p in IOVNBD_DIR.rglob("*"):
        if p.is_file():
            iovnbd_files.append(p)
            ext_counts[p.suffix.lower()] += 1

print(f"Total files in IO-VNBD: {len(iovnbd_files)}")
for ext, count in ext_counts.items():
    print(f"  {ext if ext else '[none]'}: {count}")

# Check for archives or binary logs
archives = [p for p in iovnbd_files if p.suffix.lower() in [".zip", ".tar", ".gz", ".7z", ".rar", ".bin", ".dat", ".log", ".txt", ".json", ".xml"]]
print(f"\nNon-CSV / Archive files in IO-VNBD ({len(archives)}):")
for a in archives:
    print(f"  {a.relative_to(ROOT)} ({a.stat().st_size} bytes)")

# -------------------------------------------------------------
# 2. Inspect Every Unique Raw Header in IO-VNBD
# -------------------------------------------------------------
print("\n[STEP 2] INSPECTING EVERY DISTINCT CSV HEADER IN IO-VNBD...")
csv_files = [p for p in iovnbd_files if p.suffix.lower() == ".csv"]
print(f"Found {len(csv_files)} total CSV files in IO-VNBD.")

distinct_headers = {}
header_counts = Counter()
header_file_map = defaultdict(list)

for cf in csv_files:
    try:
        with open(cf, "r", encoding="latin-1") as f:
            header_line = f.readline().strip()
        header_counts[header_line] += 1
        header_file_map[header_line].append(cf)
        if header_line not in distinct_headers:
            distinct_headers[header_line] = [c.strip() for c in header_line.split(",")]
    except Exception as e:
        print(f"Error reading {cf}: {e}")

print(f"\nDiscovered {len(distinct_headers)} distinct CSV schemas across all {len(csv_files)} files.")

# -------------------------------------------------------------
# 3. Search Header Columns for Any Pressure/Barometer Terms
# -------------------------------------------------------------
print("\n[STEP 3] SEARCHING ALL HEADERS FOR PRESSURE/BARO/ATMOSPHERIC TERMS...")
search_terms = [
    "pressure", "barometer", "baro", "atmospheric", "air pressure", "ambient",
    "hpa", "mbar", "millibar", "pa", "pascal", "bmp", "bmp280", "bmp388",
    "bme", "type_pressure", "altitude", "height", "elev"
]

all_unique_columns = set()
for h_idx, (raw_h, cols) in enumerate(distinct_headers.items(), 1):
    file_examples = [f.relative_to(ROOT) for f in header_file_map[raw_h][:2]]
    print(f"\n--- Schema {h_idx} (Appears in {header_counts[raw_h]} files, {len(cols)} columns) ---")
    print(f"    Examples: {file_examples[0]}")
    all_unique_columns.update(cols)
    
    matches = []
    for c in cols:
        c_low = c.lower()
        matched = [t for t in search_terms if t in c_low]
        if matched:
            matches.append((c, matched))
    if matches:
        print("    Matching columns:")
        for c, m in matches:
            print(f"      * '{c}' (matched: {m})")
    else:
        print("    Matching columns: NONE")

print("\n" + "=" * 80)
print(f"ALL UNIQUE COLUMNS ACROSS ALL RAW CSV FILES IN IO-VNBD ({len(all_unique_columns)}):")
for col in sorted(all_unique_columns):
    print(f"  - {col}")

# -------------------------------------------------------------
# 4. Deep Inspection of Numeric Channels for Pressure Ranges
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 4] DEEP INSPECTION OF RAW NUMERIC RANGES (Looking for 800 - 1100 hPa)")
print("=" * 80)

# Check representative smartphone runs across all drivers (A, B, C, D, E, French)
sample_runs = []
for h_line, files in header_file_map.items():
    # Pick 2 files per schema
    sample_runs.extend(files[:2])

print(f"Probing {len(sample_runs)} representative files for pressure-like ranges (800-1100)...")
candidate_pressure_cols = []

for sfile in sample_runs:
    try:
        df_sample = pd.read_csv(sfile, encoding="latin-1", nrows=500, low_memory=False)
        for col in df_sample.columns:
            # Try converting to numeric
            s_num = pd.to_numeric(df_sample[col], errors="coerce").dropna()
            if len(s_num) > 50:
                min_v = s_num.min()
                max_v = s_num.max()
                mean_v = s_num.mean()
                # Check if mean or median is in pressure range 800 to 1100 (hPa) or 80000 to 110000 (Pa)
                if (750 <= mean_v <= 1150) or (75000 <= mean_v <= 115000):
                    candidate_pressure_cols.append({
                        "file": str(sfile.relative_to(ROOT)),
                        "col": col,
                        "min": min_v,
                        "max": max_v,
                        "mean": mean_v
                    })
    except Exception as e:
        pass

if candidate_pressure_cols:
    print("\nFOUND POTENTIAL CANDIDATE NUMERIC CHANNELS IN PRESSURE RANGE:")
    for c in candidate_pressure_cols:
        print(f"  File: {c['file']}")
        print(f"    Col: '{c['col']}' | min={c['min']}, max={c['max']}, mean={c['mean']}")
else:
    print("\nNO numeric column in ANY inspected file had values in the 800-1100 hPa or 80000-110000 Pa range!")

# -------------------------------------------------------------
# 5. Check Vehicle CAN 'Brake Pressure' vs Ambient Pressure
# -------------------------------------------------------------
print("\n[STEP 5] CHECKING VEHICLE 'Brake Pressure (psi)' CHANNEL")
can_samples = [f for f in csv_files if "V-" in f.name][:3]
for cf in can_samples:
    df_can = pd.read_csv(cf, encoding="latin-1", nrows=200)
    for c in df_can.columns:
        if "pressure" in c.lower():
            vals = pd.to_numeric(df_can[c], errors="coerce").dropna()
            print(f"  {cf.name}: '{c}' min={vals.min()}, max={vals.max()}, mean={vals.mean():.2f}")

print("\nAudit script complete.")
