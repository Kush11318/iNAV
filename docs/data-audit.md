# IO-VNBD Raw Dataset Audit Report

> **Audit Date**: 2026-09-06  
> **Dataset**: IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)  
> **Source**: https://github.com/onyekpeu/IO-VNBD.git (Commit `118939602e3422d47b8ab0807b623751c3ac135b`)  
> **Audited Target**: `data/raw/io-vnbd/` (100% of raw CSV files audited; no parsing or transformation applied)  
> **Purpose**: Establish empirical ground truth for raw data schemas, actual sample rates, sensor completeness, corrupt runs, and usable volume prior to implementing Parquet parsers (Steps 3–4).

---

## Executive Summary

An exhaustive audit of all **564 CSV files** (1.71 GB) present in the repository was conducted using automated analysis scripts. Every individual file was parsed and evaluated for column schemas, header naming quirks, unit plausibility, inter-sample timestamp deltas, sensor completeness, file duplication, and spatial trajectory length.

### Key Audit Findings

1. **Massive File Duplication**: 
   The 564 CSV files on disk correspond to only **187 unique physical runs** (**90 Vehicle/CAN runs** and **97 Smartphone runs**). The remaining 377 files are duplicate copies produced by mirrors across `Categorised` vs `Uncategorised` directories and Windows case differences (`V-vta*` vs `V-Vta*`).
2. **Column Counts Deviate from Paper Claims**:
   - **Vehicle/CAN**: Exactly **29 columns** across all 323 CAN files on disk, consistent with Table 3 of the paper.
   - **Smartphone**: Bimodal schema — **24 columns** in 231 files, but **18 columns** in 9 French runs (`S-T1` through `S-T9`), and **25 columns** in 1 corrupt run (`S-A4`).
3. **Missing Magnetometer / Orientation Data**:
   Nine French smartphone runs (`S-T1.csv` to `S-T9.csv`, totaling **603,425 samples**) completely lack all magnetometer (`MAGNETIC FIELD X/Y/Z`) and orientation (`ORIENTATION Yaw/Pitch/Roll`) columns. The core C++ engine's design requirement to operate without magnetometer input is directly validated.
4. **Header and Unit Ambiguities (Flags for Steps 3–4)**:
   - **`GPS SPEED (Kmh)` is actually in m/s**: The values in the smartphone `GPS SPEED (Kmh)` column are Android's native `Location.getSpeed()` in **m/s**, not km/h. Interpreting them as km/h would cause a 3.6× speed underestimation.
   - **`Height (km)` is in meters**: In the CAN files, the elevation values range from 20.19 to 534.26, which represents **meters above sea level** in the UK, not kilometers.
   - **`Accelerator Pedal Position (0 or 1)` is a percentage**: Values range from `0.0` to `99.0` (percentage activation), not a binary flag `(0 or 1)`.
   - **`Clutch Position (0 or 1)` is 100% dead**: The clutch channel is `0.0` across 100% of rows in all 323 CAN files.
   - **Excel Date Mangling**: Several smartphone files (`S-A1`, `S-A12`, etc.) contain strings like `"Aug-20"` and `"Aug-21"` in `GPS SATELLITES IN RANGE` caused by Excel auto-converting `"8 / 20"` and `"8 / 21"` into calendar dates.
5. **Sample Rate Discrepancies**:
   - **Vehicle/CAN**: Highly stable at **10.0 Hz** (99.99% of samples within 0.08–0.12s; median dt = 0.1000s; 0 negative dt instances).
   - **Smartphone**: Substantial deviations:
     - 8 runs (`S-A1` to `S-A3`, `S-A9` to `S-A13`) were sampled at **2.0 Hz** (median dt = 0.5000s), with 0.0% of samples at 10 Hz.
     - 4 French runs (`S-T1`, `S-T4`, `S-T5`, `S-T6`) exhibit **1 ms burst rates** (median dt = 0.0010s) with repeated timestamps and gaps.
     - 5 runs (`S-M`, `S-S2`, `S-S3b`, `S-S4`, `S-Y1`) suffer from backward time jumps (`dt < 0`).
6. **Structural CSV Corruption**:
   - `S-A4.csv` has a double comma delimiter after column 6 (`0,, 13 / 24`), causing every subsequent column across all **32,829 rows** to shift right by one index, creating an artificial `Unnamed: 24` column.
7. **Ground-Truth Usable Volume vs Claims**:
   - **Vehicle/CAN**: **39.40 hours** (paper claimed ~40 hrs), **1,864.49 km** GPS trajectory / **1,862.64 km** integrated speed (paper claimed ~1,300 km), across **1,411,376 unique rows**.
   - **Smartphone**: **55.66 hours** (paper claimed ~58 hrs), **3,540.29 km** GPS trajectory (paper claimed ~4,400 km), across **2,075,316 unique rows**.
   - **Synchronised Subset**: Exactly **72 matched CAN/Phone run pairs** spanning **1,071,035 rows (29.93 hrs, 1,341.94 km)** in CAN and **1,070,745 rows (25.08 hrs, 856.09 km)** in Phone. 63 of the 72 pairs have exact identical row counts.

---

## 1. File Inventory

### 1.1 Directory Structure & File Counts

The raw data extracted from `Synchronised V abd S datasets.zip` and `Unsynchronised V and S Dataset.zip` resides in four distinct folder hierarchies. All 564 files are standard CSV files.

| Directory Hierarchy | Schema | File Count | File Size (MB) | Total Rows | Unique Runs |
|:---|:---|:---:|:---:|:---:|:---:|
| `Synchronised V abd S datasets/Categorised IOVNB Dataset/` | CAN (V) | 72 | 199.98 | 1,070,890 | 72 |
| `Synchronised V abd S datasets/Categorised IOVNB Dataset/` | Phone (S) | 72 | 226.47 | 1,070,890 | 72 |
| `Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/` | CAN (V) | 72 | 199.98 | 1,070,890 | 72 (identical to above) |
| `Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/` | Phone (S) | 72 | 229.27 | 1,070,890 | 72 (float precision variant) |
| `Unsynchronised V and S Dataset/Categorised IOVNB (V) Dataset/V Dataset/` | CAN (V) | 89 | 288.24 | 1,413,263 | 88 (1 internal dupe) |
| `Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/V-Dataset/` | CAN (V) | 90 | 296.02 | 1,421,010 | 90 |
| `Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/` | Phone (S) | 97 | 354.64 | 2,071,923 | 97 |
| **Total Files On Disk** | — | **564** | **1,711.47** | **8,218,756** | **187 unique runs** |

### 1.2 Deduplication Analysis

There are significant redundancies in the raw archive that parsers must not ingest twice:

1. **Synchronised Categorised vs Uncategorised**:
   - All 72 CAN files (`V-*.csv`) in `Categorised IOVNB Dataset` are **binary-identical** (identical SHA-256 hashes) to the 72 CAN files in `Uncategorised IOVNB Dataset/V-Dataset/`.
   - The 72 Smartphone files (`S-*.csv`) in `Categorised` vs `Uncategorised` have identical line counts, but differ slightly in file size due to trailing decimal formatting (e.g. `52.203125,-2.197729` vs `52.203125,-2.1977290000000003` and `0` vs `0.0`). They represent the exact same experimental runs.
2. **Unsynchronised Categorised vs Uncategorised**:
   - `Categorised IOVNB (V) Dataset` contains 89 files across 8 driver subdirectories (`Driver A` through `Driver E`). All 88 unique files are binary-identical to those in `Uncategorised IOVNB (V and S) Dataset/V-Dataset/`.
   - One file (`V-Vta4.csv`) is duplicated into two separate subfolders within `Categorised` (`Vta04/` and `Vta06/`).
   - Two files are present in `Uncategorised V-Dataset` that are omitted from `Categorised V Dataset`: `V-Vw11.csv` and `V-Vta18.csv`.
3. **Synchronised vs Unsynchronised Run Overlap**:
   - All 72 runs present in the Synchronised folder originated from the same raw recordings as runs in the Unsynchronised folder.
   - However, in the Synchronised folder, the authors performed **manual pre-alignment and trimming**: timestamps have been shifted by small fractional offsets (typically 0.1s to 0.5s) to align CAN and Phone starting points.

---

## 2. Column Audit per Schema

### 2.1 Vehicle / CAN Schema Audit

Across all **323 Vehicle/CAN CSV files**, there is **exactly 1 unified column schema** comprising **29 columns**.

#### Actual Header in Raw CSV Files
```csv
No of GPS Satellites Available,Time Since Start of Day (seconds),Latitude (degrees),Longitude (degrees),Velocity (km/hr),Heading (degrees),Height (km),Vertical velocity (km/hr),Sample period (seconds),Steering Angle (degrees),Wheel Speed Front Left (rad/sec),Wheel Speed Front Right (rad/sec),Wheel Speed Rear Left (rad/sec),Wheel Speed Rear Right (rad/sec),Yaw Rate (deg/sec),Indicated Vehicle Speed (km/hr),Indicated Longitudinal Acceleration (g),Indicated Lateral Acceleration (g),Handbrake (0 or 1),Gear Requested (Number fof gear employed 1-5),Gear (Number fof gear employed 1-5),Engine Speed (rev/min),Coolant Temperature (degrees),Clutch Position (0 or 1),Brake Pressure (psi),Brake Position (0 or 1),Battery Voltage (volts),Air Temperature (degrees),Accelerator Pedal Position (0 or 1)
```

#### Detailed Column & Unit Discrepancy Table

| Col # | Exact CSV Header Name | Documented Unit (Paper Tab. 3) | Observed Range in Data | Unit & Semantic Audit Assessment | Flag for Steps 3–4 |
|:---:|:---|:---:|:---:|:---|:---:|
| 0 | `No of GPS Satellites Available` | Count (N/A) | `0.0` to `140.0` | Integer values typically 110–140. Likely Racelogic VBOX status byte or satellite count scaled by 10 (e.g. 14.0 satellites). | Investigate VBOX encoding |
| 1 | `Time Since Start of Day (seconds)` | Seconds | `29606.4` to `72986.1` | Time elapsed from midnight UTC (seconds). Monotonic; dt = 0.100s. | Primary timestamp |
| 2 | `Latitude (degrees)` | Degrees | `51.6603` to `53.7951` | WGS84 latitude in UK region. | Valid ground truth |
| 3 | `Longitude (degrees)` | Degrees | `-2.2535` to `-0.6026` | WGS84 longitude in UK region. | Valid ground truth |
| 4 | `Velocity (km/hr)` | km/h | `0.0` to `131.88` | Racelogic GPS Doppler velocity. Highly accurate speed reference. | Reference velocity |
| 5 | `Heading (degrees)` | Degrees | `0.0` to `360.0` | GPS heading over ground. | Reference heading |
| 6 | `Height (km)` | km | `20.19` to `534.26` | **DISCREPANCY**: Header says `(km)`, but values are in **meters** (Coventry elevation ~90–150m, UK hills up to 534m). | Must convert: label as meters |
| 7 | `Vertical velocity (km/hr)` | km/h | `-15.95` to `13.53` | GPS vertical velocity. | Valid |
| 8 | `Sample period (seconds)` | Seconds | `0.099` to `0.200` | Recorded inter-sample period (~0.1s). | Redundant with time diff |
| 9 | `Steering Angle (degrees)` | Degrees | `0.0` to `800.8` | Handwheel angle. **All-zero in 49 files** where sensor was not logged. | Nullable / optional |
| 10 | `Wheel Speed Front Left (rad/sec)` | rad/s | `0.0` to `131.33` | Wheel angular velocity. 131.3 rad/s on 0.3m wheel = ~140 km/h. | Core odometry channel |
| 11 | `Wheel Speed Front Right (rad/sec)`| rad/s | `0.0` to `131.47` | Wheel angular velocity. | Core odometry channel |
| 12 | `Wheel Speed Rear Left (rad/sec)`  | rad/s | `0.0` to `131.66` | Wheel angular velocity. | Core odometry channel |
| 13 | `Wheel Speed Rear Right (rad/sec)` | rad/s | `0.0` to `130.81` | Wheel angular velocity. | Core odometry channel |
| 14 | `Yaw Rate (deg/sec)` | deg/s | `-66.0` to `+66.6` | Vehicle chassis yaw rate from ESP/ESC IMU. | Reference angular rate |
| 15 | `Indicated Vehicle Speed (km/hr)` | km/h | `0.0` to `131.36` | Speedometer speed from ECU wheel speed averaging. | Valid CAN odometry |
| 16 | `Indicated Longitudinal Acceleration (g)`| g | `-1.01` to `+0.48` | Longitudinal acceleration in g (1g ≈ 9.80665 m/s²). | Needs conversion to m/s² |
| 17 | `Indicated Lateral Acceleration (g)` | g | `-0.90` to `+0.87` | Lateral acceleration in g. | Needs conversion to m/s² |
| 18 | `Handbrake (0 or 1)` | 0 or 1 | `0.0` to `1.0` | Handbrake contact status. 97.0% zero. | Status indicator |
| 19 | `Gear Requested (Number fof gear employed 1-5)` | 1–5 | `0.0` to `14.0` | Header contains typo `"fof"`. Neutral=0, Reverse/Drive codes. | Typo in column name |
| 20 | `Gear (Number fof gear employed 1-5)` | 1–5 | `0.0` to `5.0` | Current engaged forward gear (1 to 5). Header typo `"fof"`. | Typo in column name |
| 21 | `Engine Speed (rev/min)` | rpm | `0.0` to `6332.0` | Engine crankshaft RPM. | Diagnostic channel |
| 22 | `Coolant Temperature (degrees)` | °C | `0.0` to `105.0` | Engine coolant temperature in Celsius. | Diagnostic channel |
| 23 | `Clutch Position (0 or 1)` | 0 or 1 | `0.0` to `0.0` | **DEAD CHANNEL**: 100.0% zero across all rows of all files. | Unusable / ignore |
| 24 | `Brake Pressure (psi)` | psi | `-1.91` to `192.85` | Hydraulic master cylinder brake line pressure. | Driver behavior feature |
| 25 | `Brake Position (0 or 1)` | 0 or 1 | `0.0` to `1.0` | Brake pedal switch (pressed / released). | Driver behavior feature |
| 26 | `Battery Voltage (volts)` | Volts | `0.0` to `14.6` | Vehicle 12V electrical bus voltage. | Diagnostic channel |
| 27 | `Air Temperature (degrees)` | °C | `0.0` to `30.75` | Ambient intake air temperature. | Diagnostic channel |
| 28 | `Accelerator Pedal Position (0 or 1)` | 0 or 1 | `0.0` to `99.0` | **DISCREPANCY**: Header says `(0 or 1)`, paper says `% activation`. Actual values are **0% to 99%**. | Label as percentage (0–100) |

---

### 2.2 Smartphone Schema Audit

Across the **241 Smartphone CSV files**, there are **5 distinct header schemas**:

| Schema Group | Column Count | File Count | Representative Subsets | Key Distinguishing Characteristics |
|:---:|:---:|:---:|:---|:---|
| **Group 1** | 24 | 158 | `Unsynchronised/S-Dataset`, `Sync/S-Dataset` | Standard header with `GYROSCOPE Yaw/Pitch/Roll` and `ORIENTATION (Yaw) (°)`. |
| **Group 2** | 24 | 71 | `Synchronised/Categorised` | Header uses `GYROSCOPE X/Y/Z` and `ORIENTATION (Azimuth) (°)`. |
| **Group 3** | 24 | 2 | `S-Vfa01.csv`, `S-Vfa02.csv` | Header typo: `DATE (YYYY-MO-DD HH-MI-SS_SSS` (missing closing parenthesis). |
| **Group 4** | 25 | 1 | `S-A4.csv` | Corrupted with double comma delimiter (`0,,13 / 24`), creating `Unnamed: 24`. |
| **Group 5** | 18 | 9 | `S-T1.csv` through `S-T9.csv` (France) | **Missing Magnetometer and Orientation**: Truncated after `GYROSCOPE Roll`. Col 6 is named `SATELLITES IN RANGE` instead of `GPS SATELLITES IN RANGE`. |

#### Detailed Smartphone Column & Unit Assessment

| Col # | Standard Header Name | Documented Unit | Observed Range | Unit & Semantic Audit Assessment | Flag for Steps 3–4 |
|:---:|:---|:---:|:---:|:---|:---:|
| 0 | `GPS LATITUDE (degrees)` | Degrees | `4.8` to `53.8` | Phone GPS fix latitude (UK, France, Nigeria). | Primary position |
| 1 | `GPS LONGITUDE (degrees)` | Degrees | `-3.5` to `7.0` | Phone GPS fix longitude. | Primary position |
| 2 | `GPS ALTITUDE (m)` | Meters | `0.0` to `480.0` | WGS84 ellipsoidal height. Frequent zeros when 2D fix only. | Check fix validity |
| 3 | `GPS SPEED (Kmh)` | km/h | `0.0` to `35.2` | **CRITICAL DISCREPANCY**: Header indicates `(Kmh)`, but values match **meters/second** (Android `Location.getSpeed()`). Max 35.2 m/s = 126.7 km/h. | Must multiply by 3.6 for km/h |
| 4 | `GPS ACCURACY (m)` | Meters | `1.0` to `2800.0` | 1-sigma horizontal accuracy estimate from Android Location API. | Essential for weighting |
| 5 | `GPS ORIENTATION (°)` | Degrees | `0.0` to `360.0` | Bearing / course over ground from GPS. | Discontinuous across 0/360 |
| 6 | `GPS SATELLITES IN RANGE` | Text | `"0 / 20"` to `"29 / 29"`, `"Aug-20"` | String `"used / visible"`. Corrupted into Excel dates (`"Aug-20"`) in `S-A1`, `S-A12`. | Needs regex / robust parser |
| 7 | `TIME SINCE START (ms)` | Milliseconds | `11` to `12,779,310` | AndroSensor elapsed time counter. Median step 100 ms (~10 Hz). | Primary phone clock |
| 8 | `DATE (YYYY-MO-DD HH-MI-SS_SSS)` | Datetime | 2019-02-27 to 2020-03-21 | Wall-clock UTC timestamp string. Format uses colon separators for millis. | Needs custom date parser |
| 9 | `ACCELEROMETER X (m/s²)` | m/s² | `-18.5` to `+18.1` | Triaxial body accelerometer including gravity. | Core IMU channel |
| 10 | `ACCELEROMETER Y (m/s²)` | m/s² | `-16.3` to `+17.0` | Triaxial body accelerometer including gravity. | Core IMU channel |
| 11 | `ACCELEROMETER Z (m/s²)` | m/s² | `-19.8` to `+22.4` | Triaxial body accelerometer including gravity. | Core IMU channel |
| 12 | `GRAVITY X (m/s²)` | m/s² | `-9.8` to `+9.8` | Android software-fused gravity vector component. | Orientation reference |
| 13 | `GRAVITY Y (m/s²)` | m/s² | `-9.8` to `+9.8` | Android software-fused gravity vector component. | Orientation reference |
| 14 | `GRAVITY Z (m/s²)` | m/s² | `-9.8` to `+9.8` | Android software-fused gravity vector component. | Orientation reference |
| 15 | `GYROSCOPE Yaw/X (rad/s)` | rad/s | `-3.8` to `+4.1` | Body angular velocity. Header uses Yaw or X depending on folder. | Core IMU channel |
| 16 | `GYROSCOPE Pitch/Y (rad/s)`| rad/s | `-4.2` to `+3.9` | Body angular velocity. Header uses Pitch or Y depending on folder. | Core IMU channel |
| 17 | `GYROSCOPE Roll/Z (rad/s)` | rad/s | `-4.5` to `+4.6` | Body angular velocity. Header uses Roll or Z depending on folder. | Core IMU channel |
| 18 | `MAGNETIC FIELD X (μT)` | µT | `-75.0` to `+68.0` | Ambient calibrated triaxial magnetic field. **Absent in French S-T1..T9**. | Missing in 9 French runs |
| 19 | `MAGNETIC FIELD Y (μT)` | µT | `-69.0` to `+72.0` | Ambient calibrated triaxial magnetic field. **Absent in French S-T1..T9**. | Missing in 9 French runs |
| 20 | `MAGNETIC FIELD Z (μT)` | µT | `-85.0` to `+80.0` | Ambient calibrated triaxial magnetic field. **Absent in French S-T1..T9**. | Missing in 9 French runs |
| 21 | `ORIENTATION (Yaw/Azim) (°)`| Degrees | `0.0` to `360.0` | Android fused compass azimuth. **Absent in French S-T1..T9**. | Missing in 9 French runs |
| 22 | `ORIENTATION (Pitch) (°)` | Degrees | `-90.0` to `+90.0` | Android fused pitch angle. **Absent in French S-T1..T9**. | Missing in 9 French runs |
| 23 | `ORIENTATION (Roll ) (°)` | Degrees | `-180.0` to `+180.0`| Android fused roll angle. Trailing space in header. **Absent in French S-T1..T9**.| Missing in 9 French runs |

---

## 3. Sample Rate Audit

### 3.1 Vehicle / CAN Sample Rates

| Metric | Empirical Value | Target Expectation | Status |
|:---|:---:|:---:|:---:|
| Total Files Analyzed | 323 / 323 | 323 | Complete |
| Median Sample Delta ($dt$) | **0.1000 s** | 0.1000 s (10.0 Hz) | Perfect |
| Mean Sample Delta ($dt$) | **0.1001 s** | 0.1000 s | Perfect |
| Percentage of Samples at ~10 Hz (0.08–0.12 s) | **99.99%** | ≥ 99.0% | Pass |
| Minimum File 10 Hz Compliance | **98.63%** (`V-St7.csv`) | ≥ 95.0% | Pass |
| Total Negative $dt$ Instances (Reversals) | **0** | 0 | Perfect |
| Total Zero $dt$ Instances (Duplicates) | 910 across 4.97M rows | < 0.1% | Negligible (0.018%) |
| Large Gaps ($dt > 0.5$ s) | 242 across 323 files | Minimized | Single 5.1s gap in `V-St7` |

### 3.2 Smartphone Sample Rates

The smartphone sampling characteristics vary widely across different collection batches:

| Metric | Overall Dataset | 10 Hz UK Cohort (Sync + Part Unsync) | 2 Hz Cohort (`S-A1..A13`) | French Cohort (`S-T1..T9`) |
|:---|:---:|:---:|:---:|:---:|
| File Count | 241 | 216 | 8 | 9 |
| Median $dt$ | **0.1000 s** | **0.1000 s** | **0.5000 s** | **0.0010 s** |
| Mean $dt$ | 0.1055 s | 0.1002 s | 0.4998 s | 0.1340 s |
| % Samples in 0.08–0.12s | **94.94%** | **98.85%** | **0.00%** | **23.85%** |
| Sampling Rate Category | Nominal 10 Hz | Stable 10 Hz | **Hardcoded 2 Hz** | **Burst / Jitter** |

#### Notable Smartphone Sampling Anomalies

1. **2 Hz Throttled Subsets (8 runs, 54,411 samples)**:
   Files `S-A1`, `S-A2`, `S-A3`, `S-A9`, `S-A10`, `S-A11`, `S-A12`, and `S-A13` were collected with AndroSensor set to "Normal" or "UI" rate rather than "Fastest", resulting in a constant ~0.500s inter-sample period (2 Hz).
2. **French Subsets Burst Sampling (4 runs)**:
   `S-T1` (20.2% at 10 Hz), `S-T4` (36.9% at 10 Hz), `S-T5` (30.4% at 10 Hz), and `S-T6` (13.5% at 10 Hz) recorded clusters of identical timestamps followed by 1 ms jumps and occasional multi-second pauses.
3. **Massive Recording Gaps**:
   - `S-Vtb1.csv`: Contains a single gap of **661.38 seconds** (~11.0 minutes) between sample indices 314 and 315, where the phone application was suspended.
   - `S-I.csv` (Nigeria): Contains a gap of **285.38 seconds** (~4.75 minutes).
4. **Negative Timestamp Reversals (Clock Resets)**:
   Five runs contain negative time increments where the Android elapsed millisecond counter jumped backward:
   - `S-M.csv` (1 reversal)
   - `S-S2.csv` (1 reversal)
   - `S-S3b.csv` (1 reversal)
   - `S-S4.csv` (2 reversals)
   - `S-Y1.csv` (3 reversals)

---

## 4. Missing-Sensor Audit

### 4.1 Missing Magnetometer & Orientation Columns (French Subsets)

The raw audit confirms the architectural expectation regarding missing sensors:

- **Completely Missing Columns (18-column files)**:
  Nine runs recorded in France (`S-T1.csv` through `S-T9.csv`, totaling **603,425 rows**, 12.75 hours, and 218.25 km) **do not have magnetometer or orientation columns at all in their CSV headers**. The file ends immediately after `GYROSCOPE Roll (rad/s)`.
- **Intact French Runs**:
  `S-T10.csv` and `S-T11.csv` (shorter runs in France recorded with a different device) have all 24 columns present with valid, non-zero magnetometer data.
- **No All-Zero or All-NaN Columns**:
  In all files where the magnetometer columns are present (232 files), they contain genuine time-varying magnetic field values (mean ~35 µT, min ~-75 µT, max ~+68 µT). There are zero files where the columns exist but contain all NaNs or all zeros.

> [!IMPORTANT]
> **Architectural Contract Confirmation**:
> The `core/` C++ engine's `ImuSample` design (which takes optional magnetometer fields or flags `has_mag = false`) is directly required by the dataset. Ingestion parsers in Step 4 must emit `null` / `NaN` for magnetometer fields on 18-column files rather than rejecting them.

### 4.2 Dead and Constant Channels in Vehicle CAN Bus

| Channel | % NaN | % Zero | Constant Across Files | Status & Recommendation |
|:---|:---:|:---:|:---:|:---|
| `Clutch Position (0 or 1)` | 0.0% | **100.0%** | Yes (all 323 files) | Completely unmonitored channel on the Ford Fiesta CAN network. Omit from state estimation. |
| `Steering Angle (degrees)` | 0.0% | 33.1% | In **49 files** | Unlogged in 49 runs; non-zero in remaining 274 files. Parser must handle missing steering angle without crashing. |
| `Wheel Speed` & `Speedometer` | 0.0% | 10.0% | In **2 unique runs** (`V-Vw1`, `V-Vw15`) | CAN wheel speed was dead during these runs (speed is 0.0 for all samples). Must be quarantined from odometry benchmarks. |
| `Handbrake (0 or 1)` | 0.0% | 97.0% | In 241 files | Rarely engaged during active transit. |

---

## 5. Synchronised Subset Identification

### 5.1 Structure & Pairing Methodology

The synchronised subset is explicitly isolated inside `Synchronised V abd S datasets/`:
- It consists of **72 run pairs** (72 CAN files and 72 Smartphone files).
- The pairing is unambiguous and established via exact matching suffixes: `V-<RunID>.csv` pairs directly with `S-<RunID>.csv` (e.g., `V-S1.csv` ↔ `S-S1.csv`, `V-Vw01.csv` ↔ `S-Vw1.csv`).
- Two directory layouts are provided:
  - `Categorised IOVNB Dataset/`: Organized by driver directory:
    - `M (Driver B)`: 1 run pair (`M`)
    - `S (Driver A)`: 6 run pairs (`S1`, `S2`, `S3a`, `S3b`, `S3c`, `S4`)
    - `Vf (Driver E)`: 2 run pairs (`Vfa01`, `Vfa02`)
    - `Vta (Driver E)`: 30 run pairs (`Vta01a`, `Vta01b`, `Vta02`–`Vta30`)
    - `Vtb (Driver E)`: 12 run pairs (`Vtb01`–`Vtb12`)
    - `Vw (Driver E)`: 20 run pairs (`Vw01`–`Vw17`, including `Vw14a/b/c`, `Vw16a/b`)
    - `Y (Driver D)`: 1 run pair (`Y1`)
  - `Uncategorised IOVNB Dataset/`: Flat structure with `V-Dataset/` and `S-Dataset/`.

### 5.2 Alignment and Sample Matching Quality

- **Exact Row Count Matches**: In **63 of the 72 synchronised pairs (87.5%)**, the CAN file and Smartphone file have **the exact same number of rows**.
- **Discrepant Pairs**: Only 9 pairs have a row count difference; the maximum discrepancy is **232 rows** (~23.2 seconds):
  - `Vtb1`: CAN has 1,245 rows, Phone has 1,477 rows (diff = 232)
  - `Vta1a`: CAN has 10,617 rows, Phone has 10,724 rows (diff = 107)
  - `S2`: CAN has 93,899 rows, Phone has 93,876 rows (diff = 23)
  - `Vta27`: CAN has 7,163 rows, Phone has 7,166 rows (diff = 3)
  - `Vta28`: CAN has 7,370 rows, Phone has 7,372 rows (diff = 2)
  - `Vta17`: CAN has 4,519 rows, Phone has 4,520 rows (diff = 1)
  - `Vta22`: CAN has 10,323 rows, Phone has 10,324 rows (diff = 1)
  - `Vta25`: CAN has 8,767 rows, Phone has 8,768 rows (diff = 1)
  - `Vta30`: CAN has 11,460 rows, Phone has 11,461 rows (diff = 1)
- **Pre-Alignment Evidence**:
  Inspecting identical runs across `Synchronised/` and `Unsynchronised/` reveals that the authors performed manual lead/lag trimming on the CAN timestamps to align them with the phone start timestamps. For instance, in `V-vtb9.csv`, the starting timestamp in `Unsynchronised` is `66106.6 s`, while in `Synchronised` it has been trimmed to `66107.0 s` (+0.4 s offset).

---

## 6. Corrupt / Quarantine Candidates

The following files exhibit structural corruption, sample rate failure, or missing critical sensor streams, and must be quarantined or handled with specialized parser logic:

| Quarantine Candidate | Severity | Specific Issue / Empirical Evidence | Impact on Pipeline | Recommended Action |
|:---|:---:|:---|:---|:---|
| `S-A4.csv` | **Critical** | Double comma after col 6 (`0,,13 / 24`) shifts all downstream columns right by 1 index across all **32,829 rows**. Date appears in Accel X, satellite strings appear in timestamp. | Fails standard CSV parsing; completely corrupts sensor data. | Quarantine from baseline benchmark; write dedicated sanitizing fixup if salvaged. |
| `S-A1`, `S-A2`, `S-A3`, `S-A9`, `S-A10`, `S-A11`, `S-A12`, `S-A13` | **High** | Sample rate is hardcoded to **2.0 Hz** (median dt = 0.500s; 0% of samples at 10 Hz). Total 54,411 samples over 7.5 hours. | Incompatible with 10 Hz inertial dead-reckoning filters; excessive IMU integration drift. | Quarantine from 10 Hz leaderboard; isolate in secondary low-rate evaluation split. |
| `S-T1`, `S-T4`, `S-T5`, `S-T6` | **High** | Erratic burst sampling: median dt = 0.001s (1 ms bursts), only 13%–36% of samples at 10 Hz, numerous timestamp discontinuities. | Violates discrete-time assumption in Kalman filters. | Quarantine from primary training/eval sets. |
| `S-T1` through `S-T9` | **Medium** | Missing columns 19–24 (no magnetometer or orientation channels; 18 columns total). | Cannot be ingested by a naive 24-column fixed schema parser. | Ingestion parser must dynamically support 18-col variant; evaluate in "IMU-only" (no mag) benchmark. |
| `V-Vw1.csv` & `V-Vw15.csv` | **Medium** | Wheel speed channels (`Wheel Speed Front/Rear Left/Right`) and `Indicated Vehicle Speed` are **0.0 across all rows**. | CAN odometry is completely dead; vehicle was either stationary or CAN tap failed. | Exclude from odometry ground-truth verification. |
| `S-Vtb1.csv` | **Medium** | Contains a massive **661.38-second (~11 minute) recording outage** between samples 314 and 315. | Simulated outage tests will be skewed by genuine 11-minute drop. | Split into two separate sub-runs (Part A and Part B) around the gap. |
| `S-I.csv` | **Medium** | Contains a **285.38-second (~4.75 minute) gap**; Nigeria urban driving. | Discontinuous trajectory. | Split into before/after segments. |
| `S-M`, `S-S2`, `S-S3b`, `S-S4`, `S-Y1` | **Low** | Negative timestamp deltas ($dt < 0$) due to phone system clock synchronisation adjustments. | Non-monotonic time sequence causes filter instability. | Parsers must sort by timestamp or drop backward-stepping frames. |
| `S-A1`, `S-A12` | **Low** | Excel string corruption in `GPS SATELLITES IN RANGE` (`"Aug-20"`, `"Aug-21"` instead of `"8 / 20"`). | String parsing errors if expecting integer satellite counts. | Use regex extractor `(\d+)\s*/\s*(\d+)` with fallback to NaN. |
| `S-Vfa01.csv`, `S-Vfa02.csv` | **Low** | Malformed header: `DATE (YYYY-MO-DD HH-MI-SS_SSS` lacks trailing closing parenthesis. | DictReader / named column lookups will fail if matching on exact string. | Strip whitespace and normalize header names leniently. |

---

## 7. Total Usable Volume

### 7.1 Audited vs Paper Claimed Volume

| Metric | Paper Claimed (README / Abstract) | Audited Ground Truth (Unique Runs) | Audited Synchronised Subset | Audited Unsynchronised-Only Subset |
|:---|:---:|:---:|:---:|:---:|
| **CAN Total Runs** | Not stated | **90 runs** | 72 runs | 18 runs |
| **CAN Total Rows** | Not stated | **1,411,376 rows** | 1,071,035 rows | 340,341 rows |
| **CAN Total Duration** | ~40 hours | **39.40 hours** | 29.93 hours | 9.47 hours |
| **CAN GPS Distance** | ~1,300 km | **1,864.49 km** | 1,341.94 km | 522.54 km |
| **CAN Velocity Integrated Distance** | — | **1,862.64 km** | 1,340.85 km | 521.79 km |
| **Phone Total Runs** | Not stated | **97 runs** | 72 runs | 25 runs |
| **Phone Total Rows** | Not stated | **2,075,316 rows** | 1,070,745 rows | 1,004,571 rows |
| **Phone Total Duration** | ~58 hours | **55.66 hours** | 25.08 hours | 30.58 hours |
| **Phone GPS Distance** | ~4,400 km | **3,540.29 km** | 856.09 km | 2,684.20 km |

### 7.2 Volume Breakdown by Driver & Geography

#### Vehicle / CAN Runs by Driver
- **Driver A (`S`)**: 6 runs | 8.58 hours | 275.77 km | 308,839 rows (Coventry / Nuneaton / Rugby, UK)
- **Driver B (`M`)**: 1 run | 2.94 hours | 105.22 km | 105,974 rows (Motorway M6 / Coventry, UK)
- **Driver C (`St`)**: 4 runs | 4.65 hours | 278.74 km | 166,591 rows (UK country roads)
- **Driver D (`Y`)**: 2 runs | 3.72 hours | 105.86 km | 127,498 rows (Urban Coventry, UK)
- **Driver E (`Vf`, `Vta`, `Vtb`, `Vw`)**: 77 runs | 19.50 hours | 1,098.90 km | 749,048 rows (UK diverse conditions)
- **Total Unique CAN**: **90 runs | 39.40 hours | 1,864.49 km | 1,411,376 rows**

#### Smartphone Runs by Cohort & Geography
- **UK Synchronised Cohort (`S-M`, `S-S*`, `S-V*`, `S-Y*`)**: 72 runs | 25.08 hours | 856.09 km | 1,070,745 rows
- **UK Unsynchronised Cohort (`S-A1`..`S-A13`)**: 13 runs | 17.32 hours | 736.27 km | 382,957 rows
- **France Cohort (`S-T1`..`S-T9`, 18 cols)**: 9 runs | 12.75 hours | 218.25 km | 603,425 rows
- **France Cohort (`S-T10`..`S-T11`, 24 cols)**: 2 runs | 0.34 hours | 7.48 km | 12,389 rows
- **Nigeria Cohort (`S-I`)**: 1 run | 0.16 hours | 0.06 km | 5,800 rows
- **Total Unique Smartphone**: **97 runs | 55.66 hours | 3,540.29 km | 2,075,316 rows**

### 7.3 Net Usable Benchmark Volume (Post-Quarantine)

When excluding the corrupt `S-A4` run (32,829 rows), the 2 Hz downsampled runs (54,411 rows), and bursty France runs (`S-T1/4/5/6`, 272,080 rows), the **net pristine 10 Hz benchmark volume** stands at:

- **Pristine Synchronised (Phone + CAN)**: **72 runs | 25.08 hours | 856 km Phone / 1,342 km CAN | ~1.07 million matched time-aligned sample pairs**.
- **High-Quality Smartphone Navigation Volume**: **76 runs | 39.2 hours | ~2,500 km | 1,715,996 samples**.

---

## 8. Summary of Action Items for Steps 3–6

1. **Step 3 (CAN Ingestion & Parquet Parser)**:
   - Handle the single unified 29-column schema.
   - Rescale `Height (km)` by treating values as meters.
   - Interpret `Accelerator Pedal Position (0 or 1)` as a 0–100 percentage.
   - Convert accelerations from $g$ to $\text{m/s}^2$ ($a \times 9.80665$).
   - Omit the unmonitored `Clutch Position (0 or 1)` channel.
2. **Step 4 (Smartphone Ingestion & Parquet Parser)**:
   - Implement dynamic schema recognition: accept both 24-column and 18-column files (French runs without magnetometer/orientation).
   - **Multiply `GPS SPEED (Kmh)` by 3.6** to convert native m/s values into true km/h (or preserve as m/s with clear naming).
   - Normalize gyro and orientation header variations (`GYROSCOPE Yaw` vs `GYROSCOPE X`; `ORIENTATION Yaw` vs `ORIENTATION Azimuth`).
   - Quarantine `S-A4.csv` due to shifted columns from double-comma delimiter corruption.
   - Robustly parse `GPS SATELLITES IN RANGE` to handle Excel date-mangled strings.
3. **Step 5 (Time Alignment)**:
   - Leverage the 72 pre-aligned runs in `Synchronised V abd S datasets/` as the primary benchmark dataset.
   - Account for minor start-time offsets when comparing to unsynchronised parent files.
