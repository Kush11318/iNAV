# Phone-Based Inertial Dead-Reckoning (IDR) Navigation Stack

[![CI](https://github.com/your-org/idr-project/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/idr-project/actions)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade, phone-based inertial dead-reckoning (IDR) navigation engine and benchmarking suite engineered for GNSS-denied environments (urban canyons, underground tunnels, jamming, and spoofing). Developed for the ISRO hackathon / Smart India Hackathon (SIH 2026).

The platform features a **dependency-free C++17 core navigation engine** consuming abstract IMU and GNSS streams, portable across an offline replay evaluation harness, an Android mobile application via JNI, and a Linux edge daemon for ~200 Hz FOG-grade IMUs. A comprehensive Python data engineering and evaluation pipeline (`dataeval`) manages dataset ingestion (IO-VNBD), multi-sensor temporal synchronization, anti-leakage train/validation/test partitioning, synthetic GNSS outage simulation, and standardized leaderboard scoring against independent ground truth.

---

## Table of Contents

1. [Key Features & System Architecture](#key-features--system-architecture)
2. [Repository Layout](#repository-layout)
3. [Engineering & Research Highlights](#engineering--research-highlights)
   - [IO-VNBD Dataset Audit & Bug Corrections](#io-vnbd-dataset-audit--bug-corrections)
   - [Time Synchronization & Sub-Sample Grid Alignment](#time-synchronization--sub-sample-grid-alignment)
   - [Leakage-Safe Partitioning (41 Atomic Groups)](#leakage-safe-partitioning-41-atomic-groups)
   - [Deterministic GNSS Outage Simulation](#deterministic-gnss-outage-simulation)
   - [C++17 Strapdown INS Mechanization](#c17-strapdown-ins-mechanization)
4. [Benchmark Leaderboard & Empirical Findings](#benchmark-leaderboard--empirical-findings)
5. [Prerequisites & Quick Start](#prerequisites--quick-start)
   - [C++ Core Engine Build & Test](#1-c-core-engine-build--test)
   - [Python Pipeline Setup & Test](#2-python-data--evaluation-pipeline)
6. [End-to-End Pipeline Execution](#end-to-end-pipeline-execution)
7. [Automated Testing & CI](#automated-testing--ci)
8. [Architectural Decision Records (ADRs)](#architectural-decision-records-adrs)
9. [Project Roadmap](#project-roadmap)

---

## Key Features & System Architecture

```
                                +---------------------------------------------------+
                                |            RAW BENCHMARK DATASETS (IO-VNBD)       |
                                |  564 CSVs | 1.71 GB | 90 CAN Drives | 97 Phones  |
                                +---------------------------------------------------+
                                                          |
                                                          v
                                +---------------------------------------------------+
                                |      STEP 1-4: INGESTION & AUDIT (dataeval/ingest)|
                                |  * Unit bug fixes (Speed m/s, Height m, Pedal %)  |
                                |  * Schema harmonization (29 CAN / 27 Phone cols)  |
                                |  * Type-safe Snappy Parquet serialization         |
                                +---------------------------------------------------+
                                                          |
                                                          v
                                +---------------------------------------------------+
                                |     STEP 5: TIME SYNCHRONIZATION (dataeval/sync)  |
                                |  * Doppler speed cross-correlation lag estimation  |
                                |  * Uniform 10 Hz grid alignment (zero extrap)     |
                                |  * Circular angle unwrapping & flag step-fill     |
                                +---------------------------------------------------+
                                                          |
                                                          v
                                +---------------------------------------------------+
                                |    STEP 6: ANTI-LEAKAGE SPLIT (dataeval/split)    |
                                |  * 41 Atomic Groups (Driver x Regional Campaign)  |
                                |  * 68.7% Train / 14.6% Val / 16.7% Test by time   |
                                |  * Enforced hard-cohort coverage in Val and Test  |
                                +---------------------------------------------------+
                                                          |
                                                          v
                                +---------------------------------------------------+
                                |   STEP 7: GNSS OUTAGE SIMULATOR (dataeval/outage) |
                                |  * Deterministic SHA-256 seeding (10s to 180s)    |
                                |  * Avoids authentic dropouts | 10s edge buffers   |
                                |  * Cross-stream ground-truth extraction (CAN GT)  |
                                +---------------------------------------------------+
                                      |                                   |
                                      v                                   v
             +--------------------------------------+   +------------------------------------+
             | PYTHON CV BASELINE (dataeval/harness)|   | C++17 REPLAY CACHE & CORE ENGINE   |
             | * Pre-outage circular mean state     |   | * export_replay_cache.py (ADR 0005)|
             | * Great-circle constant velocity     |   | * idr_replay CLI streaming reader  |
             | * Linear error divergence benchmark  |   | * Strapdown INS (Quaternion, ENU)  |
             +--------------------------------------+   +------------------------------------+
                                      |                                   |
                                      +-----------------+-----------------+
                                                        |
                                                        v
                                +---------------------------------------------------+
                                |  STEP 8-11: BENCHMARK SCORING & LEADERBOARD       |
                                |  * 7 locked metrics: Final Pos, %Dist, CEP50/95,  |
                                |    Along/Cross-track, Heading error               |
                                |  * results/leaderboard.csv & summary statistics   |
                                |  * Publication-ready comparative visualizations   |
                                +---------------------------------------------------+
```

1. **Dependency-Free C++17 Core Engine (`core/`)**:
   - Zero external runtime dependencies for maximum embeddability (Android NDK, Raspberry Pi, ARM Linux).
   - Full 3D unit quaternion attitude kinematics (`Quaternion`), 3D vector arithmetic (`Vector3d`), and local tangent plane East-North-Up (`StrapdownIns`) mechanization.
   - High-throughput streaming replay CLI (`idr_replay`) processing >100,000 samples/second.
2. **Deterministic Data & Evaluation Pipeline (`dataeval/`)**:
   - Clean, reproducible pipeline covering raw ingestion, synchronization, train/val/test partitioning, synthetic GNSS denial simulation, and metric scoring.
   - Strict data-leakage protection preventing session, vehicle, or cross-stream leakage.
3. **Rigorous Multi-Metric Evaluation**:
   - Evaluates dead-reckoning trajectories across 5 standard outage durations (10s, 30s, 60s, 120s, 180s) on 377 benchmark instances against independent vehicle CAN bus ground truth.
4. **Deployable Frontends**:
   - Offline batch evaluation harness.
   - Scaffolded JNI interface for Android mobile apps (`android/`).
   - Scaffolded Linux daemon for high-frequency FOG/MEMS units (`edge/`).

---

## Repository Layout

```
idr-project/
├── core/                                 # C++17 navigation engine (zero external dependencies)
│   ├── CMakeLists.txt                    # Core library, replay driver, and test build targets
│   ├── include/idr/                      # Public C++ API headers
│   │   ├── engine.hpp                    # Top-level IdrEngine interface
│   │   ├── gnss_fix.hpp                  # GnssFix data structure (WGS-84, speed, accuracy)
│   │   ├── imu_sample.hpp                # ImuSample data structure (accel, gyro, mag)
│   │   ├── quaternion.hpp                # Unit quaternion kinematics and attitude operations
│   │   ├── strapdown.hpp                 # StrapdownIns navigation mechanization in local ENU
│   │   └── vector3d.hpp                  # 3D vector primitives and cross/dot products
│   ├── src/                              # C++ engine implementations
│   │   ├── engine.cpp                    # IdrEngine dispatch and state tracking
│   │   ├── strapdown.cpp                 # Trapezoidal INS numerical integration and geodetic math
│   │   └── replay/
│   │       └── replay_driver.cpp         # CLI batch replay runner (streams cached CSVs to predictions)
│   └── tests/                            # C++ unit tests
│       ├── test_engine.cpp               # Engine state initialization and sample ingestion tests
│       └── test_strapdown.cpp            # Synthetic kinematic tests (straight line, circular turn, gyro bias)
│
├── dataeval/                             # Python data engineering and evaluation package
│   ├── pyproject.toml                    # PEP 518/621 build configuration
│   ├── ingest/                           # Raw dataset ingestion and sanitization
│   │   ├── run_can_ingest.py             # Vehicle CAN dataset parser CLI
│   │   ├── run_phone_ingest.py           # Smartphone sensor dataset parser CLI
│   │   ├── smartphone.py                 # 5-schema variant reconciliation & unit bug fixes
│   │   └── vehicle_can.py                # 29-column CAN parser & unit bug fixes
│   ├── harness/                          # Evaluation and benchmark orchestration
│   │   ├── baseline.py                   # Constant-velocity & constant-heading great-circle baseline
│   │   ├── cpp_predictions.py            # Loader, schema validator, and spot-checker for C++ outputs
│   │   ├── export_replay_cache.py        # Exporter generating replay cache CSVs for C++ engine
│   │   ├── make_plots.py                 # Publication-ready diagnostic visualization generator
│   │   ├── metrics.py                    # Seven locked leaderboard metrics calculation engine
│   │   ├── outage.py                     # Synthetic GNSS outage generator and ground-truth extractor
│   │   ├── pairing.py                    # CAN and Smartphone run matcher
│   │   ├── run_outage_sim.py             # CLI batch outage generator
│   │   ├── run_scoring.py                # CLI benchmark scoring and leaderboard generator
│   │   ├── run_split.py                  # CLI train/val/test partition runner
│   │   ├── run_sync.py                   # CLI time synchronization and grid alignment runner
│   │   ├── split.py                      # Leakage-safe 41-group partitioning logic
│   │   ├── summary.py                    # Grouped aggregate metrics summarizer
│   │   └── sync.py                       # GPS Doppler cross-correlation & interpolation engine
│   └── tests/                            # Python automated test suite (58 unit tests)
│       ├── test_baseline.py              # Tests for CV baseline state extraction and propagation
│       ├── test_cpp_predictions.py       # Tests for C++ trajectory prediction loading and validation
│       ├── test_metrics.py               # Tests for Haversine, CEP, along/cross track, heading error
│       ├── test_outage.py                # Tests for deterministic window placement and real-gap avoidance
│       ├── test_smartphone.py            # Tests for phone ingestion, header fixes, and unit conversions
│       ├── test_split.py                 # Tests for group assignments and partition invariants
│       ├── test_summary.py               # Tests for leaderboard aggregation and median/p95 calculations
│       ├── test_sync.py                  # Tests for cross-correlation lag estimation and interpolation
│       └── test_vehicle_can.py           # Tests for CAN ingestion, header fixes, and Parquet output
│
├── android/                              # Phase 3 Android application shell
│   └── README.md                         # Kotlin + Android NDK JNI bridge specification
├── edge/                                 # Phase 3 Linux edge daemon shell
│   └── README.md                         # POSIX daemon specification for high-frequency IMUs
│
├── data/                                 # Data storage (gitignored except .gitkeep)
│   ├── raw/                              # Original IO-VNBD raw dataset CSVs
│   └── processed/                        # Standardized Parquet files
│       ├── vehicle_can/                  # 90 sanitized CAN Parquet files + manifest.csv
│       ├── smartphone/                   # 97 sanitized smartphone Parquet files + manifest.csv
│       ├── paired/                       # 72 time-aligned 10 Hz 57-column Parquet files + manifest.csv
│       ├── splits/                       # Train/Val/Test manifest and group mappings
│       ├── outages/                      # 377 outage-masked Parquet files partitioned by split
│       ├── _cpp_replay_cache/            # Pre-exported lightweight CSV streams for C++ driver
│       └── cpp_predictions/              # Output 10 Hz trajectory predictions from C++ engine
│
├── docs/                                 # Architectural specifications and ADRs
│   ├── architecture.md                   # System architecture reference document
│   ├── data-audit.md                     # Exhaustive empirical audit of IO-VNBD dataset (31 KB)
│   └── decisions/                        # Architectural Decision Records (ADRs)
│       ├── 0001-test-framework.md        # GoogleTest choice for C++ core
│       ├── 0002-column-renames.md         # Canonical CAN column schema and unit fixes
│       ├── 0003-phone-column-renames.md   # Canonical phone schema reconciliation and speed fix
│       ├── 0004-paired-sync-alignment.md  # Cross-correlation synchronization policy
│       ├── 0004-split-grouping-key.md     # 41-group anti-leakage partitioning strategy
│       └── 0005-cpp-parquet-io.md         # Intermediate replay cache strategy for C++
│
├── results/                              # Benchmark metrics and visual artifacts
│   ├── leaderboard.csv                   # Raw instance-level benchmark scores (377 entries)
│   ├── leaderboard_summary.csv           # Grouped aggregate benchmark statistics per outage duration
│   └── plots/                            # High-resolution benchmark figures (300 DPI)
│       ├── error_vs_outage_duration.png  # Log-scale position error vs outage duration
│       ├── error_ratio_bar.png           # Bar chart of Strapdown ÷ CV divergence ratio
│       └── sample_trajectories.png       # ENU tangent plane trajectory comparisons
│
├── CMakeLists.txt                        # Top-level CMake entry point
├── .gitignore                            # Comprehensive ignore rules for build, data, and cache
└── README.md                             # Project documentation (this file)
```

---

## Engineering & Research Highlights

### IO-VNBD Dataset Audit & Bug Corrections
An empirical audit (`docs/data-audit.md`) of all 564 raw CSV files (1.71 GB) in the IO-VNBD benchmark revealed critical flaws that would invalidate dead-reckoning models if consumed naively:
1. **The 3.6× Speed Mislabelling Bug**: In 94 of the 97 smartphone runs, the column titled `GPS SPEED (Kmh)` was actually recorded in **m/s** (Android native `Location.getSpeed()`). In 3 runs (`S-A2`, `S-A9`, `S-A10`), it was recorded in true km/h. Ingesting without inspection causes a 360% velocity error.
2. **The 1000× Elevation Bug**: In all 323 CAN files, the header declared `Height (km)`, but recorded values ranged from 20.19 to 534.26, which represents **meters above sea level** in the UK. Treating them as kilometers creates a 1000× elevation error.
3. **The Pedal Position Bug**: Raw CAN declared `Accelerator Pedal Position (0 or 1)`, suggesting a boolean, but recorded values were continuous percentages (0.0% to 99.0%).
4. **Delimiter Corruption**: `S-A4.csv` contained a double-comma delimiter at column 6 across all 32,829 rows, shifting all downstream columns by one. Repaired programmatically on ingestion without data loss.
5. **Excel Date Mangling**: Spreadsheet software mangled satellite ratios (`"8 / 20"`) into date strings (`"Aug-20"`). Parsed with robust regex into nullable integers.
6. **Missing Sensors**: 9 French runs (`S-T1` through `S-T9`, 603,425 samples) completely lack magnetometer and orientation channels. Ingested with genuine PyArrow nulls (`np.nan`), verifying the core engine's requirement to operate without magnetometers.

### Time Synchronization & Sub-Sample Grid Alignment
- CAN timestamps record wall-clock seconds from midnight UTC (29,606s to 72,986s), while smartphone timestamps record elapsed milliseconds from sensor boot (~0s).
- Evaluated normalized cross-correlation of GPS velocities over a $\pm 30.0$s search window ($\pm 300$ samples) to extract optimal lag $\tau^*$ and Pearson correlation $r^*$.
- Strictly aligned on the temporal intersection $[t_{\text{start}}, t_{\text{end}}]$ on a uniform 10 Hz grid without extrapolation.
- Implemented channel-specific interpolation:
  - **Kinematics & Odometry**: Linear interpolation.
  - **Heading & Azimuth**: Circular unwrapping before interpolation to prevent boundary wrap corruption ($359^\circ \to 1^\circ$ averaging to $180^\circ$).
  - **Discrete Flags & Counters**: Forward-fill step interpolation (preventing non-existent fractional gears or booleans).

### Leakage-Safe Partitioning (41 Atomic Groups)
- Naive splitting causes session and cross-stream leakage.
- Defined atomic grouping key: $\text{Group ID} = \text{Driver} \times \text{Campaign / Multi-Leg Trip}$ yielding **41 atomic groups**.
- Solved the **Driver E imbalance** (Driver E accounted for 88.9% of all paired runs) by decomposing Driver E into 5 regional campaigns (Bradford, Peak District, Derbyshire, West Midlands).
- Target split ratios achieved by duration:
  - **Train**: 68.7% (28 groups, 48.54 hours, 2,697.9 km)
  - **Validation**: 14.6% (8 groups, 10.31 hours, 634.1 km)
  - **Test**: 16.7% (5 groups, 11.77 hours, 730.8 km)
- Guaranteed representation of all 8 hard anomaly cohorts in Validation and Test.

### Deterministic GNSS Outage Simulation
- Synthetic outage durations: **10s, 30s, 60s, 120s, 180s**.
- Placed deterministically using SHA-256 hash seeding per `(run_id, duration, index)`.
- Scans and avoids authentic recording drops ($dt > 2.0$s) to prevent confounding synthetic outages with real sensor loss.
- Enforces 10.0s edge buffers before and after outages for filter convergence and terminal drift computation.
- Sets all GNSS channels to genuine null/NaN during the outage while keeping IMU and odometry channels intact.
- Extracts independent cross-stream ground truth (CAN ground truth for phone outages).

### C++17 Strapdown INS Mechanization
- **Attitude Propagation**: Propagates unit quaternion $q_{b}^{n}$ using body angular rate vector $\mathbf{\omega}$ over interval $\Delta t$:
  $$\Delta \mathbf{\theta} = \mathbf{\omega} \Delta t, \quad \Delta q = \left[ \cos\left(\frac{\|\Delta \mathbf{\theta}\|}{2}\right), \frac{\Delta \mathbf{\theta}}{\|\Delta \mathbf{\theta}\|} \sin\left(\frac{\|\Delta \mathbf{\theta}\|}{2}\right) \right], \quad q_{k+1} = q_k \otimes \Delta q$$
- **Specific Force Resolution & Gravity Decoupling**:
  $$\mathbf{f}^{n} = R(q) \mathbf{f}^{b}, \quad \mathbf{a}^{n} = \mathbf{f}^{n} - [0, 0, g_0]^T, \quad g_0 = 9.80665 \text{ m/s}^2$$
- **Numerical Integration**: 2nd-order trapezoidal integration for velocity and position in local tangent plane ENU.
- **Geodetic Conversion**: Tangent plane ENU displacements mapped back to WGS-84 coordinates using mean Earth radius ($R = 6,371,000$ m).

---

## Benchmark Leaderboard & Empirical Findings

The dead-reckoning benchmark compares two navigation models across **377 outage instances** (253 paired instances verified against independent CAN bus ground truth):
1. **`baseline_cv_heading_v1`**: Naive Constant-Velocity / Constant-Heading great-circle propagation.
2. **`baseline_cpp_strapdown_v1`**: Dependency-free C++17 Strapdown INS integrating raw phone accelerometer and gyroscope data.

### Summary Results on Ground-Truth-Verified Instances (Paired $N=253$)

| Outage Duration | Paired $N$ | CV Median Error (m) | CV 95th Percentile (m) | C++ Strapdown Median Error (m) | C++ Strapdown 95th Percentile (m) | Strapdown ÷ CV Ratio | CV Median % of Distance | Strapdown Median % of Distance |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10s** | 67 | **34.1 m** | 121.1 m | 87.1 m | 381.4 m | **2.6×** | 34.0% | 76.9% |
| **30s** | 58 | **144.4 m** | 507.2 m | 1,037.2 m | 3,416.8 m | **7.2×** | 58.4% | 421.0% |
| **60s** | 52 | **364.9 m** | 1,111.8 m | 5,368.7 m | 12,811.7 m | **14.7×** | 61.7% | 943.7% |
| **120s** | 41 | **754.3 m** | 2,215.1 m | 26,025.0 m | 52,362.5 m | **34.5×** | 65.3% | 1,948.5% |
| **180s** | 35 | **1,142.3 m** | 3,238.9 m | 47,929.0 m | 108,790.9 m | **42.0×** | 64.4% | 2,711.0% |

### Key Scientific Insights
1. **Cubic Error Explosion in Uncorrected Strapdown INS**:
   - For short 10s outages, the C++ Strapdown INS achieves reasonable dead reckoning (87.1 m median error).
   - For longer outages, uncorrected smartphone MEMS bias causes attitude tilt, leaking Earth's gravity ($9.81 \text{ m/s}^2$) into the horizontal integration channels. This produces the classical cubic error divergence:
     $$\text{Error}(t) \approx \frac{1}{6} g \, \omega_{\text{bias}} \, t^3$$
   - At 180 seconds, uncorrected double-integration diverges to a median error of **47.9 km** (42.0× worse than constant-velocity coasting).
2. **The Constant-Velocity Baseline Floor**:
   - The naive CV baseline coasts forward assuming speed and heading remain unchanged from the pre-outage window.
   - Its error grows **linearly** ($\approx 6.3 \text{ m/s}$ error rate), providing an effective baseline floor against which advanced sensor fusion models will be evaluated.
3. **Imperative for Phase 2 Fusion**:
   - These empirical findings prove that raw double-integration of consumer MEMS IMUs without velocity aiding cannot sustain navigation past 15–20 seconds.
   - To beat the CV baseline at 30s–180s, the system requires **Zero-Velocity Updates (ZUPT)**, **Non-Holonomic Constraints (NHC)**, and **Learned Deep Inertial Odometry**.

---

## Prerequisites & Quick Start

### 1. C++ Core Engine Build & Test

**Prerequisites**: CMake ($\ge 3.16$), C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+).

```bash
# Clone the repository
git clone https://github.com/your-org/idr-project.git
cd idr-project

# Configure CMake build directory
cmake -B build

# Build library, replay CLI, and unit tests
cmake --build build --config Release

# Run C++ unit test suites via CTest
ctest --test-dir build --output-on-failure
```

### 2. Python Data & Evaluation Pipeline

**Prerequisites**: Python 3.9+ with `pip`.

```bash
# Install dataeval in editable mode
pip install -e ./dataeval

# Verify installation with unit test execution (58 tests)
python -m unittest discover -s dataeval/tests
```

---

## End-to-End Pipeline Execution

To reproduce the entire data ingestion, synchronization, partitioning, outage simulation, replay, and leaderboard scoring from scratch:

```bash
# 1. Ingest raw CAN CSV files to Parquet
python -m dataeval.ingest.run_can_ingest \
  --raw-dir data/raw/io-vnbd \
  --out-dir data/processed/vehicle_can

# 2. Ingest raw Smartphone CSV files to Parquet
python -m dataeval.ingest.run_phone_ingest \
  --raw-dir data/raw/io-vnbd \
  --out-dir data/processed/smartphone

# 3. Time-synchronize and align paired CAN/Phone runs (10 Hz grid)
python -m dataeval.harness.run_sync \
  --processed-root data/processed

# 4. Partition dataset into Train/Val/Test splits (41 atomic groups)
python -m dataeval.harness.run_split \
  --processed-root data/processed

# 5. Generate synthetic GNSS outages across splits (10s to 180s)
python -m dataeval.harness.run_outage_sim \
  --processed-root data/processed

# 6. Score Python Constant-Velocity baseline
python -m dataeval.harness.run_scoring \
  --outages-dir data/processed/outages \
  --leaderboard-csv results/leaderboard.csv \
  --config baseline_cv_heading_v1

# 7. Pre-export lightweight replay cache for C++ engine
python -m dataeval.harness.export_replay_cache \
  --outages-dir data/processed/outages \
  --output-dir data/processed/_cpp_replay_cache

# 8. Execute C++ Strapdown INS batch replay
./build/core/idr_replay \
  --cache-dir data/processed/_cpp_replay_cache \
  --out-dir data/processed/cpp_predictions \
  --batch

# 9. Score C++ Strapdown INS baseline against ground truth
python -m dataeval.harness.run_scoring \
  --outages-dir data/processed/outages \
  --leaderboard-csv results/leaderboard.csv \
  --prediction-dir data/processed/cpp_predictions \
  --config baseline_cpp_strapdown_v1

# 10. Generate summary tables and publication-ready plots
python -m dataeval.harness.make_plots \
  --leaderboard-csv results/leaderboard.csv \
  --summary-csv results/leaderboard_summary.csv \
  --outages-dir data/processed/outages \
  --prediction-dir data/processed/cpp_predictions \
  --plots-dir results/plots
```

---

## Automated Testing & CI

Continuous Integration is configured via GitHub Actions (`.github/workflows/ci.yml`), validating on every pull request and commit to `master`:
- Clean C++17 build under GCC/Clang/MSVC.
- Execution of all C++ unit tests (`idr_tests`, `idr_strapdown_tests`).
- Python package installation and full execution of the 58-test suite in `dataeval/tests/`.

### Running All Tests Locally:
```bash
# Run C++ tests
ctest --test-dir build --output-on-failure

# Run Python tests
python -m unittest discover -s dataeval/tests
```

---

## Architectural Decision Records (ADRs)

Key architectural and scientific decisions are documented under `docs/decisions/`:
- [ADR 0001: Choice of GoogleTest for Core Unit Testing](docs/decisions/0001-test-framework.md)
- [ADR 0002: Canonical Column Renaming and Unit Corrections for CAN Dataset](docs/decisions/0002-column-renames.md)
- [ADR 0003: Canonical Column Renaming and Schema Reconciliation for Smartphone Dataset](docs/decisions/0003-phone-column-renames.md)
- [ADR 0004: Paired CAN and Smartphone Time Synchronization and Common Grid Alignment](docs/decisions/0004-paired-sync-alignment.md)
- [ADR 0004: Dataset Grouping Key and Leakage-Safe Train/Val/Test Partitioning](docs/decisions/0004-split-grouping-key.md)
- [ADR 0005: Intermediate Replay Cache Strategy for C++ Driver on Windows](docs/decisions/0005-cpp-parquet-io.md)

---

## Project Roadmap

- [x] **Phase 1: Foundation & Data Infrastructure**
  - [x] IO-VNBD empirical audit and bug identification.
  - [x] Parquet ingestion pipeline with strict schema enforcement.
  - [x] Doppler cross-correlation time synchronization.
  - [x] 41-group anti-leakage train/validation/test partitioning.
  - [x] Synthetic GNSS outage simulator (10s to 180s) with real-gap avoidance.
  - [x] Constant-velocity great-circle baseline navigation model.
  - [x] Seven locked leaderboard metrics evaluation engine.
  - [x] C++17 dependency-free core Strapdown INS engine and replay driver.
  - [x] Baseline leaderboard benchmark comparing CV vs Strapdown INS.
- [ ] **Phase 2: Advanced Sensor Fusion & Estimation**
  - [ ] Error-State Extended Kalman Filter (ES-EKF) 15-state estimator (position, velocity, attitude, accel bias, gyro bias).
  - [ ] Zero-Velocity Detection (ZUPT) and Zero Angular Rate Updates (ZARU) for stationary stops.
  - [ ] Non-Holonomic Constraints (NHC) modeling wheeled vehicle lateral/vertical velocity suppression.
  - [ ] Magnetometer calibration and heading fusion.
- [ ] **Phase 3: Learning-Aided Navigation & Embedded Frontends**
  - [ ] Deep neural inertial velocity network (1D Dilated CNN / TCN / LSTM) predicting forward displacement vectors.
  - [ ] Hybrid EKF-learned velocity measurement integration.
  - [ ] Android application shell (`android/`) with Kotlin UI and JNI C++ bindings for live phone navigation.
  - [ ] Linux edge daemon (`edge/`) for ~200 Hz FOG-grade navigation hardware.
