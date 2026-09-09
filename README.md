# 🧭 iNAV: AI-ML Intelligent Dead Reckoning System

**iNAV** is an Intelligent Dead Reckoning (IDR) navigation engine designed for smartphone sensors during GNSS-denied environments (tunnels, underground parking, dense urban canyons, and forests). It fuses deep-learning displacement prediction, kinematic constraints (NHC, ZUPT), Unscented Kalman Filtering (UKF), and OpenStreetMap (OSM) Hidden Markov Model (HMM) map-matching to achieve < 10% drift without external GNSS fixes.

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph S["Sensor Input (Smartphone / OBD-II)"]
        IMU["IMU (Acc, Gyro, Mag) @ 10-100Hz"]
        GPS["GNSS Position & Velocity (when available)"]
    end

    subgraph L1["Layer 1: Preprocessing & Auto-Alignment"]
        ALIGN["Phone-to-Vehicle Alignment Engine (Gravity + PCA)"]
        LPF["Vibration Filter & Coordinate Transform (b -> v)"]
        IMU --> ALIGN --> LPF
    end

    subgraph L2["Layer 2: AI Displacement Network (Module B)"]
        WIN["2.0s Sliding Window (10Hz, 20 samples)"]
        CNN_GRU["1D-CNN + Dilated Convs + GRU"]
        HEAD1["Head 1: Forward Displacement Δd (Huber Loss)"]
        HEAD2["Head 2: Event Classifier (Cruise, Idle, Pothole, Turn)"]
        HEAD3["Head 3: Learned Uncertainty σ(Δd) (NLL)"]

        LPF --> WIN --> CNN_GRU
        CNN_GRU --> HEAD1
        CNN_GRU --> HEAD2
        CNN_GRU --> HEAD3
    end

    subgraph L3["Layer 3: Fusion & Navigation Filter (Module C & E)"]
        UKF["Unscented Kalman Filter (UKF)"]
        NHC["Non-Holonomic Constraints (Zero Lateral/Vertical Vel)"]
        ZUPT["Zero-Velocity Update (Stationary Reset)"]
        SCALE["Scale Factor k (Cross-Vehicle Adaptation)"]
        GNSS_QC["GNSS Deficit & Re-acquisition Quarantine"]

        HEAD1 --> UKF
        HEAD3 --> UKF
        HEAD2 --> ZUPT
        NHC --> UKF
        SCALE --> UKF
        GPS --> GNSS_QC --> UKF
    end

    subgraph L4["Layer 4: Map Constraints & Output (Module D & F)"]
        HMM["OSM Road Graph HMM Matcher"]
        OUT["Continuous Trajectory Solution (Lat, Lon, Heading, Vel)"]
        UKF --> HMM --> OUT
    end
```

---

## 📁 Directory Structure

```
iNAV/
├── config.py              # Central config: dataset paths, column mappings, unit conversions, splits
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/
│   ├── ingest.py          # IO-VNBD dataset parser, cleaner, and Parquet converter
│   ├── sync.py            # Phone ↔ CAN cross-correlation time-alignment & 10Hz resampling
│   └── windowize.py       # Sliding window feature extractor & train/val/test splits
├── eval/
│   ├── outage_sim.py      # GNSS blackout simulator (10, 30, 60, 120, 180s)
│   ├── replay.py          # Trajectory replay engine (runs IMU through models/filters)
│   ├── score.py           # Benchmark metrics: ATE, drift %, CEP50/95, cross/along-track
│   └── baseline.py        # Strapdown double-integration & constant velocity baselines
├── modules/
│   ├── alignment.py       # Phone-to-vehicle alignment engine (static + dynamic)
│   ├── velocity_net.py    # 1D-CNN + GRU multi-head displacement & uncertainty network
│   ├── ukf.py             # Unscented Kalman Filter with NHC, ZUPT, and scale factor k
│   └── gnss_handler.py    # GNSS health check & smooth quarantine re-acquisition
├── viz/
│   ├── plot_trajectory.py # Trajectory plotting on Folium / Matplotlib
│   └── plot_sensors.py    # Raw sensor timeseries and noise analysis
├── models/                # Saved weights and ONNX models
└── results/               # Experiment logs and leaderboard.csv
```

---

## 🎯 Benchmark Targets

| Metric | ISRO / PS Target | iNAV Engineering Target |
|---|---|---|
| Dead Reckoning Drift | < 10% distance | **< 5% distance** |
| 50m Outage (< 1 min) | < 5m drift | **< 3.5m drift** |
| 1km Outage (@ 60 km/h) | < 100m drift | **< 60m drift** |
| Update Rate | 10 Hz | **10 Hz Filter, 60 fps UI** |

---

## 🚀 Quick Start: Testing the Final Model

To verify the final **VelocityNet** Dead Reckoning model on real-world driving trajectory data:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run instant benchmark on included synchronized Parquet trajectory
python test_final_model.py

# 3. Test on full 8-minute motorway trajectory with GNSS blackouts
python test_final_model.py --parquet data/sample_test_trajectory_motorway.parquet

# 4. Verify all core pillars and integration tests
python eval/test_5pillars.py
```
