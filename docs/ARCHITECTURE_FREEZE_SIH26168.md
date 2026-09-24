# OFFICIAL ARCHITECTURE SPECIFICATION & FREEZE: SIH26168
**Project:** iNAV — Hybrid Smartphone-Inertial / CAN / Map-Constrained Personal Vehicle Navigation System  
**Status:** 🔒 **ARCHITECTURE FROZEN (DO NOT MODIFY)**  
**Date:** September 13, 2026  
**Reference Benchmark:** 56 Held-Out Real-World GNSS Outages across 20 Urban and Motorway Trajectories (IO-VNBD Dataset).

---

## 1. Frozen Production Architecture Overview

The official, frozen production navigation architecture consists of:

```
                    ┌──────────────┐
                    │     GNSS     │
                    │ Position +   │
                    │ Velocity     │
                    └──────┬───────┘
                           │
                    GNSS Health
                           │
              ┌────────────┴────────────┐
              │                         │
          GNSS Healthy              GNSS Lost
              │                         │
              ▼                         ▼
       GNSS correction          Dead Reckoning
                                        │
              ┌─────────────────────────┤
              │                         │
              ▼                         ▼
       Smartphone IMU             CAN Vehicle Speed
       Accel + Gyro                     │
              │                         │
              └───────────┬─────────────┘
                          ▼
                   ┌─────────────┐
                   │  7-State    │
                   │     UKF     │
                   └──────┬──────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        ZUPT / ZARU              OSM / HMM Map
       Stationary bias          Road association
          correction                    │
                                       ▼
                              Passive Map Protection
                              ┌───────────────────┐
                              │ Viterbi Margin    │
                              │ + 30° Gate        │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              1D Road-Normal
                              Map Constraint
                                        │
                         ┌──────────────┘
                         ▼
                  Final iNAV State
                 Position + Heading
                         │
                         ▼
                    Navigation UI
```

---

## 2. Component Specifications & Boundaries

### 2.1 GNSS Subsystem
- **Role:** Absolute position and ground speed vector reference when healthy ($3\text{D Fix}$, $\ge 5$ satellites, $\text{HDOP} \le 2.5$).
- **Pre-Outage Anchoring:** Anchors initial latitude, longitude, speed, and course angle before outage onset. Calibrates effective rolling radius $r_\text{eff}$ against wheel pulses.
- **Outage Action:** Seamless transition to dead reckoning.
- **Strict Prohibitions:**
  - **NO** dynamic Doppler $\to$ gyro-bias locking (Phase 19A proved that smartphone GPS Doppler lag of $0.8\text{s}$–$1.5\text{s}$ hallucinates false biases up to $\pm 5.7^\circ/\text{s}$).
  - **NO** pre-outage course $\to b_{gz}$ overwriting.

### 2.2 Smartphone IMU
- **Sensors:** 3-axis accelerometer and 3-axis MEMS gyroscope sampled at $\ge 100\text{ Hz}$, downsampled/filtered to $10\text{ Hz}$ synchronous update.
- **Alignment:** Pre-drive Body-to-Vehicle frame alignment ($R_b^v$) determined by gravity leveling and forward acceleration projection.
- **Role:** High-rate propagation of vehicle acceleration and angular rate.
- **Strict Prohibitions:**
  - **NO** artificial heading snapping or heuristic rate overrides.
  - Gyro and accelerometer bias states remain estimated inside the filter.

### 2.3 CAN Forward Vehicle Speed
- **Role:** Ground-truth-grade longitudinal forward velocity reference derived from rear wheel speed sensors:
  $$v_\text{CAN} = \frac{\omega_{RL} + \omega_{RR}}{2} \cdot r_\text{eff}$$
- **Slip Protection:** Gated out when wheel speed differential $|\omega_{RL} - \omega_{RR}| > 25\text{ rad/s}$ (traction loss / wheel spin).
- **Measurement Model:** 1D linear observation in vehicle forward axis with $\sigma_v = 0.15\text{ m/s}$.
- **Strict Prohibitions:**
  - **CAN speed is never replaced** by neural velocity, map arc-length, or differential wheel yaw.

### 2.4 7-State Unscented Kalman Filter (UKF)
- **State Vector:**
  $$x = \begin{bmatrix} p_N & p_E & v & \psi & a_x & \omega_z & \kappa \end{bmatrix}^T$$
  where $p_N, p_E$ are local tangent coordinates (meters), $v$ is forward speed (m/s), $\psi$ is yaw angle (rad), $a_x$ is longitudinal acceleration (m/s$^2$), $\omega_z$ is yaw rate (rad/s), and $\kappa$ is road curvature ($1/\text{m}$).
- **Process Model:** Continuous kinematic bicycle / dead-reckoning model integrated with dt $= 0.1\text{ s}$.
- **Status:** **FROZEN.** No replacement by 9-state velocity EKF or 15-state ES-EKF.

### 2.5 Standstill ZUPT / ZARU
- **Detection Criteria:** Standstill verified **only** when:
  $$|v_\text{CAN}| < 0.5\text{ m/s} \quad \text{AND} \quad \|\boldsymbol{\omega}_\text{IMU}\| < 0.05\text{ rad/s} \quad \text{for } \ge 0.5\text{ s}$$
- **Zero Velocity Update (ZUPT):** Strongly constrains forward velocity ($v = 0$, $\sigma = 0.01\text{ m/s}$).
- **Zero Angular Rate Update (ZARU):** Observes z-axis gyro bias ($b_{gz} = \omega_z$, $\sigma = 0.005\text{ rad/s}$), collapsing bias covariance by $144.2\times$.
- **Boundary:** ZARU is an auxiliary stationary correction mechanism. It is **not** assumed to hold indefinitely during prolonged highway motion.

### 2.6 OSM / HMM 1D Road-Normal Map Matching
- **Map Model:** Canonical OpenStreetMap road network loaded into spatial R-tree index.
- **Constraint Geometry:** 1D Road-Normal position update:
  $$z_\text{map} = (p_N - p_N^\text{snap}) \sin(\psi_\text{road}) - (p_E - p_E^\text{snap}) \cos(\psi_\text{road}) = 0$$
  Restricts vehicle laterally to the road corridor without altering along-track progress.
- **Update Rate:** $2\text{ Hz}$ ($\Delta t = 0.5\text{ s}$).
- **Strict Prohibitions:**
  - **NO** continuous map arc-length distance injection.
  - **NO** 2D direct position snapping.
  - **NO** UKF road-heading fusion ($z \neq \psi_\text{road}$).

### 2.7 Passive Map Protection Layer (Phase 19B)
- **Topological Gate (Viterbi Path Margin Gate):**
  $$\mathcal{M} = \frac{P_\text{winner}}{P_\text{runner-up}} \ge 3.0$$
  If the margin between the best road candidate and a competing distinct edge falls below $3.0$, the map update is **suppressed**.
- **Geometric Gate ($30^\circ$ Heading Consistency Gate):**
  $$\Delta\psi = \left|\text{wrapToPi}(\psi_\text{filter} - \psi_\text{road})\right| \le 30^\circ$$
  If filter heading differs from candidate road heading by $> 30^\circ$, the update is **suppressed**.
- **Principle:** When map evidence is ambiguous, **ABSTAIN**. Never force a correction.

---

## 3. Explicitly Killed Experiments (Quarantine Registry)

The following architectures and methods have been rigorously evaluated on the 56-outage held-out benchmark, failed objective performance and safety criteria, and are **permanently excluded** from the production pipeline:

| Phase / Method | Mechanism | Benchmark Outcome | Reason for Exclusion |
| :--- | :--- | :--- | :--- |
| **Phase 17B** | Direct UKF Road-Heading Fusion ($z = \psi_\text{map}$) | Median FPE: 134.2 m (Degraded all 14 failures) | Overwrites natural yaw dynamics with road azimuth; catastrophic on curves and forks. |
| **Phase 19 / MAPHDE** | Continuous Gyro Drift Rate Feedback ($r_\text{drift} = I / \Delta t$) | Max FPE exploded to 3135 m; worsened 9/14 failures | Held integrator acts as permanent artificial yaw rate disturbance during suspensions. |
| **Phase 19C** | Paper-Faithful MAPHDE ($\Delta\psi = 0$ in suspension) | Concept valid, but FPE 88.9 m; 10 degraded outages | Pedestrian street-alignment assumption fails on automotive ramps, curves, and merges. |
| **Phase 16B1 / 16B2** | Continuous Differential Wheel-Yaw Fusion | High-speed tire deflection induced false $0.4^\circ/\text{s}$ yaw bias | Uncalibrated tire radius differences corrupt long-duration yaw integration. |
| **Phase 19A** | Pre-Outage Doppler $b_{gz}$ Lock | Median FPE: 185.3 m, Max FPE: 3980 m | Smartphone Doppler filtering lag ($0.8$–$1.5\text{s}$) hallucinates $\pm 5.7^\circ/\text{s}$ false biases. |
| **Pillar 1 Neural Velocity** | 4s Temporal CNN / VelocityNet | Replaced by direct CAN speed fusion | CAN forward speed is physically exact ($0.15\text{ m/s}$) and drift-free. |
| **9-State EKF** | Velocity state EKF | Rejected | 7-State UKF provides superior kinematic consistency and non-linear propagation. |

---

## 4. Benchmark Performance of Frozen Production Architecture

Across the complete 56-outage held-out benchmark (durations 10s to 180s, total distance 31.8 km):

| Outage Duration | Scenario Count | Median FPE | Mean FPE | Median Drift % | Outages with Drift < 10% |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10 s** | 19 | **13.00 m** | 19.45 m | **12.2%** | 9 / 19 (47.4%) |
| **30 s** | 15 | **123.63 m** | 141.20 m | **23.4%** | 4 / 15 (26.7%) |
| **60 s** | 10 | **465.42 m** | 430.12 m | **57.4%** | 1 / 10 (10.0%) |
| **120 s** | 8 | **1112.57 m** | 1085.40 m | **62.6%** | 0 / 8 (0.0%) |
| **180 s** | 4 | **1719.60 m** | 1650.30 m | **67.3%** | 0 / 4 (0.0%) |
| **ALL OUTAGES** | **56** | **108.69 m** | **416.29 m** | **25.79%** | **14 / 56 (25.0%)** |

*With Passive Dual Protection (Phase 19B), all 14 Phase 18A catastrophic map-latching failures are safeguarded with zero regression on the 42 normal scenarios.*

---

## 5. Architectural Principle & Honest Scientific Limitations

### Core Design Principle:
> **"Use each information source only for the quantity it has demonstrated that it can reliably constrain."**
- **GNSS:** Absolute 3D position and ground velocity when healthy.
- **IMU:** High-rate ($100\text{ Hz}$) short-term relative motion.
- **CAN:** Pure longitudinal forward velocity.
- **ZUPT / ZARU:** Stationary velocity zeroing and stationary gyro bias capture.
- **OSM / HMM:** Road-corridor lateral constraint.
- **Dual Gates:** Protection against wrong-road attraction.

### Recognized Physical Limitation:
We state explicitly and honestly:
> **The frozen architecture does NOT claim to provide full continuous gyro-bias observability during long (60s–180s) uninterrupted dynamic GNSS outages.**  
> Inertial heading drift during prolonged high-speed highway driving without stops remains bounded only by the physical quality of the smartphone MEMS gyroscope and lateral road-normal boundaries. Future research branches may explore external physical references (e.g. Visual Inertial Odometry or magnetometer disturbance estimators), but such explorations must remain strictly quarantined in research branches and will not alter this frozen production baseline.
