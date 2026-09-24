# Phase 12: 9-State Velocity-State EKF Architectural Experiment Report

**Project**: iNAV Dead-Reckoning Sensor Fusion Engine  
**Experiment**: Phase 12 — 9-State Velocity-State EKF vs. 7-State UKF Baseline  
**Evaluation Set**: 56 Held-Out Real Vehicle Blackout Scenarios (IO-VNBD Benchmark)  
**Date**: September 2026  
**Status**: EXPERIMENT COMPLETE — VERDICT: **KILL / DO NOT INTEGRATE AS LONGITUDINAL DRIFT SOLUTION**

---

## 1. Executive Summary & Verdict

| Architectural Candidate | Median Drift % | Mean Drift % | P95 Drift % | Median FPE | Mean FPE | Mean Cross-Track | Drift < 10% Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: 7-State UKF Baseline (Phase 11 Model)** | **40.92%** | **54.14%** | 115.05% | 192.13 m | 455.22 m | -87.76 m | 1 / 56 (1.8%) |
| **Model B: 9-State EKF (Full NHC + AI Speed)** | **37.31%** | **54.49%** | 109.17% | 177.14 m | 439.21 m | **-23.00 m** | 1 / 56 (1.8%) |
| **Model C: 9-State EKF (Ablation: NO NHC)** | **46.33%** | **71.79%** | 194.19% | 213.69 m | 532.77 m | -60.80 m | 1 / 56 (1.8%) |

### The Core Architectural Question
> *"Did explicit world-frame velocity states $[v_E, v_N, v_U]$ make longitudinal velocity sufficiently observable to materially reduce GNSS-denied position drift to the $< 10\%$ target?"*

### **VERDICT: KILL**
**Definitive Evidence:**
1. **Along-Track Observability is Structurally Identical**: During a GNSS blackout, GNSS velocity observations $z = [v_E, v_N, v_U]$ are completely absent. The only velocity update available is scalar forward velocity from the Phase 11 neural model ($z = v_{AI}$). In the 9-state EKF, the AI measurement $h(\mathbf{x}) = v_E \sin\psi + v_N \cos\psi$ and the lateral NHC $h(\mathbf{x}) = -v_E \cos\psi + v_N \sin\psi = 0$ mathematically constrain $(v_E, v_N)$ to the forward vector $v_{AI} [\sin\psi, \cos\psi]^T$. This is **algebraically isomorphic** to the 7-state unicycle model ($v_{fwd}$). Consequently, median along-track lag did not decrease ($-70.69\text{ m}$ in 9-state vs. $-65.45\text{ m}$ in 7-state), and the target pass rate remained dead-locked at **1/56 (1.8%)**.
2. **Cross-Track Containment is Real, But Not Longitudinal**: Model B reduced mean cross-track error by $73.8\%$ (from $-87.76\text{ m}$ down to $-23.00\text{ m}$) through Kalman NHC innovation feedback. However, lateral constraint satisfaction does **not** cure accelerometer forward bias integration or neural velocity scaling errors.
3. **Severe Degradation without NHC (Ablation C)**: Removing NHC causes rapid catastrophic divergence, with P95 drift exploding to **194.19%** and mean FPE reaching **532.77 m**, proving that an unconstrained 9-state velocity filter is intrinsically ill-conditioned in dead reckoning.

**Recommendation**: Retain the lightweight, locked **7-state UKF** in production. Do not inflate the state vector or runtime complexity for zero gain in longitudinal drift.

---

## 2. Objective & Architectural Hypothesis

In automotive dead reckoning, conventional filters either use a body-frame speed state $v_{fwd}$ (such as the production 7-state UKF) or decompose velocity into world Cartesian coordinates $[v_E, v_N, v_U]$. 

The hypothesis tested in Phase 12 was:
$$\text{Hypothesis: } \mathbf{x}_{vel} = [v_E, v_N, v_U]^T \text{ anchored by GNSS velocity pre-outage and updated via AI forward speed + full NHC will eliminate longitudinal lag.}$$

Specifically, we evaluated whether maintaining separate Cartesian velocity states allows the Kalman filter to better preserve inertia, mitigate vehicle pitch/alignment tilt errors, and recover from post-outage braking or acceleration events.

---

## 3. State Formulation

The experimental 9-state Extended Kalman Filter vector is defined in local East-North-Up (ENU) coordinates:
$$\mathbf{x} = \begin{bmatrix} p_E \\ p_N \\ p_U \\ v_E \\ v_N \\ v_U \\ \psi \\ b_a \\ b_g \end{bmatrix} \in \mathbb{R}^9$$

Where:
- $p_E, p_N, p_U$: Position in local tangent plane (meters, East-North-Up).
- $v_E, v_N, v_U$: World-frame linear velocity components (m/s).
- $\psi$: Vehicle heading angle (radians, clockwise from North: $0 = \text{North}, \frac{\pi}{2} = \text{East}$).
- $b_a$: Longitudinal/body accelerometer bias ($\text{m/s}^2$).
- $b_g$: Yaw gyroscope bias ($\text{rad/s}$).

Initial covariance $\mathbf{P}_0 = \text{diag}([25, 25, 25, 1.0, 1.0, 1.0, (5^\circ \cdot \frac{\pi}{180})^2, (0.05)^2, (10^{-3})^2])$.  
Process noise $\mathbf{Q} = \text{diag}([0.01, 0.01, 0.01, 0.09, 0.09, 0.04, (0.2^\circ \cdot \frac{\pi}{180})^2, (10^{-4})^2, (10^{-6})^2])$.

---

## 4. Kinematic IMU Propagation Equations

At each IMU step ($10\text{ Hz}$, $\Delta t = 0.1\text{ s}$):
1. **De-bias sensor inputs**:
   $$\tilde{\omega}_z = \omega_{z,\text{gyro}} - b_g, \quad \tilde{a}_{fwd} = a_{fwd,\text{acc}} - b_a$$

2. **Midpoint Heading Integration**:
   $$\psi_{mid} = \psi_k + \frac{1}{2} \tilde{\omega}_z \Delta t, \quad \psi_{k+1} = \left(\psi_k + \tilde{\omega}_z \Delta t\right) \bmod 2\pi$$

3. **Coordinate Rotation into ENU**:
   $$a_E = \tilde{a}_{fwd} \sin(\psi_{mid}), \quad a_N = \tilde{a}_{fwd} \cos(\psi_{mid}), \quad a_U = 0$$

4. **Midpoint Velocity and Position Integration**:
   $$v_{E, k+1} = v_{E, k} + a_E \Delta t, \quad v_{N, k+1} = v_{N, k} + a_N \Delta t, \quad v_{U, k+1} = v_{U, k} + a_U \Delta t$$
   $$p_{E, k+1} = p_{E, k} + \frac{1}{2}(v_{E, k} + v_{E, k+1}) \Delta t, \quad p_{N, k+1} = p_{N, k} + \frac{1}{2}(v_{N, k} + v_{N, k+1}) \Delta t$$

5. **Analytical State Transition Jacobian $\mathbf{F} = \frac{\partial f}{\partial \mathbf{x}}$**:
   $$\mathbf{F} = \begin{bmatrix}
   1 & 0 & 0 & \Delta t & 0 & 0 & 0 & 0 & 0 \\
   0 & 1 & 0 & 0 & \Delta t & 0 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 & 0 & \Delta t & 0 & 0 & 0 \\
   0 & 0 & 0 & 1 & 0 & 0 & \tilde{a}_{fwd} \cos(\psi) \Delta t & -\sin(\psi) \Delta t & 0 \\
   0 & 0 & 0 & 0 & 1 & 0 & -\tilde{a}_{fwd} \sin(\psi) \Delta t & -\cos(\psi) \Delta t & 0 \\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & -\Delta t \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}$$

---

## 5. Measurement Models & Observation Jacobians

### A. Pre-Outage GNSS Updates (Healthy Phase)
- **Position Update**: $\mathbf{z}_{pos} = [p_{E,\text{GNSS}}, p_{N,\text{GNSS}}, p_{U,\text{GNSS}}]^T, \quad \mathbf{H}_{pos} = [\mathbf{I}_{3 \times 3} \quad \mathbf{0}_{3 \times 6}], \quad \mathbf{R}_{pos} = \sigma_{pos}^2 \mathbf{I}_3$
- **Velocity Update**: $\mathbf{z}_{vel} = [v_{E,\text{GNSS}}, v_{N,\text{GNSS}}, v_{U,\text{GNSS}}]^T, \quad \mathbf{H}_{vel} = [\mathbf{0}_{3 \times 3} \quad \mathbf{I}_{3 \times 3} \quad \mathbf{0}_{3 \times 3}], \quad \mathbf{R}_{vel} = \sigma_{vel}^2 \mathbf{I}_3$
- **Course/Heading Update**: $z_{hdg} = \psi_{\text{GNSS}}, \quad \mathbf{H}_{hdg} = [0, 0, 0, 0, 0, 0, 1, 0, 0], \quad R_{hdg} = \sigma_{hdg}^2$

### B. Outage Phase: Full Non-Holonomic Constraint (NHC)
To enforce vehicle chassis non-holonomic dynamics (zero lateral sideslip and zero vertical hopping):
$$\mathbf{z}_{\text{NHC}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad \mathbf{h}_{\text{NHC}}(\mathbf{x}) = \begin{bmatrix} v_{lat} \\ v_U \end{bmatrix} = \begin{bmatrix} -v_E \cos\psi + v_N \sin\psi \\ v_U \end{bmatrix}$$

**Analytical Jacobian**:
$$\mathbf{H}_{\text{NHC}} = \begin{bmatrix}
0 & 0 & 0 & -\cos\psi & \sin\psi & 0 & v_E \sin\psi + v_N \cos\psi & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}$$
*Note*: The term $\mathbf{H}_{\text{NHC}}[0, 6] = v_E \sin\psi + v_N \cos\psi = v_{forward}$ provides direct mathematical coupling between lateral innovation and heading error!

### C. Outage Phase: Phase 11 AI Forward Velocity Update
Scalar velocity predicted by the frozen Phase 11 Anchored Model ($v_{\text{AI}} = v_{anchor} + \Delta v_{\text{AI}}$):
$$z_{\text{AI}} = v_{\text{AI}}, \quad h_{\text{AI}}(\mathbf{x}) = v_{forward} = v_E \sin\psi + v_N \cos\psi$$

**Analytical Jacobian**:
$$\mathbf{H}_{\text{AI}} = \begin{bmatrix} 0 & 0 & 0 & \sin\psi & \cos\psi & 0 & v_E \cos\psi - v_N \sin\psi & 0 & 0 \end{bmatrix}$$
*Note*: The term $\mathbf{H}_{\text{AI}}[0, 6] = v_E \cos\psi - v_N \sin\psi = -v_{lat} \approx 0$ under valid NHC.

### D. Outage Phase: Zero Velocity Update (ZUPT)
When stationary detector triggers ($v_{\text{AI}} < 0.1\text{ m/s}$ or stationary event):
$$\mathbf{z}_{\text{ZUPT}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{H}_{\text{ZUPT}} = [\mathbf{0}_{3 \times 3} \quad \mathbf{I}_{3 \times 3} \quad \mathbf{0}_{3 \times 3}], \quad \mathbf{R}_{\text{ZUPT}} = (0.05)^2 \mathbf{I}_3$$

---

## 6. Implementation Checks Audit (Section 17 Verification)

Prior to running the 56-outage benchmark, all checks A through H were verified in `eval/test_phase12_ekf_sanity.py`:
- **Check A (Jacobian Parity)**: Maximum absolute difference between analytical $\mathbf{H}$ and numerical central differences ($\epsilon = 10^{-6}$) was $< 2.4 \times 10^{-7}$ for both forward velocity and NHC.
- **Check B (GNSS Velocity)**: Velocity state $[v_E, v_N]$ dynamically shifted towards observation $(0.00, 10.00) \to (0.46, 10.73)\text{ m/s}$.
- **Check C (NHC Innovation)**: $-5.0\text{ m/s}$ lateral skid produced clean $+5.00\text{ m/s}$ innovation.
- **Check D (NHC Heading Coupling)**: Induced lateral skid modified heading angle $\psi$ by $-4.5406^\circ$ purely through off-diagonal covariance coupling $P_{v,\psi}$.
- **Check E (AI Velocity Shift)**: Forward velocity update shifted forward speed from $10.00$ to $11.20\text{ m/s}$, propagating into both $v_E$ and $v_N$ in exact proportion to $[\sin 45^\circ, \cos 45^\circ]$.
- **Check F (No Clamping)**: Filter executed unbounded states without artificial ceilings.
- **Check G (Positive Semidefiniteness)**: Joseph-form covariance update preserved symmetry ($\max |P - P^T| < 10^{-12}$) and strict positive definiteness ($\lambda_{\min} = 1.06 \times 10^{-6} > 0$).
- **Check H (Synthetic 10s Outage)**: 10-second straight cruise at $20\text{ m/s}$ achieved $p_E = 200.00\text{ m}$ (expected $200.00\text{ m}$), $p_N = 0.00\text{ m}$, speed error $< 0.01\text{ m/s}$.

---

## 7. Comprehensive Benchmark Results (56 Held-Out Outages)

### A. Global A/B/C Comparative Summary

| Metric | Model A (7-State UKF) | Model B (9-State EKF + NHC) | Model C (9-State EKF No NHC) | Delta (B vs A) |
| :--- | :---: | :---: | :---: | :---: |
| **Median Drift %** | **40.92%** | **37.31%** | **46.33%** | **-3.61 pp** |
| **Mean Drift %** | **54.14%** | **54.49%** | **71.79%** | +0.35 pp |
| **P95 Drift %** | **115.05%** | **109.17%** | **194.19%** | **-5.88 pp** |
| **Median FPE (m)** | **192.13 m** | **177.14 m** | **213.69 m** | **-14.99 m** |
| **Mean FPE (m)** | **455.22 m** | **439.21 m** | **532.77 m** | **-16.01 m** |
| **P95 FPE (m)** | **1784.59 m** | **1801.92 m** | **2234.73 m** | +17.33 m |
| **Median Along-Track (m)** | **-65.45 m** | **-70.69 m** | **-35.91 m** | -5.24 m |
| **Mean Along-Track (m)** | **-187.83 m** | **-249.90 m** | **-115.96 m** | -62.07 m |
| **Median Cross-Track (m)** | **-2.92 m** | **0.59 m** | **1.71 m** | **+3.51 m** |
| **Mean Cross-Track (m)** | **-87.76 m** | **-23.00 m** | **-60.80 m** | **+64.76 m (73.8% better)** |
| **Median Heading Error** | **26.68°** | **21.82°** | **21.73°** | **-4.86°** |
| **Median Trajectory RMSE** | **123.91 m** | **111.94 m** | **115.17 m** | **-11.97 m** |
| **Median CEP50 (m)** | **108.35 m** | **100.94 m** | **103.96 m** | **-7.41 m** |
| **Median CEP95 (m)** | **173.92 m** | **156.22 m** | **202.78 m** | **-17.70 m** |
| **Drift < 10% Target Pass** | **1 / 56 (1.8%)** | **1 / 56 (1.8%)** | **1 / 56 (1.8%)** | **0.0 pp (No change)** |

---

### B. Breakdown by Outage Duration

| Duration | Outage Count | Median FPE A (m) | Median FPE B (m) | Median Drift % A | Median Drift % B | Median Cross-Track A | Median Cross-Track B |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 s** | 19 | 43.68 m | 45.63 m | 33.04% | 32.41% | -0.44 m | 1.29 m |
| **30 s** | 15 | 168.40 m | 156.31 m | 36.47% | 35.60% | 64.94 m | 34.57 m |
| **60 s** | 10 | 557.42 m | 539.93 m | 93.84% | 79.52% | -276.89 m | -126.66 m |
| **120 s** | 8 | 938.02 m | 851.16 m | 55.25% | 56.92% | -345.00 m | -254.47 m |
| **180 s** | 4 | 1183.58 m | 1307.57 m | 44.99% | 48.73% | 366.94 m | 230.42 m |

---

## 8. Observability & Regime-Change Analysis (Case A vs. Case B)

All plots referenced below were generated by the benchmark suite and are stored in `reports/phase12_plots/`:

### Case A: Steady High-Speed Cruising (Motorway 60s Outage)
- **Trajectory & Velocities**: `case_a_high_speed_cruise_1_trajectory.png`, `case_a_high_speed_cruise_3_world_velocity.png`
- **Speed & Innovations**: `case_a_high_speed_cruise_2_speed_comparison.png`, `case_a_high_speed_cruise_6_innovations.png`
- **Key Finding**: In steady high-speed cruising, Model B's lateral velocity remained tightly bounded ($|v_{lat}| < 0.2\text{ m/s}$), and cross-track error was reduced from $531.93\text{ m}$ (Model A) to $36.56\text{ m}$ (Model B). However, because the Phase 11 model slightly under-predicted the true cruising velocity, the along-track error accumulated steadily ($-477.81\text{ m}$), demonstrating that explicit $v_E, v_N$ states cannot compensate for forward velocity under-estimation.

### Case B: High-Speed Cruising into Severe Braking Deceleration (30s Outage)
- **Speed Profile**: `case_b_braking_deceleration_2_speed_comparison.png`
- **Lateral Velocity & Errors**: `case_b_braking_deceleration_4_lateral_velocity.png`, `case_b_braking_deceleration_7_along_cross_error.png`
- **Key Finding**: During deceleration from $22\text{ m/s}$ down to $8\text{ m/s}$, both Model A and Model B tracked the speed drop accurately because the Phase 11 Anchored Model provided negative $\Delta v$ updates. The 9-state filter adjusted $[v_E, v_N]$ via the AI forward velocity projection without divergence. However, the deceleration rate predicted by the AI model was slightly sluggish, resulting in a persistent longitudinal overshoot in both models.

---

## 9. Failure Case & Observability Diagnosis

Why did the 9-state EKF fail to achieve the $< 10\%$ drift target?

```
                        GNSS HEALTHY
   [vE_gnss, vN_gnss, vU_gnss] --> Directly Updates [vE, vN, vU]
                               |
                               v (GNSS Outage Begins)
                     GNSS DENIED BLACKOUT
              AI Forward Speed (z = v_AI) (1-DOF)
              Lateral NHC      (z = 0)    (1-DOF)
                               |
                               v
   Result: [vE, vN] restricted to v_AI * [sin(psi), cos(psi)]
   Along-track error = Integral(v_AI - v_GT) dt
```

1. **Rank Deficiency in Longitudinal Observation**:
   During outage, the measurement vector is:
   $$\mathbf{z}_k = \begin{bmatrix} v_{\text{AI}} \\ 0 \end{bmatrix} = \begin{bmatrix} v_E \sin\psi + v_N \cos\psi \\ -v_E \cos\psi + v_N \sin\psi \end{bmatrix}$$
   This is an orthogonal transformation of $[v_E, v_N]^T$. Therefore, the state uncertainty along the direction of travel is **100% identical** to having a single scalar state $v_{fwd}$. No new information is introduced to constrain longitudinal scale factor or accelerometer bias $b_a$.
2. **Accelerometer Bias Unobservability**:
   Under constant velocity driving, $a_{fwd} \approx 0$. The filter cannot separate accelerometer bias $b_a$ from vehicle velocity deceleration without external position anchors or wheel ticks.
3. **The Essential Role of Map Matching**:
   As proven in Phase 6B and Phase 7, longitudinal drift can only be broken below $10\%$ through **external spatial constraints** (e.g., road network link matching, intersection topological clamping, or physical wheel speed encoders). State-space reformulations within the inertial filter cannot create information out of thin air.

---

## 10. Final Architectural Conclusions

1. **Keep Production 7-State UKF**:
   The production 7-state UKF ($[p_N, p_E, v_{fwd}, \psi, b_g, b_a, k]$) is computationally lighter, inherently enforces non-holonomic vehicle constraints by construction, and provides numerically identical along-track performance ($40.92\%$ vs $37.31\%$) without the risk of lateral divergence seen in unconstrained 9-state formulations.
2. **Reject 9-State EKF as Longitudinal Solution (KILL)**:
   The hypothesis that world-frame velocity states $[v_E, v_N, v_U]$ resolve GNSS-denied longitudinal drift is **falsified**.
3. **Actionable Takeaway**:
   Longitudinal drift reduction must focus on:
   - High-fidelity wheel speed telemetry (OBD-II CAN bus integration, already wired into the Android app).
   - HMM map matching against the OpenStreetMap road graph (Phase 6B verified).
