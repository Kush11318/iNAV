# PHASE 9 — AI-ADAPTIVE NHC EXPERIMENT REPORT

**Experiment Status**: Completed  
**Benchmark Target**: 56 Held-Out Motorway Outages (IO-VNBD `vw` campaign, 10s to 180s)  
**Evaluated Systems**:
- **Baseline A**: Production 7-State UKF with fixed NHC covariance ($\mathbf{R}_{\text{NHC}} = \operatorname{diag}(0.10, 0.05)\text{ m}^2/\text{s}^2$)
- **Experiment B**: 7-State UKF with dynamic NoiseNet covariance ($\mathbf{R}_{\text{NHC}} = \text{NoiseNet}(\text{IMU})$)
- **Ablation C**: 7-State UKF with train-set mean NoiseNet covariance ($\mathbf{R}_{\text{NHC}} = \operatorname{diag}(0.05745, 0.23538)\text{ m}^2/\text{s}^2$)
- **Control Diagnostic**: 15-State ES-EKF with active 3D velocity states (A vs B vs C)

---

## 1. Exact Hypothesis

### Research Hypothesis
* **Null Hypothesis ($H_0$)**: Predicting dynamic Non-Holonomic Constraint (NHC) measurement noise covariance $\mathbf{R}_{\text{NHC}} = \operatorname{diag}(\sigma_{\text{lat}}^2, \sigma_{\text{vert}}^2)$ from raw 6-channel IMU using a lightweight neural network (NoiseNet) does not reduce vehicle dead-reckoning drift compared to fixed baseline covariance in the iNAV navigation architecture.
* **Alternative Hypothesis ($H_1$)**: Dynamically relaxing the NHC zero-velocity constraint ($v_{\text{lat}} \approx 0, v_{\text{vert}} \approx 0$) during high-dynamics turning, cornering, and rough road conditions improves longitudinal and lateral estimation accuracy.

---

## 2. Exact Baseline Configuration

### Production Filter Architecture
* **State Vector** (7 states):
  $$\mathbf{x} = [p_N, p_E, v_{\text{fwd}}, \psi, b_g, b_a, k]^T$$
* **Kinematics**: 2D planar unicycle mechanization:
  $$\dot{p}_N = v_{\text{fwd}} \cos \psi, \quad \dot{p}_E = v_{\text{fwd}} \sin \psi, \quad \dot{\psi} = \omega_z - b_g$$
* **Displacement Model**: Frozen Phase 8A 4.0-second `VelocityNet4s` (`models/phase8a_4s_velocity_net.pt`).
* **Fixed NHC Covariance Baseline A**:
  $$\mathbf{R}_{\text{NHC}} = \begin{bmatrix} 0.10 & 0 \\ 0 & 0.05 \end{bmatrix}\text{ m}^2/\text{s}^2$$
  (as established in `modules/esekf.py:456` and forensic benchmarks).

---

## 3. Exact NoiseNet Architecture

A lightweight 1D temporal convolutional neural network was implemented for real-time edge deployment:

* **Input**: 6-channel IMU tensor $(B, 6, 40)$ at 10 Hz (4.0s context window: $a_x, a_y, a_z, \omega_x, \omega_y, \omega_z$).
* **Layer 1**: `Conv1d(6, 32, kernel_size=5, padding=2)` $\rightarrow$ `BatchNorm1d(32)` $\rightarrow$ `ReLU`
* **Layer 2**: `Conv1d(32, 64, kernel_size=5, padding=2)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `ReLU`
* **Pooling**: `AdaptiveAvgPool1d(1)` $\rightarrow$ `Flatten(64)`
* **MLP Head**: `Linear(64, 32)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(32, 2)`
* **Positivity & Safety Activation**:
  $$\sigma^2 = \operatorname{clamp}(\operatorname{Softplus}(\text{raw}) + 10^{-4}, \; 10^{-4}, \; 10.0)$$
* **Parameter Count**: 13,378 parameters (52.3 KB footprint).
* **Model Artifact**: [models/phase9_noisenet.pt](file:///c:/Projects/SIH%202026/iNAV/models/phase9_noisenet.pt).

---

## 4. Training Data & Split

Strict campaign-level disjoint partitioning was maintained to eliminate data leakage:
* **Training Set**: 50 synchronized trajectories from campaigns `s`, `m`, `st`, `vfa`, `vfb`, `vta`, `vtb` (138,896 4.0s sliding windows).
* **Validation Set**: Campaigns `y` and `s2` (16,718 4.0s sliding windows).
* **Test Set**: 18 held-out motorway runs (`vw` campaign, 56 outages). **Zero test samples were used in training, validation, or hyperparameter selection.**

---

## 5. Covariance Target Construction

A physically grounded supervised target was derived directly from reference RTK/INS trajectories:
1. In the navigation frame, true velocity was computed via discrete gradient:
   $$\mathbf{v}_{\text{nav}, GT} = [\dot{p}_{N, GT}, \; \dot{p}_{E, GT}]^T$$
2. Transformed into the vehicle chassis frame using true heading $\psi_{GT}$:
   $$v_{y, GT} = -\dot{p}_{N, GT} \sin \psi_{GT} + \dot{p}_{E, GT} \cos \psi_{GT}$$
   $$v_{z, GT} = \dot{h}_{GT}$$
3. For each 40-sample window $W$, the empirical variance of the NHC residual was computed:
   $$\sigma_{\text{lat, target}}^2 = \frac{1}{W} \sum_{i \in W} (v_{y, GT}[i])^2, \quad \sigma_{\text{vert, target}}^2 = \frac{1}{W} \sum_{i \in W} (v_{z, GT}[i])^2$$
4. **Optimization Objective**: Gaussian Negative Log-Likelihood (MLE) loss:
   $$\mathcal{L} = \frac{1}{2} \left[ \frac{\sigma_{\text{lat, target}}^2}{\hat{\sigma}_{\text{lat}}^2} + \ln \hat{\sigma}_{\text{lat}}^2 + \frac{\sigma_{\text{vert, target}}^2}{\hat{\sigma}_{\text{vert}}^2} + \ln \hat{\sigma}_{\text{vert}}^2 \right]$$

### Target Statistics (Training Set, 138,896 Windows):
| Metric | $\sigma_{\text{lat}}^2$ ($\text{m}^2/\text{s}^2$) | $\sigma_{\text{vert}}^2$ ($\text{m}^2/\text{s}^2$) |
| :--- | :---: | :---: |
| **Minimum** | $0.00010$ | $0.00010$ |
| **Median** | $0.00769$ | $0.05263$ |
| **Mean** | $0.05745$ | $0.23538$ |
| **P95** | $0.09769$ | $0.93612$ |
| **Maximum** | $10.00000$ | $10.00000$ |

---

## 6. Exact Runtime Integration

At each 10 Hz filter propagation cycle, the NHC measurement model was evaluated:
* Measurement observation: $\mathbf{z} = [0, 0]^T$ (lateral and vertical velocity constraints).
* Measurement covariance:
  $$\mathbf{R}_{\text{NHC}} = \begin{bmatrix} \sigma_{\text{lat}}^2 & 0 \\ 0 & \sigma_{\text{vert}}^2 \end{bmatrix}$$
* Measurement function in the 7-state UKF:
  $$\mathbf{h}(\mathbf{x}) = \begin{bmatrix} v_{\text{lat}}(\mathbf{x}) \\ v_{\text{vert}}(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

### Mathematical Null-Op Discovery
In the 7-state UKF, forward velocity $v_{\text{fwd}}$ is parameterized along the heading vector $[\cos \psi, \sin \psi]^T$. By mathematical definition:
$$v_{\text{lat}}(\mathbf{x}) = - (v_{\text{fwd}} \cos \psi) \sin \psi + (v_{\text{fwd}} \sin \psi) \cos \psi \equiv 0$$
$$v_{\text{vert}}(\mathbf{x}) \equiv 0$$
Because $\mathbf{h}(\mathbf{s}_i) = [0, 0]^T$ for **all sigma points** $\mathbf{s}_i$:
$$\hat{\mathbf{z}} = [0, 0]^T \implies \mathbf{y} = \mathbf{z} - \hat{\mathbf{z}} = [0, 0]^T$$
$$\mathbf{P}_{xz} = \sum_{i} w_c (\mathbf{s}_i - \hat{\mathbf{x}})(\mathbf{h}(\mathbf{s}_i) - \hat{\mathbf{z}})^T \equiv \mathbf{0}_{7 \times 2}$$
$$\mathbf{K} = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1} \equiv \mathbf{0}_{7 \times 2}$$
$$\Delta \mathbf{x} = \mathbf{K} \mathbf{y} \equiv \mathbf{0}_7, \quad \Delta \mathbf{P} = -\mathbf{K} \mathbf{P}_{zz} \mathbf{K}^T \equiv \mathbf{0}_{7 \times 7}$$

**Consequently, the Kalman gain is identically zero regardless of $\mathbf{R}_{\text{NHC}}$. The 7-state UKF already hard-enforces NHC algebraically through its state parameterization.**

---

## 7. A vs B Benchmark Table

The benchmark was executed across all 56 held-out test outages:

### Primary Architecture: 7-State UKF ($N = 56$)
| Metric | Baseline A (Fixed) | Experiment B (NoiseNet) | Ablation C (Mean Train) | Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Median Drift %** | **$41.45\%$** | **$41.45\%$** | **$41.45\%$** | **$0.00\%$** (Identical) |
| **Mean Drift %** | $57.18\%$ | $57.18\%$ | $57.18\%$ | $0.00\%$ (Identical) |
| **P25 Drift %** | $27.52\%$ | $27.52\%$ | $27.52\%$ | $0.00\%$ (Identical) |
| **P75 Drift %** | $68.63\%$ | $68.63\%$ | $68.63\%$ | $0.00\%$ (Identical) |
| **Median FPE (m)** | **$236.00\text{ m}$** | **$236.00\text{ m}$** | **$236.00\text{ m}$** | **$0.00\text{ m}$** (Identical) |
| **Mean FPE (m)** | $467.52\text{ m}$ | $467.52\text{ m}$ | $467.52\text{ m}$ | $0.00\text{ m}$ (Identical) |
| **Median Along-Track (m)** | $-129.82\text{ m}$ | $-129.82\text{ m}$ | $-129.82\text{ m}$ | $0.00\text{ m}$ (Identical) |
| **Median Cross-Track (m)** | $-7.58\text{ m}$ | $-7.58\text{ m}$ | $-7.58\text{ m}$ | $0.00\text{ m}$ (Identical) |
| **Median Heading Error** | $26.55^\circ$ | $26.55^\circ$ | $26.55^\circ$ | $0.00^\circ$ (Identical) |
| **Outages $< 10\%$ Drift** | 1 / 56 | 1 / 56 | 1 / 56 | 0 change |
| **Improved / Worsened / Unchanged** | — | **0 / 0 / 56** | **0 / 0 / 56** | Complete Null Effect |

---

### Secondary Diagnostic: 15-State ES-EKF with Active 3D Velocity States ($N = 56$)
To test whether dynamic NHC covariance produces any effect when velocity states $[v_N, v_E, v_D]$ are unconstrained:

| Metric | ES-EKF Baseline A (Fixed) | ES-EKF NoiseNet B (Dynamic) | ES-EKF Ablation C (Mean) | Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Median Drift %** | **$96.68\%$** | **$97.75\%$** | $97.09\%$ | **$+1.07\%$ (Worse)** |
| **Mean Drift %** | $92.12\%$ | $94.09\%$ | $95.03\%$ | $+1.97\%$ (Worse) |
| **Median FPE (m)** | $463.80\text{ m}$ | $455.66\text{ m}$ | $456.62\text{ m}$ | $-8.14\text{ m}$ |
| **Improved / Worsened / Unchanged** | — | **24 / 32 / 0** | 26 / 30 / 0 | **Net Regressive** |

---

## 8. Per-Duration Results (7-State UKF)

| Outage Duration | $N$ | Baseline A Drift % | NoiseNet B Drift % | Baseline A FPE | NoiseNet B FPE | Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 s** | 19 | $27.74\%$ | $27.74\%$ | $41.02\text{ m}$ | $41.02\text{ m}$ | $0.00\text{ m}$ |
| **30 s** | 15 | $41.84\%$ | $41.84\%$ | $229.63\text{ m}$ | $229.63\text{ m}$ | $0.00\text{ m}$ |
| **60 s** | 10 | $60.77\%$ | $60.77\%$ | $582.25\text{ m}$ | $582.25\text{ m}$ | $0.00\text{ m}$ |
| **120 s** | 8 | $41.05\%$ | $41.05\%$ | $916.92\text{ m}$ | $916.92\text{ m}$ | $0.00\text{ m}$ |
| **180 s** | 4 | $46.24\%$ | $46.24\%$ | $1105.41\text{ m}$ | $1105.41\text{ m}$ | $0.00\text{ m}$ |

---

## 9. Per-Regime Results (7-State UKF)

| Driving Regime | $N$ | Baseline A Drift % | NoiseNet B Drift % | Baseline A FPE | NoiseNet B FPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **STRAIGHT** | 35 | $38.31\%$ | $38.31\%$ | $345.03\text{ m}$ | $345.03\text{ m}$ |
| **TURNING** | 18 | $66.88\%$ | $66.88\%$ | $153.64\text{ m}$ | $153.64\text{ m}$ |
| **ROUGH ROAD** | 3 | $28.69\%$ | $28.69\%$ | $130.11\text{ m}$ | $130.11\text{ m}$ |

---

## 10. Covariance Statistics (NoiseNet on Held-Out Test Set)

NoiseNet successfully predicted dynamic, bounded variances across all test driving conditions:

| Parameter | Minimum | Median | Maximum | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| $\sigma_{\text{lat}}^2$ ($\text{m}^2/\text{s}^2$) | $0.00086$ | $0.01507$ | $0.48345$ | Drops to $\sim 10^{-3}$ on straight highways; inflates to $0.48$ during sharp turns |
| $\sigma_{\text{vert}}^2$ ($\text{m}^2/\text{s}^2$) | $0.00624$ | $0.14321$ | $1.29711$ | Tracks vertical vibration; spikes over road irregularities and expansion joints |

The network maintained numerical stability throughout all 56 outages:
* Zero NaNs, zero infinities, zero negative variances.
* Covariances strictly bounded within $[10^{-4}, 10.0]\text{ m}^2/\text{s}^2$.

---

## 11. NIS Statistics

* **7-State UKF**: Mean NIS = $0.00000$ (Identically zero innovation because $y = z - \hat{z} = [0, 0] - [0, 0] = [0, 0]$).
* **15-State ES-EKF**: Mean NIS = $1.428$ (within the 2-DOF $\chi^2$ 95% expectation of $2.0$).

---

## 12. Failure Cases & Mechanism Analysis

### Why Brossard et al.'s AI-IMU-DR Concept Does Not Apply to iNAV
1. **Architectural Mismatch**:
   * Brossard et al. (IEEE T-RO 2020) designed NoiseNet for a **15-state 3D Cartesian velocity mechanization** ($\mathbf{v} = [v_N, v_E, v_D]$), where strapdown accelerometer integration causes lateral and vertical velocities to drift unboundedly. In that architecture, NHC is an essential external measurement constraint.
   * iNAV's production architecture is a **7-state 2D unicycle filter** ($\mathbf{x} = [p_N, p_E, v_{\text{fwd}}, \psi, b_g, b_a, k]$). Here, lateral velocity $v_{\text{lat}} \equiv 0$ and vertical velocity $v_{\text{vert}} \equiv 0$ are **hard structural constraints** built into the differential propagation equations ($\dot{\mathbf{p}} = v_{\text{fwd}} [\cos \psi, \sin \psi]^T$).
2. **Mathematical Redundancy**:
   * Applying an external measurement $z = [v_{\text{lat}}, v_{\text{vert}}] = [0, 0]$ to a filter whose state already imposes $v_{\text{lat}} \equiv 0$ yields a measurement Jacobian $H \equiv \mathbf{0}$, cross-covariance $\mathbf{P}_{xz} \equiv \mathbf{0}$, and Kalman gain $\mathbf{K} \equiv \mathbf{0}$. Modulating $\mathbf{R}_{\text{NHC}}$ has zero effect.
3. **Destabilization in Active 3D Mechanization (15-State Filter)**:
   * When evaluated on the 15-state ES-EKF where NHC is active, NoiseNet **degraded** median drift from $96.68\%$ to $97.75\%$ (32 outages worsened vs 24 improved).
   * **Physical Cause**: In consumer smartphone sensors with unmodeled accelerometer biases and hand fidgets, inflating $\sigma_{\text{lat}}^2$ during turns relaxes the lateral constraint, allowing centripetal acceleration and gravity leakage to integrate directly into runaway velocity drift.

---

## 13. Longitudinal Drift Assessment

Does Experiment B improve longitudinal drift?
* **NO**.
* Longitudinal error remains identical at $-129.82\text{ m}$ median lag across all 56 outages.
* In wheeled vehicles, the NHC constraint operates strictly in the lateral and vertical axes ($v_y \approx 0, v_z \approx 0$). It provides **zero observability into forward longitudinal speed or displacement**.

---

## 14. Final Verdict

### 🔴 KILL

**Justification**:
1. In the production 7-state UKF, NHC is already hard-enforced by the unicycle state parameterization ($v_{\text{lat}} \equiv 0, v_{\text{vert}} \equiv 0$). An external NHC measurement update is a mathematical null-op ($\mathbf{K} \equiv \mathbf{0}$), producing **exactly $0.00\%$ change** across all 56 held-out test outages (0 improved, 0 worsened, 56 unchanged).
2. In active 3D Cartesian velocity filters (15-state ES-EKF), AI-adaptive NHC covariance **degraded overall performance**, increasing median drift from $96.68\%$ to $97.75\%$ by allowing bias leakage during turns.
3. NHC provides zero longitudinal observability and cannot mitigate VelocityNet displacement drift.
4. Adding NoiseNet to the production runtime would add inference latency and memory overhead for zero empirical benefit.

---

*Experiment Artifacts:*
- Evaluated Dataset: [eval/phase9_results.csv](file:///c:/Projects/SIH%202026/iNAV/eval/phase9_results.csv)
- Trained Model: [models/phase9_noisenet.pt](file:///c:/Projects/SIH%202026/iNAV/models/phase9_noisenet.pt)
- Training Code: [eval/train_phase9_noisenet.py](file:///c:/Projects/SIH%202026/iNAV/eval/train_phase9_noisenet.py)
- Benchmark Code: [eval/evaluate_phase9_adaptive_nhc.py](file:///c:/Projects/SIH%202026/iNAV/eval/evaluate_phase9_adaptive_nhc.py)
