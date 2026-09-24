# Phase 13A: Evolving Velocity Reference Experiment Report

## Executive Summary
This document reports the findings of the **Phase 13A Research Experiment**, investigating whether an evolving neural velocity reference tracker ($v_{\text{ref}}(t+1) = \operatorname{clamp}(v_{\text{ref}}(t) + \Delta v_{\text{neural}}, \min=0.0)$) resolves longitudinal drift in GNSS-denied navigation compared to the Phase 11 frozen pre-outage anchor baseline.

---

## 1. Hypothesis
Instead of treating $v_{\text{anchor}}$ as permanently fixed during the entire GNSS blackout ($10\text{s} \dots 180\text{s}$), maintain an evolving velocity reference:
$$v_{\text{ref}}(0) = v_{\text{anchor}}$$
$$v_{\text{ref}}(t+1) = \operatorname{clamp}(v_{\text{ref}}(t) + \Delta v_{\text{neural}}(t), \min=0.0)$$
This evolving neural velocity estimate is provided with duration-inflated uncertainty to the 7-state UKF as the forward-speed measurement.

The hypothesis posits that updating $v_{\text{ref}}$ causally will eliminate the catastrophic along-track error that occurs when a vehicle enters an outage at high speed (e.g. $22.9\text{ m/s}$) and decelerates (e.g. to $2\text{ m/s}$), where the frozen anchor baseline snaps back to the pre-outage cruising speed.

---

## 2. Implementation & Experimental Isolation
- **No Production Code Alterations**: Production files `modules/ukf.py`, `modules/alignment.py`, map matching, and the Android C++/Kotlin code were left completely untouched.
- **Isolated Modules**:
  - `modules/phase13a_velocity_reference.py`: Encapsulates `Phase13AVelocityReference`, maintaining causal state, clamping negative velocities to 0.0, propagating outage-duration uncertainty $\sigma(\tau) = \max(\sigma_{\text{model}}, 0.2) \times (1 + 0.05\sqrt{\tau})$, and gating stationary conditions.
  - `eval/test_phase13a_sanity.py`: 5 synthetic and empirical sanity tests (A: Cruise, B: Accelerate, C: Brake, D: Cruise $\to$ Brake $\to$ Cruise, E: Stationary). All 5 passed before benchmarking.
  - `eval/evaluate_phase13a_navigation.py`: Rigorous A/B test harness evaluating Baseline A vs. Experiment B across all 56 held-out test outages.
- **Forbidden Heuristics Enforced**: No scale factors $k$, no RLS, no arc-length snapping, no NoiseNet, no 9-state EKF.

---

## 3. Benchmark Results: Baseline vs. Experiment (56 Held-Out Outages)

| Metric | Baseline A (Phase 11 Frozen) | Experiment B (Phase 13A Evolving) | Delta / Change |
| :--- | :---: | :---: | :---: |
| **Median Drift %** | **41.27%** | 41.33% | +0.06% (Neutral) |
| **Mean Drift %** | **51.31%** | 53.44% | +2.13% (Worse) |
| **P95 Drift %** | **111.44%** | 143.00% | +31.56% (Worse) |
| **Median FPE (m)** | **185.09 m** | 222.41 m | +37.32 m (Worse) |
| **Mean FPE (m)** | **437.10 m** | 453.35 m | +16.25 m (Worse) |
| **Median Along-Track Error (m)** | **52.11 m** | 130.48 m | **+78.37 m (+150.4% Worse)** |
| **Mean Along-Track Error (m)** | **243.02 m** | 330.40 m | **+87.38 m (+36.0% Worse)** |
| **Median Cross-Track Error (m)** | **93.57 m** | 114.27 m | +20.70 m (Worse) |
| **Median Heading Error (deg)** | **26.52°** | 26.55° | +0.03° (Identical) |
| **Scenarios Drift Improved** | — | 24 / 56 (42.9%) | Minority improved |
| **Scenarios Drift Worsened** | — | **32 / 56 (57.1%)** | Majority worsened |
| **Along-Track Error Improved** | — | 17 / 56 (30.4%) | Only 30.4% improved |
| **Along-Track Error Worsened** | — | **38 / 56 (67.9%)** | **67.9% worsened** |
| **Scenarios < 10% Drift Target** | **4 / 56 (7.1%)** | 1 / 56 (1.8%) | Dropped from 4 to 1 |

---

## 4. Breakdown by Outage Duration (Medians)

| Dur (s) | N | Drift A % | Drift B % | FPE A (m) | FPE B (m) | Along A (m) | Along B (m) | Cross A (m) | Cross B (m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 s** | 19 | 23.64% | **22.24%** | **37.53 m** | 38.51 m | **17.15 m** | 18.62 m | 33.22 m | **30.49 m** |
| **30 s** | 15 | **34.59%** | 40.48% | **173.99 m** | 213.09 m | **131.47 m** | 135.15 m | **95.84 m** | 112.17 m |
| **60 s** | 10 | 75.34% | **63.88%** | 564.12 m | **533.42 m** | **257.65 m** | 327.52 m | **465.82 m** | 469.56 m |
| **120 s** | 8 | 50.48% | **36.20%** | 867.36 m | **747.75 m** | **230.41 m** | 561.02 m | 672.85 m | **443.13 m** |
| **180 s** | 4 | 46.59% | **46.20%** | 1223.16 m | **1040.49 m** | **736.05 m** | 864.59 m | **432.66 m** | 515.34 m |

---

## 5. Along-Track Analysis & Failure Forensics

### Why Phase 13A Succeeds on Sharp Decelerations
In specific scenarios where the vehicle enters an outage at high speed and promptly decelerates, Phase 13A completely resolves the catastrophic over-speeding:
- **`vw11_outage_3` (60s outage, $22.9 \to 2.1\text{ m/s}$)**:
  - Ground truth distance: $468.8\text{ m}$
  - Phase 11 Frozen anchor distance: $918.4\text{ m}$ (Error: $+449.6\text{ m}$)
  - Phase 13A Evolving distance: $481.3\text{ m}$ (Error: $+12.6\text{ m}$)
  - Along-track error: $535.8\text{ m} \to 112.2\text{ m}$ (**$79.1\%$ reduction**)
  - FPE: $584.9\text{ m} \to 247.7\text{ m}$ (**$57.6\%$ reduction**)
- **`vw3_outage_4` (60s outage, $19.5 \to 2.9\text{ m/s}$)**:
  - Along-track error: $555.6\text{ m} \to 367.1\text{ m}$ (**$33.9\%$ reduction**)
  - FPE: $910.2\text{ m} \to 728.4\text{ m}$

### Why Phase 13A Fails on the 56-Outage Aggregate
Despite eliminating error in the deceleration failure cases, Phase 13A **worsened along-track error in 38 out of 56 outages ($67.9\%$)**.
- **The Closed-Loop Feedback Trap**:
  When $v_{\text{ref}}$ is updated recursively:
  $$v_{\text{ref}}(t+\Delta t) = v_{\text{ref}}(t) + \Delta v_{\text{neural}}(t)$$
  and fed back into `AnchoredTemporalVelocityNet` as its anchor embedding, the system creates a positive-feedback loop. Any small per-step prediction bias $\epsilon$ (e.g. $+0.15\text{ m/s}$) is not merely integrated linearly; it shifts the anchor feature passed to the multi-layer perceptron head on the *subsequent* step.
- **Anchor Drift during Steady Highway Cruising**:
  On typical motorway driving (which comprises the majority of the dataset), the true speed is relatively steady ($\approx 20\text{--}25\text{ m/s}$). In Baseline Phase 11, the frozen pre-outage anchor acts as an absolute physical peg that prevents random-walk integration. Phase 13A removes this physical peg, turning the speed estimate into an unanchored random walk.
- **Runaway Drift in Deep Outages**:
  In `vw14c_outage_4` ($180\text{s}$), along-track error surged from $470.0\text{ m}$ to $1210.9\text{ m}$. In `vw2_outage_1` ($180\text{s}$), along-track error surged from $2648.2\text{ m}$ to $3499.7\text{ m}$.

---

## 6. Data Leakage Verification
- **Runtime Integrity**: Verified 100% data leakage-free. The runtime tracker receives only the initial pre-outage speed $v_{\text{anchor}} = v_{\text{GNSS}}(t_0)$ and streaming raw IMU data.
- **No Future Labels**: Ground truth speed, GNSS positions, future trajectory, and evaluation labels were strictly isolated to the offline evaluation scoring function.
- **Frozen UKF**: All filter parameters ($Q$, $R$, state transition matrices, NIS gates) were strictly identical between Baseline A and Experiment B.

---

## 7. Conclusion & Decision

### Decision: **KILL**

**Reasoning**:
1. **Primary Metric Failed**: Phase 13A increased overall median along-track error from **$52.11\text{ m}$ to $130.48\text{ m}$ (+150.4%)** and worsened along-track error on **$67.9\%$ of held-out outages**.
2. **Failure of Unanchored Evolution**: While causal reference evolution successfully tracks sharp deceleration events when they occur, unconstrained recursive feedback converts the network into an open-loop integrator that drifts significantly on steady cruising trajectories.
3. **Strict Criteria Met**: Per the evaluation protocol ("KILL if it produces no meaningful improvement or worsens the aggregate result... Do NOT declare success based on a few cherry-picked trajectories"), unconditional velocity reference evolution must be **KILLED**.
