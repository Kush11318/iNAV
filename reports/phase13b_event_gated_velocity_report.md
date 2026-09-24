# Phase 13B: Event-Gated Neural Velocity Correction Experiment Report

## Executive Summary
This document reports the comprehensive benchmark results of the **Phase 13B Research Experiment**, investigating whether gating temporary neural velocity corrections strictly during detected acceleration/braking events—while maintaining a constant fixed pre-outage speed anchor ($v_{\text{anchor}}$)—can resolve longitudinal along-track drift during GNSS blackouts.

---

## 1. Hypothesis
Phase 11 established that a frozen pre-outage speed anchor provides high velocity stability during steady cruising, but fails when the vehicle permanently transitions to a new speed regime post-outage. Phase 13A tested recursively evolving the velocity reference, which was killed due to open-loop integration bias accumulating across 67.9% of outages.

**Phase 13B hypothesized** that keeping $v_{\text{anchor}}$ strictly constant throughout the outage while allowing a temporary neural velocity correction $c(t)$ only when strong physical/neural evidence of dynamic events exists:
$$v_{\text{measurement}} = \operatorname{clamp}\left(v_{\text{anchor}} + c_{\text{gated}}(t), \min=0.0\right)$$
would retain the stability of the frozen anchor during cruise while capturing speed changes during braking and acceleration. Once dynamic events subside, $c_{\text{gated}}(t)$ smoothly decays back toward $0.0$, returning the measurement to $v_{\text{anchor}}$ rather than permanently carrying neural integration bias.

---

## 2. Implementation & Experimental Isolation
- **No Production Modifications**: All production systems (`modules/ukf.py`, models, Android C++/Kotlin, map matchers) remained strictly untouched.
- **Isolated Tracker Module** (`modules/phase13b_event_gated_velocity.py`):
  - Constant anchor: $v_{\text{anchor}} = v_{\text{GNSS}}(t_0)$, never updated.
  - Event Gate Logic:
    - **Stationary (Class 0 / speed < 0.15)**: Gate open, drives correction to $-v_{\text{anchor}}$, triggers ZUPT.
    - **Hard Braking (Class 2 / $a_{\text{fwd}} < -1.2$)**: Gate open, tracks negative neural $\Delta v$.
    - **Acceleration ($a_{\text{fwd}} > 0.8$ & $\Delta v > 0.6$)**: Gate open, tracks positive neural $\Delta v$.
    - **Turning (Class 3 / $|\omega| > 0.15$)**: Gate closed, measurement variance inflated $4\times$.
    - **Rough Road (Class 4)**: Correction attenuated 50%, variance inflated $4\times$.
    - **Cruise (Class 1)**: Gate closed, correction decays back toward 0.0 with time constant $\tau = 2.5\text{s}$.
- **Evaluation Harness** (`eval/evaluate_phase13b_navigation.py`):
  - Evaluated on all 56 held-out test outages under identical conditions as Baseline A (Phase 11 Frozen).

---

## 3. Sanity Checks
All 7 sanity test regimes passed cleanly in `eval/test_phase13b_sanity.py`:
1. **Constant-Speed Cruise**: Phase 13B $\approx$ Phase 11 ($v_{\text{meas}} = 22.77\text{ m/s}$, gate closed, correction $= 0.0$).
2. **Hard Braking**: Gate opened, tracked deceleration drop ($v_{\text{meas}} = 7.83\text{ m/s}$, correction $= -2.98\text{ m/s}$).
3. **Acceleration**: Forward acceleration detected, gate opened ($v_{\text{meas}} = 5.35\text{ m/s}$, correction $= +1.90\text{ m/s}$).
4. **Cruise $\to$ Braking $\to$ Cruise**: Correction dropped during braking, then smoothly decayed back to within $0.12\text{ m/s}$ of anchor.
5. **Stationary**: Gate opened with `STATIONARY`, driving velocity to $0.00\text{ m/s}$.
6. **Rough Road**: Measurement uncertainty inflated to $0.43\text{ m/s}$.
7. **Turning**: Gate closed, uncertainty inflated to $0.85\text{ m/s}$.

---

## 4. Benchmark Results: Baseline vs. Experiment (56 Held-Out Outages)

| Metric | Baseline A (Phase 11 Frozen) | Experiment B (Phase 13B Event-Gated) | Delta / Change |
| :--- | :---: | :---: | :---: |
| **Median Drift %** | **41.27%** | 42.96% | +1.69% (Worse) |
| **Mean Drift %** | **51.31%** | 51.44% | +0.13% (Worse) |
| **P95 Drift %** | 111.44% | **109.91%** | -1.53% (Marginal) |
| **Median FPE (m)** | 185.09 m | **179.87 m** | -5.22 m (Marginal) |
| **Mean FPE (m)** | 437.10 m | **428.53 m** | -8.57 m (Marginal) |
| **Median Along-Track Error (m)** | **52.11 m** | 63.49 m | **+11.38 m (+21.8% Worse)** |
| **Mean Along-Track Error (m)** | 243.02 m | **236.14 m** | -6.88 m (-2.8%) |
| **Median Cross-Track Error (m)** | 93.57 m | **91.69 m** | -1.88 m |
| **Median Heading Error (deg)** | 26.52° | **26.18°** | -0.34° |
| **Scenarios Drift Improved** | — | 26 / 56 (46.4%) | Minority improved |
| **Scenarios Drift Worsened** | — | **30 / 56 (53.6%)** | Majority worsened |
| **Along-Track Improved** | — | 27 / 56 (48.2%) | Under 50% |
| **Along-Track Worsened** | — | **29 / 56 (51.8%)** | Majority worsened |
| **Scenarios < 10% Drift Target** | **4 / 56 (7.1%)** | 3 / 56 (5.4%) | Decreased |

---

## 5. Breakdown by Outage Duration (Medians)

| Dur (s) | N | Drift A % | Drift B % | FPE A (m) | FPE B (m) | Along A (m) | Along B (m) | Cross A (m) | Cross B (m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 s** | 19 | 23.64% | **22.51%** | **37.53 m** | 38.49 m | **17.15 m** | 19.92 m | 33.22 m | **32.64 m** |
| **30 s** | 15 | **34.59%** | 41.25% | **173.99 m** | 175.08 m | **131.47 m** | 140.51 m | 95.84 m | **92.63 m** |
| **60 s** | 10 | 75.34% | **72.38%** | 564.12 m | **549.33 m** | 257.65 m | **256.03 m** | 465.82 m | **459.31 m** |
| **120 s** | 8 | 50.48% | **50.17%** | 867.36 m | **863.56 m** | 230.41 m | **221.66 m** | 672.85 m | **669.40 m** |
| **180 s** | 4 | **46.59%** | 46.96% | **1223.16 m** | 1231.64 m | **736.05 m** | 767.59 m | **432.66 m** | 493.67 m |

---

## 6. Critical Analysis: Breakdown by Driving Regime

| Driving Regime | Outages (N) | Drift A % | Drift B % | Along A (m) | Along B (m) | Along Improved % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stable Cruise** | 22 | 23.06% | **22.56%** | 24.46 m | **22.41 m** | 50.0% |
| **Strong Acceleration** | 8 | **44.63%** | 45.95% | **177.42 m** | 182.07 m | **25.0%** (75% worse) |
| **Strong Deceleration** | 4 | **65.97%** | 67.27% | **182.44 m** | 203.53 m | 50.0% |
| **Cruise $\to$ Braking/Accel** | 4 | 56.34% | **54.91%** | **32.59 m** | 54.68 m | 50.0% |
| **Mixed Dynamic** | 9 | **41.78%** | 50.33% | **37.47 m** | 41.79 m | 44.4% |
| **Stationary Transition** | 9 | 92.24% | **89.01%** | **274.27 m** | 291.26 m | 66.7% |

---

## 7. Along-Track Analysis & Failure Forensics

### The Fundamental Flaw of Temporary Event Gating
Why did Phase 13B fail to produce meaningful along-track improvements (median along-track error worsened from $52.11\text{ m}$ to $63.49\text{ m}$)?

1. **Temporary Corrections Cannot Model Permanent Regime Changes**:
   When a vehicle enters an outage at highway speed ($22.9\text{ m/s}$) and decelerates over $5\text{s}$ to an off-ramp speed ($9.5\text{ m/s}$), the braking event is active for only $5\text{--}7\text{ seconds}$. During active braking, the event gate opens and applies a negative correction $c(t) < 0$.
   However, once the braking maneuver completes and the vehicle continues cruising steadily at $9.5\text{ m/s}$, the IMU window registers zero longitudinal acceleration and normal road texture. The event gate classifies the subsequent state as **Cruise**.
   Because $v_{\text{anchor}}$ was kept fixed at $22.9\text{ m/s}$, the design requirement that the correction decay back to the fixed anchor causes $c(t) \to 0.0$, forcing the measurement right back up to $22.9\text{ m/s}$!
   As demonstrated in `vw11_outage_3`:
   - Active braking drops speed briefly during $t \in [0\text{s}, 6\text{s}]$.
   - For $t \in [7\text{s}, 38\text{s}]$ (cruising at $9.5\text{ m/s}$), Phase 13B rebounds back to $22.0\text{ m/s}$.
   - Total integrated distance: Ground Truth $= 468.8\text{ m}$, Phase 11 Frozen $= 918.4\text{ m}$, Phase 13B $= 893.7\text{ m}$.
   - The temporary gate reduced overshoot by a negligible $24.7\text{ m}$ out of $449.6\text{ m}$ total error!

2. **The Duality Dilemma**:
   - If we make the correction **permanent** (Phase 13A: $v_{\text{anchor}} \leftarrow v_{\text{anchor}} + \Delta v$), open-loop neural integration bias accumulates continuously, corrupting 67.9% of steady cruising outages.
   - If we keep the anchor **fixed** and make the correction **temporary** (Phase 13B: $c(t) \to 0$ in cruise), the model forgets the deceleration and snaps right back to the obsolete pre-outage speed.

---

## 8. Data Leakage Verification
- **Runtime Isolation**: The runtime tracker receives only the initial pre-outage GNSS speed $v_{\text{anchor}}$ and causal IMU frames transformed via real-time phone-to-vehicle alignment.
- **Zero Future Information**: No post-outage GNSS fixes, future trajectory coordinates, ground-truth speed profiles, or map positions entered runtime estimation.

---

## 9. Conclusion & Decision

### Decision: **KILL**

**Reasoning**:
1. **Primary Metric Failed**: Phase 13B produced no meaningful reduction in along-track drift. Median along-track error worsened from **$52.11\text{ m}$ to $63.49\text{ m}$ (+21.8%)**, and along-track error worsened in **$51.8\%$ of outages**.
2. **Structural Ineffectiveness**: Event-gated temporary corrections cannot bridge the gap between permanent vehicle speed transitions and fixed pre-outage anchors.
3. **Mandate Followed**: As specified in the directive (*"If Phase 13B fails, STOP modifying the same velocity architecture. We will move to a different source of longitudinal observability"*), further iterations on the single-model temporal velocity net architecture are terminated. The production Phase 11 frozen anchor baseline is retained.
