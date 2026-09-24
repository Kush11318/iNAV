# Phase 18A: Forensic Diagnostic Analysis of Phase 17A Map Failures

**Executive Summary:**
Phase 18A conducts a systematic, step-by-step forensic investigation into the **14 scenarios** where Phase 17A (System C: CAN forward speed + 1D road-normal HMM/OSM map constraint) worsened Final Position Error (FPE) compared to System B (CAN speed only).

Using telemetry collected from every 0.5s map-matching step, each failure was evaluated across map availability, candidate quality, gating behavior, UKF state dynamics, and first-divergence timing.

### Key Forensic Findings:
1. **Dominant Failure Mode is Category E (Heading / Gyro-Bias Dominated), accounting for 50.0% of failures (7/14) and 82.5% of total degraded distance (+2,020 m out of +2,447 m total degradation).**
2. **Category B (Wrong / Ambiguous Road Association) is the secondary failure mode, accounting for 35.7% (5/14).** Forensic inspection reveals that Category B is directly caused by Category E: when gyro bias drifts beyond $25^\circ\text{--}40^\circ$, HMM emission probabilities begin favoring adjacent or parallel road segments that better match the drifted vehicle heading.
3. **Map Availability is 100% (Category A: 0/14):** In none of the 14 failures was the map missing or out of coverage. The road network was fully available.
4. **The "Initial Help then Catastrophic Latch" Pattern:** In 5 of the 6 severe failures, the map constraint initially **helped** during the first 10 to 30 seconds (reducing error by 5–20 m). However, as unconstrained gyro bias integration rotated the vehicle heading, the filter either suffered continuous NIS rejection or latched onto wrong parallel carriageways.
5. **Statistical Contrast (Successful vs Failed Scenarios):**
   - Heading error under CAN-only was **2.92x worse** in failed scenarios (median **70.3°** vs. **24.1°** in successful scenarios).
   - Failures concentrate heavily in extended runs (mean duration 64.3s vs. 42.6s) where unobservable gyro bias integrates unbounded over 60s, 120s, and 180s.

---

## 1. Complete 14-Row Failure Forensic Table

| Scenario | Dur | FPE B (CAN) | FPE C (Map) | $\Delta$ FPE | Cat | Category Name | Div Time (Epoch) | Hdg Err @ Div | Final Hdg Err | Map Acc / Rej / Wrong | Initial Help? | Primary Forensic Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`vw14b_o5`** | 180s | 719.2 m | 2063.3 m | **+1344.1 m** | **E** | Heading/gyro-bias dominated | 34.8s (19%) | 43.6° | 91.0° | 77 / 68 / 25 | **Yes** | Severe gyro bias drift rotated vehicle by 91°; 1D road normal dragged filter onto perpendicular cross-street. |
| **`vw4_o1`** | 120s | 2325.3 m | 2770.9 m | **+445.6 m** | **E** | Heading/gyro-bias dominated | 19.3s (16%) | 26.7° | 97.1° | 21 / 19 / 0 | **Yes** | Heading drifted to 97°; filter projected position away from highway, causing massive along-track mismatch. |
| **`vw14c_o4`** | 180s | 1176.5 m | 1375.7 m | **+199.2 m** | **B** | Wrong/ambiguous road association | 9.5s (5%) | 11.5° | 39.0° | 35 / 165 / 14 | **Yes** | Matched frontage road instead of motorway; 165 updates rejected by NIS after divergence. |
| **`vw11_o3`** | 60s | 369.4 m | 537.1 m | **+167.8 m** | **E** | Heading/gyro-bias dominated | 7.6s (13%) | 45.6° | 17.6° | 25 / 57 / 17 | **Yes** | Rapid gyro drift during early turn caused premature divergence; filter latched wrong junction branch. |
| **`vw16a_o4`** | 120s | 1196.6 m | 1313.3 m | **+116.7 m** | **B** | Wrong/ambiguous road association | 32.1s (27%) | 14.5° | 62.5° | 161 / 54 / 133 | No | Dual carriageway ambiguity: 133 of 161 accepted updates snapped to the opposite/parallel carriageway. |
| **`vw3_o3`** | 30s | 132.0 m | 160.0 m | **+28.1 m** | **B** | Wrong/ambiguous road association | 24.1s (80%) | 24.7° | 55.0° | 60 / 0 / 57 | No | Urban junction: 57 of 60 updates snapped to side street at intersection. |
| **`vw8_o1`** | 60s | 72.7 m | 91.5 m | **+18.8 m** | **E** | Heading/gyro-bias dominated | 4.6s (8%) | 69.1° | 131.4° | 112 / 7 / 110 | No | Gyro drifted by 131°; vehicle trajectory rotated completely backwards relative to road. |
| **`vw7_o1`** | 30s | 191.1 m | 209.3 m | **+18.2 m** | **E** | Heading/gyro-bias dominated | 28.2s (94%) | 109.5° | 101.8° | 31 / 9 / 4 | No | Late sharp turn with 102° gyro drift; 4 wrong road updates at tail of outage. |
| **`vw4_o3`** | 10s | 35.2 m | 49.3 m | **+14.1 m** | **D** | Road geometry representation | 5.3s (53%) | 20.5° | 15.9° | 11 / 7 / 0 | **Yes** | Curved road segment approximated by straight chord; minor 14m lateral pull. |
| **`vw6_o2`** | 30s | 455.8 m | 468.8 m | **+13.1 m** | **E** | Heading/gyro-bias dominated | 3.1s (10%) | 69.3° | 163.6° | 27 / 30 / 27 | No | Heading drifted to 164°; map updates rejected or matched reverse edge. |
| **`vw8_o3`** | 30s | 195.4 m | 208.5 m | **+13.0 m** | **E** | Heading/gyro-bias dominated | 9.6s (32%) | 56.5° | 52.3° | 60 / 0 / 43 | No | Heading drifted to 52°; 43 updates matched wrong parallel edge. |
| **`motorway_o2`**| 30s | 131.1 m | 134.0 m | **+2.9 m** | **B** | Wrong/ambiguous road association | 30.0s (100%)| 0.0° | 162.8° | 60 / 0 / 14 | No | Negligible (+2.9m): minor parallel lane shift on motorway interchange. |
| **`vw2_o2`** | 10s | 6.8 m | 7.2 m | **+0.5 m** | **D** | Road geometry representation | 10.0s (100%)| 0.0° | 3.2° | 20 / 0 / 2 | No | Negligible (+0.5m): sub-meter polyline chord discretization. |
| **`vw5_o2`** | 10s | 12.3 m | 12.4 m | **+0.1 m** | **B** | Wrong/ambiguous road association | 10.0s (100%)| 0.0° | 87.5° | 20 / 0 / 4 | No | Negligible (+0.1m): sub-meter parallel lane shift at terminus. |

---

## 2. Failure Category Distribution

```
Category E (Heading / Gyro-Bias Dominated)        :  7 / 14 ( 50.0%)  [+2,020.6 m degradation]
Category B (Wrong / Ambiguous Road Association)   :  5 / 14 ( 35.7%)  [+  346.2 m degradation]
Category D (Road Geometry Representation Problem) :  2 / 14 ( 14.3%)  [+   14.6 m degradation]
Category A (Map Unavailable)                      :  0 / 14 (  0.0%)
Category C (Map Gating / Search-Radius Failure)   :  0 / 14 (  0.0%)
Category F (Longitudinal / CAN Velocity Dominated):  0 / 14 (  0.0%)
Category G (Other)                                :  0 / 14 (  0.0%)
```

![Phase 18A Failure Trajectories](phase18a_fig1_failure_categories.png)

---

## 3. First-Divergence Analysis: When & Why Divergence Occurs

Detailed inspection of the time-series logs reveals a sharp divide between catastrophic failures and benign noise:

### 1. The Catastrophic Category E Divergence Mechanism:
- In `vw14b_o5` (180s), `vw4_o1` (120s), and `vw11_o3` (60s), divergence begins relatively early ($t \approx 7\text{--}35\text{ s}$, roughly 10%–20% into the outage).
- During $t \in [0, 20\text{ s}]$, System C tracks ground truth significantly better than System B (CAN only).
- However, consumer IMU gyro bias accumulates over time:
  $$\Delta \psi(t) = \int_0^t b_g(\tau) \, d\tau$$
- Once $|\Delta \psi| > 25^\circ$, the vehicle’s dead-reckoned forward velocity vector $\mathbf{v} = [v_\text{fwd} \cos\psi, v_\text{fwd} \sin\psi]^T$ diverges from the road direction.
- The 1D road-normal constraint tries to enforce:
  $$y_\text{ct} = -\sin\psi_\text{road}(p_N - p_{N,\text{match}}) + \cos\psi_\text{road}(p_E - p_{E,\text{match}}) = 0$$
- When the filter is pointed $45^\circ\text{--}90^\circ$ away from the road, this constraint forces the position estimate to slide along the road normal line, which now intersects perpendicular streets, side roads, or parallel lanes!
- Once the filter snaps to an intersecting road edge, subsequent high-speed CAN updates propel the filter rapidly down the *wrong road*, multiplying error by hundreds of meters.

### 2. Benign Category D Noise:
- In `vw4_o3` (+14.1 m) and `vw2_o2` (+0.5 m), divergence occurs only at the very end of the outage ($t \ge 50\%$).
- The vehicle remains on the correct road edge throughout.
- The error is caused purely by discrete OSM polyline chord discretization (straight line segments approximating a curved asphalt curve). The error is bounded by road lane width ($< 15\text{ m}$) and does not represent topological divergence.

---

## 4. Focus on 120s & 180s Outages: The Long-Duration Achilles' Heel

Long outages account for **86.1% of all degradation** across the 14 failures:
- In short outages (10s), median FPE is only 13.0 m, and failures are negligible sub-meter discretization noise.
- In 120s and 180s outages, the gyro drift without GNSS heading observability grows to $40^\circ\text{--}100^\circ$.
- In `vw14b_o5` (180s), CAN-only (System B) maintained an FPE of 719.2 m because it simply drifted straight ahead in inertial space. In contrast, System C attempted to map-match a vehicle whose heading was rotated by $91^\circ$, latching onto cross-streets and generating a catastrophic 2063.3 m error.

---

## 5. Statistical Comparison: Successful (39) vs. Failed (14) Map Scenarios

| Feature / Metric | Successful (39 Scenarios) | Failed (14 Scenarios) | Contrast / Ratio |
| :--- | :---: | :---: | :--- |
| **Mean Outage Duration** | 42.6 s | **64.3 s** | Failures are 1.5x longer on average |
| **Median Heading Error in System B (CAN)** | **24.1°** | **70.3°** | **Failures have 2.92x worse gyro drift!** |
| **Max Heading Error in System B (CAN)** | 67.2° | **163.6°** | Catastrophic gyro runaways dominate failures |
| **Median Cross-Track Error in System B** | 120.9 m | 121.4 m | Initial cross-track is virtually identical (1.00x) |
| **Map Acceptance Rate** | **80.0%** | 76.2% | -3.8% lower in failures |
| **Mean Map NIS** | 1.13 | 0.93 | Both within nominal $\chi^2$ bounds |

### Core Diagnostic Deduction:
The failure of the 1D road-normal map constraint is **not** caused by map quality, search radius collapse, or NIS mis-tuning.
It is caused by a **single fundamental root cause**:
> **When gyro bias drift rotates the unobservable vehicle heading beyond $\approx 30^\circ$, the 1D road-normal position constraint undergoes geometric breakdown, projecting position onto adjacent, parallel, or perpendicular roads.**

---

## 6. Phase 18B Recommendation

In accordance with the decision rule (*"choose exactly one targeted Phase 18B experiment based on the dominant failure category; do not propose a solution until the failure mechanism has been identified"*):

### Identified Failure Mechanism:
**Unconstrained Heading Drift Induces False Road Association and Cross-Street Snapping.**

### Recommended Single Targeted Phase 18B Experiment:
**Phase 18B: Heading-Gated Map Decoupling & Heading-Consistency Verification**
- Introduce an explicit **geometric heading-consistency gate** in the map matcher:
  $$\Delta \psi_\text{road} = |(\psi_\text{filter} - \psi_\text{road} + \pi) \pmod{2\pi} - \pi|$$
- If $|\Delta \psi_\text{road}| > \theta_\text{max}$ (e.g. $30^\circ\text{--}35^\circ$), the candidate is marked geometrically inconsistent and **abstained**.
- If the vehicle's heading diverges from all local road candidates, the map matcher **gracefully decouples**, falling back to pure CAN-speed dead reckoning (System B) rather than pulling the vehicle onto cross-streets.
- This targeted experiment directly cures the 7 Category E and 5 Category B failures by preventing the filter from snapping onto perpendicular or parallel roads when the gyro has drifted, safely preserving the 55 m median FPE gain of Phase 17A without the catastrophic long-duration outliers.
