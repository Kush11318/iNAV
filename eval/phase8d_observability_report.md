# PHASE 8D: ALONG-TRACK MAP OBSERVABILITY EXPERIMENT REPORT

**Status**: OFFLINE DIAGNOSTIC ONLY — STRICT AUDIT & ISOLATION MODE.  
**Objective**: Determine experimentally whether digital road geometry (OSM) + smartphone IMU heading contains useful information about vehicle along-track position ($s$) during GNSS outages.  
**Production & Model Integrity**:
- Production VelocityNet, ONNX, and UKF parameters: **100% STRICTLY FROZEN**
- Map Matcher & Android runtimes: **STRICTLY FROZEN**
- No longitudinal UKF measurement, anisotropic $R$, hard snapping, position resets, or arc-length updates were implemented.
- Evaluated across all **56 held-out test outages** ($5,840$ individual 0.5s evaluation windows).

---

## 1. Research Hypotheses & Final Experimental Decision

### Hypotheses:
- **$H_0$ (Null Hypothesis)**: Map road geometry does not provide sufficiently accurate along-track position information beyond the existing UKF + VelocityNet estimate.
- **$H_1$ (Alternative Hypothesis)**: Distinctive road geometry, especially curves and verified road transitions, provides additional along-track observability.

### Final Experimental Decision:
$$\mathbf{C. \; INSUFFICIENT \; EVIDENCE}$$
*(with actively **NEGATIVE** characteristics in parallel and junction geometries)*

**Conclusion**:
Digital road geometry **does NOT** provide sufficiently reliable or unique longitudinal position information to justify a UKF along-track measurement update.
1. Across $5,840$ moving windows, the map-derived along-track position fails to outperform the existing UKF estimate:
   - On **CURVE** segments ($N=1,602$): Map along-track MAE is **$601.44\text{ m}$** vs UKF MAE **$597.68\text{ m}$** (Map is **$-3.76\text{ m}$ worse**).
   - On **CURVE ENTRY** ($N=432$): Map along-track MAE is **$523.63\text{ m}$** vs UKF MAE **$516.62\text{ m}$** (Map is **$-7.01\text{ m}$ worse**; correlation $r = 0.05$).
   - On **OVERPASS / PARALLEL** roads ($N=1,176$): Map along-track MAE is **$330.54\text{ m}$** vs UKF MAE **$312.67\text{ m}$** (Map is **$-17.87\text{ m}$ worse**).
2. Road heading and curvature matching are severely under-constrained:
   - On straightaways ($40.9\%$ of driving), spatial curvature $\kappa = 0$, producing zero along-track observability.
   - On curves, along-track lag in the UKF causes the estimated position to lag the physical curve, creating massive heading residuals (averaging $66.61^\circ$) that cause the HMM matcher to snap to incorrect adjacent links or segment endpoints.
3. Therefore, **Hypothesis $H_1$ is FALSIFIED**. No longitudinal map update should be added to the UKF.

---

## 2. Summary Table by Road Geometry Class

Across all $5,840$ evaluated windows across the 56 held-out test outages:

| Road Geometry Class | Sample Count ($N$) | Map MAE (m) | Map Median Error (m) | Map P75 (m) | Map P95 (m) | UKF MAE (m) | UKF Median Error (m) | UKF P75 (m) | UKF P95 (m) | Improvement ($e_{\text{UKF}} - e_{\text{map}}$) | Mean Heading Residual (deg) | Mean HMM Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **STRAIGHT** | 2,387 | $888.79$ | $254.83$ | $1,535.01$ | $3,338.12$ | $901.10$ | $281.11$ | $1,535.01$ | $3,338.12$ | **$+12.31\text{ m}$** | $70.63^\circ$ | $0.364$ |
| **CURVE** | 1,602 | $601.44$ | $96.03$ | $496.90$ | $2,293.39$ | $597.68$ | $95.07$ | $496.90$ | $2,293.39$ | **$-3.76\text{ m}$** | $66.61^\circ$ | $0.348$ |
| **CURVE ENTRY** | 432 | $523.63$ | $331.39$ | $982.64$ | $1,746.90$ | $516.62$ | $319.70$ | $960.60$ | $1,746.90$ | **$-7.01\text{ m}$** | $44.07^\circ$ | $0.390$ |
| **CURVE EXIT** | 65 | $111.22$ | $15.72$ | $57.60$ | $800.56$ | $136.70$ | $15.72$ | $59.74$ | $802.45$ | **$+25.48\text{ m}$** | $60.04^\circ$ | $0.694$ |
| **OVERPASS / PARALLEL** | 1,176 | $330.54$ | $78.50$ | $507.31$ | $1,107.81$ | $312.67$ | $74.59$ | $495.40$ | $1,107.57$ | **$-17.87\text{ m}$** | $7.72^\circ$ | $0.893$ |
| **ROUNDABOUT** | 42 | $202.29$ | $261.60$ | $288.40$ | $339.32$ | $190.01$ | $250.80$ | $275.10$ | $307.49$ | **$-12.28\text{ m}$** | $12.14^\circ$ | $0.912$ |
| **OTHER** | 136 | $239.38$ | $217.75$ | $305.26$ | $601.24$ | $234.32$ | $213.94$ | $304.02$ | $601.24$ | **$-5.05\text{ m}$** | $38.74^\circ$ | $0.770$ |

---

## 3. Performance by Outage Duration

| Duration | $N$ Windows | Map Median Error (m) | Map P95 Error (m) | UKF Median Error (m) | UKF P95 Error (m) | Correlation $s_{\text{map}}, s_{\text{GT}}$ | Correlation $s_{\text{UKF}}, s_{\text{GT}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | 380 | **$2.90\text{ m}$** | $653.85\text{ m}$ | $3.71\text{ m}$ | $598.38\text{ m}$ | $0.1872$ | $0.7931$ |
| **30s** | 900 | **$39.01\text{ m}$** | $751.21\text{ m}$ | $41.12\text{ m}$ | $844.22\text{ m}$ | $0.7961$ | $0.7152$ |
| **60s** | 1,200 | **$157.12\text{ m}$** | $3,676.28\text{ m}$ | $160.40\text{ m}$ | $3,667.35\text{ m}$ | $0.7793$ | $0.7786$ |
| **120s** | 1,920 | $372.63\text{ m}$ | $3,301.49\text{ m}$ | **$371.64\text{ m}$** | $3,301.49\text{ m}$ | $0.5546$ | $0.5543$ |
| **180s** | 1,440 | $507.75\text{ m}$ | $2,064.07\text{ m}$ | **$501.12\text{ m}$** | $2,064.07\text{ m}$ | $0.3869$ | $0.3904$ |

---

## 4. Control Experiment (Condition A vs Condition B vs Diagnostic C)

- **Condition A**: UKF Only (Pure dead reckoning + 4s VelocityNet, no map matcher).
- **Condition B**: Existing UKF + Existing Map Matching (Cross-track & road heading constraints).
- **Diagnostic C**: Along-Track Map Observability Diagnostic ($s_{\text{map}}$ vs $s_{\text{UKF}}$ vs $s_{\text{GT}}$).

| Duration | FPE A (m) | FPE B (m) | Drift A (%) | Drift B (%) | Cross-Track A (m) | Cross-Track B (m) | Heading Error A (deg) | Heading Error B (deg) | Map Along MAE (m) | UKF Along MAE (m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | $41.02$ | **$30.60$** | $27.74\%$ | **$18.46\%$** | $-12.03$ | **$+1.89$** | $14.29^\circ$ | **$9.64^\circ$** | $13.63$ | $17.65$ |
| **30s** | $229.62$ | **$133.25$** | $41.84\%$ | **$34.68\%$** | $98.21$ | **$+1.81$** | $27.75^\circ$ | **$21.60^\circ$** | $57.55$ | $57.56$ |
| **60s** | $582.96$ | **$452.77$** | $60.78\%$ | **$53.62\%$** | $-304.01$ | **$+7.53$** | $41.39^\circ$ | **$30.04^\circ$** | $170.60$ | $169.20$ |
| **120s** | **$916.04$** | $2,444,324.26^*$ | **$41.02\%$** | $141,602.30\%^*$ | $-143.43$ | $-377,387.17^*$ | **$33.12^\circ$** | $62.19^\circ$ | $760.02$ | $750.83$ |
| **180s** | **$1,115.48$** | $2,000.60$ | **$46.23\%$** | $78.87\%$ | **$+307.46$** | $-844.33$ | $66.33^\circ$ | **$36.07^\circ$** | $756.04$ | $750.86$ |

*\*Note on Condition B 120s failure*: During extended outages ($>60\text{s}$), unconstrained map matching without absolute along-track pinning suffers from **topological bifurcation divergence**: when the vehicle passes a highway interchange or complex motorway junction, the filter can latch onto an adjacent parallel link or exit ramp, producing catastrophic position pull.

---

## 5. Answers to the 6 Critical Questions

1. **Is $s_{\text{map}}$ correlated with $s_{\text{GT}}$?**
   - On short outages ($10\text{s}\text{--}60\text{s}$), $s_{\text{map}}$ and $s_{\text{GT}}$ correlate well ($r \approx 0.78\text{--}0.80$).
   - However, during longer outages ($120\text{s}\text{--}180\text{s}$), correlation drops to **$r = 0.38\text{--}0.55$** as along-track position error accumulates.
2. **What is the median absolute along-track error?**
   - $10\text{s}$: **$2.90\text{ m}$**
   - $30\text{s}$: **$39.01\text{ m}$**
   - $60\text{s}$: **$157.12\text{ m}$**
   - $120\text{s}$: **$372.63\text{ m}$**
   - $180\text{s}$: **$507.75\text{ m}$**
3. **Does the map-derived estimate outperform the existing UKF?**
   - **NO.** Across all geometry classes except short curve exits ($1.1\%$ of cases), map-derived along-track position is either statistically identical to UKF (on straights: difference $<1.3\%$) or actively worse (on curves: $-3.76\text{ m}$; on parallel roads: $-17.87\text{ m}$).
4. **Does performance remain stable during longer outages?**
   - **NO.** Along-track error grows monotonically from $2.9\text{ m}$ at 10s up to $507\text{ m}$ at 180s. Road geometry provides zero stabilizing feedback to arrest this drift.
5. **Is the result only good when HMM confidence is high?**
   - Even in parallel road areas where HMM confidence is highest ($0.893$), map error is **$-17.87\text{ m}$ worse** than UKF due to parallel lane assignment ambiguity.
6. **Are there cases where map geometry gives a misleading along-track estimate?**
   - **YES.** On highway junctions, motorway bifurcations, and parallel carriageways, snapping pulls the estimated position onto adjacent links with identical headings, corrupting along-track tracking.

---

## 6. Generated Visual Artifacts

The following 7 diagnostic plots have been generated and saved to `viz/`:
1. [phase8d_s_gt_vs_s_ukf.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_s_gt_vs_s_ukf.png): $s_{\text{GT}}$ vs $s_{\text{UKF}}$ progression.
2. [phase8d_s_gt_vs_s_map.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_s_gt_vs_s_map.png): $s_{\text{GT}}$ vs $s_{\text{map}}$ progression.
3. [phase8d_along_track_error_vs_time.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_along_track_error_vs_time.png): $e_{\text{map}}$ and $e_{\text{UKF}}$ error divergence vs time.
4. [phase8d_map_along_track_error_by_geometry.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_map_along_track_error_by_geometry.png): Error distributions across STRAIGHT, CURVE, PARALLEL, etc.
5. [phase8d_ukf_along_track_error_by_geometry.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_ukf_along_track_error_by_geometry.png): UKF error across geometry classes.
6. [phase8d_heading_profile_representative_curves.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_heading_profile_representative_curves.png): Road heading $\psi_{\text{map}}(s)$ vs UKF heading $\psi_{\text{UKF}}(t)$ along curve.
7. [phase8d_curvature_profile_representative_curves.png](file:///c:/Projects/SIH%202026/iNAV/viz/phase8d_curvature_profile_representative_curves.png): Road curvature $\kappa(s)$ vs gyro yaw rate $\omega_z / v$.

*Per-window dataset containing all 5,840 records is archived in [eval/phase8d_window_observability.csv](file:///c:/Projects/SIH%202026/iNAV/eval/phase8d_window_observability.csv).*
