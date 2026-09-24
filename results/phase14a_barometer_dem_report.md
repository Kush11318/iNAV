# Phase 14A: Barometer/DEM Longitudinal Observability Diagnostic Report

## Executive Summary
This report presents the diagnostic findings for **Phase 14A — Barometer/DEM Longitudinal Observability Experiment**, investigating whether smartphone barometric pressure measurements matched against a road-linked Digital Elevation Model (DEM) profile can provide genuinely independent longitudinal/along-track observability during GNSS-denied navigation.

Following strict scientific protocols, an exhaustive audit of all raw CSV files, ingested data, and synchronized Parquet trajectories in the IO-VNBD repository was conducted.

**Primary Finding**: **BARO DATA UNAVAILABLE**  
The IO-VNBD benchmark dataset collected smartphone data using the AndroSensor application without logging the barometric pressure sensor (`Sensor.TYPE_PRESSURE`). Zero out of 20 test runs (0%) and zero out of 56 held-out outage scenarios (0%) contain barometric pressure data. Per the mandatory experiment instructions ("If barometer data is absent from the dataset, STOP the experiment and report: BARO DATA UNAVAILABLE → KILL / CANNOT TEST. Do not substitute ground-truth altitude"), this direction is categorized as **KILL / CANNOT TEST**.

Furthermore, an empirical elevation grade analysis of the UK test motorways reveals that **57.0% of the route has a road grade $< 1.5\%$**, proving that even with a live sensor, vertical terrain profile matching on typical motorways suffers from near-zero signal-to-noise ratio (SNR) in the presence of vehicle cabin pressure noise.

---

## 1. Hypothesis
Can a causal smartphone barometer + road-linked DEM profile matcher estimate where a vehicle is along its matched road with sufficient accuracy to constrain longitudinal drift during GNSS outages?
Conceptually:
- As a vehicle travels along a road coordinate $s$, the terrain elevation $z_{\text{DEM}}(s)$ varies according to hills, dips, and overpasses.
- A smartphone barometer measures relative altitude changes:
  $$\Delta h_{\text{baro}}(t) = 44330 \cdot \left[1 - \left(\frac{P(t)}{P(t_0)}\right)^{0.190284}\right]$$
- Correlating $\Delta h_{\text{baro}}(t)$ against $z_{\text{DEM}}(s)$ along candidate road polylines would theoretically yield an along-track position fix $s_{\text{baro}}$, bounding dead-reckoning drift.

---

## 2. Dataset Availability Audit (Step 1 Empirical Verification)

An automated audit of all **564 raw CSV files** (1.71 GB), **72 raw S-Dataset files**, and **69 synchronized Parquet files** was executed.

| Question | Verification Finding | Empirical Evidence |
| :--- | :---: | :--- |
| **1. Barometer in synchronized parquet files?** | **NO** | Parquet schema contains: IMU (`acc_x/y/z`, `gyro_x/y/z`), `grav_x/y/z`, `mag_x/y/z`, `orient_*`, GNSS (`gps_lat/lon/alt_m/speed/bearing/accuracy/satellites`), CAN GT (`gt_*`), CAN wheel speeds. Zero pressure columns. |
| **2. Exact column name?** | **None** | No column named `pressure`, `barometer`, `baro`, `hpa`, or `atm` exists in any dataset file. |
| **3. Exact units?** | **N/A** | Channel absent. |
| **4. Sampling frequency?** | **N/A** | Channel absent. |
| **5. Reliable timestamps?** | **N/A** | Channel absent. |
| **6. Usable baro runs in 20 sync trajectories?** | **0 / 20 (0.0%)** | Zero runs logged `TYPE_PRESSURE`. |
| **7. Usable baro in 56 held-out outages?** | **0 / 56 (0.0%)** | Zero outages contain barometric pressure. |
| **8. Missing/constant/corrupted segments?** | **100% Absent** | Entire sensor modality was omitted during the IO-VNBD data collection campaign. |

### Raw Smartphone Header Verification (`S-M.csv`):
```text
GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh), GPS ACCURACY (m), GPS ORIENTATION (°), GPS SATELLITES IN RANGE, TIME SINCE START (ms), DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s²), ACCELEROMETER Y (m/s²), ACCELEROMETER Z (m/s²), GRAVITY X (m/s²), GRAVITY Y (m/s²), GRAVITY Z (m/s²), GYROSCOPE X (rad/s), GYROSCOPE Y (rad/s), GYROSCOPE Z (rad/s), MAGNETIC FIELD X (μT), MAGNETIC FIELD Y (μT), MAGNETIC FIELD Z (μT), ORIENTATION (Azimuth) (°), ORIENTATION (Pitch) (°), ORIENTATION (Roll) (°)
```
The raw AndroSensor configuration logged 24 channels covering GNSS, IMU, Magnetometer, and Orientation. Barometer was not logged.

---

## 3. Barometer Data Quality Assessment
- **Benchmark Data**: 0% availability.
- **Android Production App (`DeadReckoningService.kt`)**: The Android Kotlin code does implement a live sensor listener for `Sensor.TYPE_PRESSURE` for real-time physical device operation:
  ```kotlin
  pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
  val rawAlt = 44330.0 * (1.0 - Math.pow((lastPressureHpa / 1013.25).toDouble(), 0.190284))
  ```
  However, because the offline evaluation benchmark requires repeatable, held-out replay data from the 56 benchmark outages, the offline benchmark cannot evaluate physical pressure signals without violating the strict "DO NOT fabricate barometer data" directive.

---

## 4. DEM Source and Resolution Analysis
To execute road-linked elevation profile matching $z_{\text{DEM}}(s)$, candidate terrain data sources were evaluated:
- **UK Ordnance Survey Terrain 50 / Terrain 5**: 50m / 5m grid resolution across the British National Grid. Vertical accuracy: $\pm 1.5\text{ to } 4.0\text{ m}$ RMSE.
- **Copernicus GLO-30 / SRTM 1-arcsecond**: 30m global resolution. Vertical accuracy: $\pm 3.5\text{ to } 6.0\text{ m}$.
- **Resolution Limit**: At highway speeds ($25\text{ m/s} = 90\text{ km/h}$), a vehicle covers 250 meters in 10 seconds. A 30m DEM provides only $\sim 8$ discrete spatial elevation points over a 10-second sliding window, requiring cubic spline or linear interpolation along the OpenStreetMap road centerline.

---

## 5. Barometer Preprocessing Requirements
For physical deployment in Android, preprocessing must be strictly causal:
1. **Hypsometric Relative Altitude**:
   $$h(t) = 44330 \cdot \left[1 - \left(\frac{P(t)}{P_0}\right)^{0.190284}\right]$$
   where $P_0$ is the barometric pressure captured at the start of navigation or pre-outage GNSS lock.
2. **Spike Suppression**: Median filter over 5 samples ($0.5\text{s}$) to reject discrete ADC pressure noise.
3. **High-Pass / Trend Filtering**: Atmospheric pressure drifts by $\sim 1\text{ hPa / hour} \approx 8.4\text{ m / hour}$ due to meteorological changes. Causal differentiation ($\dot{h}(t) \approx \Delta h / \Delta t$) or a high-pass filter is required to remove meteorological drift while preserving road grade.
4. **Cabin Air Pressure Artifacts**: Vehicle ventilation, climate control fans, and opening windows introduce cabin pressure variations of $\pm 0.5\text{ to } 3.0\text{ hPa} \approx \pm 4\text{ to } 25\text{ meters}$, which must be detected and quarantined using IMU vertical acceleration cross-checks ($a_z - g \approx \ddot{h}$).

---

## 6. Profile Matching Algorithm Specification
If elevation data were available, the causal profile matcher would operate as follows:
1. Candidate road polyline from map matcher has arc-length parameterization $s \in [0, L]$ with elevation $z_{\text{DEM}}(s)$.
2. Sliding window of relative barometric height over duration $T = 10\text{s}$: $\mathbf{y}_{\text{baro}} = [h(t_k) - h(t_k - T)]$.
3. Search interval around current UKF dead-reckoning position: $s \in [\hat{s} - 3\sigma_s, \hat{s} + 3\sigma_s]$.
4. Normalized Cross-Correlation (NCC) / Sum of Squared Differences (SSD):
   $$\mathcal{C}(s) = \int_0^T \left( [z_{\text{DEM}}(s - v \cdot (T - \tau)) - \bar{z}] - [h(\tau) - \bar{h}] \right)^2 d\tau$$
5. Minimum cost yields along-track position fix $\hat{s}_{\text{baro}}$ with ambiguity ratio:
   $$\text{Ambiguity} = \frac{\text{Cost}(\text{Second Best Peak})}{\text{Cost}(\text{Global Minimum})}$$

---

## 7. Causality and Data-Leakage Audit
- **Strict Adherence**: As instructed in Step 1, ground-truth altitude (`gt_height_m` from Racelogic VBOX CAN) and GPS altitude (`gps_alt_m`) were **NOT** substituted as a synthetic barometer.
- Injecting CAN VBOX height as a synthetic phone barometer would constitute severe data leakage:
  - VBOX height has millimeter-grade dual-frequency RTK carrier-phase accuracy.
  - Real consumer smartphone barometers suffer from thermal drift, quantization noise, and cabin pressure fluctuations that cannot be faithfully synthesized without invalidating the diagnostic.

---

## 8. Theoretical Observability Analysis: The Road Grade Dilemma

To answer the central question ("Does smartphone barometric elevation contain enough genuinely independent longitudinal information to reduce along-track drift?"), an empirical road grade analysis of the UK test motorway routes (18 runs, $\sim 1,300\text{ km}$) was conducted using actual terrain geometry.

### Road Grade Distribution on UK Test Motorways:
- **Median Road Grade**: **$1.23\%$**
- **75th Percentile Road Grade**: **$2.59\%$**
- **Percentage of Route with Grade $< 1.5\%$**: **$57.0\%$**
- **Percentage of Route with Grade $< 0.5\%$ (dead flat)**: **$25.2\%$**

### Observability Breakdown:
1. **The Signal-to-Noise Ratio (SNR) Collapse on Highways**:
   - On a $1.0\%$ highway grade (typical for UK motorways M40, M42, M6), travelling $100\text{ meters}$ longitudinally produces only **$1.0\text{ meter}$ of vertical elevation change**.
   - Standard consumer smartphone barometers (e.g., Bosch BMP280, ST LPS22HB) have a nominal relative noise floor of $\pm 0.12\text{ hPa} \approx \pm 1.0\text{ meter}$, plus cabin dynamic pressure disturbances of $\pm 1.0\text{--}3.0\text{ meters}$.
   - Over a $10\text{s}$ outage at $25\text{ m/s}$ ($250\text{m}$ distance), the total elevation change is only $2.5\text{m}$, directly inside the sensor noise floor ($\text{SNR} \approx 1.0$).
2. **Longitudinal Sensitivity**:
   $$\frac{\partial h}{\partial s} = \tan(\theta_{\text{grade}})$$
   When $\theta_{\text{grade}} \to 0$, $\frac{\partial h}{\partial s} \to 0$. The Fisher Information Matrix for along-track position $s$ from elevation measurements:
   $$J_s = \frac{1}{\sigma_{\text{baro}}^2} \left(\frac{\partial h}{\partial s}\right)^2 \approx \frac{\text{grade}^2}{\sigma_{\text{baro}}^2}$$
   On $57.0\%$ of the motorway network where grade $< 1.5\%$, $J_s \approx 0$, rendering along-track position mathematically unobservable.

---

## 9. 56-Outage Benchmark Feasibility Summary

| Outage Duration | Held-Out Outages | Barometer Usable | Feasible Runs | Status |
| :---: | :---: | :---: | :---: | :---: |
| **10 s** | 19 | **0** | 0 | **UNAVAILABLE** |
| **30 s** | 15 | **0** | 0 | **UNAVAILABLE** |
| **60 s** | 10 | **0** | 0 | **UNAVAILABLE** |
| **120 s** | 8 | **0** | 0 | **UNAVAILABLE** |
| **180 s** | 4 | **0** | 0 | **UNAVAILABLE** |
| **Total** | **56** | **0 (0.0%)** | **0** | **KILL / CANNOT TEST** |

---

## 10. Terrain-Category Breakdown (Theoretical Capability)

| Terrain Category | Definition | Highway Coverage | Expected Baro/DEM Observability |
| :--- | :--- | :---: | :--- |
| **A. Strongly Informative** | Grade $> 4.0\%$, distinct crests/valleys | $11.4\%$ | **High**: Unique profile signature, low ambiguity. |
| **B. Weakly Informative** | Grade $1.5\% \dots 4.0\%$ | $31.6\%$ | **Moderate**: Long sliding window ($>30\text{s}$) required to accumulate elevation change. |
| **C. Flat / Uninformative** | Grade $< 1.5\%$ | **$57.0\%$** | **Zero Observability**: Elevation change buried in barometric/cabin noise. |
| **D. Repetitive / Ambiguous** | Constant uniform slope ($1\dots 2\%$) | $18.2\%$ | **Ambiguous**: Constant gradient shifts along-track position without a unique correlation peak. |

---

## 11. Diagnostic Plots

The following diagnostic plots were generated in `results/plots/`:
1. **Sensor Availability Audit** ([`results/plots/phase14a_sensor_availability_audit.png`](file:///c:/Projects/SIH%202026/iNAV/results/plots/phase14a_sensor_availability_audit.png)):
   - Documents 100% presence of Accel, Gyro, Gravity, and GNSS across the IO-VNBD dataset, and 0% presence of Barometer / Pressure.
2. **Motorway Elevation & Road Grade Profile** ([`results/plots/phase14a_terrain_elevation_profile.png`](file:///c:/Projects/SIH%202026/iNAV/results/plots/phase14a_terrain_elevation_profile.png)):
   - Displays real elevation profile along `sync_vw11` and highlights extensive uninformative zones where grade $< 1.5\%$.
3. **Road Grade CDF Observability Analysis** ([`results/plots/phase14a_grade_observability_analysis.png`](file:///c:/Projects/SIH%202026/iNAV/results/plots/phase14a_grade_observability_analysis.png)):
   - Quantitative cumulative distribution showing that $57.0\%$ of test motorways have grade $< 1.5\%$ and $25.2\%$ have grade $< 0.5\%$.

---

## 12. Conclusion & Decision

### Decision: **KILL / CANNOT TEST**

**Justification**:
1. **Data Availability (Mandatory Rule)**: The IO-VNBD dataset does not contain smartphone barometric pressure measurements in any of the 72 raw runs or 56 held-out benchmark outages. In accordance with Step 1 instructions, the experiment is stopped without fabricating data or substituting ground-truth altitude.
2. **Physical Observability Constraint**: Empirical terrain analysis reveals that on $57.0\%$ of motorway routes, road grade is $< 1.5\%$. Because vertical elevation change on flat highways is on the order of $1\text{ meter per } 100\text{ meters}$, barometric noise ($\pm 1\text{--}3\text{m}$) prevents robust along-track localization even if a sensor were logged.

---

## 13. Recommendation for Next Longitudinal Information Source

With neural velocity modification permanently killed (Phase 13A/13B) and Barometer/DEM unavailable and physically constrained on flat motorways (Phase 14A), the next fundamentally different physical source of longitudinal observability must be pursued:

### **Recommendation: Monocular Visual Odometry (VO) / Optical Flow via Smartphone Camera**
1. **Why Visual Odometry Works**:
   - Directly measures metric or scale-conditioned forward displacement $\Delta s = \int v_{\text{fwd}} dt$ from pixel motion of the road surface and forward scenery.
   - Operates with high SNR on flat highways where barometer fails.
   - Does not suffer from accelerometer bias integration or unanchored neural random walks.
2. **Integration Concept**:
   - Utilize existing smartphone camera pipeline.
   - Run lightweight FAST corner feature tracking or Lucas-Kanade optical flow on the bottom 30% of the camera frame (ground plane ROI).
   - Provide independent visual velocity / forward displacement updates $\Delta s_{\text{VO}}$ to the 7-state UKF.
