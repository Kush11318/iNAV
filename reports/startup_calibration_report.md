# Phase 12B — 15-Second Passive Startup Calibration Report

**Date:** 2026-09-13  
**Status:** VALIDATED & INTEGRATED  
**Target:** Android App UI (`MainActivity`, `AppleClockTimerView`, `activity_main.xml`) & Core Inertial Pipeline (`StartupCalibrationManager`, `DeadReckoningService`, `NativeBridge`)

---

## 1. Executive Summary & Objective

Phase 12B introduces a **zero-maneuver, passive 15-second startup calibration** flow for the iNAV navigation platform. 

### Key UX Directives Followed:
- **No driving maneuvers required**: The driver is never asked to drive straight, brake, accelerate, rotate the phone, or execute U-turns.
- **Safety-first UI**: Prominently displays `⚠️ Complete calibration while safely parked.`
- **Authentic Apple Clock UI**: Features an Apple Watch/Clock-inspired 60-tick dial with a smooth animated sweep arc and dynamic numerical countdown.
- **Stationary Decoupling**: Gravity is strictly decoupled into pitch and roll vehicle-to-phone leveling. Yaw is preserved to be aligned dynamically with forward GNSS/odometry motion, eliminating fabricated heading errors.

---

## 2. Mathematical Calibration Architecture

```
         Hardware Sensors (100 Hz SENSOR_DELAY_FASTEST)
                           │
                           ▼
               DeadReckoningService
                           │ (onSensorChanged: ax, ay, az, gx, gy, gz, t_ns)
                           ▼
              StartupCalibrationManager
                           │
      ┌────────────────────┴─────────────────────┐
      ▼                                          ▼
Stationarity Gate                        Rolling Accumulator
|mean(a) - g| < 0.50 m/s²                1,500 samples (~15.0s)
std(a) < 0.22 m/s²                       Gyro bias bg = mean(g)
|gyro| < 0.15 rad/s                      Gravity uz = -mean(a) / |mean(a)|
      │                                          │
      ▼                                          ▼
Paused on motion                   NativeBridge.nativeApplyStaticCalibration(...)
"Please keep vehicle stationary"                 │
                                                 ▼
                                        C++ Navigation Engine
```

### Parameter Derivation

1. **Stationary Gating**:
   A rolling 0.5s window ($N = 50$ samples) checks three simultaneous criteria:
   $$\big| \|\mathbf{a}\| - 9.80665 \big| < 0.50\ \text{m/s}^2$$
   $$\sigma_a < 0.22\ \text{m/s}^2$$
   $$\max \|\boldsymbol{\omega}\| < 0.15\ \text{rad/s}\ (\approx 8.6^\circ/\text{s})$$
   If the vehicle accelerates or moves, the collection timer pauses, the dial switches to amber, and the subtitle instructs the driver to remain stationary.

2. **Gyroscope Bias Vector ($\mathbf{b}_g$)**:
   $$\mathbf{b}_g = \frac{1}{N} \sum_{i=1}^N \boldsymbol{\omega}_i, \quad \sigma_g = \sqrt{\text{Var}(\omega_x) + \text{Var}(\omega_y) + \text{Var}(\omega_z)}$$
   $$\text{Conf}_g = \text{clamp}\left(1.0 - \frac{\sigma_g}{0.03}, 0.0, 1.0\right)$$

3. **Attitude Leveling (Pitch & Roll)**:
   $$\bar{\mathbf{a}} = \frac{1}{N} \sum_{i=1}^N \mathbf{a}_i, \quad g_{\text{mag}} = \|\bar{\mathbf{a}}\|$$
   $$\mathbf{u}_z = -\frac{\bar{\mathbf{a}}}{g_{\text{mag}}}$$
   $$\theta_{\text{pitch}} = \arcsin\left(\text{clamp}(-u_{z,x}, -1.0, 1.0)\right)$$
   $$\phi_{\text{roll}} = \text{atan2}\left(u_{z,y}, -u_{z,z}\right)$$

4. **Decoupled Yaw Status**:
   Stationary gravity alone does not constrain heading. Yaw is preserved as:
   `"Stationary leveled baseline. Yaw aligns dynamically with driving vector."`

---

## 3. Operational Test Results & Verification

Evaluated using `eval/test_calibration_simulation.py` across 5 rigorous conditions:

| Test Case | Scenario | Expected | Observed | Deviation / Error | Status |
|---|---|---|---|---|---|
| **Test 1** | Flat Stationary Mount | $b_g = [2.0, -3.5, 1.0]\times 10^{-3}$, Pitch=0°, Roll=0° | $b_g = [1.93, -3.59, 1.09]\times 10^{-3}$, Pitch=0.00°, Roll=0.00° | $\|e_b\| = 0.000147$ rad/s, Tilt $< 0.01^\circ$ | **PASS** |
| **Test 2** | Windshield Cradle Mount | Pitch = 30.0°, Roll = -10.0° | Pitch = 30.01°, Roll = -10.01° | $\Delta_{\text{pitch}} = 0.01^\circ, \Delta_{\text{roll}} = 0.01^\circ$ | **PASS** |
| **Test 3** | Dynamic Motion Interruption (Vehicle moves between 5s–9s) | Timer pauses during motion, resumes when still | Elapsed wall time = 19.50s, Stationary duration = 15.01s | Zero contaminated samples included | **PASS** |
| **Test 4** | Engine Idling Vibration (25 Hz harmonic) | Stationarity detector tolerates engine idle vibration | Accel var: $0.0177\ (\text{m/s}^2)^2$, Conf: 91.3% | Completed successfully without false reject | **PASS** |
| **Test 5** | 10 Repeated Calibrations (Repeatability) | $b_g$ std $< 0.0001$ rad/s, Attitude std $< 0.05^\circ$ | $b_g$ std: $0.000052$ rad/s, Attitude std: $0.005^\circ$ | Mean confidence: 94.1% | **PASS** |

---

## 4. UI/UX Implementation Details

### Visual Hierarchy in `activity_main.xml` & `MainActivity.kt`:
1. **Preparing Navigation Screen**:
   - Header: *"Preparing Navigation"*
   - Subtitle: *"Keep your phone securely mounted.\nKeep the vehicle stationary while we calibrate the sensors."*
   - Safety Pill: *"⚠️ Complete calibration while safely parked."*
   - Apple Clock Circular Dial:
     - 60 radial tick marks with cardinal emphasis
     - Center large digit countdown (Transitions.dev spring pop-in)
     - Animated orange sweep arc
2. **Live Checklist Progression**:
   - `[✓] Sensors connected` (>20 samples)
   - `[✓] Detecting gravity` (t ≥ 3.5s)
   - `[✓] Estimating gyro bias` (t ≥ 7.5s)
   - `[✓] Finalizing alignment` (t ≥ 12.0s)
3. **Completion State**:
   - Header switches to *"Calibration Complete"*
   - Dial switches to emerald `#00E676`
   - Checklist displays:
     - `✓ Phone orientation detected`
     - `✓ Gyroscope calibrated`
     - `✓ Accelerometer calibrated`
     - `✓ Navigation ready`
   - Emerald CTA button *"START NAVIGATION"* appears with auto-advance after 2.5s.
4. **Developer Debug Panel**:
   - Expandable via `🛠 DEVELOPER DIAGNOSTICS [▲/▼]`
   - Exposes all 12 requested diagnostic metrics:
     1. Calibration Duration (s)
     2. IMU Sample Count
     3. Gyro Bias X/Y/Z (rad/s)
     4. Gyro Standard Deviation (rad/s)
     5. Mean Acceleration X/Y/Z (m/s²)
     6. Gravity Magnitude (m/s²)
     7. Acceleration Variance (m/s²)²
     8. Stationary Confidence (%)
     9. Pitch (°)
     10. Roll (°)
     11. Yaw Status string
     12. Alignment Confidence (%)

---

## 5. Parameter Pipeline Entry Points

The calibrated parameters enter the existing navigation architecture strictly through the established native bridge:

```kotlin
NativeBridge.nativeApplyStaticCalibration(
    axMean = axMean,
    ayMean = ayMean,
    azMean = azMean,
    gxBias = bgX,
    gyBias = bgY,
    gzBias = bgZ
)
```

- **Inertial Bias Initialization**: Nullifies gyroscope run-to-run drift on the 3 orthogonal axes.
- **Leveling Matrix ($R_b^v$)**: Initial pitch and roll are passed to the C++ attitude filter to align the accelerometer with local level horizon.
- **Continuous Alignment Intact**: The downstream GNSS-course-over-ground and non-holonomic constraint (NHC) heading updates continue uninterrupted to refine azimuth during transit.
