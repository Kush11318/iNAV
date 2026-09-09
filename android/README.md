# iNAV Android Mobile Application (`android/`)

Real-time, edge-native Android dead-reckoning navigation application powered by **iNAV** (1D Dilated CNN + GRU `VelocityNet` + C++17 Unscented Kalman Filter).

---

## Architecture

```
[Phone IMU Sensors: Accel + Gyro (100 Hz)]
                   │
                   ▼
       [DeadReckoningService (Foreground)]
                   │
                   ▼ JNI (NativeBridge.kt)
       ┌────────────────────────────────────────────────────────┐
       │                   libinav_core.so                      │
       │                                                        │
       │  • inav_filter.hpp: 7-State UKF Kinematic Mechanization│
       │  • inav_onnx.hpp: ONNX Runtime C++ Mobile Inference    │
       │  • velocity_net.onnx: Bundled Assets (106.3 KB)        │
       └────────────────────────────────────────────────────────┘
                   │
                   ▼ StateFlow<NavigationState>
       [MainActivity Dashboard & OsmDroid OpenStreetMap View]
       • Real-time vehicle speed (km/h) & heading (degrees)
       • Road Condition detection (Normal / Rough Road / ZUPT)
       • Live color-coded track (Green = GPS, Purple = Dead Reckoning)
       • Software GNSS Outage Simulator Toggle
```

---

## Project Structure

```
android/
├── app/
│   ├── build.gradle                   # App module build definition
│   └── src/main/
│       ├── AndroidManifest.xml        # High-rate sensor & foreground permissions
│       ├── assets/
│       │   └── velocity_net.onnx      # 106.3 KB trained displacement model
│       ├── cpp/
│       │   ├── CMakeLists.txt         # NDK C++ build configuration
│       │   ├── inav_filter.hpp        # C++17 7-state UKF navigation filter
│       │   ├── inav_onnx.hpp          # ONNX Runtime C++ wrapper
│       │   └── inav_jni.cpp           # JNI bindings
│       ├── java/com/inav/navigation/
│       │   ├── NativeBridge.kt        # JNI interface & model asset loader
│       │   ├── model/
│       │   │   └── NavigationState.kt # Immutable telemetry data class
│       │   ├── service/
│       │   │   └── DeadReckoningService.kt # 100 Hz sensor foreground service
│       │   └── ui/
│       │       └── MainActivity.kt    # OsmDroid map & dashboard UI
│       └── res/
│           ├── layout/activity_main.xml # Dark-mode dashboard layout
│           └── values/{colors, strings, themes}.xml
├── build.gradle                       # Top-level Gradle configuration
├── settings.gradle                    # Project module settings
└── gradle/wrapper/gradle-wrapper.properties
```

---

## How to Run in Android Studio

1. **Open Android Studio**:
   - Select **Open** and select the folder: `c:\Projects\SIH 2026\iNAV\android`.
2. **Gradle Sync**:
   - Android Studio will automatically recognize the project, download Gradle dependencies (OsmDroid, ONNX Runtime, AndroidX), and configure CMake for NDK compilation.
3. **Connect Device / Emulator**:
   - Connect a physical Android phone via USB debugging (recommended for real IMU sensor motion) or launch an Android Virtual Device (AVD).
4. **Build and Run (`Shift + F10`)**:
   - The application will compile `libinav_core.so`, package `velocity_net.onnx`, and deploy `iNAV.apk`.

---

## How to Test GNSS Blackout on a Real Phone

1. **Grant Permissions**:
   - On first launch, grant Location and Notification permissions.
2. **Warmup & Tracking**:
   - Drive or walk outdoors. The top badge will show **`AIDED (GNSS)`** with a green breadcrumb track on the OpenStreetMap.
3. **Simulate Outage**:
   - Tap the red button: **"Simulate GPS Blackout"**.
   - The app will immediately suppress all incoming GPS fixes and switch to **`PURE DR (AI-UKF)`** in glowing purple.
   - Watch the vehicle continue navigating smoothly along the road using solely the phone's internal accelerometer and gyroscope, updated at 100 Hz with micro-neural velocity corrections.
4. **Resume GPS**:
   - Tap **"Resume GPS Updates"** to demonstrate seamless re-acquisition and quarantine ramping.
