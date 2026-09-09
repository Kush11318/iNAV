# Architecture Reference: Phone-Based Inertial Dead-Reckoning (IDR)

This document serves as the architecture reference and single source of truth for module contracts across the system.

## System Overview

We are building a phone-based inertial dead-reckoning navigation stack for an ISRO hackathon. The architecture is fixed across four primary layers:

1. **Dependency-Free C++17 Core Engine (`core/`)**:
   - Single core engine shared across three frontends:
     - Offline evaluation harness (CSV replay)
     - Android mobile application (via JNI)
     - Linux edge CLI/daemon for ~200 Hz FOG-grade IMUs
   - Consumes abstract `ImuSample { t, ax, ay, az, gx, gy, gz, mx, my, mz }` and optional `GnssFix { t, lat, lon, alt, speed, accuracy, sat_count, valid }`.
   - Zero runtime external dependencies for portability and embedded deployment.

2. **Python Data/Evaluation Pipeline (`dataeval/`)**:
   - Ingestion: Parsing the IO-VNBD dataset, raw sensor cleaning, and standardization.
   - Harness: Synthetic GNSS outage generation (e.g. 10s, 30s, 60s dropouts) and trajectory replay scoring against ground truth.
   - Training: Machine learning pipelines for zero-velocity detection (ZUPT), dynamic step length, or orientation correction.

3. **Android Application (`android/`)**:
   - Thin Kotlin shell interfacing with the compiled C++ core library via JNI.
   - Captures hardware sensor events from Android sensor manager and feeds `ImuSample` / `GnssFix` in real time.

4. **Linux Edge CLI / Daemon (`edge/`)**:
   - Thin Linux binary shell wrapping the same C++ core for high-frequency embedded navigation units.

---

*(The full, unabridged architecture reference document and module contract definitions will be pasted here as the single source of truth).*
