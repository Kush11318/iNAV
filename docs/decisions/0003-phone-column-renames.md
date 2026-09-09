# ADR 0003: Canonical Column Renaming and Schema Reconciliation for Smartphone Dataset

## Status
Accepted

## Context
The raw IO-VNBD dataset (`data/raw/io-vnbd/`) contains 241 smartphone CSV files representing 97 unique driving runs recorded via the AndroSensor Android application across four vehicle platforms in the UK, France, and Nigeria.

In Step 2's data audit (`docs/data-audit.md`), we established that the raw smartphone CSVs exhibit five distinct schema variants:
1. **Standard 24-column**: Contains two naming conventions for identical physical quantities:
   - `GYROSCOPE Yaw/Pitch/Roll (rad/s)` vs `GYROSCOPE X/Y/Z (rad/s)`.
   - `ORIENTATION (Yaw) (°)` vs `ORIENTATION (Azimuth) (°)`.
2. **Typo Header 24-column**: Two runs (`S-Vfa01.csv`, `S-Vfa02.csv`) where `DATE (YYYY-MO-DD HH-MI-SS_SSS` is missing its closing parenthesis.
3. **Truncated 18-column**: Nine runs from France (`S-T1` through `S-T9`, totaling 603,425 samples) completely lack columns 19–24 (the magnetometer and orientation channels). The raw file ends immediately after `GYROSCOPE Roll (rad/s)`.
4. **Corrupted Delimiter 25-column**: One run (`S-A4.csv`, 32,829 rows) contains a trailing comma on the header and an empty token at index 6 (`0,,13 / 24`), causing every subsequent column to be shifted right by one.
5. **Excel Date Mangling**: Files such as `S-A1` and `S-A12` have strings like `"Aug-20"` and `"Aug-21"` in `GPS SATELLITES IN RANGE` resulting from spreadsheet software auto-converting `"8 / 20"` into calendar dates.

Additionally, our empirical speed cross-check across all 97 runs revealed:
- **94 runs** recorded raw `GPS SPEED (Kmh)` in **m/s** (Android native `Location.getSpeed()`).
- **3 runs (`S-A2`, `S-A9`, `S-A10`)** were recorded by AndroSensor in true **km/h** (their integrated speed matches GPS path distance directly with a 1.0 ratio; multiplying by 3.6 would result in impossible speeds >450 km/h).

## Decision
We define a single, unified 27-column canonical schema shared across all 97 processed smartphone Parquet files:

| # | Canonical Field Name | Dtype | Unit | Population & Transformation Rules |
|:---:|:---|:---:|:---:|:---|
| 0 | `timestamp_ms` | `int64` | ms | From `TIME SINCE START (ms)`. Monotonic integer clock counter from sensor logger. |
| 1 | `timestamp_s` | `float64` | s | Derived: `timestamp_ms / 1000.0`. Aligns with CAN `timestamp_s` for dead-reckoning fusion. |
| 2 | `date_utc` | `string` | ISO text | Normalized string representation from `DATE (YYYY-MO-DD HH-MI-SS_SSS...)`. |
| 3 | `latitude_deg` | `float64` | deg | WGS-84 latitude. |
| 4 | `longitude_deg` | `float64` | deg | WGS-84 longitude. |
| 5 | `altitude_m` | `float32` | m | GPS ellipsoidal altitude in meters. |
| 6 | `gps_speed_raw` | `float32` | raw | Exact raw speed value as recorded by AndroSensor. |
| 7 | `gps_speed_ms` | `float32` | m/s | **True speed in m/s**: Preserved as-is for the 94 m/s runs; divided by 3.6 for `S-A2`, `S-A9`, `S-A10`. |
| 8 | `gps_speed_kmh` | `float32` | km/h | **True speed in km/h**: Multiplied by 3.6 for the 94 m/s runs; preserved as-is for `S-A2`, `S-A9`, `S-A10`. |
| 9 | `gps_accuracy_m` | `float32` | m | 1-sigma horizontal accuracy estimate. |
| 10 | `gps_bearing_deg` | `float32` | deg | Course over ground bearing [0.0, 360.0]. |
| 11 | `gps_satellites_in_range` | `Int16` | count | Nullable integer. Parsed via regex `^(\d+)` from `"X / Y"`; mangled Excel dates (`"Aug-20"`) evaluate to null. |
| 12 | `accel_x_m_s2` | `float32` | m/s² | Phone body accelerometer X including gravity. |
| 13 | `accel_y_m_s2` | `float32` | m/s² | Phone body accelerometer Y including gravity. |
| 14 | `accel_z_m_s2` | `float32` | m/s² | Phone body accelerometer Z including gravity. |
| 15 | `gravity_x_m_s2` | `float32` | m/s² | Android software gravity vector X. |
| 16 | `gravity_y_m_s2` | `float32` | m/s² | Android software gravity vector Y. |
| 17 | `gravity_z_m_s2` | `float32` | m/s² | Android software gravity vector Z. |
| 18 | `gyro_x_rad_s` | `float32` | rad/s | Unified from `GYROSCOPE X` and `GYROSCOPE Yaw`. |
| 19 | `gyro_y_rad_s` | `float32` | rad/s | Unified from `GYROSCOPE Y` and `GYROSCOPE Pitch`. |
| 20 | `gyro_z_rad_s` | `float32` | rad/s | Unified from `GYROSCOPE Z` and `GYROSCOPE Roll`. |
| 21 | `mag_x_uT` | `float32` | µT | Ambient magnetic field X. **Populated with genuine nulls for French runs `S-T1`..`S-T9`**. |
| 22 | `mag_y_uT` | `float32` | µT | Ambient magnetic field Y. **Populated with genuine nulls for French runs `S-T1`..`S-T9`**. |
| 23 | `mag_z_uT` | `float32` | µT | Ambient magnetic field Z. **Populated with genuine nulls for French runs `S-T1`..`S-T9`**. |
| 24 | `orientation_azimuth_deg`| `float32` | deg | Unified from `ORIENTATION (Azimuth)` and `ORIENTATION (Yaw)`. **Null for French runs**. |
| 25 | `orientation_pitch_deg` | `float32` | deg | Android orientation pitch angle. **Null for French runs**. |
| 26 | `orientation_roll_deg` | `float32` | deg | Android orientation roll angle. **Null for French runs**. |

### Structural Repair of `S-A4.csv`
Line-by-line validation confirmed that all 32,829 data lines in `S-A4.csv` have an extraneous comma delimiter at column 6 (`0,,13 / 24`). Rather than discarding 32,829 rows of valid sensor data, the parser strips the empty 6th token on input, normalizing the line to standard 24-column format and tagging the run as `delimiter_repaired` in the manifest.

### Genuine Nulls for French Subsets
For the 9 French runs (`S-T1` through `S-T9`), columns 21–26 are stored as PyArrow nulls (`np.nan`). They are **never zero-filled**, ensuring downstream dead-reckoning models and sensor fusion engines correctly identify the absence of magnetometer hardware.

## Consequences
- All 97 smartphone runs are ingested into one identical Parquet schema without requiring schema branching downstream.
- Both true m/s and true km/h velocities are available with 100% verified correctness across all runs, eliminating the 3.6× speed bug.
- Missing magnetometer runs, low-sample-rate runs, burst-sampled runs, and clock-reset runs are clearly labeled in the manifest for downstream filtering in Steps 5 and 6.
