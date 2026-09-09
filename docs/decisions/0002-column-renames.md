# ADR 0002: Canonical Column Renaming and Unit Corrections for Vehicle/CAN Dataset

## Status
Accepted

## Context
The raw IO-VNBD dataset (`data/raw/io-vnbd/`) contains 323 vehicle/CAN CSV files representing 90 unique driving runs recorded via a Racelogic VBOX Video HD2 data logger tapped into a Ford Fiesta CAN bus.

In Step 2's empirical data audit (`docs/data-audit.md`), we verified that:
1. All 323 CAN files share an identical 29-column schema.
2. Two column headers contain typos: `Gear Requested (Number fof gear employed 1-5)` and `Gear (Number fof gear employed 1-5)` contain `"fof"` instead of `"of"`.
3. Two column headers mislabel physical units:
   - `Height (km)`: Documented as kilometers, but values range from 20.19 to 534.26, which represents **meters above sea level** in the UK (Coventry topography is ~90–150m, surrounding hills up to 534m). Retaining "km" would cause an elevation error of $1000\times$.
   - `Accelerator Pedal Position (0 or 1)`: Header suggests a binary boolean, but values range from `0.0` to `99.0` (matching the paper's description of percentage activation).
4. `Clutch Position (0 or 1)` is 100.0% zero across all rows in all 323 files. It appears to be an unmonitored channel on this vehicle's CAN network.
5. In Step 2, the smartphone's `GPS SPEED (Kmh)` was discovered to actually be recorded in m/s. We performed a cross-check of CAN speed channels (`Velocity (km/hr)` and `Indicated Vehicle Speed (km/hr)`) against the total GPS path distance (1,864.49 km):
   - Integrated `Velocity (km/hr)` = 1,862.64 km (ratio 0.9990).
   - Integrated `Indicated Vehicle Speed (km/hr)` = 1,867.53 km (ratio 1.0016).
   This confirms CAN velocity columns **genuinely represent km/h**, requiring no rescale factor.

## Decision
We standardize the 29 raw columns into a canonical snake_case naming schema with explicit physical unit suffixes where appropriate, strict casting to standard numeric types, and unit bug corrections:

| # | Raw CSV Header String | Canonical Field Name | Canonical Dtype | Physical Unit | Rationale & Transformation |
|:---:|:---|:---|:---:|:---:|:---|
| 0 | `No of GPS Satellites Available` | `gps_num_satellites` | `int32` | count | VBOX status byte or satellite count scaled by 10. |
| 1 | `Time Since Start of Day (seconds)` | `timestamp_s` | `float64` | s | Elapsed time since UTC midnight; 64-bit float preserves sub-millisecond precision. |
| 2 | `Latitude (degrees)` | `latitude_deg` | `float64` | deg | WGS-84 latitude. |
| 3 | `Longitude (degrees)` | `longitude_deg` | `float64` | deg | WGS-84 longitude. |
| 4 | `Velocity (km/hr)` | `gps_velocity_kmh` | `float32` | km/h | Racelogic Doppler velocity. Verified true km/h. |
| 5 | `Heading (degrees)` | `gps_heading_deg` | `float32` | deg | GPS course over ground [0.0, 360.0]. |
| 6 | `Height (km)` | `height_m` | `float32` | m | **Unit bug correction**: Stored in meters. Raw header falsely declared `(km)`. |
| 7 | `Vertical velocity (km/hr)` | `gps_vertical_velocity_kmh` | `float32` | km/h | GPS vertical velocity. |
| 8 | `Sample period (seconds)` | `sample_period_s` | `float32` | s | Inter-sample duration (~0.1s). |
| 9 | `Steering Angle (degrees)` | `steering_angle_deg` | `float32` | deg | Steering wheel angle. Unlogged (0.0) in 13 unique runs. |
| 10 | `Wheel Speed Front Left (rad/sec)` | `wheel_speed_fl_rad_s` | `float32` | rad/s | Wheel encoder angular velocity FL. |
| 11 | `Wheel Speed Front Right (rad/sec)` | `wheel_speed_fr_rad_s` | `float32` | rad/s | Wheel encoder angular velocity FR. |
| 12 | `Wheel Speed Rear Left (rad/sec)` | `wheel_speed_rl_rad_s` | `float32` | rad/s | Wheel encoder angular velocity RL. |
| 13 | `Wheel Speed Rear Right (rad/sec)` | `wheel_speed_rr_rad_s` | `float32` | rad/s | Wheel encoder angular velocity RR. |
| 14 | `Yaw Rate (deg/sec)` | `yaw_rate_deg_s` | `float32` | deg/s | Chassis yaw rate from vehicle ESC unit. |
| 15 | `Indicated Vehicle Speed (km/hr)` | `indicated_vehicle_speed_kmh` | `float32` | km/h | Speedometer odometry from ECU. Verified true km/h. |
| 16 | `Indicated Longitudinal Acceleration (g)` | `indicated_longitudinal_accel_g` | `float32` | g | Longitudinal chassis acceleration (1g ≈ 9.80665 m/s²). |
| 17 | `Indicated Lateral Acceleration (g)` | `indicated_lateral_accel_g` | `float32` | g | Lateral chassis acceleration. |
| 18 | `Handbrake (0 or 1)` | `handbrake` | `int8` | flag (0/1) | Handbrake status flag. |
| 19 | `Gear Requested (Number fof gear employed 1-5)` | `gear_requested` | `int8` | gear code | **Typo correction**: Removed `"fof"`. |
| 20 | `Gear (Number fof gear employed 1-5)` | `gear_actual` | `int8` | gear code | **Typo correction & disambiguation**: Removed `"fof"`; named `gear_actual` to distinguish from `gear_requested`. |
| 21 | `Engine Speed (rev/min)` | `engine_speed_rpm` | `float32` | rpm | Engine crankshaft revolutions per minute. |
| 22 | `Coolant Temperature (degrees)` | `coolant_temp_c` | `float32` | °C | Engine coolant temperature in Celsius. |
| 23 | `Clutch Position (0 or 1)` | `clutch_position` | `int8` | flag (0/1) | Preserved for schema completeness, but documented as 100% dead channel. |
| 24 | `Brake Pressure (psi)` | `brake_pressure_psi` | `float32` | psi | Hydraulic master cylinder brake line pressure. |
| 25 | `Brake Position (0 or 1)` | `brake_position` | `int8` | flag (0/1) | Brake pedal switch flag (pressed=1, released=0). |
| 26 | `Battery Voltage (volts)` | `battery_voltage_v` | `float32` | V | Vehicle 12V electrical bus voltage. |
| 27 | `Air Temperature (degrees)` | `air_temp_c` | `float32` | °C | Ambient intake air temperature. |
| 28 | `Accelerator Pedal Position (0 or 1)` | `accelerator_pedal_pct` | `float32` | % | **Unit bug correction**: Documented as percentage (0.0–100.0%). Raw header falsely indicated `(0 or 1)`. |

## Consequences
- No parser or downstream model will inherit the silent elevation bug (interpreting meters as km) or the accelerator pedal boolean misconception.
- The schema is strongly typed, memory efficient, and serialized to deterministic Parquet files with snappy compression.
- Downstream replay harnesses and training modules can consume standardized field names without ad-hoc string manipulations.
