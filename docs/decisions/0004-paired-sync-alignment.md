# ADR 0004: Paired CAN and Smartphone Time Synchronization and Common Grid Alignment

## Status
Accepted

## Context
In Step 3 and Step 4, we produced canonical, deduplicated Parquet datasets for the Vehicle/CAN bus (90 unique runs) and Smartphone (97 unique runs) streams from the IO-VNBD benchmark dataset.

Among these runs, a subset was recorded concurrently for the same physical vehicle trip. The raw dataset provided a `Synchronised V abd S datasets/` hierarchy with 72 candidate pairs. However, as established in Step 2's empirical audit:
1. **Clock Disparities**: CAN timestamps (`timestamp_s`) record wall-clock seconds from midnight UTC (29,606s to 72,986s), whereas smartphone timestamps record elapsed milliseconds from sensor logging initialization (`timestamp_ms`, starting at ~0.0s).
2. **Independent Hardware Clocks**: There was no shared hardware trigger between the CAN logger (Racelogic VBOX / CAN bus logger) and the commercial smartphone (AndroSensor Android application).
3. **Manual Pre-alignment Quirks**: While 63 of the 72 pairs were trimmed by the original authors to identical row counts, 9 pairs differ by up to 232 rows (~23.2 seconds). Furthermore, empirical cross-correlation reveals that "identical row count" does not imply zero relative time lag (e.g. `Vta3` exhibits a 22.0-second lead/lag offset between start of vehicle motion and sensor logging).
4. **Sampling & Quality Variations**: CAN streams sample at a stable 10.0 Hz, while smartphone streams contain clock resets (negative timestamp jumps in `M`, `S2`, `S3b`, `S4`, `Y1`) and two stationary parked runs (`Vw1`, `Vw15`).

To train and evaluate inertial dead-reckoning models (Steps 6–8) and replay test drives through the C++ navigation engine (Steps 9–11), we require a single, aligned 10 Hz dataset per paired drive where CAN ground-truth and smartphone IMU/GNSS observations reside side-by-side on an identical time grid without extrapolation.

---

## Decision

### 1. Pair Discovery
We match runs between `data/processed/vehicle_can/manifest.csv` and `data/processed/smartphone/manifest.csv` by identifying records sourced from the `Synchronised` subset (filtering for the `Synchronised` path segment while excluding `Unsynchronised`). Base run IDs (`S1`..`S4`, `M`, `Y1`, `Vfa*`, `Vta*`, `Vtb*`, `Vw*`) match 1-to-1, yielding **exactly 72 candidate pairs** (`pair_<base_id>`).

### 2. Time Offset Estimation via GPS Speed Cross-Correlation
To determine the time lag $\tau$ between the CAN and Smartphone recordings:
1. We resample both `can_gps_velocity_kmh` and `phone_gps_speed_kmh` onto a uniform 10 Hz grid over their common duration.
2. For stationary runs where vehicle motion is absent (`std < 0.1` km/h in `Vw1` and `Vw15`), speed cross-correlation is degenerate. The offset is set to $0.0\text{s}$, confidence to $0.0$, and the pair is tagged `stationary_run`.
3. For dynamic runs, we compute the normalized cross-correlation over a search window of $\pm 30.0$ seconds ($\pm 300$ samples).
4. The lag $\tau^*$ maximizing Pearson correlation is extracted alongside the peak correlation coefficient $r^* \in [-1, 1]$ as an objective confidence metric. Pairs with $r^* < 0.70$ are flagged `low_confidence_offset` (11 runs).
5. For runs with phone clock resets (`M`, `S2`, `S3b`, `S4`, `Y1`), the resampling timeline is constructed using monotonic sample progression ($10\text{ Hz}$ spacing) to prevent invalid overlapping windows.

### 3. Strict Intersection Alignment (No Extrapolation)
We define the common timeline on the CAN clock:
$$t_{\text{start}} = \max(0.0, -\tau^*)$$
$$t_{\text{end}} = \min(T_{\text{can}}, T_{\text{phone}} - \tau^*)$$

A uniform 10 Hz time grid $t_k \in [t_{\text{start}}, t_{\text{end}}]$ with $\Delta t = 0.1\text{s}$ is constructed. No samples are extrapolated outside the intersection. The output `timestamp_s` column represents elapsed time from start of the aligned intersection ($0.0, 0.1, 0.2, \dots$).

### 4. Per-Channel Interpolation Policy
To avoid corrupting categorical codes, discrete flags, and angular quantities, we enforce strict channel-specific interpolation policies:

| Channel Type | Channels | Method | Rationale |
|:---|:---|:---|:---|
| **Continuous Kinematic & Physical** | `can_latitude_deg`, `can_longitude_deg`, `can_gps_velocity_kmh`, `can_wheel_speed_*`, `can_indicated_*`, `can_engine_speed_rpm`, `can_accelerator_pedal_pct`, `phone_accel_*`, `phone_gyro_*`, `phone_gravity_*`, `phone_gps_speed_*` | **Linear Interpolation** | Standard resampling for continuous physical states at 10 Hz. |
| **Circular Heading / Bearing** | `can_gps_heading_deg`, `phone_gps_bearing_deg`, `phone_orientation_azimuth_deg` | **Circular (Unwrapped Radians)** | Unwrapped before linear interpolation to prevent $359^\circ \to 1^\circ$ averaging into $180^\circ$. Modulo $360^\circ$ applied after. |
| **Categorical & Discrete Flags** | `can_handbrake`, `can_gear_requested`, `can_gear_actual`, `can_clutch_position`, `can_brake_position`, `can_gps_num_satellites`, `phone_gps_satellites_in_range`, `phone_date_utc` | **Step (Forward-Fill)** | Discrete integer states and categorical labels must never be interpolated into non-existent fractional gears or booleans. |
| **Missing Sensor Channels** | `phone_mag_*`, `phone_orientation_*` | **Null Preserving** | If channels are null (e.g. absent magnetometer), genuine `np.nan` values are preserved, never zero-filled. |

---

## Output Schema (57 Columns)

Each file in `data/processed/paired/<pair_id>.parquet` contains:
- `timestamp_s` (`float64`, s): Shared 10 Hz elapsed time grid starting at 0.0.
- `can_*` (29 columns): Prefixed vehicle CAN channels.
- `phone_*` (27 columns): Prefixed smartphone sensor channels.

---

## Consequences
- All 72 paired driving runs are aligned onto a single common time grid with 100% deterministic output.
- Downstream dead-reckoning models (Step 6) and outage simulation harnesses (Step 7) can ingest paired Parquet files directly without per-sensor clock reconciliation.
- Manifest tracks offset, confidence, overlap percentages, and quality flags per pair for informed train/test partitioning.
