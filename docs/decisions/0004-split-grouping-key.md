# ADR 0004: Dataset Grouping Key and Leakage-Safe Train/Validation/Test Partitioning

## Status
Accepted

## Context
A naive row-level or random run-level split of the IO-VNBD dataset across train, validation, and test partitions introduces severe data leakage risks:
1. **Paired Inversion Leakage**: If a CAN run and its concurrent smartphone recording (which observe the exact same vehicle trajectory) end up on opposite sides of the split, test metrics become artificially optimistic and invalid.
2. **Session / Multi-Leg Leakage**: Several runs are contiguous sub-legs of a single continuous experimental trip (e.g. `S3a`, `S3b`, and `S3c` recorded along the M6 motorway and Rugby on the evening of 04/09/2019; `Vw14a`, `Vw14b`, and `Vw14c` recorded sequentially on the M5/M42 on 08/01/2020).
3. **Hardware & Calibration Leakage**: Distinct vehicle platforms (Ford Fiesta, Volvo XC70, Renault Megane, Toyota Corolla) and smartphone hardware (Huawei P20 Pro, Blackberry Priv, Motorola Moto G7 Power) exhibit unique sensor biases, vibration modes, and noise profiles.

To enable rigorous, generalizable dead-reckoning benchmark evaluation (Step 8) and replay testing (Steps 9–11), we must define a durable, leakage-safe partition grouping key.

---

## Empirical Investigation: What Metadata is Recoverable?

We conducted an exhaustive audit of repository files, folder hierarchies, and dataset documentation (`README.md`, `README_1.pdf`, and the paper appendix):

1. **Driver Identity**:
   - The paper appendix (`README_1.pdf`, Tables A1-1 through A7) explicitly catalogues **8 drivers** (`Driver A` through `Driver H`):
     - **Driver A**: 6 paired runs (`S1`, `S2`, `S3a`, `S3b`, `S3c`, `S4`) — Ford Fiesta Titanium, Huawei P20 Pro, Coventry/Rugby (England). Defensive style.
     - **Driver B**: 1 paired run (`M`) — Ford Fiesta Titanium, Huawei P20 Pro, Coventry (England). Defensive style.
     - **Driver C**: 4 CAN-only runs (`St1`, `St4`, `St6`, `St7`) — Ford Fiesta Titanium, Oxford/Kenilworth (England). Defensive style.
     - **Driver D**: 2 runs (`Y1` paired, `Y2` CAN-only) — Ford Fiesta Titanium, Coventry (England). Defensive style.
     - **Driver E**: 77 runs (`Vfa*`, `Vfb*`, `Vta*`, `Vtb*`, `Vw*`) — Ford Fiesta Titanium, Huawei P20 Pro, UK. Aggressive style.
     - **Driver F**: 11 phone-only runs (`T1` through `T11`) — Renault Megane, Motorola Moto G7 Power, France. No magnetometer.
     - **Driver G**: 1 phone-only run (`I`) — Toyota Corolla Verso, Huawei P20 Pro, Nigeria.
     - **Driver H**: 13 phone-only runs (`A1` through `A13`) — Volvo XC70, Blackberry Priv, England. 2 Hz throttled cohort.

2. **The Driver E Imbalance Challenge**:
   - While drivers are catalogued, Driver E accounts for **64 out of the 72 paired runs** (88.9% of all paired runs) and 27.5% of total dataset duration.
   - If split strictly by driver as an indivisible unit, all 64 paired runs would be locked into one single split (e.g. Train). Validation and Test would be starved of paired data (Val having 0 pairs and Test having at most 1 pair), making cross-checked dead-reckoning displacement scoring in Step 8 impossible.

3. **Hierarchical Campaign / Route-Family Sub-Sessions within Driver E**:
   - Examination of Driver E's runs in Tables A2-1 through A6-2 reveals that Driver E conducted **5 distinct experimental campaigns on different calendar dates across completely different geographic regions**:
     - `Driver_E_Vfa`: Bradford & Measham campaign (08/11/2019; A444, M1, M62; 2 paired runs).
     - `Driver_E_Vfb`: Nuthall & Bradford night campaign (08/11/2019; city centre, M606, M62; 11 CAN-only runs).
     - `Driver_E_Vta`: Peak District & Derbyshire country hills (06/11 & 14/11/2019; Nuneaton, Walton, Ashburne, Thorpe, Ilam; 28 paired runs + 1 CAN-only run).
     - `Driver_E_Vtb`: Derbyshire dales & valleys (06/11/2019; Bakewell, Tideswell, Ashford, Youlgreave, Atherstone; 12 paired runs + 1 CAN-only run).
     - `Driver_E_Vw`: West Midlands motorways (08/01/2020; M5, M40, M42, Worcester, Milton Keynes; 20 paired runs, including stationary bias calibration runs `Vw1` and `Vw15`).

---

## Decision: The Final Grouping Key

We define the primary split group as:
$$\text{Group ID} = \text{Driver} \times \text{Campaign / Multi-Leg Trip}$$

This yields **41 atomic groups** that are strictly indivisible across Train, Validation, and Test:
1. **Paired Invariance**: Both sides of every Step 5 pair (`V-<ID>` and `S-<ID>`) always participate in the same group.
2. **Multi-Leg Continuity**: Multi-leg segments (`S3a/b/c`, `Vw14a/b/c`, `Vw16a/b`, `Vfa01/02`) never cross partitions.
3. **Geographic & Session Independence**: Each group represents a distinct driving session, route family, or regional campaign.
4. **Balanced Durations**: The largest group (`Driver_E_Vw`) accounts for 10.3% of total dataset duration, enabling precise optimization towards the target split ratios.

### Target and Achieved Split Ratios

Target: **~70% Train / ~15% Validation / ~15% Test** by total recording duration (70.61 hours total).

| Split | Groups | Unique Drives | Total Duration | Duration Share | Total Distance | Paired Runs | Paired Duration Share of Split |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 28 | 79 | 48.54 hours | **68.7%** | 2,697.9 km | 50 | 36.7% |
| **VAL** | 8 | 9 | 10.31 hours | **14.6%** | 634.1 km | 2 | 44.2% |
| **TEST** | 5 | 27 | 11.77 hours | **16.7%** | 730.8 km | 20 | 62.1% |
| **TOTAL** | **41** | **115** | **70.61 hours** | **100.0%** | **4,062.8 km** | **72** | **42.0%** |

### Intentional Paired Over-Representation in Test
Paired data carries cross-checked displacement ground truth essential for leaderboard scoring (Step 8). Paired data accounts for **62.1% of Test duration** (7.30 hours across 20 runs in `Driver_E_Vw`) and **44.2% of Val duration** (4.55 hours across `S2` and `Y1`), compared to 42.0% in the general dataset.

### Mandatory Hard-Cohort Coverage in Validation and Test
Every known anomaly cohort from Steps 3–5 is guaranteed representation in Validation and/or Test:
- `phone:low_sample_rate` (2 Hz): Val (`A12`), Test (`A9`), Train (`A1`, `A2`, `A3`, `A10`, `A11`, `A13`).
- `phone:burst_sampling` (1 ms rate): Val (`T5`, `T6`), Test (`T1`), Train (`T4`).
- `phone:no_magnetometer`: Val (`T5`, `T6`, `T8`), Test (`T1`, `T9`), Train (`T2`, `T3`, `T4`, `T7`).
- `stationary_run` & `zero_wheel_speed`: Test (`Vw1`, `Vw15` in `Driver_E_Vw`).
- `phone:clock_reset`: Val (`S2`, `Y1`), Train (`S3b`, `S4`, `M`).
- `can:timestamp_gap`: Val (`Y1`), Train (`St7`).
- `phone:delimiter_repaired`: Val (`A4`).
- `raw_speed_kmh`: Test (`A9`), Train (`A2`, `A10`).
- `low_confidence_offset`: Val (`Y1`), Test (`Driver_E_Vw`), Train (`S4`, `Vta`, `Vtb`).

---

## Residual Leakage Risks & Limitations

1. **Intra-Driver Variation for Driver E**:
   - Driver E appears in Train (`Vfa`, `Vfb`, `Vta`, `Vtb`) and in Test (`Vw`). While they took place on different calendar dates (November 2019 vs January 2020) and in distinct geographic regions (Peak District valleys vs West Midlands motorways), Driver E's individual driving habits (e.g. throttle aggression, steering smoothness) could theoretically leak slight behavioral priors. However, partitioning Driver E by regional campaign was mathematically necessary to prevent locking 89% of all paired test data into a single split.
2. **Single Vehicle Model for CAN**:
   - All CAN recordings were obtained using the same Ford Fiesta Titanium. CAN wheel speed and steering models cannot evaluate cross-vehicle generalization on CAN odometry alone; cross-vehicle evaluation occurs on the smartphone side (Renault Megane, Volvo XC70, Toyota Corolla).

---

## Consequences
- Every run and pair is deterministically mapped to one and only one split group.
- Manifest `data/processed/splits/manifest.csv` provides a single durable source of truth for Steps 7, 8, and Phase 1 training.
