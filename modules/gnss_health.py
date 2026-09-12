"""
iNAV GNSS Health Manager & Outage/Reacquisition State Machine (Phase 5)
Handles:
  1. 4-State Health Machine (AIDED, DEGRADED, PURE_DR, QUARANTINE)
  2. Outage Detection with Monotonic Clock Timeout (1.5s)
  3. Candidate Quarantine (N=3 consecutive valid fixes)
  4. Smooth Reacquisition Trust Ramp (5.0s covariance scaling)
  5. Innovation / NIS Consistency Verification
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, Optional
import math

from modules.sensor_types import GnssSample, GnssValidity


class GnssHealthState(IntEnum):
    PURE_DR = 0
    QUARANTINE = 1
    AIDED = 2
    DEGRADED = 3


@dataclass
class GnssHealthConfig:
    outage_timeout_ns: int = 1_500_000_000       # 1.5 seconds missing timeout
    quarantine_required_epochs: int = 3          # 3 consecutive consistent valid fixes
    trust_ramp_duration_ns: int = 5_000_000_000  # 5.0 seconds covariance ramp
    initial_ramp_scale: float = 9.0              # R_eff starts at (1.0 + 9.0) = 10x R_base
    degraded_accuracy_m: float = 15.0            # Acc >= 15m enters DEGRADED
    unusable_accuracy_m: float = 50.0            # Acc >= 50m rejected as invalid
    speed_heading_min_mps: float = 2.5           # Heading course updates active only when v > 2.5 m/s
    nis_pos_2d_threshold: float = 9.21           # 2-DOF Chi-Square 99% threshold
    nis_scalar_1d_threshold: float = 6.635       # 1-DOF Chi-Square 99% threshold


class GnssHealthManager:
    def __init__(self, config: Optional[GnssHealthConfig] = None):
        self.config = config or GnssHealthConfig()
        self.state = GnssHealthState.PURE_DR
        self.last_valid_gnss_time_ns: int = 0
        self.quarantine_count: int = 0
        self.ramp_start_time_ns: int = 0
        self.is_ramp_active: bool = False
        self.total_accepted_fixes: int = 0
        self.total_rejected_fixes: int = 0

    def reset(self) -> None:
        self.state = GnssHealthState.PURE_DR
        self.last_valid_gnss_time_ns = 0
        self.quarantine_count = 0
        self.ramp_start_time_ns = 0
        self.is_ramp_active = False
        self.total_accepted_fixes = 0
        self.total_rejected_fixes = 0

    def validate_sample(self, sample: GnssSample) -> bool:
        if not sample.has_basic_fix:
            return False
        if sample.horizontal_accuracy_m <= 0.0 or sample.horizontal_accuracy_m >= self.config.unusable_accuracy_m:
            return False
        return True

    def start_quarantine(self) -> None:
        self.state = GnssHealthState.QUARANTINE
        self.quarantine_count = 0

    def get_effective_r_scale(self, current_time_ns: int) -> float:
        ramp_factor = 1.0
        if self.is_ramp_active:
            elapsed_ns = current_time_ns - self.ramp_start_time_ns
            if elapsed_ns >= self.config.trust_ramp_duration_ns or elapsed_ns < 0:
                ramp_factor = 1.0
            else:
                progress = float(elapsed_ns) / float(self.config.trust_ramp_duration_ns)
                ramp_factor = 1.0 + self.config.initial_ramp_scale * (1.0 - progress)
        quality_inflation = 4.0 if self.state == GnssHealthState.DEGRADED else 1.0
        return ramp_factor * quality_inflation

    def process_candidate(self, sample: GnssSample) -> Tuple[bool, float]:
        """
        Evaluates candidate GNSS sample, handles state transitions,
        and returns (should_update, effective_r_scale).
        """
        # 1. Basic fix and accuracy validity check
        if not sample.has_basic_fix:
            self.total_rejected_fixes += 1
            return False, 1.0

        if sample.horizontal_accuracy_m <= 0.0 or sample.horizontal_accuracy_m >= self.config.unusable_accuracy_m:
            self.total_rejected_fixes += 1
            return False, 1.0

        # 2. Transition from PURE_DR to QUARANTINE
        if self.state == GnssHealthState.PURE_DR:
            self.state = GnssHealthState.QUARANTINE
            self.quarantine_count = 0

        # 3. Quarantine handling
        if self.state == GnssHealthState.QUARANTINE:
            self.quarantine_count += 1
            if self.quarantine_count < self.config.quarantine_required_epochs:
                # Hold candidate in quarantine; do NOT update filter yet
                self.last_valid_gnss_time_ns = sample.timestamp_ns
                return False, 1.0

            # Quarantine completed! Transition to AIDED / DEGRADED and start trust ramp
            if sample.horizontal_accuracy_m >= self.config.degraded_accuracy_m:
                self.state = GnssHealthState.DEGRADED
            else:
                self.state = GnssHealthState.AIDED
            self.ramp_start_time_ns = sample.timestamp_ns
            self.is_ramp_active = True
            self.quarantine_count = 0

        # 4. Normal AIDED / DEGRADED state updates
        if self.state in (GnssHealthState.AIDED, GnssHealthState.DEGRADED):
            if sample.horizontal_accuracy_m >= self.config.degraded_accuracy_m:
                self.state = GnssHealthState.DEGRADED
            else:
                self.state = GnssHealthState.AIDED

        # 5. Compute effective covariance scaling (Trust ramp + Quality inflation)
        ramp_factor = 1.0
        if self.is_ramp_active:
            elapsed_ns = sample.timestamp_ns - self.ramp_start_time_ns
            if elapsed_ns >= self.config.trust_ramp_duration_ns or elapsed_ns < 0:
                self.is_ramp_active = False
                ramp_factor = 1.0
            else:
                progress = float(elapsed_ns) / float(self.config.trust_ramp_duration_ns)
                ramp_factor = 1.0 + self.config.initial_ramp_scale * (1.0 - progress)

        quality_inflation = 4.0 if self.state == GnssHealthState.DEGRADED else 1.0
        effective_scale = ramp_factor * quality_inflation

        self.last_valid_gnss_time_ns = sample.timestamp_ns
        self.total_accepted_fixes += 1
        return True, effective_scale

    def check_timeout(self, current_time_ns: int) -> None:
        """Checks for outage timeout during IMU prediction."""
        if self.last_valid_gnss_time_ns == 0:
            return
        elapsed_ns = current_time_ns - self.last_valid_gnss_time_ns
        if elapsed_ns > self.config.outage_timeout_ns:
            if self.state != GnssHealthState.PURE_DR:
                self.state = GnssHealthState.PURE_DR
                self.quarantine_count = 0
                self.is_ramp_active = False

    def on_measurement_rejected(self) -> None:
        """Called when a measurement fails NIS gating."""
        self.total_rejected_fixes += 1
        if self.state == GnssHealthState.QUARANTINE:
            self.quarantine_count = 0
