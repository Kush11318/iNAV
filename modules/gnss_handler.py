"""
iNAV Seamless GNSS Deficit Handler (Module E)
Manages navigation operating modes (AIDED, DEGRADED, PURE_DR),
monitors satellite geometry and health, and implements smooth
re-acquisition quarantine to prevent vehicle teleportation on UI.
"""

import enum
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("iNAV.gnss")


class NavigationMode(enum.Enum):
    AIDED = "AIDED"               # Healthy GNSS + INS fusion
    DEGRADED = "DEGRADED"         # Poor geometry / weak fix, inflated covariance
    PURE_DR = "PURE_DR"           # Complete blackout, pure AI dead reckoning


class GnssHandler:
    """
    GNSS Signal Quality Monitor & Seamless Transition Manager.
    Enforces outlier rejection, mode switching, and smooth quarantine ramping.
    """

    def __init__(
        self,
        min_satellites: int = 5,
        max_acceptable_accuracy_m: float = 18.0,
        quarantine_count: int = 3,
        ramp_duration_sec: float = 2.0
    ):
        self.min_satellites = min_satellites
        self.max_accuracy_m = max_acceptable_accuracy_m
        self.quarantine_count = quarantine_count
        self.ramp_duration_sec = ramp_duration_sec

        self.current_mode: NavigationMode = NavigationMode.AIDED
        self.previous_mode: NavigationMode = NavigationMode.AIDED

        # Re-acquisition quarantine state
        self.quarantine_buffer: list = []
        self.quarantine_active: bool = False
        self.ramp_start_time: Optional[float] = None

    def assess_health(
        self,
        lat: float,
        lon: float,
        accuracy_m: float,
        sat_count: float,
        timestamp_s: float
    ) -> NavigationMode:
        """
        Evaluate raw GNSS observables and determine operating mode.
        """
        # Complete signal absence
        if np.isnan(lat) or np.isnan(lon):
            return NavigationMode.PURE_DR

        # Check satellites and accuracy
        has_sats = not np.isnan(sat_count)
        has_acc = not np.isnan(accuracy_m)

        if has_sats and sat_count < 4:
            return NavigationMode.PURE_DR

        if has_acc and accuracy_m > 35.0:
            return NavigationMode.PURE_DR

        if (has_sats and sat_count < self.min_satellites) or (has_acc and accuracy_m > self.max_accuracy_m):
            return NavigationMode.DEGRADED

        return NavigationMode.AIDED

    def process_fix(
        self,
        lat: float,
        lon: float,
        accuracy_m: float,
        sat_count: float,
        timestamp_s: float
    ) -> Tuple[NavigationMode, float]:
        """
        Process incoming epoch and return:
          - (current_mode, covariance_inflation_factor)
        covariance_inflation_factor scales the GNSS measurement covariance R.
        """
        raw_mode = self.assess_health(lat, lon, accuracy_m, sat_count, timestamp_s)

        # Transition into blackout
        if raw_mode == NavigationMode.PURE_DR:
            if self.current_mode != NavigationMode.PURE_DR:
                logger.info(f"GNSS Blackout initiated at t={timestamp_s:.2f}s -> Switching to PURE_DR")
                self.previous_mode = self.current_mode
                self.current_mode = NavigationMode.PURE_DR
            self.quarantine_buffer.clear()
            self.quarantine_active = False
            return NavigationMode.PURE_DR, 1.0e6  # effectively infinite variance

        # Transition out of blackout (Signal re-acquired)
        if self.current_mode == NavigationMode.PURE_DR and raw_mode in [NavigationMode.AIDED, NavigationMode.DEGRADED]:
            logger.info(f"GNSS signal detected after blackout at t={timestamp_s:.2f}s -> Entering Quarantine")
            self.quarantine_active = True
            self.quarantine_buffer.clear()
            self.current_mode = NavigationMode.DEGRADED

        # During quarantine: buffer fixes and check kinematic plausibility
        if self.quarantine_active:
            self.quarantine_buffer.append((lat, lon, timestamp_s, accuracy_m))

            if len(self.quarantine_buffer) < self.quarantine_count:
                # Still buffering, continue relying on DR
                return NavigationMode.PURE_DR, 1.0e6

            # Check mutual consistency of buffered fixes
            p1 = self.quarantine_buffer[0]
            p2 = self.quarantine_buffer[-1]
            dt = max(p2[2] - p1[2], 1e-3)
            # Rough distance in meters
            dlat = (p2[0] - p1[0]) * 111000.0
            dlon = (p2[1] - p1[1]) * 111000.0 * np.cos(np.radians(p1[0]))
            dist_m = np.hypot(dlat, dlon)
            implied_speed = dist_m / dt

            if implied_speed > 55.0:  # > 200 km/h is implausible multipath
                logger.warning(f"Quarantined fixes failed velocity check ({implied_speed:.1f} m/s) -> rejecting fix")
                self.quarantine_buffer.pop(0)
                return NavigationMode.PURE_DR, 1.0e6

            # Fixes passed quarantine! Start smooth covariance ramp
            self.quarantine_active = False
            self.ramp_start_time = timestamp_s
            self.current_mode = raw_mode
            logger.info(f"GNSS fixes cleared quarantine at t={timestamp_s:.2f}s -> Beginning smooth covariance ramp")

        # Smooth covariance ramping after quarantine exit
        if self.ramp_start_time is not None:
            elapsed = timestamp_s - self.ramp_start_time
            if elapsed < self.ramp_duration_sec:
                # Linearly interpolate inflation from 50x down to 1.0x
                frac = elapsed / self.ramp_duration_sec
                inflation = 50.0 * (1.0 - frac) + 1.0 * frac
                return self.current_mode, float(inflation)
            else:
                self.ramp_start_time = None

        if raw_mode == NavigationMode.DEGRADED:
            self.current_mode = NavigationMode.DEGRADED
            return NavigationMode.DEGRADED, 10.0  # inflate covariance by 10x

        self.current_mode = NavigationMode.AIDED
        return NavigationMode.AIDED, 1.0
