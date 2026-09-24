"""
Phase 13B: Event-Gated Neural Velocity Correction Module
Maintains a STRICTLY FIXED v_anchor while allowing a temporary neural velocity
correction only during detected dynamic events (hard braking, forward acceleration, etc.):
    v_anchor = constant (NEVER updated recursively)
    v_measurement(t) = clamp(v_anchor + gated_correction(t), min=0.0)

Event Gating Rules:
- CRUISE (Class 1): Gate closed. Gated correction decays to 0.0 -> measurement is fixed anchor.
- HARD BRAKING (Class 2 / a_fwd < -1.5): Gate open. Allows negative neural correction.
- ACCELERATION (a_fwd > 0.8 & delta_v > 0.8): Gate open. Allows positive neural correction.
- TURNING (Class 3): Conservative. Gate closed, uncertainty inflated (R_scale = 4.0).
- ROUGH ROAD (Class 4): Conservative. Correction attenuated 50%, uncertainty inflated (R_scale = 4.0).
- STATIONARY (Class 0 / speed < 0.15): Triggers ZUPT, forces speed to 0.0.
"""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch


class Phase13BEventGatedVelocityTracker:
    """
    Event-Gated Neural Velocity Correction Tracker for Phase 13B.
    """

    def __init__(
        self,
        tau_decay_s: float = 2.5,
        tau_attack_s: float = 1.0,
        min_sigma: float = 0.2,
        zupt_threshold_ms: float = 0.15
    ):
        self.tau_decay = tau_decay_s
        self.tau_attack = tau_attack_s
        self.min_sigma = min_sigma
        self.zupt_threshold = zupt_threshold_ms

        self.v_anchor: float = 0.0
        self.current_correction: float = 0.0
        self.outage_elapsed_s: float = 0.0
        self.is_active: bool = False
        self.step_count: int = 0

    def initialize(self, v_anchor: float, initial_sigma: float = 0.2) -> None:
        """
        Initialize the tracker with the pre-outage GNSS speed.
        v_anchor remains constant for the entire outage.
        """
        self.v_anchor = max(0.0, float(v_anchor))
        self.current_correction = 0.0
        self.outage_elapsed_s = 0.0
        self.is_active = True
        self.step_count = 0

    def update_step(
        self,
        model: torch.nn.Module,
        imu_window_40x6: np.ndarray,
        a_fwd_mean: float,
        yaw_rate_mean: float,
        dt_elapsed: float = 0.5,
        device: str = "cpu"
    ) -> Tuple[float, float, int, bool, str, float]:
        """
        Processes one 4-second IMU window using the frozen anchor and evaluates the event gate.

        Parameters:
            model: AnchoredTemporalVelocityNet
            imu_window_40x6: raw IMU measurements (40, 6)
            a_fwd_mean: vehicle-frame forward acceleration (mean over recent window)
            yaw_rate_mean: vehicle-frame yaw rate (mean over recent window)
            dt_elapsed: elapsed time since last update (s)
            device: torch device ('cpu')

        Returns:
            v_meas: temporary corrected velocity measurement (m/s)
            sigma_meas: measurement standard deviation (m/s)
            event_class: classified driving condition
            is_gate_open: boolean indicating whether neural correction was permitted
            gate_type: human-readable event label
            correction_val: current value of gated_correction (m/s)
        """
        if not self.is_active:
            raise RuntimeError("Tracker must be initialized before update_step.")

        self.outage_elapsed_s += dt_elapsed
        self.step_count += 1

        # Format input tensor: (1, 6, 40)
        bx = torch.from_numpy(imu_window_40x6.transpose(1, 0)).unsqueeze(0).float().to(device)
        # Anchor is strictly the FIXED pre-outage speed
        b_anc = torch.tensor([[self.v_anchor]], dtype=torch.float32).to(device)

        with torch.no_grad():
            delta_v_seq, v_pred_seq, sig_seq, event_logits = model(bx, b_anc)
            
            p_v = v_pred_seq.cpu().numpy().squeeze()      # (20,)
            p_del = delta_v_seq.cpu().numpy().squeeze()   # (20,)
            p_sig = sig_seq.cpu().numpy().squeeze()       # (20,)
            event_class = int(torch.argmax(event_logits, dim=1).item())

            raw_delta_v = float(np.mean(p_del))
            raw_sigma = float(np.mean(p_sig))
            v_reconstructed = float(np.mean(p_v))

        # -------------------------------------------------------------
        # Event Gate Evaluation
        # -------------------------------------------------------------
        is_gate_open = False
        gate_type = "CRUISE"
        target_correction = 0.0
        r_scale = 1.0

        # 1. Stationary (Class 0 or near-zero speed)
        if event_class == 0 or (self.v_anchor + raw_delta_v < self.zupt_threshold) or (abs(a_fwd_mean) < 0.1 and raw_delta_v < -self.v_anchor * 0.8):
            is_gate_open = True
            gate_type = "STATIONARY"
            target_correction = -self.v_anchor
            r_scale = 1.0

        # 2. Hard Braking (Class 2 or forward braking acceleration)
        elif event_class == 2 or a_fwd_mean < -1.2 or raw_delta_v < -0.8:
            is_gate_open = True
            gate_type = "BRAKING"
            target_correction = min(raw_delta_v, a_fwd_mean * 1.5)
            r_scale = 1.0

        # 3. Acceleration (Physical forward acceleration and positive model response)
        elif event_class not in (3, 4) and (a_fwd_mean > 0.8 or raw_delta_v > 0.6):
            is_gate_open = True
            gate_type = "ACCELERATION"
            target_correction = max(raw_delta_v, a_fwd_mean * 1.2)
            r_scale = 1.5

        # 4. Turning (Class 3 or high yaw rate) -> Conservative: Gate closed, inflate R
        elif event_class == 3 or abs(yaw_rate_mean) > 0.15:
            is_gate_open = False
            gate_type = "TURNING"
            target_correction = 0.0
            r_scale = 4.0

        # 5. Rough Road (Class 4) -> Conservative: Attenuate correction 50%, inflate R
        elif event_class == 4:
            is_gate_open = True
            gate_type = "ROUGH_ROAD"
            target_correction = 0.5 * raw_delta_v
            r_scale = 4.0

        # 6. Default Cruise: Gate closed, target correction = 0.0 (anchor dominates)
        else:
            is_gate_open = False
            gate_type = "CRUISE"
            target_correction = 0.0
            r_scale = 1.0

        # -------------------------------------------------------------
        # Causal Temporal Smoothing & Decay
        # -------------------------------------------------------------
        if is_gate_open:
            alpha_attack = 1.0 - math.exp(-dt_elapsed / max(0.1, self.tau_attack))
            self.current_correction += alpha_attack * (target_correction - self.current_correction)
        else:
            alpha_decay = math.exp(-dt_elapsed / max(0.1, self.tau_decay))
            self.current_correction *= alpha_decay

        # Calculate final temporary velocity measurement
        v_meas = max(0.0, self.v_anchor + self.current_correction)

        # Scale measurement uncertainty
        sigma_meas = max(self.min_sigma, float(raw_sigma * math.sqrt(r_scale)))

        return v_meas, sigma_meas, event_class, is_gate_open, gate_type, self.current_correction

    def reset(self) -> None:
        """Reset tracker state."""
        self.v_anchor = 0.0
        self.current_correction = 0.0
        self.outage_elapsed_s = 0.0
        self.is_active = False
        self.step_count = 0
