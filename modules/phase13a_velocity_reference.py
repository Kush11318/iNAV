"""
Phase 13A: Evolving Velocity Reference Module
Maintains an evolving forward-speed reference during GNSS outages:
    v_ref(0) = v_anchor (last reliable GNSS speed)
    v_ref(t + dt) = clamp(v_ref(t) + delta_v_neural, min=0.0)

Safety Guardrails:
- Clamp negative speeds to 0.0
- Outage duration uncertainty propagation: sigma(tau) = sigma_base * (1.0 + alpha * sqrt(tau))
- ZUPT triggering on stationary detection or v_ref < 0.1 m/s
- Zero production code modification (isolated module).
"""

import math
from typing import Tuple, Optional
import numpy as np
import torch


class Phase13AVelocityReference:
    """
    Evolving Neural Velocity Reference Tracker for GNSS Outages.
    """

    def __init__(
        self,
        alpha_uncertainty_growth: float = 0.05,
        min_sigma: float = 0.2,
        zupt_speed_threshold: float = 0.15
    ):
        self.alpha = alpha_uncertainty_growth
        self.min_sigma = min_sigma
        self.zupt_threshold = zupt_speed_threshold

        self.v_ref: float = 0.0
        self.sigma_ref: float = min_sigma
        self.outage_elapsed_s: float = 0.0
        self.is_active: bool = False
        self.step_count: int = 0

    def initialize(self, v_initial: float, initial_sigma: float = 0.2) -> None:
        """
        Initialize the reference with the last reliable GNSS speed before outage.
        """
        self.v_ref = max(0.0, float(v_initial))
        self.sigma_ref = max(self.min_sigma, float(initial_sigma))
        self.outage_elapsed_s = 0.0
        self.is_active = True
        self.step_count = 0

    def update_with_model(
        self,
        model: torch.nn.Module,
        imu_window_40x6: np.ndarray,
        dt_elapsed: float = 0.5,
        device: str = "cpu"
    ) -> Tuple[float, float, int]:
        """
        Perform a causal step update of the velocity reference using the neural model.

        Parameters:
            model: AnchoredTemporalVelocityNet instance
            imu_window_40x6: IMU measurements over the 4.0-second window (40, 6)
            dt_elapsed: time elapsed since last update (s), typically 0.5s (5 IMU steps @ 10 Hz)
            device: torch device ('cpu')

        Returns:
            v_ref: Updated evolving forward speed (m/s)
            sigma_v: Propagated uncertainty (m/s)
            event_class: Driving condition class (0: stationary, 1: normal, etc.)
        """
        if not self.is_active:
            raise RuntimeError("Phase13AVelocityReference must be initialized before updating.")

        self.outage_elapsed_s += dt_elapsed
        self.step_count += 1

        # Format input tensor: (1, 6, 40)
        bx = torch.from_numpy(imu_window_40x6.transpose(1, 0)).unsqueeze(0).float().to(device)
        b_ref = torch.tensor([[self.v_ref]], dtype=torch.float32).to(device)

        with torch.no_grad():
            delta_v_seq, v_pred_seq, sig_seq, event_logits = model(bx, b_ref)

            end_speed = float(v_pred_seq[0, -1].item())
            raw_sigma = float(sig_seq[0, -1].item())
            event_class = int(torch.argmax(event_logits, dim=1).item())

        # Physical safety: Clamp impossible negative vehicle speeds
        self.v_ref = max(0.0, end_speed)

        # Propagate uncertainty with outage duration: sigma(tau) = max(sigma_model, min_sigma) * (1 + alpha * sqrt(tau))
        sigma_base = max(raw_sigma, self.min_sigma)
        time_factor = 1.0 + self.alpha * math.sqrt(max(0.0, self.outage_elapsed_s))
        self.sigma_ref = float(sigma_base * time_factor)

        # Stationary logic: clamp to 0.0 only when below stationary threshold
        if self.v_ref < self.zupt_threshold:
            self.v_ref = 0.0

        return self.v_ref, self.sigma_ref, event_class

    def get_measurement(self) -> Tuple[float, float]:
        """Returns current (v_ref, sigma_ref)."""
        return self.v_ref, self.sigma_ref

    def reset(self) -> None:
        """Reset tracker state."""
        self.v_ref = 0.0
        self.sigma_ref = self.min_sigma
        self.outage_elapsed_s = 0.0
        self.is_active = False
        self.step_count = 0
