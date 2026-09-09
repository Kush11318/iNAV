# iNAV Jury Defense & Q&A Cheatsheet (SIH 2026)

This document contains the exact answers to the **6 killer questions** ISRO evaluators and technical juries ask during defense rounds, referenced directly from the iNAV Technical Design Architecture.

---

### Q1: "Why did you choose an Unscented / Error-State filter (UKF / ES-EKF) over a standard Extended Kalman Filter (EKF)?"
**The Jury's Angle:** They want to see if you actually understand nonlinear estimation or just picked a library at random.
* **The Winning Answer:**
  > *"Heading enters dead-reckoning equations through trigonometric sines and cosines, which are severely nonlinear. An EKF relies on first-order Taylor series linearisation with Jacobians. During extended tunnel outages, heading uncertainty grows, and first-order linearisation degrades catastrophically. 
  > Instead, our filter propagates deterministic sigma points (in UKF) or works in the tangent Lie algebra space (in ES-EKF) through the exact nonlinear equations without computing analytical Jacobians. It provides superior covariance consistency during extended blackouts while adding negligible computational overhead (< 0.5 microseconds per step)."*

---

### Q2: "Is this really an AI project, or just a classical Kalman filter with a neural network bolted on?"
**The Jury's Angle:** They want to verify where the AI lives and whether each component has an empirical ablation.
* **The Winning Answer:**
  > *"AI is not merely 'bolted on'; it replaces the physically impossible parts of the system where classical physics fails. We have three distinct learned components, each answering an exact ablation question:
  > 1. **SPECTRA / VelocityNet (Module B, Head 1):** Maps 6-axis IMU vibration spectra directly to 2-second forward displacement $\Delta d$. Classical strapdown double-integration explodes by $50\times$; our neural displacement remains bounded by $\sqrt{t}$.
  > 2. **Event Classifier (Module B, Head 2):** Classifies road regimes (cruise, pothole, dynamic turn, stationary). When a pothole occurs, naive integration would absorb a 4g shock as forward speed; our event head inflates measurement covariance $20\times$ to gate it out. At red lights, it asserts ZUPT to zero velocity error.
  > 3. **Learned Uncertainty Head $\sigma(\Delta d)$ (Module B, Head 3):** The network predicts its own confidence. On smooth motorways it outputs $\pm 2\%$; on rough roads it outputs $\pm 15\%$, dynamically tuning the filter's Kalman gain $R$ matrix."*

---

### Q3: "How does the system transition between GNSS and Dead-Reckoning in zero milliseconds without the icon jumping?"
**The Jury's Angle:** Evaluators hate hard `if-else` mode switches that jerk or teleport the cursor.
* **The Winning Answer:**
  > *"Because our architecture is built around an error-state filter, **the mode transition is mathematically free.** There is no 'switch to DR mode' code branch. 
  > Every 100 ms epoch, the filter asks: 'Do we have a trustworthy GNSS fix this epoch?'
  > - If yes, it applies the GNSS innovation update and refines sensor biases.
  > - If no (tunnel entry), it simply skips the GNSS update step and propagates on learned displacement, NHC physics, and gyro heading alone.
  > On tunnel exit, our **Re-acquisition Quarantine** holds the first noisy fixes for 2 seconds to reject tunnel-mouth multipath, then ramps covariance down smoothly so the vehicle cursor settles gracefully without snapping."*

---

### Q4: "How does your model generalize to cars it has never seen in training (e.g., Bolero vs Swift)?"
**The Jury's Angle:** Probing whether you overfitted to the training car or phone mount.
* **The Winning Answer:**
  > *"We solve cross-vehicle generalization at two distinct layers:
  > 1. **Offline 3D Rotation Augmentation in $SO(3)$:** We train the network by rotating accelerometer and gyroscope windows through random 3D rotation matrices, making the feature extractor mount-agnostic.
  > 2. **Online Adaptive Scale Factor $k$:** Inside the filter, we maintain a state variable $k$ (initialized to 1.0). During the first 30 seconds of normal GPS driving, the filter observes the ratio between learned displacement and true GPS distance and automatically learns $k$ (e.g., $k=1.04$ for larger tyres). When the tunnel hits, the model is already pre-calibrated to that specific vehicle."*

---

### Q5: "What happens if the phone is knocked out of the cradle mid-drive?"
**The Jury's Angle:** Real-world robustness test.
* **The Winning Answer:**
  > *"Module A continuously monitors the gravity unit vector. A sudden gravity angle jump (> 5°) indicates the phone was disturbed or picked up. The system immediately drops alignment confidence to zero, switches the UI to 'RE-ALIGNING', and automatically executes a dynamic PCA alignment on the very next acceleration event to re-identify the vehicle forward axis, rather than silently producing drifting garbage."*

---

### Q6: "Does this only run on Android phones, or can it run on autonomous vehicles / edge computers?"
**The Jury's Angle:** Probing Deliverable #2 from the SIH problem statement.
* **The Winning Answer:**
  > *"We built both deliverables required by the problem statement:
  > 1. **Mobile App:** An Android application with real-time road snapping, live telemetry, and side-by-side Ghost Trace visualization.
  > 2. **Standalone Edge Engine (`inav_edge_engine`):** A zero-dependency C++17 binary compiled for ARM64 and x86_64 Linux. On an ARM processor, it processes **200 Hz FOG sensor streams at 0.414 microseconds per epoch**—over 8,800× faster than real-time, ready for immediate deployment on autonomous rovers, drones, or vehicle telemetry ECUs."*
