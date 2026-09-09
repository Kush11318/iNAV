import math

class PedestrianStepEngineSim:
    def __init__(self):
        self.step_count = 0
        self.total_dist_m = 0.0
        self.last_step_ts_ms = 0
        self.is_waiting_for_trough = False
        self.current_peak_acc = 9.81
        self.current_trough_acc = 9.81
        
        # 1-D Kalman Filter
        self.x_est = 9.81
        self.p_cov = 1.0
        self.q = 0.02
        self.r = 0.15
        
        self.threshold_high = 11.2
        self.threshold_low = 8.6
        self.min_step_interval_ms = 260
        self.k_weinberg = 0.44

    def process_sample(self, ax, ay, az, ts_ms):
        raw_mag = math.sqrt(ax*ax + ay*ay + az*az)
        
        # Kalman
        self.p_cov += self.q
        k = self.p_cov / (self.p_cov + self.r)
        self.x_est += k * (raw_mag - self.x_est)
        self.p_cov *= (1.0 - k)
        filtered = self.x_est
        
        self.current_peak_acc = max(self.current_peak_acc, filtered)
        self.current_trough_acc = min(self.current_trough_acc, filtered)
        
        step_detected = False
        stride_length = 0.75
        
        if not self.is_waiting_for_trough:
            if filtered > self.threshold_high:
                self.is_waiting_for_trough = True
                self.current_peak_acc = filtered
        else:
            if filtered < self.threshold_low:
                dt = ts_ms - self.last_step_ts_ms
                if dt >= self.min_step_interval_ms:
                    step_detected = True
                    self.last_step_ts_ms = ts_ms
                    self.step_count += 1
                    bounce = max(0.5, self.current_peak_acc - self.current_trough_acc)
                    stride_length = max(0.50, min(1.05, self.k_weinberg * (bounce ** 0.25)))
                    self.total_dist_m += stride_length
                self.is_waiting_for_trough = False
                self.current_peak_acc = 9.81
                self.current_trough_acc = 9.81
                
        return step_detected, stride_length, self.step_count, self.total_dist_m

def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def initial_bearing_deg(from_lat, from_lon, to_lat, to_lon):
    phi1 = math.radians(from_lat)
    phi2 = math.radians(to_lat)
    dlambda = math.radians(to_lon - from_lon)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    deg = math.degrees(math.atan2(y, x))
    return (deg + 360.0) % 360.0

def bearing_to_cardinal(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int(((deg + 11.25) % 360.0) / 22.5) % 16
    return dirs[idx]

def run_tests():
    print("=" * 65)
    print("iNAV PEDESTRIAN DEAD RECKONING & FOREST BACKTRACK TEST")
    print("=" * 65)
    
    engine = PedestrianStepEngineSim()
    
    # Test 1: Desk Shake Immunity Test (vibrations around 9.81 without full foot bounce)
    desk_steps = 0
    ts = 1000
    for i in range(200): # 2 seconds at 100 Hz
        # small high freq jitter: +/- 0.4 m/s^2
        ax = 0.2 * math.sin(i * 0.5)
        ay = 0.2 * math.cos(i * 0.5)
        az = 9.81 + 0.3 * math.sin(i * 0.8)
        step, _, count, _ = engine.process_sample(ax, ay, az, ts)
        if step: desk_steps += 1
        ts += 10
    
    print(f"Test 1 [Desk Shake Immunity]: Detected steps = {desk_steps} (Expected: 0) -> {'PASS' if desk_steps == 0 else 'FAIL'}")
    
    # Test 2: Walking Gait (1.8 Hz cadence, peak ~ 12.5, trough ~ 7.5)
    walk_steps = 0
    for step_i in range(50):
        # 550 ms per step (approx 1.82 steps/s)
        for t_ms in range(0, 550, 10):
            phase = 2 * math.pi * (t_ms / 550.0)
            # footstrike profile
            az = 9.81 + 3.2 * math.sin(phase)
            ax = 0.8 * math.sin(phase)
            ay = 1.0 * math.cos(phase)
            step, stride, count, dist = engine.process_sample(ax, ay, az, ts)
            if step:
                walk_steps += 1
            ts += 10
            
    print(f"Test 2 [Hike Step Counting]: Steps simulated = 50, Detected = {walk_steps} -> {'PASS' if abs(walk_steps - 50) <= 1 else 'FAIL'}")
    print(f"        Total distance covered: {engine.total_dist_m:.2f} m (Average Stride: {engine.total_dist_m / max(1, engine.step_count):.2f} m)")
    
    # Test 3: Forest Trail Backtrack Navigation
    origin_lat, origin_lon = 22.730227, 75.833604
    # Hiker walks Northeast into the deep forest for 1,200 meters at 45 degrees
    dist_walked_m = 1200.0
    hike_azimuth_deg = 45.0
    
    dlat = (dist_walked_m * math.cos(math.radians(hike_azimuth_deg))) / 111132.954
    cur_lat = origin_lat + dlat
    cur_lon = origin_lon + (dist_walked_m * math.sin(math.radians(hike_azimuth_deg))) / (111412.84 * math.cos(math.radians(origin_lat)))
    
    return_dist_m = haversine_m(cur_lat, cur_lon, origin_lat, origin_lon)
    return_bearing = initial_bearing_deg(cur_lat, cur_lon, origin_lat, origin_lon)
    return_cardinal = bearing_to_cardinal(return_bearing)
    
    print(f"Test 3 [Forest Backtrack Return Vector]:")
    print(f"        Trailhead Origin: ({origin_lat:.6f}, {origin_lon:.6f})")
    print(f"        Lost in Forest:   ({cur_lat:.6f}, {cur_lon:.6f})")
    print(f"        Distance to Safety: {return_dist_m:.1f} m (Expected ~1200 m)")
    print(f"        Return Azimuth:     {return_bearing:.1f}° {return_cardinal} (Expected ~225° SW)")
    
    bearing_err = abs(return_bearing - 225.0)
    dist_err = abs(return_dist_m - 1200.0)
    passed = bearing_err < 1.0 and dist_err < 5.0
    print(f"        Status: {'PASS' if passed else 'FAIL'} (Bearing err: {bearing_err:.2f}°, Dist err: {dist_err:.2f}m)")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
