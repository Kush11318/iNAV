package com.inav.navigation.model

enum class NavigationMode(val displayName: String) {
    AIDED("AIDED (GNSS + INS)"),
    DEGRADED("DEGRADED (Weak GNSS)"),
    PURE_DR("PURE DR (Dead Reckoning)")
}

enum class RoadEvent(val displayName: String) {
    STATIONARY("Stationary (ZUPT Active)"),
    NORMAL("Normal Cruise"),
    ROUGH_ROAD("Rough Road / Pothole Detected"),
    DYNAMIC("Dynamic Maneuver / Cornering")
}

data class NavigationState(
    // Core Position & Velocity
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val speedKmh: Double = 0.0,
    val speedMs: Double = 0.0,
    val headingDeg: Double = 0.0,
    val mode: NavigationMode = NavigationMode.AIDED,
    val roadEvent: RoadEvent = RoadEvent.NORMAL,
    val uncertaintySigmaM: Double = 0.5,

    // Outage Simulation & Ghost Trace
    val isOutageSimulated: Boolean = false,
    val outageDurationSec: Double = 0.0,
    val snappedLatitude: Double = 0.0,
    val snappedLongitude: Double = 0.0,
    val snappedHeadingDeg: Double = 0.0,
    val roadName: String = "Detecting Road...",
    val isRoadSnapped: Boolean = false,
    val naiveLatitude: Double = 0.0,
    val naiveLongitude: Double = 0.0,
    val driftDistanceM: Double = 0.0,
    val driftPercentage: Double = 0.0,
    val outageDistanceTravelledM: Double = 0.0,

    // Attitude & Elevation
    val pitchDeg: Double = 0.0,
    val rollDeg: Double = 0.0,
    val altitudeM: Double = 0.0,
    val gradePct: Double = 0.0,
    val isRoadAnomaly: Boolean = false,
    val roadAnomalyType: Int = 0, // 0 = Normal, 1 = Pothole (Crater Drop), 2 = Speed Breaker (Bump)
    val filteredBumpMs2: Double = 0.0,

    // Pedestrian PDR & Forest Wilderness Survival
    val isPedestrianMode: Boolean = false,
    val pedestrianStepCount: Int = 0,
    val pedestrianDistanceM: Double = 0.0,
    val pedestrianCadenceSpm: Int = 0,
    val backtrackBearingDeg: Double = 0.0,
    val backtrackDistanceM: Double = 0.0,
    val backtrackCardinal: String = "N",
    val elevationGainM: Double = 0.0,

    // OBD-II Data
    val obdSpeedKmh: Double? = null,
    val isObdConnected: Boolean = false,

    // LIVE SENSORS: 1. Accelerometer (3-Axis m/s²)
    val accelX: Float = 0f,
    val accelY: Float = 0f,
    val accelZ: Float = 9.81f,
    val linAccelFwd: Float = 0f,
    val linAccelLat: Float = 0f,
    val linAccelVert: Float = 0f,

    // LIVE SENSORS: 2. Gyroscope (3-Axis deg/s & rad/s)
    val gyroXDeg: Float = 0f,
    val gyroYDeg: Float = 0f,
    val gyroZDeg: Float = 0f,
    val gyroBiasDegPerSec: Double = 0.0,

    // LIVE SENSORS: 3. Magnetometer (3-Axis µT & Compass)
    val magX: Float = 0f,
    val magY: Float = 0f,
    val magZ: Float = 0f,
    val magneticHeadingDeg: Float = 0f,
    val cardinalDirection: String = "N",

    // LIVE SENSORS: 4. Barometer (Pressure & Vertical Climb)
    val pressureHpa: Float = 1013.25f,
    val baroAltitudeM: Float = 0f,
    val verticalSpeedMs: Float = 0f,

    // LIVE SENSORS: 5. Ambient Light (Lux)
    val lightLux: Float = 350f,
    val isTunnelLighting: Boolean = false,

    // LIVE SENSORS: 6. GNSS Raw Hardware Status
    val satellitesInView: Int = 18,
    val satellitesUsed: Int = 12,
    val gnssAccuracyM: Float = 1.8f,
    val gnssFixType: String = "3D Fix (DGPS)",

    // Google Maps Turn Guidance State
    val hasActiveRoute: Boolean = false,
    val nextManeuverText: String = "Continue straight on course",
    val nextManeuverIcon: String = "⬆",
    val distanceToManeuverM: Double = 250.0,
    val remainingDistanceM: Double = 0.0,
    val etaMinutes: Int = 14,
    val speedLimitKmh: Int = 60,
    val destinationName: String = "",
    val isArrivedAtDestination: Boolean = false,

    val timestampMs: Long = System.currentTimeMillis()
)
