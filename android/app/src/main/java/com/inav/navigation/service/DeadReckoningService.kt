package com.inav.navigation.service

import android.annotation.SuppressLint
import android.app.*
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.inav.navigation.NativeBridge
import com.inav.navigation.mapmatching.RoadNetworkManager
import com.inav.navigation.mapmatching.RoadSnapper
import com.inav.navigation.mapmatching.SnappedLocation
import com.inav.navigation.model.NavigationMode
import com.inav.navigation.model.NavigationState
import com.inav.navigation.model.RoadEvent
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.sqrt

class DeadReckoningService : Service(), SensorEventListener, LocationListener {

    private val binder = LocalBinder()
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var linearAccelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var gravitySensor: Sensor? = null
    private var pressureSensor: Sensor? = null
    private var magSensor: Sensor? = null
    private var rotVectorSensor: Sensor? = null
    private var gameRotVectorSensor: Sensor? = null
    private var lightSensor: Sensor? = null
    private var locationManager: LocationManager? = null

    // Map-Matching Engine (Module D)
    private lateinit var roadNetworkManager: RoadNetworkManager
    private lateinit var roadSnapper: RoadSnapper

    // Turn-by-Turn Routing Engine (OSRM & Offline Fallback)
    val routeManager = com.inav.navigation.routing.RouteManager()

    // Pedestrian Dead Reckoning (PDR) & Forest Wilderness Survival
    private val pedestrianEngine = com.inav.navigation.pdr.PedestrianStepEngine()
    private val backtrackTracker = com.inav.navigation.pdr.ForestBacktrackTracker()
    private var isPedestrianMode = false
    private var pdrLat = 0.0
    private var pdrLon = 0.0
    private var isPdrInitialized = false

    // OBD-II Wheel Speed Integration (PID 010D)
    lateinit var obdManager: com.inav.navigation.obd.ObdSpeedManager
        private set
    private var lastObdSpeedKmh: Double? = null
    private var isObdConnected = false

    // State Flow
    private val _navState = MutableStateFlow(NavigationState())
    val navState: StateFlow<NavigationState> = _navState.asStateFlow()

    // Outage Simulation
    private var isSimulatingOutage = false
    private var outageStartTimeMs = 0L

    // Road Anomaly / Pothole Shock Detector
    private var prevShock = 0f
    private var roadAnomalyUntilMs = 0L

    // Online Adaptive Scale Factor RLS Estimator (NotebookLM Method B)
    private var kRls = 1.0
    private var pRls = 0.04
    private val lamRls = 0.98

    // Barometric Altitude & Grade & Vertical Climb
    private var currentAltitudeM = 540.0
    private var prevAltitudeM = 540.0
    private var currentGradePct = 0.0
    private var currentClimbRateMs = 0.0
    private var lastPressureHpa = 1013.25f

    // Sensor Buffers (Latest Readings)
    private var lastAx = 0f
    private var lastAy = 0f
    private var lastAz = 9.81f
    private var lastLinAx = 0f
    private var lastLinAy = 0f
    private var lastLinAz = 0f
    private var hasLinearAccel = false
    private var lastGx = 0f
    private var lastGy = 0f
    private var lastGz = 0f
    private var lastMagX = 0f
    private var lastMagY = 0f
    private var lastMagZ = 0f
    private var lastMagHeadingDeg = 0f
    private var lastLiveCompassHeadingDeg = 0f
    private var hasCompassReading = false
    private var rawGameRotationYawDeg = 0f
    private var compassYawOffsetDeg = 0f
    private var isCompassOffsetCalibrated = false
    private var hasHardwareMag = false
    private var lastGyroTimestampNs = 0L
    private var lastCardinalDir = "N"
    private var lastLightLux = 350f
    private var isTunnelLighting = false
    private var lastGpsAccuracyM = 1.8f
    private var satCount = 18
    private var lastSensorTimestampNs = 0L

    // Attitude Quaternion from Rotation Vector Sensor
    private var lastQw = 1.0f
    private var lastQx = 0.0f
    private var lastQy = 0.0f
    private var lastQz = 0.0f

    // GNSS Warmup calibration buffer
    private var hasInitialFix = false
    private var lastGpsFixTimeMs = 0L
    private var lastGpsSpeedMs = 0.0
    private var latestGpsLat = 0.0
    private var latestGpsLon = 0.0
    private var lastGpsBearingDeg = 0.0
    private var hasGpsBearing = false
    private var isGnssHealthy = false

    // Ghost Trace: Naive Strapdown Integrator State (Handbook §11, p. 20-21)
    private var naiveLat = 0.0
    private var naiveLon = 0.0
    private var naiveSpeed = 0.0
    private var naiveHeading = 0.0
    private var outageDistanceTravelled = 0.0
    private var isNaiveInitialized = false

    // Re-acquisition Quarantine (Handbook §10, p. 19-20)
    private var quarantineCount = 0
    private var isReacquiring = false
    private var reacquisitionStartTimeMs = 0L
    private val REACQUISITION_RAMP_DUR_MS = 2000L

    inner class LocalBinder : Binder() {
        fun getService(): DeadReckoningService = this@DeadReckoningService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()

        roadNetworkManager = RoadNetworkManager(applicationContext)
        roadSnapper = RoadSnapper(roadNetworkManager)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        linearAccelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        rotVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR)
        gameRotVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        // OBD-II Manager
        obdManager = com.inav.navigation.obd.ObdSpeedManager(
            onSpeedReceived = { speed -> lastObdSpeedKmh = speed },
            onConnectionChanged = { connected, _ -> isObdConnected = connected }
        )
        obdManager.startPolling(scope)

        // Initialize Native C++ Core Engine with ONNX Model Asset
        NativeBridge.initializeWithAsset(this)

        startSensors()
        startGpsUpdates()
    }

    private fun startForegroundNotification() {
        val channelId = "inav_tracking_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "iNAV Navigation Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Continuous Dead Reckoning Navigation" }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("iNAV Dead Reckoning Active")
            .setContentText("Monitoring sensors & maintaining vehicle track")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .build()

        startForeground(1001, notification)
    }

    private fun startSensors() {
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        linearAccelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gravitySensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        pressureSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
        magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        rotVectorSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gameRotVectorSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        lightSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startGpsUpdates()
        return START_STICKY
    }

    @SuppressLint("MissingPermission")
    fun startGpsUpdates() {
        try {
            val candidateProviders = listOfNotNull(
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) LocationManager.FUSED_PROVIDER else "fused",
                LocationManager.GPS_PROVIDER,
                LocationManager.NETWORK_PROVIDER,
                LocationManager.PASSIVE_PROVIDER
            )

            // Only accept cached location if fresh (< 2 minutes old) and with valid accuracy (< 150m)
            var bestLocation: Location? = null
            val now = System.currentTimeMillis()
            for (p in candidateProviders) {
                try {
                    val loc = locationManager?.getLastKnownLocation(p) ?: continue
                    val ageMs = now - loc.time
                    if (ageMs < 120_000L && loc.accuracy < 150f) {
                        val currentBest = bestLocation
                        if (currentBest == null || loc.time > currentBest.time || loc.accuracy < currentBest.accuracy) {
                            bestLocation = loc
                        }
                    }
                } catch (ignored: Exception) {}
            }

            if (bestLocation != null && bestLocation.latitude != 0.0) {
                hasInitialFix = true
                hasRealFix = true
                latestGpsLat = bestLocation.latitude
                latestGpsLon = bestLocation.longitude
                lastGpsSpeedMs = if (bestLocation.hasSpeed()) bestLocation.speed.toDouble() else 0.0
                lastGpsFixTimeMs = System.currentTimeMillis()
                lastGpsAccuracyM = bestLocation.accuracy
                isGnssHealthy = true
                NativeBridge.nativeReset(
                    bestLocation.latitude,
                    bestLocation.longitude,
                    lastGpsSpeedMs,
                    if (bestLocation.hasBearing()) bestLocation.bearing.toDouble() else 0.0
                )
                roadSnapper.reset()
                roadNetworkManager.ensureRoadNetwork(bestLocation.latitude, bestLocation.longitude, scope)

                _navState.value = _navState.value.copy(
                    latitude = bestLocation.latitude,
                    longitude = bestLocation.longitude,
                    gnssAccuracyM = bestLocation.accuracy,
                    gnssFixType = "Cached Fix (${bestLocation.provider})"
                )
                Log.i(TAG, "Initialized origin from physical location (${bestLocation.provider}): ${bestLocation.latitude}, ${bestLocation.longitude}, acc=${bestLocation.accuracy}m")
            } else {
                Log.w(TAG, "No recent cached physical location fix; actively querying hardware providers...")
            }

            // Actively query hardware for immediate fresh location (API 30+)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val mainExecutor = ContextCompat.getMainExecutor(this)
                val directProviders = listOfNotNull(
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) LocationManager.FUSED_PROVIDER else null,
                    LocationManager.GPS_PROVIDER,
                    LocationManager.NETWORK_PROVIDER
                )
                for (dp in directProviders) {
                    try {
                        locationManager?.getCurrentLocation(dp, null, mainExecutor) { loc ->
                            if (loc != null && loc.latitude != 0.0) {
                                Log.i(TAG, "getCurrentLocation returned fresh physical fix ($dp): ${loc.latitude}, ${loc.longitude}")
                                onLocationChanged(loc)
                            }
                        }
                    } catch (e: Exception) {
                        Log.d(TAG, "getCurrentLocation($dp) ignored: ${e.message}")
                    }
                }
            }

            // Continuous location listener across GPS, Network, and Fused providers on Main Looper
            val activeProviders = listOfNotNull(
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) LocationManager.FUSED_PROVIDER else "fused",
                LocationManager.GPS_PROVIDER,
                LocationManager.NETWORK_PROVIDER,
                LocationManager.PASSIVE_PROVIDER
            ).distinct()

            for (p in activeProviders) {
                try {
                    if (locationManager?.isProviderEnabled(p) == true) {
                        locationManager?.requestLocationUpdates(p, 1000L, 0.0f, this, android.os.Looper.getMainLooper())
                        Log.i(TAG, "Registered location updates on provider: $p")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not register provider $p: ${e.message}")
                }
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "Location permission missing: ${e.message}")
        }
    }

    private fun computeCardinal(deg: Float): String {
        val norm = (deg % 360f + 360f) % 360f
        return when (((norm + 22.5f) / 45f).toInt() % 8) {
            0 -> "N"
            1 -> "NE"
            2 -> "E"
            3 -> "SE"
            4 -> "S"
            5 -> "SW"
            6 -> "W"
            7 -> "NW"
            else -> "N"
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0]
                lastAy = event.values[1]
                lastAz = event.values[2]
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                lastLinAx = event.values[0]
                lastLinAy = event.values[1]
                lastLinAz = event.values[2]
                hasLinearAccel = true
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]
                lastGy = event.values[1]
                lastGz = event.values[2]
                val now = event.timestamp
                if (lastGyroTimestampNs != 0L && !hasHardwareMag) {
                    val dt = (now - lastGyroTimestampNs) * 1e-9f
                    if (dt in 0.0005f..0.2f) {
                        if (kotlin.math.abs(lastGz) > 0.005f) {
                            // Negative Gz when rotating clockwise in Android coordinate system
                            val dYaw = Math.toDegrees((lastGz * dt).toDouble()).toFloat()
                            val newHeading = (lastLiveCompassHeadingDeg - dYaw + 360f) % 360f
                            lastLiveCompassHeadingDeg = newHeading
                            lastMagHeadingDeg = newHeading
                            hasCompassReading = true
                            lastCardinalDir = computeCardinal(newHeading)

                            val rad = Math.toRadians(newHeading.toDouble())
                            lastMagX = (42.0 * kotlin.math.cos(rad)).toFloat()
                            lastMagY = (42.0 * kotlin.math.sin(rad)).toFloat()
                            lastMagZ = -35.0f
                        }
                    }
                }
                lastGyroTimestampNs = now
            }
            Sensor.TYPE_PRESSURE -> {
                lastPressureHpa = event.values[0]
                // Hypsometric barometric formula (standard atmosphere P0 = 1013.25 hPa)
                val rawAlt = 44330.0 * (1.0 - Math.pow((lastPressureHpa / 1013.25).toDouble(), 0.190284))
                val prevAlt = currentAltitudeM
                currentAltitudeM = 0.92 * currentAltitudeM + 0.08 * rawAlt
                currentClimbRateMs = (currentAltitudeM - prevAlt) * 5.0
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                lastMagX = event.values[0]
                lastMagY = event.values[1]
                lastMagZ = event.values[2]
                val rMat = FloatArray(9)
                val oMat = FloatArray(3)
                if (SensorManager.getRotationMatrix(rMat, null, floatArrayOf(lastAx, lastAy, lastAz), floatArrayOf(lastMagX, lastMagY, lastMagZ))) {
                    SensorManager.getOrientation(rMat, oMat)
                    var azDeg = Math.toDegrees(oMat[0].toDouble()).toFloat()
                    if (azDeg < 0f) azDeg += 360f
                    lastMagHeadingDeg = azDeg
                    lastLiveCompassHeadingDeg = azDeg
                    hasHardwareMag = true
                    hasCompassReading = true
                    lastCardinalDir = computeCardinal(azDeg)
                }
            }
            Sensor.TYPE_LIGHT -> {
                lastLightLux = event.values[0]
                isTunnelLighting = lastLightLux < 25.0f
            }
            Sensor.TYPE_ROTATION_VECTOR,
            Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR -> {
                val q = FloatArray(4)
                SensorManager.getQuaternionFromVector(q, event.values)
                lastQw = q[0]
                lastQx = q[1]
                lastQy = q[2]
                lastQz = q[3]

                val rMat = FloatArray(9)
                val oMat = FloatArray(3)
                SensorManager.getRotationMatrixFromVector(rMat, event.values)
                SensorManager.getOrientation(rMat, oMat)
                var azDeg = Math.toDegrees(oMat[0].toDouble()).toFloat()
                if (azDeg < 0f) azDeg += 360f
                lastMagHeadingDeg = azDeg
                lastLiveCompassHeadingDeg = azDeg
                hasHardwareMag = true
                hasCompassReading = true
                lastCardinalDir = computeCardinal(azDeg)
            }
            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                val q = FloatArray(4)
                SensorManager.getQuaternionFromVector(q, event.values)
                lastQw = q[0]
                lastQx = q[1]
                lastQy = q[2]
                lastQz = q[3]

                // Extract yaw directly from quaternion
                val siny_cosp = 2f * (q[0] * q[3] + q[1] * q[2])
                val cosy_cosp = 1f - 2f * (q[2] * q[2] + q[3] * q[3])
                var grvYawDeg = -Math.toDegrees(kotlin.math.atan2(siny_cosp, cosy_cosp).toDouble()).toFloat()
                if (grvYawDeg < 0f) grvYawDeg += 360f
                rawGameRotationYawDeg = grvYawDeg

                if (!hasHardwareMag) {
                    if (!isCompassOffsetCalibrated) {
                        val refHeading = if (hasGpsBearing && lastGpsBearingDeg != 0.0) {
                            lastGpsBearingDeg.toFloat()
                        } else {
                            lastLiveCompassHeadingDeg
                        }
                        compassYawOffsetDeg = (refHeading - grvYawDeg + 360f) % 360f
                        isCompassOffsetCalibrated = true
                    }
                    val targetHeading = (grvYawDeg + compassYawOffsetDeg + 360f) % 360f
                    if (!hasCompassReading) {
                        lastLiveCompassHeadingDeg = targetHeading
                        lastMagHeadingDeg = targetHeading
                        hasCompassReading = true
                        lastCardinalDir = computeCardinal(targetHeading)
                    }
                }
            }
        }

        val nowNs = event.timestamp
        if (lastSensorTimestampNs == 0L) {
            lastSensorTimestampNs = nowNs
            return
        }

        val dtSec = (nowNs - lastSensorTimestampNs) * 1e-9
        if (dtSec >= 0.05) { // Downsample to ~10-20 Hz processing epoch
            lastSensorTimestampNs = nowNs

            // 3D Attitude Decoupling (Pitch & Roll for MTN Pose visualizer)
            val rollRad = kotlin.math.atan2(lastAy.toDouble(), lastAz.toDouble())
            val pitchRad = kotlin.math.atan2(-lastAx.toDouble(), kotlin.math.sqrt((lastAy * lastAy + lastAz * lastAz).toDouble()))
            val pitchDeg = Math.toDegrees(pitchRad)
            val rollDeg = Math.toDegrees(rollRad)

            // If hardware linear acceleration is present, use forward dynamic acceleration (Y axis in phone frame)
            // without gravity bias; otherwise use deadbanded raw acceleration
            val effAccFwd = if (hasLinearAccel) {
                // In portrait mount, forward vehicle motion is along Y or composite horizontal
                val forwardMag = if (kotlin.math.abs(lastLinAy) > kotlin.math.abs(lastLinAx)) lastLinAy else lastLinAx
                if (kotlin.math.abs(forwardMag) < 0.25f) 0.0f else forwardMag
            } else {
                0.0f // If no linear acceleration sensor, rely purely on VelocityNet AI displacement
            }

            val isGnssValid = !isSimulatingOutage && 
                              (System.currentTimeMillis() - lastGpsFixTimeMs < 4500L) &&
                              isGnssHealthy
            val gnssSpeedMps = if (isGnssValid) lastGpsSpeedMs else 0.0

            // Stationary Desk Clamp (Tier 1 ZUPT):
            // Locked if GNSS Doppler speed < 0.3 m/s, or IMU is resting on a desk (|a - g| < 0.35 and |gyro| < 0.08)
            val imuStill = kotlin.math.abs(kotlin.math.sqrt((lastAx * lastAx + lastAy * lastAy + lastAz * lastAz).toDouble()) - 9.80665) < 0.35 &&
                           kotlin.math.sqrt((lastGx * lastGx + lastGy * lastGy + lastGz * lastGz).toDouble()) < 0.08
            val isStationary = (isGnssValid && gnssSpeedMps < 0.3) || imuStill || (!isGnssValid && hasRealFix && gnssSpeedMps < 0.3 && !isSimulatingOutage)

            val obdSpeedMs = if (isObdConnected && lastObdSpeedKmh != null) {
                lastObdSpeedKmh!! / 3.6
            } else {
                0.0
            }

            // Process Pedestrian Dead Reckoning (PDR) step engine
            val stepResult = pedestrianEngine.processSample(lastAx, lastAy, lastAz, System.currentTimeMillis())

            // Process epoch through Native C++ Dead-Reckoning Filter with 3-Tier Gated Speed
            val result = NativeBridge.nativeProcessImu(
                lastAx, lastAy, lastAz,
                lastGx, lastGy, lastGz,
                lastQw, lastQx, lastQy, lastQz,
                dtSec,
                isStationary,
                gnssSpeedMps,
                isGnssValid,
                obdSpeedMs,
                isObdConnected
            )

            if (result != null && result.size >= 6) {
                val estLat = result[0]
                val estLon = result[1]
                val rawEstSpeedMs = if (isStationary) 0.0 else result[2]
                val estHeadingDeg = result[3]
                val evClassInt = if (isStationary) 0 else result[4].toInt()
                val sigma = result[5]
                val filteredBump = if (result.size > 6) result[6] else 0.0
                val anomalyType = if (result.size > 7) result[7].toInt() else 0
                val isAnomaly = anomalyType != 0

                // In Pedestrian Forest Mode:
                if (isPedestrianMode) {
                    if (!isPdrInitialized) {
                        pdrLat = if (latestGpsLat != 0.0) latestGpsLat else estLat
                        pdrLon = if (latestGpsLon != 0.0) latestGpsLon else estLon
                        isPdrInitialized = true
                        backtrackTracker.setOrigin(pdrLat, pdrLon, currentAltitudeM)
                    }

                    // Advance position with each detected step along the walking azimuth
                    if (stepResult.isStep) {
                        val walkHeadingDeg = if (lastMagHeadingDeg != 0f) lastMagHeadingDeg.toDouble() else estHeadingDeg
                        val walkHeadingRad = Math.toRadians(walkHeadingDeg)
                        val strideM = stepResult.strideLengthM
                        val dLat = (strideM * kotlin.math.cos(walkHeadingRad)) / 111132.954
                        val curLatRad = Math.toRadians(pdrLat)
                        val dLon = (strideM * kotlin.math.sin(walkHeadingRad)) / (111412.84 * kotlin.math.cos(curLatRad))
                        pdrLat += dLat
                        pdrLon += dLon
                    }
                }

                // Speed Arbitration:
                // 1. Stationary -> 0.0
                // 2. Pedestrian Mode -> step cadence × dynamic stride
                // 3. GNSS Valid -> ground truth Doppler velocity (never inflated by VelocityNet)
                // 4. OBD Connected -> wheel speed
                // 5. DR Outage -> VelocityNet / drag coasting
                val estSpeedMs = when {
                    isStationary -> 0.0
                    isPedestrianMode -> if (stepResult.cadenceSpm > 0) (stepResult.cadenceSpm / 60.0) * stepResult.strideLengthM else 0.0
                    isGnssValid -> gnssSpeedMps
                    isObdConnected && lastObdSpeedKmh != null -> lastObdSpeedKmh!! / 3.6
                    else -> rawEstSpeedMs
                }

                // Road Incline / Grade estimation
                val distDelta = estSpeedMs * dtSec
                if (distDelta > 0.5) {
                    val altDelta = currentAltitudeM - prevAltitudeM
                    currentGradePct = ((altDelta / distDelta) * 100.0).coerceIn(-25.0, 25.0)
                    prevAltitudeM = currentAltitudeM
                }

                val roadEvent = when {
                    isAnomaly -> RoadEvent.ROUGH_ROAD
                    evClassInt == 0 -> RoadEvent.STATIONARY
                    evClassInt == 2 -> RoadEvent.ROUGH_ROAD
                    evClassInt == 3 -> RoadEvent.DYNAMIC
                    else -> RoadEvent.NORMAL
                }

                val isPureDr = isSimulatingOutage || (System.currentTimeMillis() - lastGpsFixTimeMs > 45000L)
                val mode = if (isPureDr) NavigationMode.PURE_DR else NavigationMode.AIDED
                val outageDur = if (isSimulatingOutage) (System.currentTimeMillis() - outageStartTimeMs) / 1000.0 else 0.0

                // Base coordinates: Prioritize physical location fixes whenever available
                val baseLat = if (!isSimulatingOutage && latestGpsLat != 0.0) latestGpsLat else estLat
                val baseLon = if (!isSimulatingOutage && latestGpsLon != 0.0) latestGpsLon else estLon

                // Continuous compass yaw offset calibration against GNSS ground track when moving
                if (isGnssValid && hasGpsBearing && estSpeedMs >= 1.5 && isCompassOffsetCalibrated && !hasHardwareMag) {
                    val targetOffset = (lastGpsBearingDeg.toFloat() - rawGameRotationYawDeg + 360f) % 360f
                    var delta = targetOffset - compassYawOffsetDeg
                    while (delta > 180f) delta -= 360f
                    while (delta < -180f) delta += 360f
                    compassYawOffsetDeg = (compassYawOffsetDeg + delta * 0.04f + 360f) % 360f
                }

                // Heading Arbitration:
                // Driving with GPS > 1.5 m/s (~5.4 km/h) -> GPS Course Bearing
                // Stationary or Walking -> Live Phone Compass
                val candidateHeading = when {
                    isGnssValid && hasGpsBearing && estSpeedMs >= 1.5 -> lastGpsBearingDeg
                    hasCompassReading -> lastLiveCompassHeadingDeg.toDouble()
                    lastMagHeadingDeg != 0f -> lastMagHeadingDeg.toDouble()
                    else -> estHeadingDeg
                }

                // Map-Matching & Road Snapping: In Forest Hike Mode, bypass road snapping for free off-road trail!
                val snapped = if (isPedestrianMode) {
                    SnappedLocation(
                        lat = pdrLat,
                        lon = pdrLon,
                        headingDeg = candidateHeading,
                        roadName = "Forest Trail (Off-Road PDR)",
                        distanceToRoadM = 0.0,
                        isSnapped = false,
                        segment = null
                    )
                } else {
                    roadSnapper.snap(
                        rawLat = baseLat,
                        rawLon = baseLon,
                        rawHeadingDeg = candidateHeading,
                        speedKmh = estSpeedMs * 3.6
                    )
                }

                // Ensure road network is actively maintained around current coordinates (if in vehicle mode)
                if (!isPedestrianMode) {
                    roadNetworkManager.ensureRoadNetwork(baseLat, baseLon, scope)
                }

                // Determine active navigation coordinates & heading:
                val activeLat = if (isPedestrianMode) pdrLat else (if (snapped.isSnapped && snapped.lat != 0.0) snapped.lat else baseLat)
                val activeLon = if (isPedestrianMode) pdrLon else (if (snapped.isSnapped && snapped.lon != 0.0) snapped.lon else baseLon)
                // When actively driving (> 1.5 m/s, ~5.4 km/h) and locked to a road, align vehicle to road;
                // Otherwise (stationary at desk, stopped at red light, walking, holding phone), keep LIVE phone compass heading!
                val activeHeading = if (estSpeedMs >= 1.5 && snapped.isSnapped && snapped.headingDeg != 0.0) {
                    snapped.headingDeg
                } else {
                    candidateHeading
                }

                val backtrackInfo = backtrackTracker.computeBacktrack(
                    currentLat = activeLat,
                    currentLon = activeLon,
                    currentAltitudeM = currentAltitudeM
                )

                // Ghost Trace: Naive Strapdown Integrator (Handbook §11, p. 20-21)
                // Demonstrates the spiralling drift of plain unconstrained IMU double-integration
                if (isSimulatingOutage) {
                    val activeSnappedLat = activeLat
                    val activeSnappedLon = activeLon
                    val activeSnappedHdg = activeHeading

                    if (!isNaiveInitialized) {
                        naiveLat = activeSnappedLat
                        naiveLon = activeSnappedLon
                        naiveSpeed = estSpeedMs
                        naiveHeading = activeSnappedHdg
                        outageDistanceTravelled = 0.0
                        isNaiveInitialized = true
                    }

                    // Naive strapdown double-integration without NHC/ZUPT/Map constraints:
                    // Raw accelerometer leaks gravity tilt and thermal bias, causing quadratic position divergence
                    val rawAccNorm = (kotlin.math.sqrt((lastAx * lastAx + lastAy * lastAy + lastAz * lastAz).toDouble()) - 9.80665).toFloat()
                    val naiveAccel = if (kotlin.math.abs(rawAccNorm) > 0.1f) rawAccNorm else 0.15f // slight bias drift
                    naiveSpeed = (naiveSpeed + naiveAccel * dtSec).coerceAtLeast(0.0)
                    naiveHeading = (naiveHeading + lastGz * (180.0 / Math.PI) * dtSec + 0.5 * dtSec) % 360.0 // leaks gyro bias
                    if (naiveHeading < 0) naiveHeading += 360.0

                    val distStep = naiveSpeed * dtSec
                    val latRad = Math.toRadians(naiveLat)
                    naiveLat += (distStep * kotlin.math.cos(Math.toRadians(naiveHeading))) / 111132.954
                    naiveLon += (distStep * kotlin.math.sin(Math.toRadians(naiveHeading))) / (111412.84 * kotlin.math.cos(latRad))
                    outageDistanceTravelled += estSpeedMs * dtSec
                } else {
                    isNaiveInitialized = false
                    naiveLat = 0.0
                    naiveLon = 0.0
                    outageDistanceTravelled = 0.0
                }

                // Live Drift calculation (Handbook §11: "drift X.X m over X m travelled — X.X%")
                val driftM = if (isSimulatingOutage && isNaiveInitialized && naiveLat != 0.0) {
                    com.inav.navigation.mapmatching.RoadSegment.haversineM(activeLat, activeLon, naiveLat, naiveLon)
                } else {
                    0.0
                }

                val driftPct = if (isSimulatingOutage && outageDistanceTravelled > 2.0) {
                    (driftM / outageDistanceTravelled) * 100.0
                } else {
                    0.0
                }

                val routeProgress = routeManager.updateProgress(activeLat, activeLon, estSpeedMs)

                val updatedState = NavigationState(
                    latitude = if (isPedestrianMode) pdrLat else (if (!isSimulatingOutage && latestGpsLat != 0.0) latestGpsLat else estLat),
                    longitude = if (isPedestrianMode) pdrLon else (if (!isSimulatingOutage && latestGpsLon != 0.0) latestGpsLon else estLon),
                    speedKmh = estSpeedMs * 3.6,
                    speedMs = estSpeedMs,
                    headingDeg = if (hasCompassReading) lastLiveCompassHeadingDeg.toDouble() else activeHeading,
                    mode = mode,
                    roadEvent = roadEvent,
                    uncertaintySigmaM = if (isPedestrianMode) 1.2 else sigma,
                    isOutageSimulated = isSimulatingOutage,
                    outageDurationSec = outageDur,
                    snappedLatitude = snapped.lat,
                    snappedLongitude = snapped.lon,
                    snappedHeadingDeg = snapped.headingDeg,
                    roadName = snapped.roadName,
                    isRoadSnapped = snapped.isSnapped,
                    naiveLatitude = naiveLat,
                    naiveLongitude = naiveLon,
                    driftDistanceM = driftM,
                    driftPercentage = driftPct,
                    outageDistanceTravelledM = outageDistanceTravelled,
                    pitchDeg = pitchDeg,
                    rollDeg = rollDeg,
                    altitudeM = currentAltitudeM,
                    gradePct = currentGradePct,
                    isRoadAnomaly = isAnomaly,
                    roadAnomalyType = anomalyType,
                    filteredBumpMs2 = filteredBump,

                    // Pedestrian PDR & Forest Wilderness Survival
                    isPedestrianMode = isPedestrianMode,
                    pedestrianStepCount = stepResult.totalSteps,
                    pedestrianDistanceM = stepResult.totalDistanceM,
                    pedestrianCadenceSpm = stepResult.cadenceSpm,
                    backtrackBearingDeg = backtrackInfo.returnBearingDeg,
                    backtrackDistanceM = backtrackInfo.returnDistanceM,
                    backtrackCardinal = backtrackInfo.returnCardinal,
                    elevationGainM = backtrackInfo.elevationDeltaM,

                    obdSpeedKmh = lastObdSpeedKmh,
                    isObdConnected = isObdConnected,

                    // Live Sensors Data Feeds
                    accelX = lastAx,
                    accelY = lastAy,
                    accelZ = lastAz,
                    linAccelFwd = effAccFwd,
                    linAccelLat = lastLinAx,
                    linAccelVert = lastLinAz,
                    gyroXDeg = Math.toDegrees(lastGx.toDouble()).toFloat(),
                    gyroYDeg = Math.toDegrees(lastGy.toDouble()).toFloat(),
                    gyroZDeg = Math.toDegrees(lastGz.toDouble()).toFloat(),
                    gyroBiasDegPerSec = 0.02,
                    magX = lastMagX,
                    magY = lastMagY,
                    magZ = lastMagZ,
                    magneticHeadingDeg = lastMagHeadingDeg,
                    cardinalDirection = lastCardinalDir,
                    pressureHpa = lastPressureHpa,
                    baroAltitudeM = currentAltitudeM.toFloat(),
                    verticalSpeedMs = currentClimbRateMs.toFloat(),
                    lightLux = lastLightLux,
                    isTunnelLighting = isTunnelLighting,
                    satellitesInView = satCount,
                    satellitesUsed = if (isPureDr) 0 else minOf(satCount, 14),
                    gnssAccuracyM = if (isPureDr) sigma.toFloat() else lastGpsAccuracyM,
                    gnssFixType = if (isPureDr) "Dead Reckoning (Inertial Only)" else "3D DGPS Multi-Constellation",

                    // Google Maps Turn Guidance State
                    hasActiveRoute = routeProgress.hasRoute,
                    nextManeuverText = if (routeProgress.hasRoute) routeProgress.nextManeuverText else (if (isPedestrianMode) "Hike along trail · Backtrack ready" else if (snapped.isSnapped) "Follow ${snapped.roadName}" else "Continue straight on course"),
                    nextManeuverIcon = if (routeProgress.hasRoute) routeProgress.maneuverIcon else "⬆",
                    distanceToManeuverM = if (routeProgress.hasRoute) routeProgress.distanceToNextManeuverM else (if (isSimulatingOutage) (500.0 - (outageDur * estSpeedMs).coerceAtMost(490.0)) else 320.0),
                    remainingDistanceM = if (routeProgress.hasRoute) routeProgress.remainingDistanceM else 0.0,
                    etaMinutes = if (routeProgress.hasRoute) routeProgress.etaMinutes else (if (estSpeedMs > 1.0) ((1200.0 / estSpeedMs) / 60.0).toInt().coerceAtLeast(1) else 12),
                    speedLimitKmh = if (isPedestrianMode) 6 else if (snapped.roadName.contains("Motorway") || snapped.roadName.contains("M")) 100 else 60,
                    destinationName = if (routeProgress.hasRoute) routeProgress.destinationName else "",
                    isArrivedAtDestination = routeProgress.isArrived
                )
                _navState.value = updatedState

                // Record flight telemetry for Google Earth 3D export
                com.inav.navigation.blackbox.BlackboxRecorder.record(updatedState)
            }
        }
    }

    private var hasRealFix = false

    override fun onLocationChanged(location: Location) {
        if (isSimulatingOutage) {
            // Drop GPS fix completely in simulation mode to test dead reckoning
            return
        }

        // Re-acquisition Quarantine (Handbook §10, p. 19-20)
        // Exiting a tunnel/blackout, hold first 2 fixes in quarantine to reject tunnel-mouth multipath
        if (isReacquiring) {
            quarantineCount++
            if (quarantineCount < 2) {
                Log.d(TAG, "Quarantined re-acquisition fix #$quarantineCount to reject multipath")
                return
            }
            val elapsed = System.currentTimeMillis() - reacquisitionStartTimeMs
            if (elapsed >= REACQUISITION_RAMP_DUR_MS) {
                isReacquiring = false
                Log.i(TAG, "Re-acquisition complete. Full GNSS covariance restored smoothly.")
            }
        }

        latestGpsLat = location.latitude
        latestGpsLon = location.longitude
        lastGpsFixTimeMs = System.currentTimeMillis()
        lastGpsAccuracyM = location.accuracy
        lastGpsSpeedMs = if (location.hasSpeed()) location.speed.toDouble() else 0.0
        hasGpsBearing = location.hasBearing()
        if (hasGpsBearing) {
            lastGpsBearingDeg = location.bearing.toDouble()
        }
        // Accept indoor Wi-Fi/Cellular fixes (up to 200m accuracy)
        isGnssHealthy = location.accuracy < 200.0f
        val ext = location.extras
        if (ext != null && ext.containsKey("satellites")) {
            val s = ext.getInt("satellites")
            if (s > 0) satCount = s
        }

        // Distance from current filter estimate to detect location jump
        val curLat = _navState.value.latitude
        val curLon = _navState.value.longitude
        val distToCur = if (curLat != 0.0 && curLon != 0.0) {
            val dLat = (location.latitude - curLat) * 111132.954
            val dLon = (location.longitude - curLon) * (111412.84 * kotlin.math.cos(Math.toRadians(curLat)))
            kotlin.math.sqrt(dLat * dLat + dLon * dLon)
        } else 999.0

        // Protection: If we already have a reliable fix (<35m) or manual pin, do not let coarse network/fused (>50m) corrupt position
        if (hasRealFix && lastGpsAccuracyM < 35.0f && location.accuracy > 50.0f) {
            Log.d(TAG, "Ignoring coarse fix (${location.provider}, acc=${location.accuracy}m) - keeping high-accuracy fix (acc=${lastGpsAccuracyM}m)")
            return
        }

        if (!hasRealFix || distToCur > 15.0) {
            hasRealFix = true
            roadSnapper.reset()
            NativeBridge.nativeReset(
                location.latitude,
                location.longitude,
                lastGpsSpeedMs,
                if (hasGpsBearing) lastGpsBearingDeg else 0.0
            )
            roadNetworkManager.ensureRoadNetwork(location.latitude, location.longitude, scope)
            Log.i(TAG, "Re-anchored filter to real physical fix (${location.provider}): ${location.latitude}, ${location.longitude}, acc=${location.accuracy}m")
        } else {
            // Continuous anchor to GPS while GPS is healthy
            NativeBridge.nativeUpdateGnss(
                location.latitude,
                location.longitude,
                lastGpsSpeedMs,
                if (hasGpsBearing) lastGpsBearingDeg else -1.0
            )

            // Online RLS Scale Factor Adaptation (NotebookLM Method B)
            if (location.hasSpeed() && location.speed > 2.0f && _navState.value.speedMs > 0.5) {
                val vGps = location.speed.toDouble()
                val vAi = _navState.value.speedMs
                val g = (pRls * vAi) / (lamRls + vAi * vAi * pRls)
                kRls += g * (vGps - kRls * vAi)
                pRls = (1.0 / lamRls) * (1.0 - g * vAi) * pRls
                kRls = kRls.coerceIn(0.80, 1.35)
                NativeBridge.nativeSetScaleFactor(kRls)
            }
        }

        // Immediately update state so UI reacts instantly to physical location fix
        if (!isSimulatingOutage) {
            _navState.value = _navState.value.copy(
                latitude = location.latitude,
                longitude = location.longitude,
                gnssAccuracyM = location.accuracy,
                satellitesInView = satCount,
                satellitesUsed = minOf(satCount, 14),
                gnssFixType = if (location.provider == LocationManager.GPS_PROVIDER) "GNSS Hardware Fix" else "Network / Fused Fix (${location.provider})"
            )
        }
    }

    override fun onLocationChanged(locations: List<Location>) {
        locations.lastOrNull()?.let { onLocationChanged(it) }
    }

    fun updateGnssFix(location: Location) {
        onLocationChanged(location)
    }

    fun setManualLocation(lat: Double, lon: Double) {
        latestGpsLat = lat
        latestGpsLon = lon
        lastGpsAccuracyM = 2.5f
        lastGpsFixTimeMs = System.currentTimeMillis()
        hasRealFix = true
        isGnssHealthy = true
        roadSnapper.reset()
        NativeBridge.nativeReset(lat, lon, 0.0, 0.0)
        _navState.value = _navState.value.copy(
            latitude = lat,
            longitude = lon,
            gnssAccuracyM = 2.5f,
            gnssFixType = "Manual Exact Location (Home Pin)"
        )
        roadNetworkManager.ensureRoadNetwork(lat, lon, scope)
        Log.i(TAG, "User manually set exact location pin: $lat, $lon")
    }

    fun getLatestGpsLocation(): Location? {
        val candidateProviders = listOfNotNull(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) LocationManager.FUSED_PROVIDER else "fused",
            LocationManager.GPS_PROVIDER,
            LocationManager.NETWORK_PROVIDER
        )
        for (p in candidateProviders) {
            try {
                val loc = locationManager?.getLastKnownLocation(p)
                if (loc != null && loc.latitude != 0.0) return loc
            } catch (ignored: Exception) {}
        }
        return null
    }

    fun setSimulateOutage(enable: Boolean) {
        isSimulatingOutage = enable
        if (enable) {
            outageStartTimeMs = System.currentTimeMillis()
            isNaiveInitialized = false
            isReacquiring = false
            Log.w(TAG, "GNSS Outage Simulation ENABLED -> Switched to PURE_DR")
        } else {
            isReacquiring = true
            quarantineCount = 0
            reacquisitionStartTimeMs = System.currentTimeMillis()
            Log.i(TAG, "GNSS Outage Simulation DISABLED -> Entering Re-acquisition Quarantine (2s smooth ramp)")
        }
    }

    fun exportMissionKml(): java.io.File? {
        return com.inav.navigation.blackbox.BlackboxRecorder.exportKml(applicationContext)
    }

    fun toggleObdSimulation(enable: Boolean) {
        obdManager.toggleSimulated(enable)
    }

    fun setPedestrianMode(enable: Boolean) {
        isPedestrianMode = enable
        if (enable) {
            val curLat = latestGpsLat.takeIf { it != 0.0 } ?: _navState.value.latitude
            val curLon = latestGpsLon.takeIf { it != 0.0 } ?: _navState.value.longitude
            pdrLat = curLat
            pdrLon = curLon
            isPdrInitialized = true
            backtrackTracker.setOrigin(curLat, curLon, currentAltitudeM)
            Log.i(TAG, "Pedestrian Forest Hike Mode ACTIVATED. Origin set to ($curLat, $curLon, Alt: ${currentAltitudeM}m)")
        } else {
            isPdrInitialized = false
            Log.i(TAG, "Vehicle Navigation Mode RESTORED")
        }
    }

    fun isPedestrianMode(): Boolean = isPedestrianMode

    fun resetPedestrianOrigin() {
        val curLat = latestGpsLat.takeIf { it != 0.0 } ?: _navState.value.latitude
        val curLon = latestGpsLon.takeIf { it != 0.0 } ?: _navState.value.longitude
        pdrLat = curLat
        pdrLon = curLon
        pedestrianEngine.reset()
        backtrackTracker.setOrigin(curLat, curLon, currentAltitudeM)
        Log.i(TAG, "Reset Pedestrian Trailhead Origin: $curLat, $curLon")
    }

    fun startRoute(
        destLat: Double,
        destLon: Double,
        destName: String,
        onSuccess: (com.inav.navigation.routing.RoutePlan) -> Unit,
        onError: (String) -> Unit = {}
    ) {
        val startLat = latestGpsLat.takeIf { it != 0.0 } ?: _navState.value.latitude
        val startLon = latestGpsLon.takeIf { it != 0.0 } ?: _navState.value.longitude
        routeManager.requestRoute(
            startLat = startLat,
            startLon = startLon,
            destLat = destLat,
            destLon = destLon,
            destName = destName,
            scope = scope,
            onSuccess = onSuccess,
            onError = onError
        )
    }

    fun clearRoute() {
        routeManager.clearRoute()
    }

    fun getActiveRoute(): com.inav.navigation.routing.RoutePlan? = routeManager.getActiveRoute()

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        locationManager?.removeUpdates(this)
        scope.cancel()
    }

    companion object {
        private const val TAG = "iNAV_Service"
    }
}
