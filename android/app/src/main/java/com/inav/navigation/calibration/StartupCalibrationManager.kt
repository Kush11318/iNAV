package com.inav.navigation.calibration

import android.util.Log
import com.inav.navigation.NativeBridge
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.*

data class CalibrationResult(
    val durationSec: Double,
    val sampleCount: Int,
    val gyroBiasX: Double,
    val gyroBiasY: Double,
    val gyroBiasZ: Double,
    val gyroStdDev: Double,
    val meanAccelX: Double,
    val meanAccelY: Double,
    val meanAccelZ: Double,
    val gravityMagnitude: Double,
    val accelVariance: Double,
    val pitchDeg: Double,
    val rollDeg: Double,
    val stationaryConfidence: Double,
    val gyroConfidence: Double,
    val accelConfidence: Double,
    val alignmentConfidence: Double,
    val yawStatus: String = "Stationary leveled baseline. Yaw aligns dynamically with driving vector."
)

enum class CalibrationStatus {
    IDLE,
    COLLECTING,
    PAUSED_MOTION,
    INSUFFICIENT_DATA,
    COMPLETED
}

data class CalibrationState(
    val status: CalibrationStatus = CalibrationStatus.IDLE,
    val progress: Float = 0f, // 0.0 to 1.0
    val elapsedStationarySec: Double = 0.0,
    val targetDurationSec: Double = 15.0,
    val remainingSec: Int = 15,
    val isStationary: Boolean = true,
    val samplesCollected: Int = 0,
    val statusText: String = "Preparing Navigation",
    val checklistSensors: Boolean = false,
    val checklistGravity: Boolean = false,
    val checklistGyro: Boolean = false,
    val checklistAlignment: Boolean = false,
    val result: CalibrationResult? = null
)

/**
 * StartupCalibrationManager
 * Phase 12B: 15-second passive vehicle calibration.
 * Collects ~100 Hz IMU samples to nullify gyro drift biases,
 * calculate phone-to-vehicle leveling (pitch & roll), and prime the UKF alignment engine.
 */
class StartupCalibrationManager {

    private val _state = MutableStateFlow(CalibrationState())
    val state: StateFlow<CalibrationState> = _state.asStateFlow()

    private var isCalibrating = false
    private var isCompleted = false

    private data class Sample(
        val ax: Float, val ay: Float, val az: Float,
        val gx: Float, val gy: Float, val gz: Float,
        val timestampNs: Long
    )

    private val samples = ArrayList<Sample>(2000)
    private val rollingWindow = ArrayDeque<Sample>(60)
    private var tiltShiftCounter = 0
    private var calibratedAtTimestampMs: Long = 0L

    private var firstSampleTimestampNs = 0L
    private var lastSampleTimestampNs = 0L
    private var totalStationaryDurationSec = 0.0
    private var lastStateEmitTimeNs = 0L

    companion object {
        private const val TAG = "iNAV_Calibration"
        const val TARGET_DURATION_SEC = 15.0
        const val MAX_COLLECTION_TIMEOUT_SEC = 45.0
        const val MAX_CALIBRATION_AGE_MS = 15 * 60 * 1000L // 15 mins while idle/parked
        private const val G_REF = 9.80665 // m/s^2
    }

    @Synchronized
    fun startCalibration() {
        if (isCompleted) return
        reset()
        isCalibrating = true
        _state.value = CalibrationState(
            status = CalibrationStatus.COLLECTING,
            statusText = "Calibrating sensors...",
            targetDurationSec = TARGET_DURATION_SEC,
            remainingSec = TARGET_DURATION_SEC.toInt()
        )
        Log.i(TAG, "=== 15s STARTUP CALIBRATION STARTED ===")
        Log.i(TAG, "[BEFORE CALIBRATION] Alignment confidence: 0.00 | Gyro bias estimate: [0.000000, 0.000000, 0.000000] rad/s")
    }

    @Synchronized
    fun forceRecalibration(reason: String = "User requested") {
        Log.i(TAG, "Triggering recalibration: $reason")
        reset()
        startCalibration()
    }

    @Synchronized
    fun reset() {
        isCalibrating = false
        isCompleted = false
        calibratedAtTimestampMs = 0L
        samples.clear()
        rollingWindow.clear()
        firstSampleTimestampNs = 0L
        lastSampleTimestampNs = 0L
        totalStationaryDurationSec = 0.0
        tiltShiftCounter = 0
        _state.value = CalibrationState()
    }

    fun isCalibrationExpired(): Boolean {
        if (!isCompleted || calibratedAtTimestampMs == 0L) return false
        return (System.currentTimeMillis() - calibratedAtTimestampMs) > MAX_CALIBRATION_AGE_MS
    }

    fun getTimeSinceCalibrationMs(): Long {
        return if (calibratedAtTimestampMs == 0L) 0L else System.currentTimeMillis() - calibratedAtTimestampMs
    }

    /**
     * Checks if the phone orientation has shifted substantially while stationary
     */
    @Synchronized
    fun checkOrientationShift(currentAx: Float, currentAy: Float, currentAz: Float): Boolean {
        if (!isCompleted) return false
        val lastResult = _state.value.result ?: return false
        val gMag = lastResult.gravityMagnitude
        val currentMag = sqrt((currentAx * currentAx + currentAy * currentAy + currentAz * currentAz).toDouble())
        if (abs(currentMag - G_REF) > 1.2) return false

        val dot = (currentAx * lastResult.meanAccelX + currentAy * lastResult.meanAccelY + currentAz * lastResult.meanAccelZ) / (currentMag * gMag)
        val angleDiffDeg = Math.toDegrees(acos(dot.coerceIn(-1.0, 1.0)))
        if (angleDiffDeg > 10.0) {
            Log.i(TAG, "Orientation shift detected (%.1f°) -> recalibrating".format(angleDiffDeg))
            forceRecalibration("Phone unmounted/orientation changed (shift: %.1f°)".format(angleDiffDeg))
            return true
        }
        return false
    }

    @Synchronized
    fun onImuSample(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        timestampNs: Long
    ) {
        if (isCompleted) {
            // Demonstration Mode:
            // When calibration is complete and maps are showing, phone disturbances,
            // handling, tilts, or movements must NEVER automatically trigger recalibration.
            // Recalibration is strictly controlled manually via the dedicated Recalibrate button.
            return
        }

        if (!isCalibrating) {
            startCalibration()
        }

        val sample = Sample(ax, ay, az, gx, gy, gz, timestampNs)
        if (firstSampleTimestampNs == 0L) {
            firstSampleTimestampNs = timestampNs
            lastSampleTimestampNs = timestampNs
        }

        val dtSec = ((timestampNs - lastSampleTimestampNs) * 1e-9).coerceIn(0.0, 0.2)
        lastSampleTimestampNs = timestampNs

        // Maintain rolling window for live stationarity
        rollingWindow.addLast(sample)
        if (rollingWindow.size > 50) {
            rollingWindow.removeFirst()
        }

        val isStationaryNow = evaluateStationarity(rollingWindow)

        if (isStationaryNow) {
            samples.add(sample)
            totalStationaryDurationSec += dtSec
        }

        // Checklist progression
        val sensorsOk = samples.size > 20
        val gravityOk = totalStationaryDurationSec >= 3.5
        val gyroOk = totalStationaryDurationSec >= 7.5
        val alignmentOk = totalStationaryDurationSec >= 12.0

        val progress = (totalStationaryDurationSec / TARGET_DURATION_SEC).toFloat().coerceIn(0f, 1f)
        val remainingSec = ceil(max(0.0, TARGET_DURATION_SEC - totalStationaryDurationSec)).toInt()

        val status = when {
            !isStationaryNow -> CalibrationStatus.PAUSED_MOTION
            totalStationaryDurationSec >= TARGET_DURATION_SEC && samples.size < 600 -> CalibrationStatus.INSUFFICIENT_DATA
            else -> CalibrationStatus.COLLECTING
        }

        val statusText = when {
            !isStationaryNow -> "Please keep the vehicle stationary for calibration."
            totalStationaryDurationSec >= TARGET_DURATION_SEC && samples.size < 600 -> "Calibration needs a little more data"
            totalStationaryDurationSec < 3.5 -> "Sensors connected, detecting gravity..."
            totalStationaryDurationSec < 7.5 -> "Estimating gyroscope drift bias..."
            totalStationaryDurationSec < 12.0 -> "Calculating vehicle leveling horizon..."
            else -> "Finalizing alignment reference..."
        }

        val shouldEmitState = (timestampNs - lastStateEmitTimeNs) >= 100_000_000L ||
                              remainingSec != _state.value.remainingSec ||
                              status != _state.value.status

        if (shouldEmitState) {
            lastStateEmitTimeNs = timestampNs
            _state.value = _state.value.copy(
                status = status,
                progress = progress,
                elapsedStationarySec = totalStationaryDurationSec,
                remainingSec = remainingSec,
                isStationary = isStationaryNow,
                samplesCollected = samples.size,
                statusText = statusText,
                checklistSensors = sensorsOk,
                checklistGravity = gravityOk,
                checklistGyro = gyroOk,
                checklistAlignment = alignmentOk
            )
        }

        // Check completion condition: full target duration reached, sufficient samples, and stationary
        if (totalStationaryDurationSec >= TARGET_DURATION_SEC && samples.size >= 800 && isStationaryNow) {
            finalizeCalibration()
        } else if (totalStationaryDurationSec >= MAX_COLLECTION_TIMEOUT_SEC && samples.size >= 300) {
            // Safety fallback at maximum timeout
            finalizeCalibration()
        }
    }

    private fun evaluateStationarity(window: Collection<Sample>): Boolean {
        if (window.size < 10) return true

        var sumAcc = 0.0
        var sumAccSq = 0.0
        var maxGyroNorm = 0.0

        for (s in window) {
            val accNorm = sqrt((s.ax * s.ax + s.ay * s.ay + s.az * s.az).toDouble())
            sumAcc += accNorm
            sumAccSq += accNorm * accNorm

            val gyroNorm = sqrt((s.gx * s.gx + s.gy * s.gy + s.gz * s.gz).toDouble())
            if (gyroNorm > maxGyroNorm) maxGyroNorm = gyroNorm
        }

        val n = window.size.toDouble()
        val meanAcc = sumAcc / n
        val accVar = (sumAccSq / n) - (meanAcc * meanAcc)
        val accStd = sqrt(max(0.0, accVar))

        // Stationarity conditions:
        // 1. Acceleration norm close to 1g (9.80665 m/s^2 +/- 0.50)
        // 2. Acceleration standard deviation < 0.22 m/s^2
        // 3. Gyroscope norm < 0.15 rad/s (~8.6 deg/s)
        val accNear1g = abs(meanAcc - G_REF) < 0.50
        val accQuiet = accStd < 0.22
        val gyroQuiet = maxGyroNorm < 0.15

        return accNear1g && accQuiet && gyroQuiet
    }

    @Synchronized
    private fun finalizeCalibration() {
        if (isCompleted) return
        isCalibrating = false
        isCompleted = true

        val n = samples.size.toDouble()
        if (n < 200) {
            Log.w(TAG, "Calibration aborted: insufficient samples ($n)")
            _state.value = _state.value.copy(
                status = CalibrationStatus.INSUFFICIENT_DATA,
                statusText = "Calibration needs a little more data"
            )
            isCompleted = false
            isCalibrating = true
            return
        }

        // 1. Gyroscope Bias computation: bg = 1/N * sum(g_i)
        var sumGx = 0.0
        var sumGy = 0.0
        var sumGz = 0.0
        var sumGxSq = 0.0
        var sumGySq = 0.0
        var sumGzSq = 0.0

        // 2. Accelerometer Mean: a_mean = 1/N * sum(a_i)
        var sumAx = 0.0
        var sumAy = 0.0
        var sumAz = 0.0
        var sumAxSq = 0.0
        var sumAySq = 0.0
        var sumAzSq = 0.0

        for (s in samples) {
            sumGx += s.gx
            sumGy += s.gy
            sumGz += s.gz
            sumGxSq += s.gx * s.gx
            sumGySq += s.gy * s.gy
            sumGzSq += s.gz * s.gz

            sumAx += s.ax
            sumAy += s.ay
            sumAz += s.az
            sumAxSq += s.ax * s.ax
            sumAySq += s.ay * s.ay
            sumAzSq += s.az * s.az
        }

        val bgX = sumGx / n
        val bgY = sumGy / n
        val bgZ = sumGz / n

        val varGx = max(0.0, (sumGxSq / n) - (bgX * bgX))
        val varGy = max(0.0, (sumGySq / n) - (bgY * bgY))
        val varGz = max(0.0, (sumGzSq / n) - (bgZ * bgZ))
        val gyroStdDev = sqrt(varGx + varGy + varGz)

        val axMean = sumAx / n
        val ayMean = sumAy / n
        val azMean = sumAz / n

        val varAx = max(0.0, (sumAxSq / n) - (axMean * axMean))
        val varAy = max(0.0, (sumAySq / n) - (ayMean * ayMean))
        val varAz = max(0.0, (sumAzSq / n) - (azMean * azMean))
        val accelVar = varAx + varAy + varAz

        val gMag = sqrt(axMean * axMean + ayMean * ayMean + azMean * azMean)

        // Attitude from gravity: uz = -a_mean / |a_mean|
        val uzX = -axMean / gMag
        val uzY = -ayMean / gMag
        val uzZ = -azMean / gMag

        val pitchRad = asin((-uzX).coerceIn(-1.0, 1.0))
        val rollRad = atan2(uzY, -uzZ)
        val pitchDeg = Math.toDegrees(pitchRad)
        val rollDeg = Math.toDegrees(rollRad)

        // Quality and confidence scores in [0, 1]
        val gyroConf = (1.0 - (gyroStdDev / 0.03)).coerceIn(0.0, 1.0)
        val accelConf = (1.0 - (abs(gMag - G_REF) / 0.40)).coerceIn(0.0, 1.0)
        val stationaryConf = (1.0 - (sqrt(accelVar) / 0.15)).coerceIn(0.0, 1.0)
        val alignmentConf = (0.5 * accelConf + 0.5 * gyroConf).coerceIn(0.0, 1.0)

        val result = CalibrationResult(
            durationSec = totalStationaryDurationSec,
            sampleCount = samples.size,
            gyroBiasX = bgX,
            gyroBiasY = bgY,
            gyroBiasZ = bgZ,
            gyroStdDev = gyroStdDev,
            meanAccelX = axMean,
            meanAccelY = ayMean,
            meanAccelZ = azMean,
            gravityMagnitude = gMag,
            accelVariance = accelVar,
            pitchDeg = pitchDeg,
            rollDeg = rollDeg,
            stationaryConfidence = stationaryConf,
            gyroConfidence = gyroConf,
            accelConfidence = accelConf,
            alignmentConfidence = alignmentConf
        )

        // Apply static calibration to the native navigation engine
        NativeBridge.nativeApplyStaticCalibration(
            axMean, ayMean, azMean,
            bgX, bgY, bgZ
        )

        Log.i(TAG, "=== 15s STARTUP CALIBRATION COMPLETE ===")
        Log.i(TAG, "[AFTER CALIBRATION] Duration: %.2f s | Samples: %d".format(totalStationaryDurationSec, samples.size))
        Log.i(TAG, "[AFTER CALIBRATION] Gyro Bias: [%.6f, %.6f, %.6f] rad/s (StdDev: %.5f)".format(bgX, bgY, bgZ, gyroStdDev))
        Log.i(TAG, "[AFTER CALIBRATION] Gravity Vector: [%.4f, %.4f, %.4f] m/s^2 (|g|: %.4f)".format(axMean, ayMean, azMean, gMag))
        Log.i(TAG, "[AFTER CALIBRATION] Attitude: Pitch = %.2f°, Roll = %.2f°".format(pitchDeg, rollDeg))
        Log.i(TAG, "[AFTER CALIBRATION] Alignment Confidence: %.1f%% (Gyro: %.1f%%, Accel: %.1f%%)".format(
            alignmentConf * 100, gyroConf * 100, accelConf * 100
        ))

        calibratedAtTimestampMs = System.currentTimeMillis()

        _state.value = _state.value.copy(
            status = CalibrationStatus.COMPLETED,
            progress = 1.0f,
            remainingSec = 0,
            statusText = "Calibration Complete",
            checklistSensors = true,
            checklistGravity = true,
            checklistGyro = true,
            checklistAlignment = true,
            result = result
        )
    }

    fun isComplete(): Boolean = isCompleted
    fun getLatestResult(): CalibrationResult? = _state.value.result
}
