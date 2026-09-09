package com.inav.navigation.pdr

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * @brief Pedestrian Dead Reckoning (PDR) Step Detector & Dynamic Stride Estimator
 * Implements Kalman/low-pass smoothed peak-to-trough detection and the Weinberg
 * dynamic stride length model for GPS-denied pedestrian navigation (forests, trails, indoor).
 */
class PedestrianStepEngine {

    companion object {
        private const val MIN_STEP_INTERVAL_MS = 260L   // Max ~3.8 steps/sec
        private const val THRESHOLD_HIGH = 11.2         // Upward acceleration peak
        private const val THRESHOLD_LOW = 8.6           // Downward compression trough
        private const val WEINBERG_K = 0.44             // Stride scaling constant
    }

    private var stepCount = 0
    private var totalDistanceM = 0.0
    private var lastStepTimestampMs = 0L

    private var isWaitingForTrough = false
    private var currentPeakAcc = 9.81
    private var currentTroughAcc = 9.81

    // 1-D Kalman Filter for accelerometer smoothing
    private var kalmanEstimate = 9.81
    private var kalmanErrorCov = 1.0
    private val processNoiseQ = 0.02
    private val measurementNoiseR = 0.15

    data class StepResult(
        val isStep: Boolean,
        val strideLengthM: Double,
        val totalSteps: Int,
        val totalDistanceM: Double,
        val cadenceSpm: Int
    )

    fun processSample(
        ax: Float, ay: Float, az: Float,
        timestampMs: Long
    ): StepResult {
        val rawMagnitude = sqrt((ax * ax + ay * ay + az * az).toDouble())

        // 1-D Kalman smoothing
        kalmanErrorCov += processNoiseQ
        val kalmanGain = kalmanErrorCov / (kalmanErrorCov + measurementNoiseR)
        kalmanEstimate += kalmanGain * (rawMagnitude - kalmanEstimate)
        kalmanErrorCov *= (1.0 - kalmanGain)

        val filteredAcc = kalmanEstimate

        // Track local peak and trough
        currentPeakAcc = max(currentPeakAcc, filteredAcc)
        currentTroughAcc = min(currentTroughAcc, filteredAcc)

        var stepDetected = false
        var strideLength = 0.75

        if (!isWaitingForTrough) {
            if (filteredAcc > THRESHOLD_HIGH) {
                isWaitingForTrough = true
                currentPeakAcc = filteredAcc
            }
        } else {
            if (filteredAcc < THRESHOLD_LOW) {
                val dt = timestampMs - lastStepTimestampMs
                if (dt >= MIN_STEP_INTERVAL_MS) {
                    stepDetected = true
                    lastStepTimestampMs = timestampMs
                    stepCount++

                    // Weinberg Dynamic Stride Length Model:
                    // Stride = k * (a_max - a_min)^(1/4)
                    val bounceRange = max(0.5, currentPeakAcc - currentTroughAcc)
                    strideLength = (WEINBERG_K * Math.pow(bounceRange, 0.25)).coerceIn(0.50, 1.05)
                    totalDistanceM += strideLength
                }
                isWaitingForTrough = false
                currentPeakAcc = 9.81
                currentTroughAcc = 9.81
            }
        }

        val cadence = if (lastStepTimestampMs > 0 && timestampMs - lastStepTimestampMs < 3000L) {
            val recentDt = max(300L, timestampMs - lastStepTimestampMs)
            (60000L / recentDt).toInt().coerceIn(40, 180)
        } else {
            0
        }

        return StepResult(
            isStep = stepDetected,
            strideLengthM = strideLength,
            totalSteps = stepCount,
            totalDistanceM = totalDistanceM,
            cadenceSpm = cadence
        )
    }

    fun reset() {
        stepCount = 0
        totalDistanceM = 0.0
        lastStepTimestampMs = 0L
        isWaitingForTrough = false
        currentPeakAcc = 9.81
        currentTroughAcc = 9.81
        kalmanEstimate = 9.81
        kalmanErrorCov = 1.0
    }
}
