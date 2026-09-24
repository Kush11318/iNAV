package com.inav.navigation

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

object NativeBridge {
    private const val TAG = "iNAV_NativeBridge"
    var isLoaded = false
        private set

    init {
        try {
            System.loadLibrary("c++_shared")
        } catch (t: Throwable) {
            // Optional when statically linked
        }
        try {
            System.loadLibrary("inav_core")
            isLoaded = true
            Log.i(TAG, "Successfully loaded libinav_core.so")
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load libinav_core.so: ${t.message}")
            isLoaded = false
        }
    }

    /**
     * Copy the ONNX model from assets to internal storage and initialize native engine.
     */
    fun initializeWithAsset(context: Context, assetName: String = "velocity_net.onnx"): Boolean {
        if (!isLoaded) return false
        return try {
            val modelFile = File(context.filesDir, assetName)
            if (!modelFile.exists() || modelFile.length() == 0L) {
                context.assets.open(assetName).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "Copied $assetName to ${modelFile.absolutePath} (${modelFile.length()} bytes)")
            }
            jniInit(modelFile.absolutePath)
        } catch (t: Throwable) {
            Log.e(TAG, "Error initializing ONNX asset: ${t.message}")
            false
        }
    }

    fun nativeReset(initLat: Double, initLon: Double, initSpeedMs: Double, initHeadingDeg: Double) {
        if (!isLoaded) return
        try {
            jniReset(initLat, initLon, initSpeedMs, initHeadingDeg)
        } catch (t: Throwable) {
            Log.e(TAG, "nativeReset caught: ${t.message}")
        }
    }

    fun nativeUpdateGnss(
        lat: Double,
        lon: Double,
        speedMs: Double,
        headingDeg: Double,
        accuracyM: Double,
        timestampNs: Long
    ): Boolean {
        if (!isLoaded) return false
        return try {
            jniUpdateGnss(lat, lon, speedMs, headingDeg, accuracyM, timestampNs)
        } catch (t: Throwable) {
            Log.e(TAG, "nativeUpdateGnss caught: ${t.message}")
            false
        }
    }

    fun nativeGetGnssHealthState(): Int {
        if (!isLoaded) return 0
        return try {
            jniGetGnssHealthState()
        } catch (t: Throwable) {
            Log.e(TAG, "nativeGetGnssHealthState caught: ${t.message}")
            0
        }
    }

    fun nativeSetScaleFactor(scaleK: Double) {
        if (!isLoaded) return
        try {
            jniSetScaleFactor(scaleK)
        } catch (t: Throwable) {
            Log.e(TAG, "nativeSetScaleFactor caught: ${t.message}")
        }
    }

    fun nativeProcessImu(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        qw: Float, qx: Float, qy: Float, qz: Float,
        dt: Double,
        isStationary: Boolean,
        gnssSpeedMs: Double,
        isGnssHealthy: Boolean,
        obdSpeedMs: Double,
        isObdConnected: Boolean
    ): DoubleArray? {
        if (!isLoaded) return null
        return try {
            jniProcessImu(
                ax, ay, az,
                gx, gy, gz,
                qw, qx, qy, qz,
                dt,
                isStationary,
                gnssSpeedMs,
                isGnssHealthy,
                obdSpeedMs,
                isObdConnected
            )
        } catch (t: Throwable) {
            Log.e(TAG, "nativeProcessImu caught: ${t.message}")
            null
        }
    }

    fun nativeApplyStaticCalibration(
        axMean: Double, ayMean: Double, azMean: Double,
        gxBias: Double, gyBias: Double, gzBias: Double
    ) {
        if (!isLoaded) return
        try {
            jniApplyStaticCalibration(axMean, ayMean, azMean, gxBias, gyBias, gzBias)
        } catch (t: Throwable) {
            Log.e(TAG, "nativeApplyStaticCalibration caught: ${t.message}")
        }
    }

    fun nativeGetAlignmentStatus(): DoubleArray? {
        if (!isLoaded) return null
        return try {
            jniGetAlignmentStatus()
        } catch (t: Throwable) {
            Log.e(TAG, "nativeGetAlignmentStatus caught: ${t.message}")
            null
        }
    }

    // Raw JNI endpoints (wrapped by safe public Kotlin methods above)
    @JvmStatic private external fun jniInit(modelPath: String): Boolean
    @JvmStatic private external fun jniReset(initLat: Double, initLon: Double, initSpeedMs: Double, initHeadingDeg: Double)
    @JvmStatic private external fun jniApplyStaticCalibration(
        axMean: Double, ayMean: Double, azMean: Double,
        gxBias: Double, gyBias: Double, gzBias: Double
    )
    @JvmStatic private external fun jniGetAlignmentStatus(): DoubleArray
    @JvmStatic private external fun jniUpdateGnss(
        lat: Double,
        lon: Double,
        speedMs: Double,
        headingDeg: Double,
        accuracyM: Double,
        timestampNs: Long
    ): Boolean
    @JvmStatic private external fun jniGetGnssHealthState(): Int
    @JvmStatic private external fun jniSetScaleFactor(scaleK: Double)
    @JvmStatic private external fun jniProcessImu(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        qw: Float, qx: Float, qy: Float, qz: Float,
        dt: Double,
        isStationary: Boolean,
        gnssSpeedMs: Double,
        isGnssHealthy: Boolean,
        obdSpeedMs: Double,
        isObdConnected: Boolean
    ): DoubleArray?
}
