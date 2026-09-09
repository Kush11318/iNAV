package com.inav.navigation

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

object NativeBridge {
    private const val TAG = "iNAV_NativeBridge"
    private var isLoaded = false

    init {
        try {
            System.loadLibrary("inav_core")
            isLoaded = true
            Log.i(TAG, "Successfully loaded libinav_core.so")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library: ${e.message}")
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
            nativeInit(modelFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX asset: ${e.message}")
            false
        }
    }

    external fun nativeInit(modelPath: String): Boolean
    external fun nativeReset(initLat: Double, initLon: Double, initSpeedMs: Double, initHeadingDeg: Double)
    external fun nativeUpdateGnss(lat: Double, lon: Double, speedMs: Double, headingDeg: Double)
    external fun nativeSetScaleFactor(scaleK: Double)
    external fun nativeProcessImu(
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
