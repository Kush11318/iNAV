package com.inav.navigation.obd

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.util.Log
import kotlinx.coroutines.*
import java.io.InputStream
import java.io.OutputStream
import java.util.UUID

/**
 * OBD-II Bluetooth CAN-Bus Wheel Speed Integration (PID 010D).
 * Fuses true physical wheel speed into the 15-state ES-EKF measurement update.
 */
class ObdSpeedManager(
    private val onSpeedReceived: (speedKmh: Double) -> Unit,
    private val onConnectionChanged: (connected: Boolean, deviceName: String?) -> Unit
) {
    companion object {
        private const val TAG = "ObdSpeedManager"
        private val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }

    private var socket: BluetoothSocket? = null
    private var inputStream: InputStream? = null
    private var outputStream: OutputStream? = null

    private var isRunning = false
    private var job: Job? = null

    var isSimulated: Boolean = false
        private set

    fun toggleSimulated(enabled: Boolean) {
        isSimulated = enabled
        if (isSimulated) {
            onConnectionChanged(true, "Simulated CAN-Bus (ELM327)")
        } else if (socket == null || !socket!!.isConnected) {
            onConnectionChanged(false, null)
        }
    }

    fun startPolling(scope: CoroutineScope) {
        isRunning = true
        job = scope.launch(Dispatchers.IO) {
            while (isRunning) {
                if (isSimulated) {
                    // Simulate normal vehicle cruising speed with slight noise
                    val simulatedSpeed = 45.0 + (Math.sin(System.currentTimeMillis() / 2000.0) * 3.0)
                    onSpeedReceived(simulatedSpeed)
                } else if (socket != null && socket!!.isConnected) {
                    try {
                        // Send OBD-II PID 010D (Current Vehicle Speed)
                        outputStream?.write("010D\r".toByteArray())
                        outputStream?.flush()

                        val buffer = ByteArray(64)
                        val bytes = inputStream?.read(buffer) ?: 0
                        if (bytes > 0) {
                            val response = String(buffer, 0, bytes).trim()
                            val parsedSpeed = parseSpeedResponse(response)
                            if (parsedSpeed != null) {
                                onSpeedReceived(parsedSpeed)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "OBD-II communication error: ${e.message}")
                        disconnect()
                    }
                }
                delay(100L) // 10 Hz wheel speed update
            }
        }
    }

    private fun parseSpeedResponse(raw: String): Double? {
        // Standard OBD response: "41 0D XX" where XX is hex speed in km/h
        val clean = raw.replace(" ", "").replace("\r", "").replace("\n", "")
        val idx = clean.indexOf("410D")
        if (idx != -1 && clean.length >= idx + 6) {
            val hexSpeed = clean.substring(idx + 4, idx + 6)
            return hexSpeed.toIntOrNull(16)?.toDouble()
        }
        return null
    }

    fun disconnect() {
        isRunning = false
        job?.cancel()
        try {
            socket?.close()
        } catch (ignored: Exception) {}
        socket = null
        onConnectionChanged(false, null)
    }
}
