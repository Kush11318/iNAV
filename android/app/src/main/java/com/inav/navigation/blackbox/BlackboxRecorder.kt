package com.inav.navigation.blackbox

import android.content.Context
import android.os.Environment
import android.util.Log
import com.inav.navigation.model.NavigationState
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * Blackbox Flight Telemetry Recorder & Google Earth 3D KML Exporter.
 * Logs ground truth, road-snapped iNAV trajectory, and naive strapdown Ghost drift.
 */
object BlackboxRecorder {

    private const val TAG = "BlackboxRecorder"

    data class TelemetryPoint(
        val timestampMs: Long,
        val lat: Double,
        val lon: Double,
        val altM: Double,
        val speedKmh: Double,
        val headingDeg: Double,
        val isOutage: Boolean,
        val isSnapped: Boolean,
        val ghostLat: Double,
        val ghostLon: Double
    )

    private val records = Collections.synchronizedList(mutableListOf<TelemetryPoint>())
    private var lastRecordTimeMs = 0L

    fun record(state: NavigationState) {
        val now = System.currentTimeMillis()
        // Record at max 2 Hz to keep KML clean and fast
        if (now - lastRecordTimeMs < 500L) return
        lastRecordTimeMs = now

        val activeLat = if (state.isRoadSnapped && state.snappedLatitude != 0.0) state.snappedLatitude else state.latitude
        val activeLon = if (state.isRoadSnapped && state.snappedLongitude != 0.0) state.snappedLongitude else state.longitude

        if (activeLat == 0.0 || activeLon == 0.0) return

        records.add(
            TelemetryPoint(
                timestampMs = now,
                lat = activeLat,
                lon = activeLon,
                altM = state.altitudeM,
                speedKmh = state.speedKmh,
                headingDeg = state.headingDeg,
                isOutage = state.isOutageSimulated,
                isSnapped = state.isRoadSnapped,
                ghostLat = state.naiveLatitude,
                ghostLon = state.naiveLongitude
            )
        )

        // Keep last 10,000 points (~1.5 hours)
        if (records.size > 10000) {
            records.removeAt(0)
        }
    }

    fun getRecordCount(): Int = records.size

    fun exportKml(context: Context): File? {
        if (records.isEmpty()) return null

        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val filename = "iNAV_mission_${sdf.format(Date())}.kml"

        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val file = if (downloadsDir.exists() && downloadsDir.canWrite()) {
            File(downloadsDir, filename)
        } else {
            File(context.getExternalFilesDir(null), filename)
        }

        try {
            FileWriter(file).use { writer ->
                writer.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
                writer.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n")
                writer.write("  <Document>\n")
                writer.write("    <name>iNAV Vehicular Mission Telemetry</name>\n")
                writer.write("    <description>Side-by-side export: iNAV Dead-Reckoning vs Naive Unconstrained Ghost Drift</description>\n\n")

                // Styles
                // iNAV Fused Track: Cyan (AABBGGRR: ff00e5ff)
                writer.write("    <Style id=\"inavStyle\">\n")
                writer.write("      <LineStyle>\n")
                writer.write("        <color>ffe50000</color>\n") // Blue/cyan in KML
                writer.write("        <width>5</width>\n")
                writer.write("      </LineStyle>\n")
                writer.write("    </Style>\n")

                // Ghost Track: Semi-transparent Red (AABBGGRR: 990000ff)
                writer.write("    <Style id=\"ghostStyle\">\n")
                writer.write("      <LineStyle>\n")
                writer.write("        <color>990000ff</color>\n")
                writer.write("        <width>3</width>\n")
                writer.write("      </LineStyle>\n")
                writer.write("    </Style>\n\n")

                // Folder: iNAV Road Snapped / Fused Track
                writer.write("    <Folder>\n")
                writer.write("      <name>iNAV Fused Road-Snapped Track</name>\n")
                writer.write("      <Placemark>\n")
                writer.write("        <name>iNAV Trajectory</name>\n")
                writer.write("        <styleUrl>#inavStyle</styleUrl>\n")
                writer.write("        <LineString>\n")
                writer.write("          <tessellate>1</tessellate>\n")
                writer.write("          <altitudeMode>relativeToGround</altitudeMode>\n")
                writer.write("          <coordinates>\n")

                synchronized(records) {
                    for (pt in records) {
                        writer.write("            ${pt.lon},${pt.lat},${String.format(Locale.US, "%.1f", pt.altM)}\n")
                    }
                }

                writer.write("          </coordinates>\n")
                writer.write("        </LineString>\n")
                writer.write("      </Placemark>\n")
                writer.write("    </Folder>\n\n")

                // Folder: Ghost Unconstrained Track (Only during outage)
                writer.write("    <Folder>\n")
                writer.write("      <name>Naive Unconstrained Strapdown Ghost Drift</name>\n")
                writer.write("      <Placemark>\n")
                writer.write("        <name>Ghost Drift Trajectory</name>\n")
                writer.write("        <styleUrl>#ghostStyle</styleUrl>\n")
                writer.write("        <LineString>\n")
                writer.write("          <tessellate>1</tessellate>\n")
                writer.write("          <altitudeMode>relativeToGround</altitudeMode>\n")
                writer.write("          <coordinates>\n")

                synchronized(records) {
                    for (pt in records) {
                        if (pt.isOutage && pt.ghostLat != 0.0 && pt.ghostLon != 0.0) {
                            writer.write("            ${pt.ghostLon},${pt.ghostLat},${String.format(Locale.US, "%.1f", pt.altM)}\n")
                        }
                    }
                }

                writer.write("          </coordinates>\n")
                writer.write("        </LineString>\n")
                writer.write("      </Placemark>\n")
                writer.write("    </Folder>\n")

                writer.write("  </Document>\n")
                writer.write("</kml>\n")
            }
            Log.i(TAG, "Successfully exported mission KML: ${file.absolutePath}")
            return file
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting KML: ${e.message}")
            return null
        }
    }
}
