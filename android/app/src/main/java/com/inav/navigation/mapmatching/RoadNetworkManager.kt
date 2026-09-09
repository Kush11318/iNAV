package com.inav.navigation.mapmatching

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.abs

class RoadNetworkManager(private val context: Context) {

    private val segments = CopyOnWriteArrayList<RoadSegment>()
    private var lastFetchLat: Double = 0.0
    private var lastFetchLon: Double = 0.0
    private var isFetching = false

    private val cacheFile: File
        get() = File(context.cacheDir, "osm_roads_cache.json")

    init {
        // Attempt to load from offline cache on startup
        loadFromCache()
    }

    fun ensureRoadNetwork(lat: Double, lon: Double, scope: CoroutineScope) {
        if (lat == 0.0 && lon == 0.0) return

        val distSinceLast = RoadSegment.haversineM(lat, lon, lastFetchLat, lastFetchLon)
        if (segments.isEmpty() || (distSinceLast > 600.0 && !isFetching)) {
            scope.launch(Dispatchers.IO) {
                fetchRoadsFromOverpass(lat, lon)
            }
        }
    }

    private fun loadFromCache() {
        try {
            if (cacheFile.exists() && cacheFile.length() > 0) {
                val jsonStr = cacheFile.readText()
                parseOsmJson(jsonStr)
                Log.i(TAG, "Loaded ${segments.size} road segments from disk cache")
            } else {
                // Try reading pre-bundled assets
                try {
                    context.assets.open("osm_roads_cache.json").use { input ->
                        val jsonStr = input.bufferedReader().use { it.readText() }
                        parseOsmJson(jsonStr)
                        Log.i(TAG, "Loaded ${segments.size} road segments from pre-bundled assets")
                    }
                } catch (e: Exception) {
                    Log.d(TAG, "No pre-bundled road assets found: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed loading cache: ${e.message}")
        }
    }

    private fun fetchRoadsFromOverpass(lat: Double, lon: Double) {
        if (isFetching) return
        isFetching = true
        Log.i(TAG, "Fetching road network around ($lat, $lon)...")

        try {
            val query = """
                [out:json][timeout:15];
                way["highway"](around:1000,$lat,$lon);
                out geom;
            """.trimIndent()

            val postData = "data=" + URLEncoder.encode(query, "UTF-8")
            val url = URL("https://overpass-api.de/api/interpreter")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                setRequestProperty("User-Agent", "iNAV-Android/1.0 (dead-reckoning)")
                setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
            }

            conn.outputStream.use { os ->
                os.write(postData.toByteArray(Charsets.UTF_8))
                os.flush()
            }

            val responseCode = conn.responseCode
            if (responseCode == 200) {
                val responseText = conn.inputStream.bufferedReader().use { it.readText() }
                // Save to offline cache
                cacheFile.writeText(responseText)
                parseOsmJson(responseText)
                lastFetchLat = lat
                lastFetchLon = lon
                Log.i(TAG, "Successfully updated road graph: ${segments.size} segments cached")
            } else {
                Log.w(TAG, "Overpass API returned HTTP $responseCode")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not fetch Overpass roads online (using offline cache if available): ${e.message}")
        } finally {
            isFetching = false
        }
    }

    private fun parseOsmJson(jsonStr: String) {
        try {
            val root = JSONObject(jsonStr)
            val elements = root.optJSONArray("elements") ?: return

            val newSegments = mutableListOf<RoadSegment>()

            for (i in 0 until elements.length()) {
                val elem = elements.getJSONObject(i)
                val type = elem.optString("type")
                if (type != "way") continue

                val wayId = elem.optLong("id")
                val tags = elem.optJSONObject("tags")
                val roadName = tags?.optString("name")?.ifBlank { null }
                    ?: tags?.optString("ref")?.ifBlank { null }
                    ?: tags?.optString("highway")?.replace('_', ' ')?.replaceFirstChar { it.uppercase() }
                    ?: "Street"

                val hwType = tags?.optString("highway") ?: "residential"
                val geom = elem.optJSONArray("geometry") ?: continue

                if (geom.length() < 2) continue

                for (j in 0 until geom.length() - 1) {
                    val p1 = geom.getJSONObject(j)
                    val p2 = geom.getJSONObject(j + 1)
                    val nodeA = RoadNode(p1.getDouble("lat"), p1.getDouble("lon"))
                    val nodeB = RoadNode(p2.getDouble("lat"), p2.getDouble("lon"))
                    newSegments.add(
                        RoadSegment(
                            start = nodeA,
                            end = nodeB,
                            wayId = wayId,
                            name = roadName,
                            highwayType = hwType
                        )
                    )
                }
            }

            if (newSegments.isNotEmpty()) {
                segments.clear()
                segments.addAll(newSegments)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing OSM road json: ${e.message}")
        }
    }

    /**
     * Get candidate road segments near coordinate using rough bounding-box filter
     */
    fun getCandidateSegments(lat: Double, lon: Double, radiusM: Double = 80.0): List<RoadSegment> {
        val dLat = radiusM / 111000.0
        val dLon = radiusM / (111000.0 * kotlin.math.cos(Math.toRadians(lat)))

        return segments.filter { seg ->
            val minLat = minOf(seg.start.lat, seg.end.lat) - dLat
            val maxLat = maxOf(seg.start.lat, seg.end.lat) + dLat
            val minLon = minOf(seg.start.lon, seg.end.lon) - dLon
            val maxLon = maxOf(seg.start.lon, seg.end.lon) + dLon
            lat in minLat..maxLat && lon in minLon..maxLon
        }
    }

    fun hasRoads(): Boolean = segments.isNotEmpty()

    companion object {
        private const val TAG = "RoadNetworkManager"
    }
}
