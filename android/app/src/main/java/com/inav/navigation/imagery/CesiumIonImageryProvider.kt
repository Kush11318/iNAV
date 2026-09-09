package com.inav.navigation.imagery

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.OnlineTileSourceBase
import org.osmdroid.tileprovider.tilesource.TileSourcePolicy
import org.osmdroid.tileprovider.tilesource.XYTileSource
import org.osmdroid.util.MapTileIndex
import java.net.HttpURLConnection
import java.net.URL

/**
 * Cesium Ion Imagery & Satellite Layer Provider for iNAV.
 *
 * Provides:
 * 1. Cesium Ion REST API Integration (/v1/assets/{assetId}/endpoint)
 * 2. High-Resolution Bing Aerial (Asset 2) & Bing Hybrid (Asset 3) via Ion
 * 3. Sentinel-2 Cloudless Satellite (Asset 3954)
 * 4. Instant Keyless Esri World Satellite Imagery (ArcGIS fallback matching God's Eye View stack)
 * 5. Automatic Bearer token injection and credential lifecycle management
 */
object CesiumIonImageryProvider {

    private const val TAG = "CesiumIonImagery"
    private const val PREFS_NAME = "inav_prefs"
    private const val KEY_CESIUM_TOKEN = "cesium_ion_token"
    private const val CESIUM_API_BASE = "https://api.cesium.com/v1"

    // Well-Known Cesium Ion Asset IDs
    const val ASSET_BING_AERIAL = 2
    const val ASSET_BING_HYBRID = 3
    const val ASSET_SENTINEL2 = 3954

    data class IonEndpointResult(
        val type: String,
        val url: String,
        val accessToken: String,
        val attribution: String,
        val isTms: Boolean = true
    )

    enum class ImageryLayerType {
        STREET_MAP,
        SATELLITE_ESRI,
        CESIUM_ION_AERIAL,
        CESIUM_ION_HYBRID
    }

    private var activeIonSessionToken: String? = null

    fun getSavedToken(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_CESIUM_TOKEN, "")?.trim() ?: ""
    }

    fun saveToken(context: Context, token: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_CESIUM_TOKEN, token.trim()).apply()
    }

    fun hasValidToken(context: Context): Boolean {
        return getSavedToken(context).isNotEmpty()
    }

    /**
     * Verifies if a given token is valid against Cesium Ion API.
     */
    suspend fun validateToken(token: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val url = URL("$CESIUM_API_BASE/assets?limit=1")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                setRequestProperty("Authorization", "Bearer ${token.trim()}")
                connectTimeout = 8000
                readTimeout = 8000
            }

            val code = conn.responseCode
            if (code == 200) {
                Result.success("Token verified successfully")
            } else {
                Result.failure(Exception("Cesium Ion returned HTTP $code: Invalid token or insufficient scopes"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Token validation error", e)
            Result.failure(e)
        }
    }

    /**
     * Fetches the dynamic streaming endpoint for an asset from Cesium Ion.
     */
    suspend fun fetchAssetEndpoint(token: String, assetId: Int): Result<IonEndpointResult> = withContext(Dispatchers.IO) {
        try {
            val url = URL("$CESIUM_API_BASE/assets/$assetId/endpoint")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                setRequestProperty("Authorization", "Bearer ${token.trim()}")
                connectTimeout = 8000
                readTimeout = 8000
            }

            val code = conn.responseCode
            if (code != 200) {
                return@withContext Result.failure(Exception("Failed to fetch endpoint for asset $assetId (HTTP $code)"))
            }

            val responseBody = conn.inputStream.bufferedReader().use { it.readText() }
            val json = JSONObject(responseBody)
            val type = json.optString("type", "IMAGERY")
            val endpointUrl = json.getString("url")
            val accessToken = json.optString("accessToken", token)

            var attribution = "© Cesium ion"
            val attributionsArray = json.optJSONArray("attributions")
            if (attributionsArray != null && attributionsArray.length() > 0) {
                attribution = attributionsArray.getJSONObject(0).optString("html", attribution)
            }

            Result.success(
                IonEndpointResult(
                    type = type,
                    url = endpointUrl,
                    accessToken = accessToken,
                    attribution = attribution
                )
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to resolve Cesium Ion endpoint for asset $assetId", e)
            Result.failure(e)
        }
    }

    /**
     * Builds an OSMDroid tile source backed by Cesium Ion imagery.
     */
    fun createCesiumIonTileSource(endpoint: IonEndpointResult, assetId: Int): OnlineTileSourceBase {
        activeIonSessionToken = endpoint.accessToken

        // Inject Authorization header for all tile requests
        Configuration.getInstance().additionalHttpRequestProperties["Authorization"] =
            "Bearer ${endpoint.accessToken}"

        val base = if (endpoint.url.endsWith("/")) endpoint.url else "${endpoint.url}/"

        return object : OnlineTileSourceBase(
            "CesiumIon_$assetId",
            0,
            19,
            256,
            ".png",
            arrayOf(base),
            endpoint.attribution,
            TileSourcePolicy(8, 0)
        ) {
            override fun getTileURLString(pMapTileIndex: Long): String {
                val zoom = MapTileIndex.getZoom(pMapTileIndex)
                val x = MapTileIndex.getX(pMapTileIndex)
                // TMS specification uses inverted Y axis: y_tms = (1 shl zoom) - 1 - y_osm
                val yTms = (1 shl zoom) - 1 - MapTileIndex.getY(pMapTileIndex)
                return "$baseUrl$zoom/$x/$yTms.png"
            }
        }
    }

    /**
     * Keyless Esri World Imagery (ArcGIS Online) satellite tile source.
     * Direct equivalent of God's Eye View's default satellite stack.
     * Works with 0 configuration.
     */
    fun createEsriSatelliteTileSource(): OnlineTileSourceBase {
        // Clear any leftover Ion Authorization header to avoid rejected tile requests
        Configuration.getInstance().additionalHttpRequestProperties.remove("Authorization")

        return object : OnlineTileSourceBase(
            "EsriWorldImagery",
            0,
            19,
            256,
            "",
            arrayOf(
                "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/",
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
            ),
            "Powered by Esri — Source: Esri, Maxar, Earthstar Geographics",
            TileSourcePolicy(8, 0)
        ) {
            override fun getTileURLString(pMapTileIndex: Long): String {
                val zoom = MapTileIndex.getZoom(pMapTileIndex)
                val y = MapTileIndex.getY(pMapTileIndex)
                val x = MapTileIndex.getX(pMapTileIndex)
                // ArcGIS MapServer tile format is: {baseUrl}{z}/{y}/{x}
                return "$baseUrl$zoom/$y/$x"
            }
        }
    }

    /**
     * Clears tile authorization when returning to standard OSM or dark tiles.
     */
    fun clearAuthHeader() {
        Configuration.getInstance().additionalHttpRequestProperties.remove("Authorization")
        activeIonSessionToken = null
    }
}
