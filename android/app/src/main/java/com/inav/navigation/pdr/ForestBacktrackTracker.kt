package com.inav.navigation.pdr

import kotlin.math.*

/**
 * @brief Forest Survival Backtrack Tracker
 * Computes return azimuth and straight-line range back to the origin
 * (where the hiker entered the forest or where GPS signal was lost).
 */
class ForestBacktrackTracker {

    private var originLat = 0.0
    private var originLon = 0.0
    private var originAltitudeM = 0.0
    private var isAnchorSet = false

    data class BacktrackInfo(
        val isAnchorActive: Boolean,
        val originLat: Double,
        val originLon: Double,
        val returnBearingDeg: Double,
        val returnDistanceM: Double,
        val returnCardinal: String,
        val elevationDeltaM: Double
    )

    fun setOrigin(lat: Double, lon: Double, altitudeM: Double) {
        if (lat != 0.0 && lon != 0.0) {
            originLat = lat
            originLon = lon
            originAltitudeM = altitudeM
            isAnchorSet = true
        }
    }

    fun computeBacktrack(currentLat: Double, currentLon: Double, currentAltitudeM: Double): BacktrackInfo {
        if (!isAnchorSet || currentLat == 0.0 || currentLon == 0.0) {
            return BacktrackInfo(
                isAnchorActive = false,
                originLat = 0.0,
                originLon = 0.0,
                returnBearingDeg = 0.0,
                returnDistanceM = 0.0,
                returnCardinal = "N",
                elevationDeltaM = 0.0
            )
        }

        val distM = haversineM(currentLat, currentLon, originLat, originLon)
        val bearingDeg = initialBearingDeg(currentLat, currentLon, originLat, originLon)
        val cardinal = bearingToCardinal(bearingDeg)
        val elevDelta = currentAltitudeM - originAltitudeM

        return BacktrackInfo(
            isAnchorActive = true,
            originLat = originLat,
            originLon = originLon,
            returnBearingDeg = bearingDeg,
            returnDistanceM = distM,
            returnCardinal = cardinal,
            elevationDeltaM = elevDelta
        )
    }

    fun reset() {
        originLat = 0.0
        originLon = 0.0
        originAltitudeM = 0.0
        isAnchorSet = false
    }

    companion object {
        fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val r = 6371000.0
            val dLat = Math.toRadians(lat2 - lat1)
            val dLon = Math.toRadians(lon2 - lon1)
            val a = sin(dLat / 2).pow(2) + cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
            val c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return r * c
        }

        fun initialBearingDeg(fromLat: Double, fromLon: Double, toLat: Double, toLon: Double): Double {
            val phi1 = Math.toRadians(fromLat)
            val phi2 = Math.toRadians(toLat)
            val deltaLambda = Math.toRadians(toLon - fromLon)

            val y = sin(deltaLambda) * cos(phi2)
            val x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(deltaLambda)
            val thetaRad = atan2(y, x)
            var bearing = Math.toDegrees(thetaRad)
            if (bearing < 0) bearing += 360.0
            return bearing
        }

        fun bearingToCardinal(bearingDeg: Double): String {
            val directions = arrayOf("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
            val index = ((bearingDeg + 11.25) % 360.0 / 22.5).toInt().coerceIn(0, 15)
            return directions[index]
        }
    }
}
