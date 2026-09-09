package com.inav.navigation.mapmatching

import kotlin.math.*

data class RoadNode(
    val lat: Double,
    val lon: Double
)

data class RoadSegment(
    val start: RoadNode,
    val end: RoadNode,
    val wayId: Long,
    val name: String,
    val highwayType: String
) {
    // Road bearing from start to end in degrees [0, 360)
    val bearingDeg: Double by lazy {
        val lat1 = Math.toRadians(start.lat)
        val lat2 = Math.toRadians(end.lat)
        val dLon = Math.toRadians(end.lon - start.lon)
        val y = sin(dLon) * cos(lat2)
        val x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
        val deg = Math.toDegrees(atan2(y, x))
        (deg + 360.0) % 360.0
    }

    // Segment length in meters
    val lengthM: Double by lazy {
        haversineM(start.lat, start.lon, end.lat, end.lon)
    }

    companion object {
        fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val r = 6371000.0 // Earth radius in meters
            val dLat = Math.toRadians(lat2 - lat1)
            val dLon = Math.toRadians(lon2 - lon1)
            val a = sin(dLat / 2).pow(2) +
                    cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                    sin(dLon / 2).pow(2)
            val c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return r * c
        }
    }
}

data class SnappedLocation(
    val lat: Double,
    val lon: Double,
    val headingDeg: Double,
    val roadName: String,
    val distanceToRoadM: Double,
    val isSnapped: Boolean,
    val segment: RoadSegment?
)
