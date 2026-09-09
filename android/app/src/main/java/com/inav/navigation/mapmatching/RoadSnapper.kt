package com.inav.navigation.mapmatching

import kotlin.math.*

class RoadSnapper(private val roadNetworkManager: RoadNetworkManager) {

    private var lastSnappedSegment: RoadSegment? = null
    private var lastSnappedWayId: Long? = null
    private var lastSnappedLat: Double = 0.0
    private var lastSnappedLon: Double = 0.0
    private var lastSnappedHeading: Double = 0.0

    fun reset() {
        lastSnappedSegment = null
        lastSnappedWayId = null
        lastSnappedLat = 0.0
        lastSnappedLon = 0.0
        lastSnappedHeading = 0.0
    }

    fun snap(
        rawLat: Double,
        rawLon: Double,
        rawHeadingDeg: Double,
        speedKmh: Double
    ): SnappedLocation {
        // Fallback if no valid location
        if (rawLat == 0.0 && rawLon == 0.0) {
            return SnappedLocation(
                lat = rawLat,
                lon = rawLon,
                headingDeg = rawHeadingDeg,
                roadName = "No Location",
                distanceToRoadM = 0.0,
                isSnapped = false,
                segment = null
            )
        }

        // Distance from previous snapped anchor
        val latMidRad = Math.toRadians(rawLat)
        val mPerLat = 111132.954 - 559.822 * cos(2 * latMidRad)
        val mPerLon = 111412.84 * cos(latMidRad)
        val distFromSnapM = if (lastSnappedLat != 0.0) {
            val dx = (rawLon - lastSnappedLon) * mPerLon
            val dy = (rawLat - lastSnappedLat) * mPerLat
            sqrt(dx * dx + dy * dy)
        } else 0.0

        // When stationary or walking slowly (< 2.0 km/h), DO NOT snap to road centerline!
        // This ensures the blue dot sits at the user's true home/building, not on a nearby street.
        if (speedKmh < 2.0) {
            return SnappedLocation(
                lat = rawLat,
                lon = rawLon,
                headingDeg = rawHeadingDeg,
                roadName = "Current Location",
                distanceToRoadM = 0.0,
                isSnapped = false,
                segment = null
            )
        }

        // Search for nearby road segments within 60 meters
        val candidates = roadNetworkManager.getCandidateSegments(rawLat, rawLon, radiusM = 60.0)
        if (candidates.isEmpty()) {
            return SnappedLocation(
                lat = rawLat,
                lon = rawLon,
                headingDeg = rawHeadingDeg,
                roadName = if (roadNetworkManager.hasRoads()) "Off-Road" else "Loading Road Graph...",
                distanceToRoadM = 999.0,
                isSnapped = false,
                segment = null
            )
        }

        var bestSegment: RoadSegment? = null
        var bestProjLat = rawLat
        var bestProjLon = rawLon
        var bestAlignedHeading = rawHeadingDeg
        var bestDistanceM = Double.MAX_VALUE
        var minCost = Double.MAX_VALUE

        for (seg in candidates) {
            // Local Cartesian orthogonal projection onto segment AB
            val sx = (seg.end.lon - seg.start.lon) * mPerLon
            val sy = (seg.end.lat - seg.start.lat) * mPerLat
            val segLenSq = sx * sx + sy * sy
            if (segLenSq < 1.0) continue // Skip micro-segments

            val px = (rawLon - seg.start.lon) * mPerLon
            val py = (rawLat - seg.start.lat) * mPerLat

            val t = ((px * sx + py * sy) / segLenSq).coerceIn(0.0, 1.0)

            val projLat = seg.start.lat + t * (seg.end.lat - seg.start.lat)
            val projLon = seg.start.lon + t * (seg.end.lon - seg.start.lon)

            val distM = RoadSegment.haversineM(rawLat, rawLon, projLat, projLon)

            // Segment orientation and bidirectional alignment
            val segHeading = seg.bearingDeg
            val diffForward = angleDifference(rawHeadingDeg, segHeading)
            val diffReverse = angleDifference(rawHeadingDeg, (segHeading + 180.0) % 360.0)

            val (alignedHeading, diffHeading) = if (diffForward <= diffReverse) {
                Pair(segHeading, diffForward)
            } else {
                Pair((segHeading + 180.0) % 360.0, diffReverse)
            }

            // Speed-scaled heading angle weighting (Pillar 5 Roadmap)
            // Low speed (<15 km/h): w_phi = 0.15 (turns/reversing possible)
            // Mid speed (15-50 km/h): w_phi = 0.4 (urban street guidance)
            // High speed (>50 km/h): w_phi = 0.8 (highway strict tangent enforcement)
            val wPhi = when {
                speedKmh > 50.0 -> 0.80
                speedKmh > 15.0 -> 0.40
                else -> 0.15
            }

            // Cost: orthogonal distance in meters + speed-weighted heading penalty
            var cost = distM + (diffHeading * wPhi)

            // Hysteresis bonus: prioritize continuing on the current road
            if (seg.wayId == lastSnappedWayId) {
                cost -= 8.0
            }

            if (cost < minCost) {
                minCost = cost
                bestDistanceM = distM
                bestSegment = seg
                bestProjLat = projLat
                bestProjLon = projLon
                bestAlignedHeading = alignedHeading
            }
        }

        // Snapping decision threshold:
        // Must be close to the road (<= 18m) and moving along the road direction (<= 45 deg).
        // Only allow low-speed retention if already locked to the exact same road segment.
        val bestDiffHeading = angleDifference(rawHeadingDeg, bestAlignedHeading)
        val isContinuingSameRoad = (lastSnappedSegment != null && bestSegment?.wayId == lastSnappedWayId && bestDistanceM <= 15.0)
        val canSnap = bestSegment != null && bestDistanceM <= 18.0 && (bestDiffHeading <= 45.0 || isContinuingSameRoad)

        return if (canSnap && bestSegment != null) {
            lastSnappedSegment = bestSegment
            lastSnappedWayId = bestSegment.wayId
            lastSnappedLat = bestProjLat
            lastSnappedLon = bestProjLon
            lastSnappedHeading = bestAlignedHeading

            SnappedLocation(
                lat = bestProjLat,
                lon = bestProjLon,
                headingDeg = bestAlignedHeading,
                roadName = bestSegment.name,
                distanceToRoadM = bestDistanceM,
                isSnapped = true,
                segment = bestSegment
            )
        } else {
            // Off-road fallback: use dead reckoning position directly
            SnappedLocation(
                lat = rawLat,
                lon = rawLon,
                headingDeg = rawHeadingDeg,
                roadName = if (bestDistanceM < 50.0) "Near ${bestSegment?.name ?: "Road"}" else "Off-Road Area",
                distanceToRoadM = bestDistanceM,
                isSnapped = false,
                segment = null
            )
        }
    }

    private fun angleDifference(a: Double, b: Double): Double {
        val diff = abs(a - b) % 360.0
        return if (diff > 180.0) 360.0 - diff else diff
    }
}
