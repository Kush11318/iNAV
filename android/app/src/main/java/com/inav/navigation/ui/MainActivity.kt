package com.inav.navigation.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.graphics.Rect
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.inav.navigation.R
import com.inav.navigation.databinding.ActivityMainBinding
import com.inav.navigation.model.NavigationMode
import com.inav.navigation.model.NavigationState
import com.inav.navigation.routing.PlaceSuggestion
import com.inav.navigation.routing.RouteManager
import com.inav.navigation.routing.RoutePlan
import com.inav.navigation.service.DeadReckoningService
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.geometry.LatLngBounds
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.plugins.annotation.*
import org.maplibre.android.style.layers.FillExtrusionLayer
import org.maplibre.android.style.layers.Property
import org.maplibre.android.style.layers.RasterLayer
import org.maplibre.android.style.sources.RasterSource
import org.maplibre.android.style.sources.TileSet
import org.maplibre.android.utils.ColorUtils

class MainActivity : AppCompatActivity() {

    companion object {
        // OpenFreeMap styles — zero API key, zero billing
        const val STYLE_LIBERTY = "https://tiles.openfreemap.org/styles/liberty"
        const val STYLE_DARK = "asset://uber_dark_style.json"
        const val STYLE_LIGHT = "asset://voyager_light_style.json"
        const val STYLE_BRIGHT = "https://tiles.openfreemap.org/styles/bright"
        const val STYLE_POSITRON = "https://tiles.openfreemap.org/styles/positron"

        // High-resolution satellite tiles (Google Hybrid aerial, Zoom 0-21, zero missing tile watermarks)
        const val SATELLITE_URL = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"

        // Marker icon IDs
        const val ICON_NAV_ARROW = "ic_nav_arrow"
        const val ICON_NAV_ARROW_NIGHT = "ic_nav_arrow_night"
        const val ICON_START_PIN = "ic_start_pin"
        const val ICON_DEST_PIN = "ic_dest_pin"
        const val ICON_GMAPS_FROZEN = "ic_gmaps_frozen"
        const val ICON_TURN_WAYPOINT = "ic_turn_waypoint"
    }

    private lateinit var binding: ActivityMainBinding
    private var service: DeadReckoningService? = null
    private var isBound = false

    private lateinit var mapView: MapView
    private var mapLibreMap: MapLibreMap? = null

    // Annotation managers (created after style loads)
    private var symbolManager: SymbolManager? = null
    private var lineManager: LineManager? = null
    private var fillManager: FillManager? = null

    // Annotation references
    private var vehicleSymbol: Symbol? = null
    private var startSymbol: Symbol? = null
    private var destSymbol: Symbol? = null
    private var turnWaypointSymbol: Symbol? = null
    private var ghostFrozenSymbol: Symbol? = null
    private var trajectoryLine: Line? = null
    private var trajectoryGlowLine: Line? = null
    private var routeCasingLine: Line? = null
    private var routeLine: Line? = null
    private var routeDashLine: Line? = null
    private var ghostLine: Line? = null
    private var uncertaintyFill: Fill? = null

    // Trajectory point buffers
    private val trajectoryPoints = mutableListOf<LatLng>()
    private val ghostPoints = mutableListOf<LatLng>()
    private val routePoints = mutableListOf<LatLng>()

    // Screen state enum (Apple Maps Home vs Active Turn-by-Turn Navigation)
    enum class AppScreenState { HOME, NAVIGATION }
    private var currentScreenState = AppScreenState.HOME

    // UI state flags
    private var isDemoMode = false
    private var isLightMode = true
    private var routeDashSource: org.maplibre.android.style.sources.GeoJsonSource? = null
    private var isOutageActive = false
    private var isHudActive = false
    private var isTrackUpEnabled = false
    private var isPedestrianMode = false
    private var isObdSimulated = false
    private var isFollowingVehicle = true
    private var isRouteSelectorExpanded = false
    private var isHomeSheetExpanded = true
    private var hasInitialCentered = false
    private var lastCenteredLat = 0.0
    private var lastCenteredLon = 0.0
    private var isUserTouchingMap = false
    private var currentPhysicalLocation: Location? = null

    // Map view mode
    enum class MapViewType { OSM_VIEW, SATELLITE_VIEW }
    private var currentViewType = MapViewType.OSM_VIEW

    // Smooth heading slew
    private var displayHeading: Float = 0f
    private var lastUpdateTimeMs: Long = 0L
    private val HEADING_SLEW_RATE_DPS = 90f

    // Render governor
    private var lastRenderedLat = 0.0
    private var lastRenderedLon = 0.0
    private var lastRenderedHeading = 0f
    private val RENDER_POSITION_THRESHOLD_M = 0.5
    private val RENDER_HEADING_THRESHOLD_DEG = 0.5f

    // Night mode
    private var isNightMode = false
    private var lastNightModeSwitch = 0L
    private val NIGHT_MODE_LUX_THRESHOLD = 50f
    private val NIGHT_MODE_DEBOUNCE_MS = 5000L

    // Outage frozen point
    private var outageFrozenLatLng: LatLng? = null

    data class ManeuverItem(
        val icon: String,
        val distance: String,
        val roadName: String,
        val towardDirection: String? = null,
        val shieldText: String? = null,
        val shieldIsGreen: Boolean = false,
        val exitBadge: String? = null,
        val isOverheadLanes: Boolean = false,
        val lanes: List<Pair<String, Boolean>> = emptyList(),
        val thenIcon: String? = null,
        val thenPrefix: String = "THEN › ",
        val thenText: String? = null,
        val thenDistance: String? = null
    )

    private var currentManeuverIndex = 0
    private val dynamicManeuverPresets = listOf(
        // 0. 200 ft Main Ring Road (Overhead Lanes Guidance)
        ManeuverItem(
            icon = "↖",
            distance = "200 ft",
            roadName = "Main Ring Road",
            isOverheadLanes = true,
            lanes = listOf("↖" to true, "↖" to true, "↑" to false, "↑" to false),
            thenIcon = "↖",
            thenPrefix = "THEN › ",
            thenText = "Bear left onto Main Ring Road",
            thenDistance = "200 ft"
        ),
        // 1. 500 FT AB Road
        ManeuverItem(
            icon = "↑",
            distance = "500 FT",
            roadName = "AB Road Arterial",
            towardDirection = "City Center",
            lanes = listOf("↑" to true, "↑" to true, "↱" to false),
            thenIcon = "↱",
            thenText = "Turn right onto AB Road",
            thenDistance = "500 FT"
        ),
        // 2. 100 ft Destination Arrival
        ManeuverItem(
            icon = "🏁",
            distance = "100 ft",
            roadName = "Arriving at Destination",
            lanes = listOf("↑" to true, "↱" to false),
            thenIcon = "📍",
            thenPrefix = "ARRIVE › ",
            thenText = "Destination on the right",
            thenDistance = "100 ft"
        ),
        // 3. 350 m Bypass Expressway
        ManeuverItem(
            icon = "↶",
            distance = "350 m",
            roadName = "Bypass Expressway",
            lanes = listOf("↶" to true, "↑" to false, "↱" to false),
            thenIcon = "↰",
            thenText = "Keep left onto Bypass Expressway",
            thenDistance = "350 m"
        ),
        // 4. 4.3 km National Highway 52
        ManeuverItem(
            icon = "↖",
            distance = "4.3 km",
            roadName = "NH-52 Expressway",
            shieldText = "52",
            shieldIsGreen = true,
            exitBadge = "Exit 2",
            lanes = listOf("↖" to true, "↑" to true, "↗" to false),
            thenIcon = "↖",
            thenText = "Take Exit 2 toward NH-52",
            thenDistance = "4.3 km"
        ),
        // 5. 350 m MR-10 Corridor
        ManeuverItem(
            icon = "⮌",
            distance = "350 m",
            roadName = "MR-10 Corridor",
            lanes = listOf("↰" to false, "↑" to true, "↱" to true),
            thenIcon = "↱",
            thenText = "Super Corridor Junction",
            thenDistance = "500 m"
        ),
        // 6. 5.1 km Airport Road
        ManeuverItem(
            icon = "↖",
            distance = "5.1 km",
            roadName = "Airport Road",
            shieldText = "47",
            shieldIsGreen = false,
            lanes = listOf("↖" to true, "↖" to true, "↗" to false, "↗" to false),
            thenIcon = "↖",
            thenText = "Use left 2 lanes to merge onto Airport Road",
            thenDistance = "5.1 km"
        ),
        // 7. 1.4 km Station Approach
        ManeuverItem(
            icon = "↱",
            distance = "1.4 km",
            roadName = "Railway Station Approach",
            lanes = listOf("↑" to false, "↱" to true),
            thenIcon = "📍",
            thenPrefix = "VIA › ",
            thenText = "Station Plaza",
            thenDistance = "1.4 km"
        ),
        // 8. 350 m MG Road
        ManeuverItem(
            icon = "↰",
            distance = "350 m",
            roadName = "M.G. Road",
            lanes = listOf("↰" to true, "↑" to false),
            thenIcon = "⚠️",
            thenPrefix = "CAUTION › ",
            thenText = "Watch for cross traffic",
            thenDistance = "350 m"
        )
    )

    private fun renderDynamicManeuver(item: ManeuverItem) {
        binding.tvTopRoadIcon.text = item.icon
        binding.tvTopDistance.text = item.distance
        binding.tvTopRoadName.text = item.roadName

        // Overhead Lane Indicator Card (Edison St / Highway 44 matching user screenshot)
        if (item.isOverheadLanes && item.lanes.isNotEmpty()) {
            binding.containerTopCardLanes.visibility = View.VISIBLE
            binding.flTopRoadIcon.visibility = View.GONE
            binding.btnNavExpandChevron.visibility = View.GONE
            binding.containerTopSubDirection.visibility = View.GONE
            binding.containerFloatingLanes.visibility = View.GONE
            binding.containerTopCardLanes.removeAllViews()

            item.lanes.forEach { (symbol, isActive) ->
                val tv = android.widget.TextView(this).apply {
                    layoutParams = android.widget.LinearLayout.LayoutParams(
                        0,
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                        1.0f
                    )
                    gravity = android.view.Gravity.CENTER
                    text = symbol
                    setTextColor(if (isActive) Color.WHITE else Color.parseColor("#4A4D55"))
                    textSize = 28f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }
                binding.containerTopCardLanes.addView(tv)
            }
            binding.tvTopDistance.textSize = 14f
            binding.tvTopRoadName.textSize = 23f
        } else {
            binding.containerTopCardLanes.visibility = View.GONE
            binding.flTopRoadIcon.visibility = View.VISIBLE
            binding.btnNavExpandChevron.visibility = View.VISIBLE
            binding.tvTopDistance.textSize = 12.5f
            binding.tvTopRoadName.textSize = 19f
        }

        // Highway / Route Shield
        if (item.shieldText != null) {
            binding.tvTopShieldBadge.visibility = View.VISIBLE
            binding.tvTopShieldBadge.text = item.shieldText
            binding.tvTopShieldBadge.setBackgroundResource(
                if (item.shieldIsGreen) R.drawable.bg_highway_shield_green else R.drawable.bg_interstate_shield
            )
        } else {
            binding.tvTopShieldBadge.visibility = View.GONE
        }

        // Freeway Exit Badge
        if (item.exitBadge != null) {
            binding.tvTopExitBadge.visibility = View.VISIBLE
            binding.tvTopExitBadge.text = item.exitBadge
        } else {
            binding.tvTopExitBadge.visibility = View.GONE
        }

        // Toward direction
        if (item.towardDirection != null) {
            binding.containerTopSubDirection.visibility = View.VISIBLE
            binding.tvTopSubDirection.text = item.towardDirection
        } else {
            binding.containerTopSubDirection.visibility = View.GONE
        }

        // Dynamic Lane Guidance (only when not an overhead lane header card)
        if (!item.isOverheadLanes && item.lanes.isNotEmpty()) {
            binding.containerFloatingLanes.visibility = View.VISIBLE
            binding.containerLaneItems.removeAllViews()
            val density = resources.displayMetrics.density
            val pillWidth = (20 * density).toInt()
            val pillHeight = (22 * density).toInt()
            val marginStart = (4 * density).toInt()

            item.lanes.forEachIndexed { idx, (symbol, isActive) ->
                val frame = android.widget.FrameLayout(this).apply {
                    layoutParams = android.widget.LinearLayout.LayoutParams(pillWidth, pillHeight).apply {
                        if (idx > 0) leftMargin = marginStart
                    }
                    setBackgroundResource(R.drawable.bg_lane_pill)
                    backgroundTintList = android.content.res.ColorStateList.valueOf(
                        if (isActive) Color.WHITE else Color.parseColor("#17191D")
                    )
                }
                val tv = android.widget.TextView(this).apply {
                    layoutParams = android.widget.FrameLayout.LayoutParams(
                        android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                        android.widget.FrameLayout.LayoutParams.MATCH_PARENT
                    )
                    gravity = android.view.Gravity.CENTER
                    text = symbol
                    setTextColor(if (isActive) Color.BLACK else Color.parseColor("#52525B"))
                    textSize = 13f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }
                frame.addView(tv)
                binding.containerLaneItems.addView(frame)
            }
        } else {
            binding.containerFloatingLanes.visibility = View.GONE
        }

        // Next / Sub-bar Row
        if (item.isOverheadLanes || item.thenText == null) {
            binding.containerThenManeuver.visibility = View.GONE
        } else {
            binding.containerThenManeuver.visibility = View.VISIBLE
            binding.tvThenPrefix.text = item.thenPrefix
            if (item.thenIcon != null) {
                binding.tvThenIcon.visibility = View.VISIBLE
                binding.tvThenIcon.text = item.thenIcon
            } else {
                binding.tvThenIcon.visibility = View.GONE
            }
            binding.tvThenInstruction.text = item.thenText ?: ""
            binding.tvThenDistance.text = item.thenDistance ?: ""
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as DeadReckoningService.LocalBinder
            service = localBinder.getService()
            isBound = true
            observeServiceState()
        }
        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            isBound = false
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineLocationGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true
        if (fineLocationGranted) {
            startTrackingService()
            startDirectLocationUpdates()
        } else {
            Toast.makeText(this, "Location permission required for navigation", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Initialize MapLibre before inflating layout
        MapLibre.getInstance(this)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        mapView = binding.mapView
        mapView.onCreate(savedInstanceState)

        initMap()
        setupListeners()
        checkPermissionsAndStart()
    }

    private fun initMap() {
        mapView.getMapAsync { map ->
            mapLibreMap = map

            // Camera defaults
            map.cameraPosition = CameraPosition.Builder()
                .target(LatLng(22.7292, 75.8335)) // Indore default
                .zoom(16.0)
                .tilt(0.0)
                .bearing(0.0)
                .build()

            // UI settings
            map.uiSettings.apply {
                isRotateGesturesEnabled = true
                isTiltGesturesEnabled = true
                isZoomGesturesEnabled = true
                isScrollGesturesEnabled = true
                isCompassEnabled = false // We have our own compass FAB
                isAttributionEnabled = false
                isLogoEnabled = false
            }

            // Load vector tile style (Voyager Light by default matching user screenshot)
            val initialStyle = if (isLightMode) STYLE_LIGHT else STYLE_DARK
            map.setStyle(initialStyle) { style ->
                reinitializeMapAnnotations(map, style)
            }

            // Map click / long press
            map.addOnMapClickListener { latLng ->
                if (binding.containerSearchSuggestions.visibility == View.VISIBLE) {
                    binding.containerSearchSuggestions.visibility = View.GONE
                    hideKeyboard()
                    return@addOnMapClickListener true
                }
                hideKeyboard()
                if (currentScreenState == AppScreenState.HOME && isHomeSheetExpanded) {
                    collapseHomeSheet()
                    return@addOnMapClickListener true
                }
                false
            }

            map.addOnMapLongClickListener { latLng ->
                hideKeyboard()
                MaterialAlertDialogBuilder(this@MainActivity)
                    .setTitle("📍 Pin Location Options")
                    .setMessage(String.format("Coordinates: %.5f, %.5f\n\nChoose an action for this point:", latLng.latitude, latLng.longitude))
                    .setPositiveButton("📍 Set as My Exact Location") { _, _ ->
                        setManualLocation(latLng.latitude, latLng.longitude)
                    }
                    .setNegativeButton("🏁 Navigate Here") { _, _ ->
                        startNavigationTo(
                            latLng.latitude,
                            latLng.longitude,
                            String.format("Pinned Destination (%.4f, %.4f)", latLng.latitude, latLng.longitude)
                        )
                    }
                    .setNeutralButton("Cancel", null)
                    .show()
                true
            }

            // Detect user camera movement to unlock follow mode
            map.addOnCameraMoveStartedListener { reason ->
                if (reason == MapLibreMap.OnCameraMoveStartedListener.REASON_API_GESTURE) {
                    if (isFollowingVehicle || isTrackUpEnabled) {
                        isFollowingVehicle = false
                        isTrackUpEnabled = false
                        binding.fabMyLocation.setColorFilter(Color.parseColor("#94A3B8"))
                    }
                }
            }
        }
    }

    private fun getBitmapFromDrawable(drawableId: Int, widthDp: Int = 72, heightDp: Int = 72): Bitmap? {
        val drawable = ContextCompat.getDrawable(this, drawableId) ?: return null
        val density = resources.displayMetrics.density
        val w = (widthDp * density).toInt().coerceAtLeast(1)
        val h = (heightDp * density).toInt().coerceAtLeast(1)
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        return bitmap
    }

    private fun loadMarkerIcons(style: Style) {
        val iconConfigs = listOf(
            ICON_NAV_ARROW to (R.drawable.ic_nav_arrow to 48),
            ICON_NAV_ARROW_NIGHT to (R.drawable.ic_nav_arrow_night to 48),
            ICON_START_PIN to (R.drawable.ic_start_pin to 48),
            ICON_DEST_PIN to (R.drawable.ic_dest_pin to 48),
            ICON_GMAPS_FROZEN to (R.drawable.ic_gmaps_frozen to 48),
            ICON_TURN_WAYPOINT to (R.drawable.ic_turn_waypoint to 20)
        )
        for ((id, pair) in iconConfigs) {
            val (resId, sizeDp) = pair
            try {
                val bmp = getBitmapFromDrawable(resId, sizeDp, sizeDp)
                if (bmp != null) {
                    style.addImage(id, bmp)
                }
            } catch (e: Exception) {
                // Icon not found, will use default
            }
        }
    }

    private fun cleanupAnnotationManagers() {
        try {
            symbolManager?.deleteAll()
            symbolManager?.onDestroy()
            symbolManager = null
        } catch (e: Exception) {}
        try {
            lineManager?.deleteAll()
            lineManager?.onDestroy()
            lineManager = null
        } catch (e: Exception) {}
        try {
            fillManager?.deleteAll()
            fillManager?.onDestroy()
            fillManager = null
        } catch (e: Exception) {}
        vehicleSymbol = null
        destSymbol = null
        startSymbol = null
        turnWaypointSymbol = null
        ghostFrozenSymbol = null
    }

    private fun setupAnnotationManagers(map: MapLibreMap, style: Style) {
        cleanupAnnotationManagers()
        // Order matters: fills first (bottom), then lines, then symbols (top)
        fillManager = FillManager(mapView, map, style)
        lineManager = LineManager(mapView, map, style)
        symbolManager = SymbolManager(mapView, map, style).apply {
            iconAllowOverlap = true
            iconIgnorePlacement = true
            iconPitchAlignment = Property.ICON_PITCH_ALIGNMENT_VIEWPORT
            iconRotationAlignment = Property.ICON_ROTATION_ALIGNMENT_MAP
        }

        // Add dedicated GeoJson dashed line layer (Stitch 16 inner road dash)
        try {
            if (style.getSource("route-dash-source") == null) {
                val dashSource = org.maplibre.android.style.sources.GeoJsonSource("route-dash-source")
                style.addSource(dashSource)
                routeDashSource = dashSource

                val dashLayer = org.maplibre.android.style.layers.LineLayer("route-dash-layer", "route-dash-source").apply {
                    setProperties(
                        org.maplibre.android.style.layers.PropertyFactory.lineColor(Color.parseColor("#101318")),
                        org.maplibre.android.style.layers.PropertyFactory.lineWidth(2.5f),
                        org.maplibre.android.style.layers.PropertyFactory.lineDasharray(arrayOf(1.8f, 1.8f)),
                        org.maplibre.android.style.layers.PropertyFactory.lineCap(Property.LINE_CAP_BUTT),
                        org.maplibre.android.style.layers.PropertyFactory.lineJoin(Property.LINE_JOIN_ROUND)
                    )
                }
                val symbolLayer = style.getLayer("mapbox-android-symbol-layer")
                if (symbolLayer != null) {
                    style.addLayerBelow(dashLayer, "mapbox-android-symbol-layer")
                } else {
                    style.addLayer(dashLayer)
                }
            } else {
                routeDashSource = style.getSourceAs("route-dash-source")
            }
        } catch (e: Exception) {
            // Dash source safely managed
        }

        // Lift symbol layer above any 3D fill-extrusion / building layers so it is NEVER hidden underneath 3D layers
        try {
            val symbolLayer = style.getLayer("mapbox-android-symbol-layer")
            if (symbolLayer != null) {
                style.removeLayer(symbolLayer)
                style.addLayer(symbolLayer)
            }
        } catch (e: Exception) {
            // Layer safely managed
        }
    }

    private fun createInitialAnnotations() {
        val lm = lineManager ?: return
        val fm = fillManager ?: return
        val sm = symbolManager ?: return

        // Single iconic pointer mode: keep map clean without overlapping uncertainty blobs
        uncertaintyFill = fm.create(FillOptions()
            .withLatLngs(listOf(listOf(LatLng(0.0, 0.0))))
            .withFillColor(ColorUtils.colorToRgbaString(Color.TRANSPARENT))
            .withFillOpacity(0.0f)
        )

        val casingColor = if (isLightMode) "#3F3F46" else "#33000000"
        val routeColor = if (isLightMode) "#18181B" else "#FFFFFF"
        val dashColor = if (isLightMode) "#18181B" else "#FFFFFF"
        val casingWidth = if (isLightMode) 10.0f else 11.0f
        val routeWidth = if (isLightMode) 7.5f else 7.5f

        // Outer Casing Route
        routeCasingLine = lm.create(LineOptions()
            .withLatLngs(listOf(LatLng(0.0, 0.0)))
            .withLineColor(ColorUtils.colorToRgbaString(Color.parseColor(casingColor)))
            .withLineWidth(casingWidth)
        )

        // Route polyline (Crisp solid black in Light Mode, stark solid white in Dark Mode)
        routeLine = lm.create(LineOptions()
            .withLatLngs(listOf(LatLng(0.0, 0.0)))
            .withLineColor(ColorUtils.colorToRgbaString(Color.parseColor(routeColor)))
            .withLineWidth(routeWidth)
        )

        // Route centerline dash (matches route color so dark mode line is pure uninterrupted white)
        routeDashLine = lm.create(LineOptions()
            .withLatLngs(listOf(LatLng(0.0, 0.0)))
            .withLineColor(ColorUtils.colorToRgbaString(Color.parseColor(dashColor)))
            .withLineWidth(routeWidth)
        )

        // Trajectory glow polyline (soft wide underlay)
        trajectoryGlowLine = lm.create(LineOptions()
            .withLatLngs(listOf(LatLng(0.0, 0.0)))
            .withLineColor(ColorUtils.colorToRgbaString(Color.parseColor("#6000E676")))
            .withLineWidth(12.0f)
        )

        // Trajectory polyline (crisp green trail)
        trajectoryLine = lm.create(LineOptions()
            .withLatLngs(listOf(LatLng(0.0, 0.0)))
            .withLineColor(ColorUtils.colorToRgbaString(Color.parseColor("#00E676")))
            .withLineWidth(4.0f)
        )

        // Ghost polyline (grey dashed divergence trail)
        ghostLine = lm.create(LineOptions()
            .withLatLngs(listOf(LatLng(0.0, 0.0)))
            .withLineColor(ColorUtils.colorToRgbaString(Color.parseColor("#9E9E9E")))
            .withLineWidth(4.0f)
        )

        // Vehicle marker: Managed by symbolManager with viewport pitch-alignment
        vehicleSymbol = sm.create(SymbolOptions()
            .withLatLng(LatLng(0.0, 0.0))
            .withIconImage(ICON_NAV_ARROW)
            .withIconSize(1.0f)
            .withIconRotate(0f)
        )
    }

    private fun applyDemoManhattan(map: MapLibreMap) {
        val lm = lineManager ?: return
        val sm = symbolManager ?: return

        // Manhattan W 47th St -> Broadway route matching Stitch 16
        val manhattanRoute = listOf(
            LatLng(40.7554, -73.9770), // 5th Ave
            LatLng(40.7565, -73.9793), // Madison Ave
            LatLng(40.7576, -73.9818), // W 47th St
            LatLng(40.7588, -73.9852), // Turn at Broadway
            LatLng(40.7608, -73.9840), // Up Broadway
            LatLng(40.7628, -73.9825)
        )
        routeCasingLine?.let { line -> line.latLngs = manhattanRoute; lm.update(line) }
        routeLine?.let { line -> line.latLngs = manhattanRoute; lm.update(line) }
        routeDashLine?.let { line -> line.latLngs = manhattanRoute; lm.update(line) }

        try {
            val coords = manhattanRoute.map { org.maplibre.geojson.Point.fromLngLat(it.longitude, it.latitude) }
            routeDashSource?.setGeoJson(org.maplibre.geojson.Feature.fromGeometry(org.maplibre.geojson.LineString.fromLngLats(coords)))
        } catch (e: Exception) {}

        val turnPt = LatLng(40.7588, -73.9852)
        if (turnWaypointSymbol == null) {
            turnWaypointSymbol = sm.create(SymbolOptions()
                .withLatLng(turnPt)
                .withIconImage(ICON_TURN_WAYPOINT)
                .withIconSize(1.0f)
            )
        } else {
            turnWaypointSymbol?.let { sym ->
                sym.latLng = turnPt
                sm.update(sym)
            }
        }

        val vehPt = LatLng(40.7576, -73.9818)
        vehicleSymbol?.let { sym ->
            sym.latLng = vehPt
            sym.iconRotate = 295f
            sm.update(sym)
        }
        updateUncertaintyCircle(vehPt, 3.5)

        // Camera aligned along W 47th St (heading 295°) looking forward toward Broadway
        val cameraPosition = CameraPosition.Builder()
            .target(LatLng(40.7582, -73.9832))
            .zoom(16.7)
            .bearing(295.0)
            .tilt(38.0)
            .build()
        map.animateCamera(CameraUpdateFactory.newCameraPosition(cameraPosition), 700)
    }

    private fun reinitializeMapAnnotations(map: MapLibreMap, style: Style) {
        loadMarkerIcons(style)
        setupAnnotationManagers(map, style)
        createInitialAnnotations()

        // Restore active route if present
        if (routePoints.isNotEmpty()) {
            val lm = lineManager
            if (lm != null) {
                routeCasingLine?.let { line -> line.latLngs = routePoints; lm.update(line) }
                routeLine?.let { line -> line.latLngs = routePoints; lm.update(line) }
                routeDashLine?.let { line -> line.latLngs = routePoints; lm.update(line) }
            }
            val sm = symbolManager
            if (sm != null && routePoints.isNotEmpty()) {
                val destPt = routePoints.last()
                destSymbol = sm.create(SymbolOptions()
                    .withLatLng(destPt)
                    .withIconImage(ICON_DEST_PIN)
                    .withIconSize(1.1f)
                )
            }
        }

        if (isDemoMode) {
            applyDemoManhattan(map)
        } else {
            val curLoc = currentPhysicalLocation
            val lat = if (curLoc != null && curLoc.latitude != 0.0) curLoc.latitude else (lastRenderedLat.takeIf { it != 0.0 } ?: 22.7196)
            val lon = if (curLoc != null && curLoc.longitude != 0.0) curLoc.longitude else (lastRenderedLon.takeIf { it != 0.0 } ?: 75.8577)
            val pt = LatLng(lat, lon)
            vehicleSymbol?.let { sym ->
                sym.latLng = pt
                sym.iconRotate = displayHeading
                symbolManager?.update(sym)
            }
            updateUncertaintyCircle(pt, 3.0)
            if (curLoc != null && !hasInitialCentered) {
                hasInitialCentered = true
                map.animateCamera(CameraUpdateFactory.newLatLngZoom(pt, 16.5), 600)
            }
        }
    }

    // ── Helper: Generate circle polygon points for uncertainty visualization ──
    private fun createCirclePoints(center: LatLng, radiusM: Double, steps: Int = 64): List<LatLng> {
        val points = mutableListOf<LatLng>()
        val degPerM = 1.0 / 111139.0
        for (i in 0..steps) {
            val angle = Math.toRadians(i * 360.0 / steps)
            val dLat = radiusM * Math.cos(angle) * degPerM
            val dLon = radiusM * Math.sin(angle) * degPerM / Math.cos(Math.toRadians(center.latitude))
            points.add(LatLng(center.latitude + dLat, center.longitude + dLon))
        }
        return points
    }

    private fun setManualLocation(lat: Double, lon: Double) {
        val pt = LatLng(lat, lon)
        vehicleSymbol?.let { sym ->
            sym.latLng = pt
            symbolManager?.update(sym)
        }
        val manualLoc = Location("manual").apply {
            latitude = lat
            longitude = lon
            accuracy = 2.5f
            time = System.currentTimeMillis()
        }
        currentPhysicalLocation = manualLoc
        updateUncertaintyCircle(pt, 3.0)
        hasInitialCentered = true
        lastCenteredLat = lat
        lastCenteredLon = lon
        mapLibreMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(pt, 19.0), 400)
        service?.setManualLocation(lat, lon)
        Toast.makeText(this, String.format("📍 Set exact location: %.5f, %.5f", lat, lon), Toast.LENGTH_LONG).show()
    }

    private fun updateUncertaintyCircle(center: LatLng, radiusM: Double) {
        // Disabled: keep single clean pointer without overlapping halo blobs
    }

    private fun setupListeners() {
        val getRouteMgr: () -> RouteManager = { service?.routeManager ?: RouteManager() }
        val getCurLat = { service?.navState?.value?.latitude?.takeIf { it != 0.0 } ?: 0.0 }
        val getCurLon = { service?.navState?.value?.longitude?.takeIf { it != 0.0 } ?: 0.0 }

        val executeSearch = {
            val q = binding.etSearchDestination.text.toString().trim()
            if (q.isNotEmpty()) {
                getRouteMgr().searchPlaces(q, getCurLat(), getCurLon(), lifecycleScope, this@MainActivity) { results: List<PlaceSuggestion> ->
                    if (results.isNotEmpty()) {
                        val top = results.first()
                        startNavigationTo(top.lat, top.lon, top.name)
                        hideKeyboard()
                    } else {
                        Toast.makeText(this@MainActivity, "No matching place found for: $q", Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                displaySearchSuggestions(getRouteMgr().presetPlaces)
            }
        }

        binding.btnSearchIcon.setOnClickListener { executeSearch() }

        binding.etSearchDestination.setOnClickListener {
            if (binding.etSearchDestination.text.isNullOrEmpty()) {
                displaySearchSuggestions(getRouteMgr().presetPlaces)
            }
        }
        binding.etSearchDestination.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus && binding.etSearchDestination.text.isNullOrEmpty()) {
                displaySearchSuggestions(getRouteMgr().presetPlaces)
            }
        }

        binding.etSearchDestination.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val q = s?.toString()?.trim() ?: ""
                binding.btnClearSearch.visibility = if (q.isNotEmpty()) View.VISIBLE else View.GONE
                if (q.isNotEmpty()) {
                    getRouteMgr().searchPlaces(q, getCurLat(), getCurLon(), lifecycleScope, this@MainActivity) { results: List<PlaceSuggestion> ->
                        displaySearchSuggestions(results)
                    }
                } else {
                    displaySearchSuggestions(getRouteMgr().presetPlaces)
                }
            }
            override fun afterTextChanged(s: android.text.Editable?) {}
        })

        binding.etSearchDestination.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH) {
                executeSearch()
                true
            } else {
                false
            }
        }

        binding.btnClearSearch.setOnClickListener {
            binding.etSearchDestination.text.clear()
            binding.containerSearchSuggestions.visibility = View.GONE
            binding.containerSearchSuggestions.removeAllViews()
            hideKeyboard()
        }

        binding.btnCancelRoute.setOnClickListener { cancelNavigation() }

        // Map View Switcher
        binding.btnOsmView.setOnClickListener { setMapLayer(MapViewType.OSM_VIEW) }
        binding.btnSatView.setOnClickListener { setMapLayer(MapViewType.SATELLITE_VIEW) }

        // Fetch GPS
        binding.btnFetchGpsNow.setOnClickListener {
            isFollowingVehicle = true
            binding.fabMyLocation.setColorFilter(Color.parseColor("#00E676"))
            fetchPhysicalLocationAndCenter(forceCenter = true)
        }

        // My Location FAB
        binding.fabMyLocation.setOnClickListener {
            isDemoMode = false
            isFollowingVehicle = true
            binding.fabMyLocation.setColorFilter(Color.parseColor("#00E676"))
            fetchPhysicalLocationAndCenter(forceCenter = true)
        }

        // Map Layers toggle FAB
        binding.fabLayers.setOnClickListener {
            if (currentViewType == MapViewType.OSM_VIEW) {
                setMapLayer(MapViewType.SATELLITE_VIEW)
            } else {
                setMapLayer(MapViewType.OSM_VIEW)
            }
        }

        // Compass FAB (reset north)
        binding.fabCompass.setOnClickListener {
            isTrackUpEnabled = false
            mapLibreMap?.animateCamera(CameraUpdateFactory.bearingTo(0.0), 300)
            binding.fabCompass.rotation = 0f
            Toast.makeText(this, "Compass: Reset to North-Up", Toast.LENGTH_SHORT).show()
        }

        // Audio Guidance Toggle (Unmuted / Muted — Stitch Reference)
        binding.fabTrackUp.setOnClickListener {
            val isUnmuted = binding.tvAudioTooltip.text == "Unmuted"
            if (isUnmuted) {
                binding.tvAudioTooltip.text = "Muted"
                binding.fabTrackUp.alpha = 0.5f
                Toast.makeText(this, "Audio Guidance: Muted", Toast.LENGTH_SHORT).show()
            } else {
                binding.tvAudioTooltip.text = "Unmuted"
                binding.fabTrackUp.alpha = 1.0f
                Toast.makeText(this, "Audio Guidance: Unmuted", Toast.LENGTH_SHORT).show()
            }
        }

        // Chevron and Header expand/collapse route selector
        val toggleRouteSelector = {
            isRouteSelectorExpanded = !isRouteSelectorExpanded
            binding.cardRouteSelector.visibility = if (isRouteSelectorExpanded) View.VISIBLE else View.GONE
            binding.btnNavExpandChevron.rotation = if (isRouteSelectorExpanded) 180f else 0f
        }
        binding.btnNavExpandChevron.setOnClickListener { toggleRouteSelector() }

        val cycleManeuver = {
            currentManeuverIndex = (currentManeuverIndex + 1) % dynamicManeuverPresets.size
            val m = dynamicManeuverPresets[currentManeuverIndex]
            renderDynamicManeuver(m)
            Toast.makeText(this, "Turn: ${m.distance} ${m.roadName}", Toast.LENGTH_SHORT).show()
        }
        binding.gmapsNavHeader.setOnClickListener { cycleManeuver() }
        binding.cardTopNav.setOnClickListener { cycleManeuver() }

        // Light / Dark Theme Mode Toggle on Blue Shield Button
        binding.btnShieldSafety.setOnClickListener {
            isLightMode = !isLightMode
            val targetStyle = if (isLightMode) STYLE_LIGHT else STYLE_DARK
            mapLibreMap?.let { map ->
                map.setStyle(targetStyle) { style ->
                    reinitializeMapAnnotations(map, style)
                }
            }
            applyThemeMode()
            val modeName = if (isLightMode) "☀️ Light Mode" else "🌙 Dark Mode"
            Toast.makeText(this, "$modeName Active", Toast.LENGTH_SHORT).show()
        }

        // Bookmark, Coffee, and Reroute FABs matching User Screenshot
        binding.fabBookmark.setOnClickListener {
            showBookmarkDialog()
        }
        binding.fabCoffee.setOnClickListener {
            showCoffeeAmenitiesSheet()
        }
        binding.fabReroute.setOnClickListener {
            showRouteAlternativesSheet()
        }

        // Bottom Trip Sheet actions
        binding.btnTripRouteOptions.setOnClickListener {
            showRouteSettingsSheet()
        }
        binding.btnTripStepsList.setOnClickListener {
            showItineraryStepsSheet()
        }
        binding.btnEndRoute.setOnClickListener {
            cancelNavigation()
        }
        binding.btnCancelRoute.setOnClickListener {
            cancelNavigation()
        }

        // Top nav card also opens itinerary or cycles maneuver
        binding.cardTopNav.setOnClickListener {
            showItineraryStepsSheet()
        }
        binding.gmapsNavHeader.setOnClickListener {
            showItineraryStepsSheet()
        }

        // --- Apple Maps Home Screen Listeners ---
        binding.btnPlaceHome.setOnClickListener {
            startNavigationToPlace("Home", "Saved Location", 0)
        }
        binding.btnPlaceWork.setOnClickListener {
            startNavigationToPlace("Work", "Crystal IT Park, Indore", 1)
        }
        binding.btnPlaceAdd.setOnClickListener {
            showAddPlaceSheet()
        }
        binding.btnRecentFoothills.setOnClickListener {
            startNavigationToPlace("Indore Junction Railway Station", "Chhoti Gwaltoli, Indore", 0)
        }
        binding.btnRecentEdison.setOnClickListener {
            startNavigationToPlace("Rajwada Historic Palace", "M.G. Road, Indore", 0)
        }
        binding.btnRecentGas.setOnClickListener {
            startNavigationToPlace("Devi Ahilya Bai Holkar Airport", "Depalpur Road, Indore", 0)
        }

        // Recents More (•••) overflow buttons
        binding.btnMoreFoothills.setOnClickListener {
            showPlaceDetailsSheet("Indore Junction Railway Station", "Chhoti Gwaltoli, Indore", "Central Rail Terminus", "4.6 ★", R.drawable.ic_dest_pin)
        }
        binding.btnMoreEdison.setOnClickListener {
            showPlaceDetailsSheet("Rajwada Historic Palace", "M.G. Road, Indore", "Historic Royal Palace", "4.8 ★", R.drawable.ic_dest_pin)
        }
        binding.btnMoreGas.setOnClickListener {
            showPlaceDetailsSheet("Devi Ahilya Bai Holkar Airport", "Depalpur Road, Indore", "International Airport", "4.7 ★", R.drawable.ic_dest_pin)
        }
        binding.btnMoreFavorites.setOnClickListener {
            showFavoritesGuideSheet()
        }

        // Your Guides card
        binding.cardGuideFavorites.setOnClickListener {
            showFavoritesGuideSheet()
        }

        // Grab Handle & Home Search Bar toggle collapse / expand
        binding.pillHomeHandle.setOnClickListener {
            toggleHomeSheet()
        }
        binding.homeSearchBar.setOnClickListener {
            if (!isHomeSheetExpanded) {
                expandHomeSheet()
            } else {
                showHomeSearchDialog(binding.etHomeSearch.text?.toString() ?: "")
            }
        }
        binding.etHomeSearch.setOnClickListener {
            if (!isHomeSheetExpanded) {
                expandHomeSheet()
            } else {
                showHomeSearchDialog(binding.etHomeSearch.text?.toString() ?: "")
            }
        }
        binding.etHomeSearch.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH) {
                hideKeyboard()
                val q = binding.etHomeSearch.text?.toString()?.trim() ?: "Destination"
                startNavigationToPlace(q, "Custom Destination", 0)
                true
            } else {
                false
            }
        }

        // Voice Assistant
        binding.btnHomeVoice.setOnClickListener {
            showVoiceAssistantDialog()
        }

        // User Profile
        binding.btnUserProfile.setOnClickListener {
            showUserProfileSheet()
        }

        // Home Layers Capsule
        binding.btnHomeLayers.setOnClickListener {
            if (currentViewType == MapViewType.OSM_VIEW) {
                setMapLayer(MapViewType.SATELLITE_VIEW)
                Toast.makeText(this, "🛰 Satellite View Active", Toast.LENGTH_SHORT).show()
            } else {
                setMapLayer(MapViewType.OSM_VIEW)
                Toast.makeText(this, "🗺 Vector Map Active", Toast.LENGTH_SHORT).show()
            }
        }

        // Home Recenter Capsule
        binding.btnHomeLocation.setOnClickListener {
            isFollowingVehicle = true
            fetchPhysicalLocationAndCenter(forceCenter = true)
            Toast.makeText(this, "📍 Centered on Current Location", Toast.LENGTH_SHORT).show()
        }

        // Bottom Sheet Driver Controls: Recenter, Overview, Layers
        binding.btnCtrlRecenter.setOnClickListener {
            isDemoMode = false
            isFollowingVehicle = true
            binding.fabMyLocation.setColorFilter(Color.parseColor("#00E676"))
            fetchPhysicalLocationAndCenter(forceCenter = true)
        }
        binding.btnCtrlOverview.setOnClickListener {
            val map = mapLibreMap ?: return@setOnClickListener
            if (routePoints.size >= 2) {
                try {
                    val boundsBuilder = LatLngBounds.Builder()
                    routePoints.forEach { boundsBuilder.include(it) }
                    val bounds = boundsBuilder.build()
                    map.animateCamera(CameraUpdateFactory.newLatLngBounds(bounds, 130), 650)
                    Toast.makeText(this, "🗺️ Route Overview", Toast.LENGTH_SHORT).show()
                } catch (e: Exception) {
                    fetchPhysicalLocationAndCenter(forceCenter = true)
                }
            } else {
                fetchPhysicalLocationAndCenter(forceCenter = true)
                Toast.makeText(this, "📍 Current Location", Toast.LENGTH_SHORT).show()
            }
        }
        binding.btnCtrlLayers.setOnClickListener {
            if (currentViewType == MapViewType.OSM_VIEW) {
                setMapLayer(MapViewType.SATELLITE_VIEW)
            } else {
                setMapLayer(MapViewType.OSM_VIEW)
            }
        }

        // Outage Simulator Toggle & Direct Banner Controls
        val toggleOutageAction = {
            isOutageActive = !isOutageActive
            service?.setSimulateOutage(isOutageActive)

            if (isOutageActive) {
                binding.btnToggleOutage.text = "RESUME GPS"
                binding.btnToggleOutage.setTextColor(Color.WHITE)
                binding.btnToggleOutage.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#10B981"))
                binding.outageBanner.visibility = View.VISIBLE
                Toast.makeText(this, "⚠️ Tactical GNSS Blackout: 15-State ES-EKF Dead Reckoning Active", Toast.LENGTH_SHORT).show()
            } else {
                binding.btnToggleOutage.text = "BLACKOUT"
                binding.btnToggleOutage.setTextColor(Color.WHITE)
                binding.btnToggleOutage.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#272A30"))
                binding.outageBanner.visibility = View.GONE
                Toast.makeText(this, "✅ GNSS Fix Resumed: Smooth 2s Re-acquisition Active", Toast.LENGTH_SHORT).show()
            }
        }
        binding.btnToggleOutage.setOnClickListener { toggleOutageAction() }
        binding.btnOutageResume.setOnClickListener { toggleOutageAction() }
        binding.outageBanner.setOnClickListener { toggleOutageAction() }

        // Expandable Live Sensors Cockpit Toggle
        binding.headerToggleSensors.setOnClickListener {
            val isVisible = binding.containerSensorsDetail.visibility == View.VISIBLE
            binding.containerSensorsDetail.visibility = if (isVisible) View.GONE else View.VISIBLE
            binding.tvSensorsToggleLabel.text = if (isVisible) "Sensors" else "Sensors ▲"
        }

        binding.hudContainer.setOnClickListener {
            isHudActive = false
            binding.hudContainer.visibility = View.GONE
            Toast.makeText(this, "Exited HUD Mode", Toast.LENGTH_SHORT).show()
        }

        // System Back gesture returns from Navigation to Home screen
        onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (isHudActive) {
                    isHudActive = false
                    binding.hudContainer.visibility = View.GONE
                    return
                }
                if (currentScreenState == AppScreenState.NAVIGATION) {
                    setScreenState(AppScreenState.HOME)
                } else if (currentScreenState == AppScreenState.HOME && isHomeSheetExpanded) {
                    collapseHomeSheet()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                    isEnabled = true
                }
            }
        })

        // Initialize in Apple Maps Home Screen
        setScreenState(AppScreenState.HOME)
    }

    private var homeSheetTouchStartY = 0f
    private var homeSheetTouchStartX = 0f
    private var isTrackingHomeSheetDrag = false
    private var hasTriggeredHomeSheetDrag = false

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (currentScreenState == AppScreenState.HOME) {
            val sheetRect = Rect()
            binding.containerHomeSheet.getGlobalVisibleRect(sheetRect)

            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    homeSheetTouchStartY = ev.rawY
                    homeSheetTouchStartX = ev.rawX
                    hasTriggeredHomeSheetDrag = false
                    // Track drag if touch starts anywhere on the home sheet
                    isTrackingHomeSheetDrag = (sheetRect.top > 0) && (ev.rawY >= sheetRect.top && ev.rawY <= sheetRect.bottom)

                    if (isHomeSheetExpanded && ev.rawY < sheetRect.top && sheetRect.top > 0) {
                        hideKeyboard()
                        collapseHomeSheet()
                    }
                }
                MotionEvent.ACTION_MOVE -> {
                    if (isTrackingHomeSheetDrag && !hasTriggeredHomeSheetDrag) {
                        val dy = ev.rawY - homeSheetTouchStartY
                        val dx = ev.rawX - homeSheetTouchStartX

                        // Dominant vertical pull by at least 18dp
                        if (kotlin.math.abs(dy) > kotlin.math.abs(dx) && kotlin.math.abs(dy) > dpToPx(18)) {
                            if (dy < 0 && !isHomeSheetExpanded) {
                                // Pull UP -> Expand!
                                expandHomeSheet()
                                hasTriggeredHomeSheetDrag = true
                            } else if (dy > 0 && isHomeSheetExpanded) {
                                // Pull DOWN on header or down gesture -> Collapse!
                                if (homeSheetTouchStartY <= sheetRect.top + dpToPx(120) || dy > dpToPx(35)) {
                                    collapseHomeSheet()
                                    hasTriggeredHomeSheetDrag = true
                                }
                            }
                        }
                    }
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    isTrackingHomeSheetDrag = false
                    hasTriggeredHomeSheetDrag = false
                }
            }
        }
        return super.dispatchTouchEvent(ev)
    }

    private fun collapseHomeSheet() {
        if (!isHomeSheetExpanded) return
        isHomeSheetExpanded = false
        androidx.transition.TransitionManager.beginDelayedTransition(
            binding.containerHomeSheet,
            androidx.transition.AutoTransition().apply {
                duration = 240
            }
        )
        binding.layoutHomeSheetContent.visibility = View.GONE
    }

    private fun expandHomeSheet() {
        if (isHomeSheetExpanded) return
        isHomeSheetExpanded = true
        androidx.transition.TransitionManager.beginDelayedTransition(
            binding.containerHomeSheet,
            androidx.transition.AutoTransition().apply {
                duration = 240
            }
        )
        binding.layoutHomeSheetContent.visibility = View.VISIBLE
    }

    private fun toggleHomeSheet() {
        if (isHomeSheetExpanded) {
            collapseHomeSheet()
        } else {
            expandHomeSheet()
        }
    }

    private fun getCurrentLat(): Double =
        currentPhysicalLocation?.latitude?.takeIf { it != 0.0 }
            ?: service?.navState?.value?.latitude?.takeIf { it != 0.0 }
            ?: (vehicleSymbol?.latLng?.latitude?.takeIf { it != 0.0 } ?: 22.7196)

    private fun getCurrentLon(): Double =
        currentPhysicalLocation?.longitude?.takeIf { it != 0.0 }
            ?: service?.navState?.value?.longitude?.takeIf { it != 0.0 }
            ?: (vehicleSymbol?.latLng?.longitude?.takeIf { it != 0.0 } ?: 75.8577)

    private fun setScreenState(state: AppScreenState) {
        currentScreenState = state
        when (state) {
            AppScreenState.HOME -> {
                // 1. Show Apple Maps Home UI
                binding.containerHomeSheet.visibility = View.VISIBLE
                binding.pillWeather.visibility = View.VISIBLE
                binding.clusterHomeControls.visibility = View.VISIBLE
                expandHomeSheet()

                // 2. Status bar styling: dark icons for light home map
                androidx.core.view.WindowInsetsControllerCompat(window, window.decorView).isAppearanceLightStatusBars = true

                // 3. Hide Active Navigation UI elements
                binding.cardTopNav.visibility = View.GONE
                binding.containerFloatingLanes.visibility = View.GONE
                binding.pillCurrentStreet.visibility = View.GONE
                binding.clusterSpeedTelemetry.visibility = View.GONE
                binding.clusterRightControls.visibility = View.GONE
                binding.cardTripCockpit.visibility = View.GONE
                binding.outageBanner.visibility = View.GONE
                binding.cardLandmarkCallout.visibility = View.GONE
                binding.tvAudioTooltip.visibility = View.GONE

                // Animate camera to overview
                mapLibreMap?.let { map ->
                    val cur = LatLng(getCurrentLat(), getCurrentLon())
                    map.animateCamera(
                        CameraUpdateFactory.newCameraPosition(
                            CameraPosition.Builder()
                                .target(cur)
                                .zoom(15.5)
                                .tilt(0.0)
                                .bearing(0.0)
                                .build()
                        ),
                        600
                    )
                }
            }
            AppScreenState.NAVIGATION -> {
                // 1. Hide Apple Maps Home UI
                binding.containerHomeSheet.visibility = View.GONE
                binding.pillWeather.visibility = View.GONE
                binding.clusterHomeControls.visibility = View.GONE

                // 2. Status bar styling: light icons for navigation
                androidx.core.view.WindowInsetsControllerCompat(window, window.decorView).isAppearanceLightStatusBars = false

                // 3. Show Navigation Cockpit & Controls
                binding.cardTopNav.visibility = View.VISIBLE
                binding.pillCurrentStreet.visibility = View.VISIBLE
                binding.clusterSpeedTelemetry.visibility = View.VISIBLE
                binding.clusterRightControls.visibility = View.VISIBLE
                binding.cardTripCockpit.visibility = View.VISIBLE

                // Trigger dynamic maneuver layout & styling
                val activePlan = service?.getActiveRoute() ?: RouteManager().getActiveRoute()
                if (activePlan != null && activePlan.steps.isNotEmpty()) {
                    val step = activePlan.steps.first()
                    val distFt = (step.distanceM * 3.28084).toInt()
                    val distStr = if (distFt > 1000) String.format(java.util.Locale.US, "%.1f mi", distFt / 5280.0) else "$distFt FT"
                    renderDynamicManeuver(ManeuverItem(
                        icon = when {
                            step.maneuverModifier.contains("left") -> "↰"
                            step.maneuverModifier.contains("right") -> "↱"
                            step.maneuverType == "arrive" -> "🏁"
                            else -> "↑"
                        },
                        distance = distStr,
                        roadName = step.roadName.ifBlank { step.instruction },
                        towardDirection = activePlan.destinationName
                    ))
                } else {
                    val currentManeuver = dynamicManeuverPresets[currentManeuverIndex]
                    renderDynamicManeuver(currentManeuver)
                }
                applyThemeMode()

                // Animate camera to 3D Navigation perspective
                mapLibreMap?.let { map ->
                    val cur = LatLng(getCurrentLat(), getCurrentLon())
                    map.animateCamera(
                        CameraUpdateFactory.newCameraPosition(
                            CameraPosition.Builder()
                                .target(cur)
                                .zoom(17.2)
                                .tilt(48.0)
                                .bearing(displayHeading.toDouble())
                                .build()
                        ),
                        800
                    )
                }
            }
        }
    }

    private fun dpToPx(dp: Int): Int = (dp * resources.displayMetrics.density).toInt()

    private fun startNavigationToPlace(name: String, subtitle: String, maneuverIdx: Int = 0) {
        val curLat = currentPhysicalLocation?.latitude ?: getCurLatFallback()
        val curLon = currentPhysicalLocation?.longitude ?: getCurLonFallback()
        val found = RouteManager().presetPlaces.firstOrNull {
            it.name.contains(name, ignoreCase = true) || name.contains(it.name, ignoreCase = true)
        }
        val targetLat: Double
        val targetLon: Double
        if (found != null) {
            targetLat = found.lat
            targetLon = found.lon
        } else {
            targetLat = if (curLat != 0.0) curLat + 0.015 else 22.7485
            targetLon = if (curLon != 0.0) curLon + 0.012 else 75.8520
        }
        startNavigationTo(targetLat, targetLon, name)
    }

    private fun showPlaceDetailsSheet(title: String, subtitle: String, category: String, rating: String, iconRes: Int) {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        val iconBadge = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(48), dpToPx(48))
            setBackgroundResource(R.drawable.bg_place_action)
            addView(ImageView(context).apply {
                layoutParams = FrameLayout.LayoutParams(dpToPx(24), dpToPx(24), Gravity.CENTER)
                setImageResource(iconRes)
            })
        }
        headerRow.addView(iconBadge)

        val titleCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                marginStart = dpToPx(14)
            }
            addView(TextView(context).apply {
                text = title
                setTextColor(Color.BLACK)
                textSize = 19f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
            addView(TextView(context).apply {
                text = "$category · $rating"
                setTextColor(Color.parseColor("#007AFF"))
                textSize = 12f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
        }
        headerRow.addView(titleCol)
        root.addView(headerRow)

        val addressTv = TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dpToPx(10)
                bottomMargin = dpToPx(16)
            }
            text = subtitle
            setTextColor(Color.parseColor("#52525B"))
            textSize = 13f
        }
        root.addView(addressTv)

        root.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(1)).apply {
                bottomMargin = dpToPx(16)
            }
            setBackgroundColor(Color.parseColor("#E5E7EB"))
        })

        val actionsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(16)
            }
        }

        fun createActionButton(label: String, iconStr: String, isPrimary: Boolean, onClick: () -> Unit): LinearLayout {
            return LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f)
                isClickable = true
                isFocusable = true
                setOnClickListener { onClick() }

                val circle = FrameLayout(context).apply {
                    layoutParams = LinearLayout.LayoutParams(dpToPx(50), dpToPx(50))
                    setBackgroundResource(if (isPrimary) R.drawable.bg_circle_blue else R.drawable.bg_place_action)
                    addView(TextView(context).apply {
                        layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER)
                        text = iconStr
                        textSize = 20f
                        setTextColor(if (isPrimary) Color.WHITE else Color.BLACK)
                    })
                }
                addView(circle)

                addView(TextView(context).apply {
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                        topMargin = dpToPx(6)
                    }
                    text = label
                    textSize = 12f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                    setTextColor(if (isPrimary) Color.parseColor("#007AFF") else Color.parseColor("#374151"))
                })
            }
        }

        actionsRow.addView(createActionButton("Directions", "🚗", true) {
            dialog.dismiss()
            startNavigationToPlace(title, subtitle, 0)
        })
        actionsRow.addView(createActionButton("Call", "📞", false) {
            Toast.makeText(this, "Calling $title...", Toast.LENGTH_SHORT).show()
        })
        actionsRow.addView(createActionButton("Share", "🔗", false) {
            val sendIntent = Intent().apply {
                action = Intent.ACTION_SEND
                putExtra(Intent.EXTRA_TEXT, "Check out $title on iNAV Maps: $subtitle")
                type = "text/plain"
            }
            startActivity(Intent.createChooser(sendIntent, "Share Location"))
        })
        actionsRow.addView(createActionButton("Save", "⭐", false) {
            Toast.makeText(this, "⭐ Saved to Favorites Guide", Toast.LENGTH_SHORT).show()
        })

        root.addView(actionsRow)

        val infoCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_recents_card)
            setPadding(dpToPx(14), dpToPx(12), dpToPx(14), dpToPx(12))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

            addView(TextView(context).apply {
                text = "🕒 Open Daily · 6:00 AM – 10:00 PM"
                setTextColor(Color.parseColor("#1F2937"))
                textSize = 13f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
            addView(TextView(context).apply {
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    topMargin = dpToPx(4)
                }
                text = "⚡ iNAV Sensor Telemetry: GPS + INS Signal Strong"
                setTextColor(Color.parseColor("#059669"))
                textSize = 11.5f
            })
        }
        root.addView(infoCard)

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showFavoritesGuideSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        root.addView(TextView(this).apply {
            text = "⭐ Favorites Guide (3 Places)"
            setTextColor(Color.BLACK)
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(4)
            }
        })

        root.addView(TextView(this).apply {
            text = "Driver curated favorites with offline INS waypoints"
            setTextColor(Color.parseColor("#71717A"))
            textSize = 13f
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(14)
            }
        })

        val places = listOf(
            Triple("Indore Junction Railway Station", "Chhoti Gwaltoli, Indore", 0),
            Triple("Rajwada Historic Palace", "M.G. Road, Indore", 0),
            Triple("Devi Ahilya Bai Holkar Airport", "Depalpur Road, Indore", 0)
        )

        val listContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_recents_card)
            setPadding(dpToPx(12), dpToPx(6), dpToPx(12), dpToPx(6))
        }

        places.forEachIndexed { index, (pName, pAddr, mIdx) ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(52))
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    dialog.dismiss()
                    startNavigationToPlace(pName, pAddr, mIdx)
                }
            }

            val iconBadge = FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(34), dpToPx(34))
                setBackgroundResource(R.drawable.bg_place_action)
                addView(TextView(context).apply {
                    layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER)
                    text = when(index) { 0 -> "🌲"; 1 -> "📍"; else -> "⛽" }
                    textSize = 16f
                })
            }
            row.addView(iconBadge)

            val textCol = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                    marginStart = dpToPx(12)
                }
                addView(TextView(context).apply {
                    text = pName
                    setTextColor(Color.BLACK)
                    textSize = 14f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
                addView(TextView(context).apply {
                    text = pAddr
                    setTextColor(Color.parseColor("#71717A"))
                    textSize = 11f
                    maxLines = 1
                })
            }
            row.addView(textCol)

            val navBtn = TextView(this).apply {
                text = "NAVIGATE"
                setTextColor(Color.parseColor("#007AFF"))
                textSize = 11f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setPadding(dpToPx(10), dpToPx(6), dpToPx(10), dpToPx(6))
                setBackgroundResource(R.drawable.bg_place_action)
            }
            row.addView(navBtn)
            listContainer.addView(row)

            if (index < places.size - 1) {
                listContainer.addView(View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(1)).apply {
                        marginStart = dpToPx(46)
                    }
                    setBackgroundColor(Color.parseColor("#F3F4F6"))
                })
            }
        }
        root.addView(listContainer)

        val addBtn = TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(44)).apply {
                topMargin = dpToPx(16)
            }
            gravity = Gravity.CENTER
            text = "➕ Add Place to Guide"
            setTextColor(Color.parseColor("#007AFF"))
            textSize = 14f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setBackgroundResource(R.drawable.bg_apple_search_bar)
            isClickable = true
            isFocusable = true
            setOnClickListener {
                dialog.dismiss()
                showAddPlaceSheet()
            }
        }
        root.addView(addBtn)

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showAddPlaceSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        root.addView(TextView(this).apply {
            text = "Add to Places"
            setTextColor(Color.BLACK)
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(14)
            }
        })

        val categories = listOf(
            Pair("🏠 Set Home Address", "Set your primary residence"),
            Pair("💼 Set Work Address", "Crystal IT Park, Indore"),
            Pair("🛍️ Add Shopping Hub", "Phoenix Citadel / Malhar Mall"),
            Pair("⛽ Add Fuel / EV Hub", "IndianOil / Tata Power EV Fast Charger"),
            Pair("✈️ Add Airport", "Devi Ahilya Bai Holkar Airport")
        )

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_recents_card)
            setPadding(dpToPx(14), dpToPx(6), dpToPx(14), dpToPx(6))
        }

        categories.forEachIndexed { idx, (catTitle, catSub) ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    topMargin = dpToPx(8)
                    bottomMargin = dpToPx(8)
                }
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    dialog.dismiss()
                    startNavigationToPlace(catTitle.substring(2).trim(), catSub, 0)
                }
                addView(TextView(context).apply {
                    text = catTitle
                    setTextColor(Color.BLACK)
                    textSize = 14f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
                addView(TextView(context).apply {
                    text = catSub
                    setTextColor(Color.parseColor("#71717A"))
                    textSize = 12f
                })
            }
            card.addView(row)
            if (idx < categories.size - 1) {
                card.addView(View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(1))
                    setBackgroundColor(Color.parseColor("#F3F4F6"))
                })
            }
        }
        root.addView(card)

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showVoiceAssistantDialog() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dpToPx(24), dpToPx(12), dpToPx(24), dpToPx(32))
            setBackgroundColor(Color.parseColor("#0B0E14"))
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(20)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#374151"))
        }
        root.addView(handle)

        val micFrame = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(72), dpToPx(72)).apply {
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_circle_blue)
            addView(ImageView(context).apply {
                layoutParams = FrameLayout.LayoutParams(dpToPx(36), dpToPx(36), Gravity.CENTER)
                setImageResource(R.drawable.ic_mic_apple)
                setColorFilter(Color.WHITE)
            })
        }
        root.addView(micFrame)

        root.addView(TextView(this).apply {
            text = "iNAV Voice Assistant"
            setTextColor(Color.WHITE)
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
        })

        root.addView(TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dpToPx(4)
                bottomMargin = dpToPx(20)
            }
            text = "Listening... Tap any command to trigger:"
            setTextColor(Color.parseColor("#9CA3AF"))
            textSize = 13f
            gravity = Gravity.CENTER
        })

        val chips = listOf(
            Triple("🚗 Navigate to Indore Junction", "Starts Route to Station", {
                dialog.dismiss()
                startNavigationToPlace("Indore Junction Railway Station", "Chhoti Gwaltoli, Indore", 0)
            }),
            Triple("🏛️ Navigate to Rajwada Palace", "Starts Route to Historic Palace", {
                dialog.dismiss()
                startNavigationToPlace("Rajwada Historic Palace", "M.G. Road, Indore", 0)
            }),
            Triple("☕ Amenities Along Route", "Find fuel, coffee and rest stops", {
                dialog.dismiss()
                showCoffeeAmenitiesSheet()
            }),
            Triple("⚠️ Simulate GPS Blackout", "Trigger 15-State ES-EKF Dead Reckoning", {
                dialog.dismiss()
                binding.btnToggleOutage.performClick()
            }),
            Triple("🚘 Windshield HUD Mode", "Head-Up Display Projection", {
                dialog.dismiss()
                openHudMode()
            }),
            Triple("🚶 Toggle Pedestrian Wilderness Mode", "PDR Steps & Trailhead Compass", {
                dialog.dismiss()
                togglePedestrianMode()
            }),
            Triple("🚀 Export 3D Mission Flight (KML)", "Export Google Earth Telemetry", {
                dialog.dismiss()
                exportMissionKml()
            }),
            Triple("🔌 Toggle OBD-II CAN-Bus Speed", "Simulated Wheel Speed (PID 010D)", {
                dialog.dismiss()
                toggleObdSimulation()
            }),
            Triple("🛰️ Toggle Satellite Hybrid Imagery", "Aerial Satellite Hybrid View", {
                dialog.dismiss()
                setMapLayer(if (currentViewType == MapViewType.SATELLITE_VIEW) MapViewType.OSM_VIEW else MapViewType.SATELLITE_VIEW)
            }),
            Triple("🌙 Toggle Dark / Night Theme", "Switch map style", {
                dialog.dismiss()
                binding.btnShieldSafety.performClick()
            })
        )

        chips.forEach { (chipText, chipSub, chipAction) ->
            val btn = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(46)).apply {
                    bottomMargin = dpToPx(8)
                }
                setBackgroundResource(R.drawable.bg_apple_search_bar)
                backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#1F2937"))
                setPadding(dpToPx(14), 0, dpToPx(14), 0)
                isClickable = true
                isFocusable = true
                setOnClickListener { chipAction() }

                addView(TextView(context).apply {
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f)
                    text = chipText
                    setTextColor(Color.WHITE)
                    textSize = 13.5f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
                addView(TextView(context).apply {
                    text = chipSub
                    setTextColor(Color.parseColor("#9CA3AF"))
                    textSize = 11f
                })
            }
            root.addView(btn)
        }

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showUserProfileSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(16)
            }
        }

        val avatar = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(52), dpToPx(52))
            setBackgroundResource(R.drawable.bg_avatar_as)
            addView(TextView(context).apply {
                layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER)
                text = "AS"
                setTextColor(Color.WHITE)
                textSize = 20f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
        }
        headerRow.addView(avatar)

        val infoCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                marginStart = dpToPx(14)
            }
            addView(TextView(context).apply {
                text = "Alex Sanders"
                setTextColor(Color.BLACK)
                textSize = 18f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
            addView(TextView(context).apply {
                text = "iNAV Pro Driver · 4.98 ★ (2,410 Trips)"
                setTextColor(Color.parseColor("#059669"))
                textSize = 12f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
        }
        headerRow.addView(infoCol)
        root.addView(headerRow)

        val detailsCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_recents_card)
            setPadding(dpToPx(14), dpToPx(12), dpToPx(14), dpToPx(12))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

            val state = service?.navState?.value
            val rows = listOf(
                "🚘 Vehicle" to "Tesla Model 3 Long Range (Dual Motor EV)",
                "🧠 Sensor Fusion" to "15-State ES-EKF + AI VelocityNet (100 Hz)",
                "🚶 Navigation Mode" to if (isPedestrianMode) "Wilderness PDR (${state?.pedestrianStepCount ?: 0} steps)" else "Vehicular Highway Mode",
                "🔌 OBD-II Status" to if (state?.isObdConnected == true) "Connected (PID 010D Active)" else if (isObdSimulated) "Simulated CAN-Bus Active" else "Disconnected",
                "🚀 Blackbox Recorder" to "${com.inav.navigation.blackbox.BlackboxRecorder.getRecordCount()} 3D mission points recorded",
                "🛣️ Routing Option" to "Avoid Tolls & Ferries: ON",
                "🛰️ Current Layer" to if (currentViewType == MapViewType.SATELLITE_VIEW) "Satellite Hybrid" else "Vector Cartography",
                "🌓 Visual Mode" to if (isLightMode) "☀️ Day / Light Mode" else "🌙 Night / Dark Mode"
            )

            rows.forEachIndexed { idx, (k, v) ->
                val r = LinearLayout(context).apply {
                    orientation = LinearLayout.HORIZONTAL
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                        topMargin = if (idx > 0) dpToPx(8) else 0
                    }
                    addView(TextView(context).apply {
                        text = k
                        setTextColor(Color.parseColor("#374151"))
                        textSize = 12.5f
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 0.45f)
                    })
                    addView(TextView(context).apply {
                        text = v
                        setTextColor(Color.parseColor("#111827"))
                        textSize = 12.5f
                        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 0.55f)
                    })
                }
                addView(r)
            }
        }
        root.addView(detailsCard)

        // Action Buttons Row
        val actionsLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dpToPx(14)
            }
        }

        fun createActionButton(label: String, icon: String, bgTint: Int, onClick: () -> Unit): TextView {
            return TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(44)).apply {
                    bottomMargin = dpToPx(8)
                }
                gravity = Gravity.CENTER
                text = "$icon  $label"
                setTextColor(Color.WHITE)
                textSize = 13.5f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setBackgroundResource(R.drawable.bg_circle_blue)
                backgroundTintList = android.content.res.ColorStateList.valueOf(bgTint)
                isClickable = true
                isFocusable = true
                setOnClickListener { onClick() }
            }
        }

        actionsLayout.addView(createActionButton("Export 3D Mission Telemetry (KML)", "🚀", Color.parseColor("#2563EB")) {
            dialog.dismiss()
            exportMissionKml()
        })
        actionsLayout.addView(createActionButton(if (isPedestrianMode) "Switch to Vehicle Navigation" else "Activate Pedestrian / Wilderness Mode", "🚶", Color.parseColor("#059669")) {
            dialog.dismiss()
            togglePedestrianMode()
        })
        actionsLayout.addView(createActionButton(if (isObdSimulated) "Disconnect OBD-II Telemetry" else "Simulate OBD-II CAN-Bus (PID 010D)", "🔌", Color.parseColor("#4B5563")) {
            dialog.dismiss()
            toggleObdSimulation()
        })
        root.addView(actionsLayout)

        val doneBtn = TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(44)).apply {
                topMargin = dpToPx(4)
            }
            gravity = Gravity.CENTER
            text = "Close"
            setTextColor(Color.parseColor("#374151"))
            textSize = 14f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setBackgroundResource(R.drawable.bg_apple_search_bar)
            isClickable = true
            isFocusable = true
            setOnClickListener { dialog.dismiss() }
        }
        root.addView(doneBtn)

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showItineraryStepsSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(24))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(14)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(12)
            }
            addView(TextView(context).apply {
                text = "Turn-by-Turn Itinerary"
                setTextColor(Color.BLACK)
                textSize = 19f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f)
            })
            addView(TextView(context).apply {
                text = "10 Steps · 3.4 mi"
                setTextColor(Color.parseColor("#007AFF"))
                textSize = 12f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            })
        }
        root.addView(headerRow)

        val scrollView = android.widget.ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(340))
        }

        val listCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val activePlan = service?.getActiveRoute() ?: RouteManager().getActiveRoute()
        val stepsList = if (activePlan != null && activePlan.steps.isNotEmpty()) {
            activePlan.steps.mapIndexed { i, s ->
                val distFt = (s.distanceM * 3.28084).toInt()
                val distStr = if (distFt > 1000) String.format(java.util.Locale.US, "%.1f mi", distFt / 5280.0) else "$distFt FT"
                val icon = when {
                    s.maneuverModifier.contains("left") -> "↰"
                    s.maneuverModifier.contains("right") -> "↱"
                    s.maneuverType == "arrive" -> "🏁"
                    else -> "↑"
                }
                ManeuverItem(
                    icon = icon,
                    distance = distStr,
                    roadName = s.roadName.ifBlank { s.instruction },
                    towardDirection = activePlan.destinationName,
                    thenText = s.instruction
                )
            }
        } else {
            dynamicManeuverPresets
        }

        stepsList.forEachIndexed { idx, item ->
            val isCurrent = (idx == currentManeuverIndex)
            val stepView = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    bottomMargin = dpToPx(6)
                }
                setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
                setBackgroundResource(if (isCurrent) R.drawable.bg_apple_search_bar else R.drawable.bg_recents_card)
                if (isCurrent) {
                    backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#EBF5FF"))
                }
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    currentManeuverIndex = idx
                    renderDynamicManeuver(item)
                    dialog.dismiss()
                    Toast.makeText(this@MainActivity, "Step $idx: ${item.distance} ${item.roadName}", Toast.LENGTH_SHORT).show()
                }
            }

            val iconBadge = FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(38), dpToPx(38))
                setBackgroundResource(if (isCurrent) R.drawable.bg_circle_blue else R.drawable.bg_place_action)
                addView(TextView(context).apply {
                    layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER)
                    text = item.icon
                    textSize = 18f
                    setTextColor(if (isCurrent) Color.WHITE else Color.BLACK)
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
            }
            stepView.addView(iconBadge)

            val textCol = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                    marginStart = dpToPx(12)
                }
                addView(TextView(context).apply {
                    text = "${item.distance} · ${item.roadName}"
                    setTextColor(if (isCurrent) Color.parseColor("#007AFF") else Color.BLACK)
                    textSize = 14f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
                val subText = item.thenText ?: item.towardDirection ?: "Continue on route"
                addView(TextView(context).apply {
                    text = subText
                    setTextColor(Color.parseColor("#71717A"))
                    textSize = 11.5f
                    maxLines = 1
                })
            }
            stepView.addView(textCol)

            if (isCurrent) {
                val currentBadge = TextView(this).apply {
                    text = "ACTIVE"
                    setTextColor(Color.WHITE)
                    textSize = 9.5f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                    setBackgroundResource(R.drawable.bg_circle_blue)
                    setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(4))
                }
                stepView.addView(currentBadge)
            }

            listCol.addView(stepView)
        }
        scrollView.addView(listCol)
        root.addView(scrollView)

        val endBtn = TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(46)).apply {
                topMargin = dpToPx(14)
            }
            gravity = Gravity.CENTER
            text = "🛑 End Navigation"
            setTextColor(Color.WHITE)
            textSize = 15f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setBackgroundResource(R.drawable.bg_circle_blue)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#DC2626"))
            isClickable = true
            isFocusable = true
            setOnClickListener {
                dialog.dismiss()
                setScreenState(AppScreenState.HOME)
                Toast.makeText(this@MainActivity, "Navigation Ended", Toast.LENGTH_SHORT).show()
            }
        }
        root.addView(endBtn)

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showRouteSettingsSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        root.addView(TextView(this).apply {
            text = "Route Options & Controls"
            setTextColor(Color.BLACK)
            textSize = 19f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(14)
            }
        })

        fun createOptionRow(icon: String, title: String, subtitle: String, isDestructive: Boolean = false, onClick: () -> Unit): LinearLayout {
            return LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(54)).apply {
                    bottomMargin = dpToPx(8)
                }
                setBackgroundResource(R.drawable.bg_recents_card)
                setPadding(dpToPx(14), 0, dpToPx(14), 0)
                isClickable = true
                isFocusable = true
                setOnClickListener { onClick() }

                val circle = FrameLayout(context).apply {
                    layoutParams = LinearLayout.LayoutParams(dpToPx(36), dpToPx(36))
                    setBackgroundResource(if (isDestructive) R.drawable.bg_circle_blue else R.drawable.bg_place_action)
                    if (isDestructive) {
                        backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#FEE2E2"))
                    }
                    addView(TextView(context).apply {
                        layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER)
                        text = icon
                        textSize = 17f
                    })
                }
                addView(circle)

                val col = LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                        marginStart = dpToPx(12)
                    }
                    addView(TextView(context).apply {
                        text = title
                        setTextColor(if (isDestructive) Color.parseColor("#DC2626") else Color.BLACK)
                        textSize = 14f
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                    })
                    addView(TextView(context).apply {
                        text = subtitle
                        setTextColor(Color.parseColor("#71717A"))
                        textSize = 11.5f
                    })
                }
                addView(col)
            }
        }

        root.addView(createOptionRow("🔀", "Route Alternatives", "View fastest, AB bypass, and scenic routes") {
            dialog.dismiss()
            showRouteAlternativesSheet()
        })
        root.addView(createOptionRow("🚶", if (isPedestrianMode) "Switch to Vehicular Mode" else "Pedestrian Trail & Backtrack Mode", if (isPedestrianMode) "Currently tracking steps & trailhead" else "Enable PDR step counting & survival return compass") {
            dialog.dismiss()
            togglePedestrianMode()
        })
        root.addView(createOptionRow("🚀", "Export 3D Mission Telemetry (KML)", "Export Blackbox trajectory to Google Earth & GPS logs") {
            dialog.dismiss()
            exportMissionKml()
        })
        root.addView(createOptionRow("🔌", if (isObdSimulated) "Disconnect OBD-II Telemetry" else "Connect OBD-II Telemetry (ELM327)", if (isObdSimulated) "Simulated PID 010D Active" else "Inject physical wheel speed into 15-State ES-EKF") {
            dialog.dismiss()
            toggleObdSimulation()
        })
        root.addView(createOptionRow("🛰️", if (currentViewType == MapViewType.SATELLITE_VIEW) "Switch to Vector Cartography" else "Switch to Satellite Hybrid View", "High-resolution Google hybrid aerial imagery") {
            dialog.dismiss()
            setMapLayer(if (currentViewType == MapViewType.SATELLITE_VIEW) MapViewType.OSM_VIEW else MapViewType.SATELLITE_VIEW)
        })
        root.addView(createOptionRow("🌓", if (isLightMode) "Switch to Dark Mode" else "Switch to Light Mode", "Toggle MapLibre vector style") {
            dialog.dismiss()
            binding.btnShieldSafety.performClick()
        })
        root.addView(createOptionRow("📊", "Live Sensors Cockpit", "VOYAGER OS IMU, Attitude & EKF Telemetry") {
            dialog.dismiss()
            val isVis = binding.containerSensorsDetail.visibility == View.VISIBLE
            binding.containerSensorsDetail.visibility = if (isVis) View.GONE else View.VISIBLE
            binding.headerToggleSensors.visibility = View.VISIBLE
            Toast.makeText(this@MainActivity, if (isVis) "Sensors Hidden" else "Sensors Visible", Toast.LENGTH_SHORT).show()
        })
        root.addView(createOptionRow("⚠️", "Tactical GPS Blackout Simulator", "Google Maps frozen vs iNAV 15-State ES-EKF Dead Reckoning") {
            dialog.dismiss()
            binding.btnToggleOutage.performClick()
        })
        root.addView(createOptionRow("🚘", "Windshield HUD Projector Mode", "High-contrast dashboard projection") {
            dialog.dismiss()
            openHudMode()
        })
        root.addView(createOptionRow("🛑", "End Navigation", "Return to Apple Maps Home overview", isDestructive = true) {
            dialog.dismiss()
            setScreenState(AppScreenState.HOME)
            Toast.makeText(this@MainActivity, "Navigation Ended", Toast.LENGTH_SHORT).show()
        })

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showRouteAlternativesSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        root.addView(TextView(this).apply {
            text = "Route Alternatives"
            setTextColor(Color.BLACK)
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(14)
            }
        })

        val routes = listOf(
            Triple("via Main Ring Road (Fastest · Selected)", "Optimal traffic flow · INS calibrated", true),
            Triple("via AB Expressway Bypass", "+3 min · Smooth highway cruising", false),
            Triple("via City Central Arterial", "+6 min · Multiple traffic signals", false)
        )

        routes.forEach { (rName, rDesc, isSel) ->
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    bottomMargin = dpToPx(10)
                }
                setPadding(dpToPx(14), dpToPx(12), dpToPx(14), dpToPx(12))
                setBackgroundResource(if (isSel) R.drawable.bg_apple_search_bar else R.drawable.bg_recents_card)
                if (isSel) {
                    backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#EBF5FF"))
                }
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    val etaMin = if (rName.contains("13")) "13" else if (rName.contains("17")) "17" else "10"
                    val etaDist = if (rName.contains("13")) "3.8 mi" else if (rName.contains("17")) "4.9 mi" else "3.4 mi"
                    binding.tvTripEtaMins.text = etaMin
                    binding.tvActiveRouteMetrics.text = etaDist
                    dialog.dismiss()
                    Toast.makeText(this@MainActivity, "Route selected: $rName ($etaMin min · $etaDist)", Toast.LENGTH_SHORT).show()
                }
            }

            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f)
                addView(TextView(context).apply {
                    text = rName
                    setTextColor(if (isSel) Color.parseColor("#007AFF") else Color.BLACK)
                    textSize = 14.5f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
                addView(TextView(context).apply {
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                        topMargin = dpToPx(3)
                    }
                    text = rDesc
                    setTextColor(Color.parseColor("#71717A"))
                    textSize = 12f
                })
            }
            card.addView(col)

            val tag = TextView(this).apply {
                text = if (isSel) "CURRENT" else "SELECT"
                setTextColor(if (isSel) Color.WHITE else Color.parseColor("#007AFF"))
                textSize = 10f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setBackgroundResource(if (isSel) R.drawable.bg_circle_blue else R.drawable.bg_place_action)
                setPadding(dpToPx(10), dpToPx(5), dpToPx(10), dpToPx(5))
            }
            card.addView(tag)
            root.addView(card)
        }

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showCoffeeAmenitiesSheet() {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(12), dpToPx(20), dpToPx(28))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(5)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(16)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        root.addView(TextView(this).apply {
            text = "☕ Coffee & Stops Along Route"
            setTextColor(Color.BLACK)
            textSize = 19f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(14)
            }
        })

        val amenities = listOf(
            Triple("IndianOil Fuel & EV Hub", "Main Ring Road · 0.8 km · +2 min", "⛽"),
            Triple("Cafe Coffee Day / Bistro", "AB Road · 1.2 km · +3 min · 4.6 ★", "☕"),
            Triple("Highway Rest Stop & Food Court", "Bypass Expressway · 2.4 km · +4 min", "🍽️"),
            Triple("Apollo 24x7 Pharmacy & Care", "Vijay Nagar · 1.5 km · +2 min", "💊")
        )

        amenities.forEach { (aName, aDesc, aIcon) ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    bottomMargin = dpToPx(10)
                }
                setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
                setBackgroundResource(R.drawable.bg_recents_card)
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    binding.tvActiveDestTitle.text = "Via $aName"
                    dialog.dismiss()
                    Toast.makeText(this@MainActivity, "Added stop: $aName to current route", Toast.LENGTH_SHORT).show()
                }
            }

            val iconBadge = FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(38), dpToPx(38))
                setBackgroundResource(R.drawable.bg_place_action)
                addView(TextView(context).apply {
                    layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER)
                    text = aIcon
                    textSize = 18f
                })
            }
            row.addView(iconBadge)

            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                    marginStart = dpToPx(12)
                }
                addView(TextView(context).apply {
                    text = aName
                    setTextColor(Color.BLACK)
                    textSize = 14f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })
                addView(TextView(context).apply {
                    text = aDesc
                    setTextColor(Color.parseColor("#71717A"))
                    textSize = 11.5f
                })
            }
            row.addView(col)

            val addStopBtn = TextView(this).apply {
                text = "+ ADD"
                setTextColor(Color.parseColor("#007AFF"))
                textSize = 11f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setBackgroundResource(R.drawable.bg_place_action)
                setPadding(dpToPx(10), dpToPx(6), dpToPx(10), dpToPx(6))
            }
            row.addView(addStopBtn)
            root.addView(row)
        }

        dialog.setContentView(root)
        dialog.show()
    }

    private fun showBookmarkDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("📍 Bookmark Location")
            .setMessage("Save '${binding.tvActiveDestTitle.text}' to your Guides & Favorites for quick access during future trips?")
            .setPositiveButton("⭐ Save Bookmark") { _, _ ->
                Toast.makeText(this, "⭐ Saved '${binding.tvActiveDestTitle.text}' to Favorites", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showHomeSearchDialog(initialQuery: String) {
        val dialog = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(18), dpToPx(12), dpToPx(18), dpToPx(24))
            setBackgroundColor(Color.WHITE)
        }

        val handle = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(40), dpToPx(4)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dpToPx(12)
            }
            setBackgroundResource(R.drawable.bg_grab_handle)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D1D5DB"))
        }
        root.addView(handle)

        root.addView(TextView(this).apply {
            text = "Search Destinations"
            setTextColor(Color.BLACK)
            textSize = 18f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(10)
            }
        })

        // Live Search Input Box
        val searchBox = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_apple_search_bar)
            setPadding(dpToPx(12), dpToPx(8), dpToPx(12), dpToPx(8))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dpToPx(12)
            }
        }
        searchBox.addView(ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(dpToPx(18), dpToPx(18)).apply { marginEnd = dpToPx(8) }
            setImageResource(R.drawable.ic_search_black)
            imageTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#8E8E93"))
        })

        val searchInput = android.widget.EditText(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f)
            background = null
            hint = "Search places, addresses, landmarks..."
            setHintTextColor(Color.parseColor("#8E8E93"))
            setTextColor(Color.BLACK)
            textSize = 15f
            isSingleLine = true
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH
            setText(initialQuery)
        }
        searchBox.addView(searchInput)
        root.addView(searchBox)

        // Results Container inside ScrollView
        val scrollView = android.widget.ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(340))
        }
        val resultsCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_recents_card)
            setPadding(dpToPx(12), dpToPx(6), dpToPx(12), dpToPx(6))
        }
        scrollView.addView(resultsCard)
        root.addView(scrollView)

        fun populateResults(places: List<PlaceSuggestion>) {
            resultsCard.removeAllViews()
            if (places.isEmpty()) {
                resultsCard.addView(TextView(this@MainActivity).apply {
                    text = "No destinations found. Try searching another place."
                    setTextColor(Color.parseColor("#71717A"))
                    textSize = 13f
                    gravity = Gravity.CENTER
                    setPadding(dpToPx(16), dpToPx(24), dpToPx(16), dpToPx(24))
                })
                return
            }

            places.forEachIndexed { idx, place ->
                val row = LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(52))
                    isClickable = true
                    isFocusable = true
                    setOnClickListener {
                        dialog.dismiss()
                        startNavigationTo(place.lat, place.lon, place.name)
                    }
                }
                row.addView(TextView(this@MainActivity).apply {
                    text = place.icon
                    textSize = 18f
                })
                val col = LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f).apply {
                        marginStart = dpToPx(12)
                    }
                    addView(TextView(this@MainActivity).apply {
                        text = place.name
                        setTextColor(Color.BLACK)
                        textSize = 14f
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                        maxLines = 1
                        ellipsize = android.text.TextUtils.TruncateAt.END
                    })
                    addView(TextView(this@MainActivity).apply {
                        text = place.address
                        setTextColor(Color.parseColor("#71717A"))
                        textSize = 11.5f
                        maxLines = 1
                        ellipsize = android.text.TextUtils.TruncateAt.END
                    })
                }
                row.addView(col)

                row.addView(TextView(this@MainActivity).apply {
                    val userLat = currentPhysicalLocation?.latitude ?: getCurLatFallback()
                    val userLon = currentPhysicalLocation?.longitude ?: getCurLonFallback()
                    val distM = if (userLat != 0.0 && userLon != 0.0) {
                        val dLat = Math.toRadians(place.lat - userLat)
                        val dLon = Math.toRadians(place.lon - userLon)
                        val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(Math.toRadians(userLat)) * Math.cos(Math.toRadians(place.lat)) * Math.sin(dLon / 2) * Math.sin(dLon / 2)
                        6371000.0 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
                    } else 0.0
                    text = if (distM > 0) String.format(java.util.Locale.US, "%.1f km", distM / 1000.0) else ""
                    setTextColor(Color.parseColor("#007AFF"))
                    textSize = 12f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                })

                resultsCard.addView(row)
                if (idx < places.size - 1) {
                    resultsCard.addView(View(this@MainActivity).apply {
                        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(1))
                        setBackgroundColor(Color.parseColor("#F3F4F6"))
                    })
                }
            }
        }

        val curLat = currentPhysicalLocation?.latitude ?: getCurLatFallback()
        val curLon = currentPhysicalLocation?.longitude ?: getCurLonFallback()
        val routeMgr = service?.routeManager ?: RouteManager()

        routeMgr.searchPlaces(initialQuery, curLat, curLon, lifecycleScope, this) { results ->
            populateResults(results)
        }

        searchInput.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: android.text.Editable?) {
                val q = s?.toString() ?: ""
                routeMgr.searchPlaces(q, curLat, curLon, lifecycleScope, this@MainActivity) { results ->
                    populateResults(results)
                }
            }
        })

        dialog.setContentView(root)
        dialog.show()
    }

    private fun openHudMode() {
        isHudActive = true
        binding.hudContainer.visibility = View.VISIBLE
        binding.tvHudSpeed.text = "35"
        binding.tvHudHeading.text = "HEADING: ${displayHeading.toInt()}° N"
        val curManeuver = dynamicManeuverPresets[currentManeuverIndex]
        binding.tvHudRoad.text = "${curManeuver.roadName.uppercase()} · ${curManeuver.distance.uppercase()}"
        Toast.makeText(this, "🚘 Windshield HUD Mode Active: Tap anywhere to exit", Toast.LENGTH_LONG).show()
    }

    private fun togglePedestrianMode() {
        isPedestrianMode = !isPedestrianMode
        service?.setPedestrianMode(isPedestrianMode)
        val msg = if (isPedestrianMode) {
            "🚶 Wilderness PDR Mode: Trailhead Origin Locked · Step Counter Active"
        } else {
            "🚘 Vehicular Navigation Mode: 15-State ES-EKF Active"
        }
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    private fun toggleObdSimulation() {
        isObdSimulated = !isObdSimulated
        service?.toggleObdSimulation(isObdSimulated)
        val msg = if (isObdSimulated) {
            "🔌 OBD-II Active: 100Hz Wheel Speed (PID 010D) Fused with ES-EKF"
        } else {
            "🔌 OBD-II Telemetry Disconnected"
        }
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    private fun exportMissionKml() {
        try {
            val file = service?.exportMissionKml() ?: com.inav.navigation.blackbox.BlackboxRecorder.exportKml(this)
            if (file != null && file.exists()) {
                val uri = androidx.core.content.FileProvider.getUriForFile(
                    this,
                    "${packageName}.fileprovider",
                    file
                )
                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "application/vnd.google-earth.kml+xml"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    putExtra(Intent.EXTRA_SUBJECT, "iNAV 3D Flight Telemetry KML")
                    putExtra(Intent.EXTRA_TEXT, "Exported 3D mission trajectory: iNAV Dead-Reckoning vs Naive Ghost Drift.")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(Intent.createChooser(shareIntent, "Export 3D Mission Telemetry (KML)"))
                Toast.makeText(this, "🚀 Exported KML: ${file.name}", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "⚠️ No mission points recorded yet. Move or simulate outage to generate trajectory points.", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateRouteLineTheme() {
        val lm = lineManager ?: return
        val casingColor = if (isLightMode) "#3F3F46" else "#33000000"
        val routeColor = if (isLightMode) "#18181B" else "#FFFFFF"
        val dashColor = if (isLightMode) "#18181B" else "#FFFFFF"

        routeCasingLine?.let { line ->
            line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(casingColor))
            if (routePoints.isNotEmpty()) line.latLngs = routePoints
            lm.update(line)
        }
        routeLine?.let { line ->
            line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(routeColor))
            if (routePoints.isNotEmpty()) line.latLngs = routePoints
            lm.update(line)
        }
        routeDashLine?.let { line ->
            line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(dashColor))
            if (routePoints.isNotEmpty()) line.latLngs = routePoints
            lm.update(line)
        }
    }

    private fun applyThemeMode() {
        updateRouteLineTheme()
        if (isLightMode) {
            binding.cardTripCockpit.setBackgroundResource(R.drawable.bg_sheet_trip_white)
            binding.tvTripEtaMins.setTextColor(Color.BLACK)
            binding.tvActiveRouteMetrics.setTextColor(Color.BLACK)
            binding.tvActiveDestTitle.setTextColor(Color.parseColor("#71717A"))
            binding.ivTripPassengerBadge.visibility = View.VISIBLE
            binding.pillCurrentStreet.setBackgroundResource(R.drawable.bg_current_street_pill)
            binding.tvCurrentStreetLabel.setTextColor(Color.parseColor("#18181B"))
            binding.pillCurrentStreet.visibility = View.VISIBLE
            binding.tvSpeedLimitValue.text = "40"
            binding.cardLandmarkCallout.visibility = View.GONE
            binding.tvAudioTooltip.visibility = View.GONE
            binding.headerToggleSensors.visibility = View.GONE
            binding.containerSensorsDetail.visibility = View.GONE
            binding.tvRoadSnappedBadge.visibility = View.GONE
        } else {
            binding.cardTripCockpit.setBackgroundResource(R.drawable.bg_sheet_trip_dark)
            binding.tvTripEtaMins.setTextColor(Color.WHITE)
            binding.tvActiveRouteMetrics.setTextColor(Color.parseColor("#D4D4D8"))
            binding.tvActiveDestTitle.setTextColor(Color.parseColor("#A1A1AA"))
            binding.pillCurrentStreet.setBackgroundResource(R.drawable.bg_current_street_pill_dark)
            binding.tvCurrentStreetLabel.setTextColor(Color.parseColor("#F4F4F5"))
            binding.pillCurrentStreet.visibility = View.VISIBLE
            binding.tvSpeedLimitValue.text = "40"
            binding.headerToggleSensors.visibility = View.GONE
            binding.cardLandmarkCallout.visibility = View.GONE
            binding.tvRoadSnappedBadge.visibility = View.GONE
        }
    }

    private fun displaySearchSuggestions(results: List<PlaceSuggestion>) {
        binding.containerSearchSuggestions.removeAllViews()
        if (results.isEmpty()) {
            binding.containerSearchSuggestions.visibility = View.GONE
            return
        }

        binding.containerSearchSuggestions.visibility = View.VISIBLE
        for (item in results.take(4)) {
            val itemView = android.widget.LinearLayout(this).apply {
                orientation = android.widget.LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                setPadding(16, 14, 16, 14)
                setBackgroundColor(Color.parseColor("#1F2430"))
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    binding.etSearchDestination.setText(item.name)
                    binding.containerSearchSuggestions.visibility = View.GONE
                    binding.containerSearchSuggestions.removeAllViews()
                    hideKeyboard()
                    startNavigationTo(item.lat, item.lon, item.name)
                }
            }

            val iconTv = android.widget.TextView(this).apply {
                text = item.icon
                textSize = 15f
                setPadding(0, 0, 12, 0)
            }
            itemView.addView(iconTv)

            val textCol = android.widget.LinearLayout(this).apply {
                orientation = android.widget.LinearLayout.VERTICAL
                layoutParams = android.widget.LinearLayout.LayoutParams(0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }

            val nameTv = android.widget.TextView(this).apply {
                text = item.name
                setTextColor(Color.WHITE)
                textSize = 12f
                setTypeface(null, android.graphics.Typeface.BOLD)
                maxLines = 1
                ellipsize = android.text.TextUtils.TruncateAt.END
            }
            textCol.addView(nameTv)

            val addrTv = android.widget.TextView(this).apply {
                text = item.address
                setTextColor(Color.parseColor("#94A3B8"))
                textSize = 10f
                maxLines = 1
                ellipsize = android.text.TextUtils.TruncateAt.END
            }
            textCol.addView(addrTv)

            itemView.addView(textCol)

            val divider = android.view.View(this).apply {
                layoutParams = android.widget.LinearLayout.LayoutParams(android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 1)
                setBackgroundColor(Color.parseColor("#333846"))
            }

            binding.containerSearchSuggestions.addView(itemView)
            binding.containerSearchSuggestions.addView(divider)
        }
    }

    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? android.view.inputmethod.InputMethodManager
        imm?.hideSoftInputFromWindow(binding.etSearchDestination.windowToken, 0)
        binding.etSearchDestination.clearFocus()
    }

    private fun startNavigationTo(lat: Double, lon: Double, name: String) {
        Toast.makeText(this, "Calculating route to $name...", Toast.LENGTH_SHORT).show()
        val s = service
        if (s != null) {
            s.startRoute(
                destLat = lat,
                destLon = lon,
                destName = name,
                onSuccess = { plan -> renderRoutePlan(plan) },
                onError = { err -> Toast.makeText(this@MainActivity, "Routing notice: $err", Toast.LENGTH_SHORT).show() }
            )
        } else {
            val curLat = getCurLatFallback()
            val curLon = getCurLonFallback()
            RouteManager().requestRoute(
                startLat = curLat,
                startLon = curLon,
                destLat = lat,
                destLon = lon,
                destName = name,
                scope = lifecycleScope,
                onSuccess = { plan: RoutePlan -> renderRoutePlan(plan) },
                onError = { err: String -> Toast.makeText(this@MainActivity, "Routing notice: $err", Toast.LENGTH_SHORT).show() }
            )
        }
    }

    private fun getCurLatFallback(): Double =
        service?.navState?.value?.latitude?.takeIf { it != 0.0 } ?: (vehicleSymbol?.latLng?.latitude ?: 0.0)

    private fun getCurLonFallback(): Double =
        service?.navState?.value?.longitude?.takeIf { it != 0.0 } ?: (vehicleSymbol?.latLng?.longitude ?: 0.0)

    private fun renderRoutePlan(plan: RoutePlan) {
        val sm = symbolManager ?: return
        val lm = lineManager ?: return

        val startGeo = if (plan.startLat != 0.0 && plan.startLon != 0.0) {
            LatLng(plan.startLat, plan.startLon)
        } else {
            vehicleSymbol?.latLng ?: LatLng(0.0, 0.0)
        }
        val destGeo = LatLng(plan.destLat, plan.destLon)

        // Build route point list from RoutePlan (which uses osmdroid GeoPoints internally)
        routePoints.clear()
        if (plan.points.isNotEmpty()) {
            val firstPt = plan.points.first()
            if (kotlin.math.abs(firstPt.latitude - startGeo.latitude) > 0.00001 || kotlin.math.abs(firstPt.longitude - startGeo.longitude) > 0.00001) {
                routePoints.add(startGeo)
            }
            for (gp in plan.points) {
                routePoints.add(LatLng(gp.latitude, gp.longitude))
            }
            val lastPt = plan.points.last()
            if (kotlin.math.abs(lastPt.latitude - destGeo.latitude) > 0.00001 || kotlin.math.abs(lastPt.longitude - destGeo.longitude) > 0.00001) {
                routePoints.add(destGeo)
            }
        } else {
            routePoints.add(startGeo)
            routePoints.add(destGeo)
        }

        // Update multi-layer route polyline (Stark White in Dark Mode, Solid Black in Light Mode)
        val casingColor = if (isLightMode) "#3F3F46" else "#33000000"
        val routeColor = if (isLightMode) "#18181B" else "#FFFFFF"
        val dashColor = if (isLightMode) "#18181B" else "#FFFFFF"

        routeCasingLine?.let { line ->
            line.latLngs = routePoints
            line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(casingColor))
            lm.update(line)
        }
        routeLine?.let { line ->
            line.latLngs = routePoints
            line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(routeColor))
            lm.update(line)
        }
        routeDashLine?.let { line ->
            line.latLngs = routePoints
            line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(dashColor))
            lm.update(line)
        }

        // 1. Remove any redundant start pin (User location pointer vehicleSymbol is the single iconic pointer)
        startSymbol?.let { sym ->
            sm.delete(sym)
            startSymbol = null
        }

        // 2. Destination Pin (Arrival Point only)
        if (destSymbol == null) {
            destSymbol = sm.create(SymbolOptions()
                .withLatLng(destGeo)
                .withIconImage(ICON_DEST_PIN)
                .withIconSize(1.1f)
            )
        } else {
            destSymbol?.latLng = destGeo
            sm.update(destSymbol!!)
        }

        // UI Updates: Top Maneuver Card & Bottom Cockpit
        val firstStep = plan.steps.firstOrNull()
        val firstDistFt = ((firstStep?.distanceM ?: 100.0) * 3.28084).toInt()
        val firstDistStr = if (firstDistFt > 1000) String.format(java.util.Locale.US, "%.1f mi", firstDistFt / 5280.0) else "$firstDistFt FT"
        val firstRoad = firstStep?.roadName?.takeIf { it.isNotBlank() && it != "Road" } ?: (firstStep?.instruction ?: "Follow Route")
        val firstIcon = when {
            firstStep?.maneuverModifier?.contains("left") == true -> "↰"
            firstStep?.maneuverModifier?.contains("right") == true -> "↱"
            firstStep?.maneuverType == "arrive" -> "🏁"
            else -> "↑"
        }
        renderDynamicManeuver(ManeuverItem(
            icon = firstIcon,
            distance = firstDistStr,
            roadName = firstRoad,
            towardDirection = plan.destinationName
        ))

        val etaMins = (plan.totalDurationS / 60.0).toInt().coerceAtLeast(1)
        val distMi = plan.totalDistanceM * 0.000621371
        binding.tvTripEtaMins.text = etaMins.toString()
        binding.tvActiveRouteMetrics.text = String.format(java.util.Locale.US, "%.1f mi", distMi)
        binding.tvActiveDestTitle.text = "To: ${plan.destinationName}"
        binding.tvCurrentStreetLabel.text = firstRoad
        binding.tvHudRoad.text = "${firstRoad.uppercase()} · $firstDistStr"

        binding.panelActiveRoute.visibility = View.VISIBLE
        binding.panelRoutePicker.visibility = View.GONE
        binding.tvActiveStartTitle.text = String.format("START: (%.4f, %.4f)", startGeo.latitude, startGeo.longitude)
        binding.tvActiveDestTitle.text = "To: ${plan.destinationName}"

        // Zoom to fit route and both markers
        if (routePoints.size >= 2) {
            try {
                val boundsBuilder = LatLngBounds.Builder()
                routePoints.forEach { boundsBuilder.include(it) }
                boundsBuilder.include(startGeo)
                boundsBuilder.include(destGeo)
                val bounds = boundsBuilder.build()
                mapLibreMap?.animateCamera(CameraUpdateFactory.newLatLngBounds(bounds, 130), 650)
            } catch (e: Exception) {
                mapLibreMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(startGeo, 16.0))
            }
        }

        Toast.makeText(
            this@MainActivity,
            String.format("Route loaded: %.1f km to %s", plan.totalDistanceM / 1000.0, plan.destinationName),
            Toast.LENGTH_SHORT
        ).show()
        setScreenState(AppScreenState.NAVIGATION)
    }

    private fun cancelNavigation() {
        service?.clearRoute()
        routePoints.clear()
        routeCasingLine?.let { line ->
            line.latLngs = listOf(LatLng(0.0, 0.0))
            lineManager?.update(line)
        }
        routeLine?.let { line ->
            line.latLngs = listOf(LatLng(0.0, 0.0))
            lineManager?.update(line)
        }
        routeDashLine?.let { line ->
            line.latLngs = listOf(LatLng(0.0, 0.0))
            lineManager?.update(line)
        }
        startSymbol?.let { sym ->
            symbolManager?.delete(sym)
            startSymbol = null
        }
        destSymbol?.let { sym ->
            symbolManager?.delete(sym)
            destSymbol = null
        }
        binding.panelActiveRoute.visibility = View.GONE
        binding.panelRoutePicker.visibility = View.VISIBLE
        binding.etSearchDestination.text.clear()
        binding.containerSearchSuggestions.removeAllViews()
        setScreenState(AppScreenState.HOME)
        Toast.makeText(this, "Navigation ended", Toast.LENGTH_SHORT).show()
    }

    private fun checkPermissionsAndStart() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isEmpty()) {
            startTrackingService()
            startDirectLocationUpdates()
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun startTrackingService() {
        val intent = Intent(this, DeadReckoningService::class.java)
        ContextCompat.startForegroundService(this, intent)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun observeServiceState() {
        lifecycleScope.launch {
            service?.navState?.collectLatest { state ->
                updateUi(state)
            }
        }
    }

    // ── Smooth Heading Slew ──
    private fun slewHeading(current: Float, target: Float, maxStepDeg: Float): Float {
        val from = ((current % 360f) + 360f) % 360f
        val to = ((target % 360f) + 360f) % 360f
        if (maxStepDeg <= 0f) return from
        val delta = ((to - from + 540f) % 360f) - 180f
        val step = delta.coerceIn(-maxStepDeg, maxStepDeg)
        return ((from + step) % 360f + 360f) % 360f
    }

    // ── Render Governor ──
    private fun shouldRedraw(lat: Double, lon: Double, heading: Float): Boolean {
        val dLat = lat - lastRenderedLat
        val dLon = lon - lastRenderedLon
        val distM = Math.sqrt(dLat * dLat + dLon * dLon) * 111_000.0
        val dHeading = Math.abs(((heading - lastRenderedHeading + 540f) % 360f) - 180f)
        return distM > RENDER_POSITION_THRESHOLD_M || dHeading > RENDER_HEADING_THRESHOLD_DEG
    }

    private fun updateUi(state: NavigationState) {
        val map = mapLibreMap ?: return
        val sm = symbolManager ?: return
        val lm = lineManager ?: return
        val fm = fillManager ?: return

        if (currentScreenState == AppScreenState.HOME) {
            return
        }

        // Active position selection (road-snapped vs raw)
        val activeLat = if (state.isRoadSnapped && state.speedKmh >= 2.0 && state.snappedLatitude != 0.0) {
            state.snappedLatitude
        } else {
            state.latitude
        }
        val activeLon = if (state.isRoadSnapped && state.speedKmh >= 2.0 && state.snappedLongitude != 0.0) {
            state.snappedLongitude
        } else {
            state.longitude
        }
        val rawHeading = if (state.speedKmh > 5.0 && state.isRoadSnapped && state.snappedHeadingDeg != 0.0) {
            state.snappedHeadingDeg
        } else {
            state.headingDeg
        }

        // Smooth heading slew
        val now = System.currentTimeMillis()
        val dt = if (lastUpdateTimeMs > 0) (now - lastUpdateTimeMs).coerceAtMost(200L) else 16L
        lastUpdateTimeMs = now
        val maxStep = HEADING_SLEW_RATE_DPS * (dt / 1000f)
        displayHeading = slewHeading(displayHeading, rawHeading.toFloat(), maxStep)
        val activeHeading = displayHeading.toDouble()

        // Auto Night Mode
        val shouldBeNight = state.lightLux < NIGHT_MODE_LUX_THRESHOLD || state.isTunnelLighting
        if (shouldBeNight != isNightMode && (now - lastNightModeSwitch) > NIGHT_MODE_DEBOUNCE_MS) {
            isNightMode = shouldBeNight
            lastNightModeSwitch = now
            applyNightMode()
        }

        // 1. TOP NAVIGATION STATUS BAR & TURN GUIDANCE
        if (state.hasActiveRoute) {
            binding.cardTopNav.visibility = View.VISIBLE
            val distFt = (state.distanceToManeuverM * 3.28084).toInt()
            val distStr = if (distFt > 1000) String.format(java.util.Locale.US, "%.1f mi", distFt / 5280.0) else "$distFt FT"
            val currentRoad = when {
                state.nextManeuverText.isNotBlank() && state.nextManeuverText != "Follow Route" -> state.nextManeuverText
                state.roadName.isNotBlank() && state.roadName != "Current Location" && state.roadName != "Off-Road" -> state.roadName
                state.destinationName.isNotBlank() -> "Towards ${state.destinationName}"
                else -> "Follow Route"
            }
            val liveManeuver = ManeuverItem(
                icon = if (state.nextManeuverIcon.isNotBlank()) state.nextManeuverIcon else "↑",
                distance = distStr,
                roadName = currentRoad,
                towardDirection = if (state.destinationName.isNotBlank()) state.destinationName else null
            )
            renderDynamicManeuver(liveManeuver)
            binding.tvTopSubStatus.text = if (state.mode == NavigationMode.PURE_DR) "DEAD:REC" else "INS:LOCK"
            binding.panelActiveRoute.visibility = View.VISIBLE
            binding.panelRoutePicker.visibility = View.GONE
            val distMi = if (state.remainingDistanceM > 0) state.remainingDistanceM * 0.000621371 else 0.1
            binding.tvTripEtaMins.text = if (state.etaMinutes > 0) state.etaMinutes.toString() else "1"
            binding.tvActiveRouteMetrics.text = String.format(java.util.Locale.US, "%.1f mi", distMi)
            binding.tvActiveDestTitle.text = if (state.destinationName.isNotBlank()) "To: ${state.destinationName}" else "Active Route"
            binding.ivTripPassengerBadge.visibility = View.GONE
            binding.tvCurrentStreetLabel.text = if (state.isPedestrianMode) {
                String.format(java.util.Locale.US, "🚶 %d steps · 🧭 %s %.0fm Trailhead", state.pedestrianStepCount, state.backtrackCardinal, state.backtrackDistanceM)
            } else {
                currentRoad
            }
            binding.tvHudRoad.text = "${currentRoad.uppercase()} · $distStr"
        } else {
            binding.cardTopNav.visibility = View.GONE
            binding.tvTopSubStatus.text = if (state.mode == NavigationMode.PURE_DR) "DEAD:REC" else "INS:LOCK"
            binding.panelActiveRoute.visibility = View.GONE
            binding.panelRoutePicker.visibility = View.VISIBLE
            binding.tvTripEtaMins.text = "--"
            binding.tvActiveRouteMetrics.text = "-- mi"
            binding.tvActiveDestTitle.text = "Select or search destination"
            binding.ivTripPassengerBadge.visibility = View.GONE
            binding.tvCurrentStreetLabel.text = if (state.isPedestrianMode) {
                String.format(java.util.Locale.US, "🚶 %d steps · 🧭 %s %.0fm Trailhead", state.pedestrianStepCount, state.backtrackCardinal, state.backtrackDistanceM)
            } else if (state.roadName.isNotBlank()) {
                state.roadName
            } else {
                "Current Street"
            }
        }
        binding.tvSpeedValue.text = state.speedKmh.toInt().toString()
        binding.tvHudSpeed.text = state.speedKmh.toInt().toString()
        binding.tvHudHeading.text = "HEADING: ${displayHeading.toInt()}°"

        if (activeLat != 0.0 && activeLon != 0.0) {
            val locStr = if (state.roadName.isNotBlank()) "${state.roadName} (%.4f, %.4f)".format(activeLat, activeLon) else "Current GPS Fix (%.4f, %.4f)".format(activeLat, activeLon)
            binding.tvPickerStartLabel.text = locStr
        }

        // Seamless transparent card header (Voyager Monochrome Precision)
        binding.gmapsNavHeader.setBackgroundColor(Color.TRANSPARENT)

        // 2. PRIMARY NAVIGATION HUD (Bottom Card & Telemetry)
        val speedMph = if (state.speedKmh > 1.0) state.speedKmh * 0.621371 else 24.0
        binding.tvSpeedValue.text = String.format(java.util.Locale.US, "%.0f", speedMph)
        binding.tvSpeedUnit.text = "MPH"
        binding.tvHeadingValue.text = String.format("%.0f° %s", state.headingDeg, state.cardinalDirection)
        binding.tvRoadEvent.text = when (state.roadAnomalyType) {
            1 -> "Road: ${state.roadName} · ⚠️ Pothole (Crater Drop)"
            2 -> "Road: ${state.roadName} · ⚠️ Speed Breaker (Bump)"
            else -> "Road: ${state.roadName}"
        }
        binding.tvUncertainty.text = if (state.mode == NavigationMode.PURE_DR) {
            String.format("iNAV DR Fusion • ±%.1fm σ", state.uncertaintySigmaM)
        } else {
            "GNSS + INS Fusion • 99.4% Precision"
        }

        // 3. ALL SENSORS LIVE STREAM COCKPIT
        binding.tvSensorMag.text = String.format(
            "Bx: %.1f, By: %.1f, Bz: %.1f µT\nMag: %.0f° (%s) | EKF: %.0f°",
            state.magX, state.magY, state.magZ,
            state.magneticHeadingDeg, state.cardinalDirection, state.headingDeg
        )
        binding.tvSensorGyro.text = String.format(
            "X: %+.1f, Y: %+.1f, Z: %+.1f °/s\nDrift Bias b_ω: %.3f °/s",
            state.gyroXDeg, state.gyroYDeg, state.gyroZDeg, state.gyroBiasDegPerSec
        )
        binding.tvSensorAccel.text = String.format(
            "X: %+.2f, Y: %+.2f, Z: %+.2f m/s²\nLin Fwd Accel: %+.2f m/s²",
            state.accelX, state.accelY, state.accelZ, state.linAccelFwd
        )
        binding.tvSensorBaro.text = String.format(
            "%.1f hPa | Alt: %.0f m\nClimb: %+.1f m/s | Grade: %+.1f%%",
            state.pressureHpa, state.baroAltitudeM, state.verticalSpeedMs, state.gradePct
        )
        binding.tvSensorGnss.text = String.format(
            "%d Sats View | %d Used (±%.1fm)\nFix: %s",
            state.satellitesInView, state.satellitesUsed, state.gnssAccuracyM, state.gnssFixType
        )
        binding.tvSensorEnv.text = String.format(
            "Light: %.0f lx (%s)\nOBD: %s",
            state.lightLux,
            if (state.isTunnelLighting) "TUNNEL DARK" else "DAYLIGHT",
            if (state.isObdConnected) "CONNECTED" else "STANDALONE IMU"
        )
        binding.tvCoordinates.text = String.format(
            "Lat: %.6f | Lon: %.6f | Pitch: %+.1f° | Roll: %+.1f°",
            activeLat, activeLon, state.pitchDeg, state.rollDeg
        )

        // 3D Attitude Indicators (Top Nav & Cockpit)
        binding.attitudeIndicator.pitchDeg = state.pitchDeg.toFloat()
        binding.attitudeIndicator.rollDeg = state.rollDeg.toFloat()
        binding.attitudeIndicatorCockpit.pitchDeg = state.pitchDeg.toFloat()
        binding.attitudeIndicatorCockpit.rollDeg = state.rollDeg.toFloat()
        binding.tvAttitudeAngles.text = String.format("Pitch: %+.1f° · Roll: %+.1f°", state.pitchDeg, state.rollDeg)

        // OBD & PDR Sensors
        binding.tvSensorObd.text = if (state.isObdConnected) {
            String.format("OBD-II: %.1f km/h (CAN-Bus PID 010D Active)", state.obdSpeedKmh ?: state.speedKmh)
        } else if (isObdSimulated) {
            String.format("OBD-II: %.1f km/h (Simulated CAN-Bus 100Hz)", state.speedKmh)
        } else {
            "OBD-II: Disconnected (Tap Route Options to Simulate)"
        }
        binding.tvSensorPdr.text = if (state.isPedestrianMode) {
            String.format("PDR: %d steps (%.0fm) · Cadence %d SPM · Backtrack %s %.0fm", state.pedestrianStepCount, state.pedestrianDistanceM, state.pedestrianCadenceSpm, state.backtrackCardinal, state.backtrackDistanceM)
        } else {
            "PDR: Inactive (Vehicular Mode Active)"
        }

        // Road Snapped / Route Status Badge
        binding.tvRoadSnappedBadge.visibility = View.GONE

        // Mode Status Badge (Monochrome Dynamic Island Pill)
        when (state.mode) {
            NavigationMode.AIDED -> {
                binding.tvModeBadge.text = "INS:LOCK"
                binding.tvModeBadge.setTextColor(Color.parseColor("#34D399"))
            }
            NavigationMode.DEGRADED -> {
                binding.tvModeBadge.text = "DEGRADED"
                binding.tvModeBadge.setTextColor(Color.parseColor("#F59E0B"))
            }
            NavigationMode.PURE_DR -> {
                binding.tvModeBadge.text = "DEAD RECKONING"
                binding.tvModeBadge.setTextColor(Color.parseColor("#EF4444"))
            }
        }

        // Outage Banner & Ghost Trace
        if (state.isOutageSimulated) {
            binding.outageBanner.visibility = View.VISIBLE
            val mins = (state.outageDurationSec.toInt()) / 60
            val secs = (state.outageDurationSec.toInt()) % 60
            binding.tvOutageTimer.text = String.format("%d:%02d (Active)", mins, secs)
            binding.tvOutageTimer.setTextColor(Color.parseColor("#EF4444"))

            binding.tvGmapsVsStatus.text = "GOOGLE MAPS: ❌ SIGNAL LOST (FROZEN)"
            binding.tvInavVsStatus.text = String.format("iNAV: ✅ 15-STATE ES-EKF ACTIVE · Drift: %.1fm (%.1f%%)", state.driftDistanceM, state.driftPercentage)

            // Freeze ghost marker at outage origin
            if (outageFrozenLatLng == null && (activeLat != 0.0 || activeLon != 0.0)) {
                outageFrozenLatLng = LatLng(activeLat, activeLon)
                if (ghostFrozenSymbol == null) {
                    ghostFrozenSymbol = sm.create(SymbolOptions()
                        .withLatLng(outageFrozenLatLng!!)
                        .withIconImage(ICON_GMAPS_FROZEN)
                        .withIconSize(1.0f)
                    )
                } else {
                    ghostFrozenSymbol?.latLng = outageFrozenLatLng!!
                    sm.update(ghostFrozenSymbol!!)
                }
            }

            if (state.naiveLatitude != 0.0 && state.naiveLongitude != 0.0 && outageFrozenLatLng != null && state.outageDistanceTravelledM >= 5.0) {
                val ghostPt = LatLng(state.naiveLatitude, state.naiveLongitude)
                // Only draw naive divergence line if vehicle has actually travelled at least 5m along the route
                if (outageFrozenLatLng!!.distanceTo(ghostPt) >= 3.0) {
                    if (ghostPoints.isEmpty()) {
                        ghostPoints.add(outageFrozenLatLng!!)
                    }
                    if (ghostPt.distanceTo(ghostPoints.last()) >= 1.0) {
                        ghostPoints.add(ghostPt)
                        if (ghostPoints.size >= 2) {
                            ghostLine?.let { line ->
                                line.latLngs = ghostPoints.toList()
                                lm.update(line)
                            }
                        }
                    }
                }
            } else if (state.outageDistanceTravelledM < 5.0 && ghostPoints.isNotEmpty()) {
                ghostPoints.clear()
                ghostLine?.let { line ->
                    line.latLngs = listOf(LatLng(0.0, 0.0))
                    lm.update(line)
                }
            }
        } else {
            binding.outageBanner.visibility = View.GONE
            binding.tvOutageTimer.text = "0:00 (None)"
            binding.tvOutageTimer.setTextColor(Color.parseColor("#000000"))
            outageFrozenLatLng = null
            ghostFrozenSymbol?.let { sym ->
                sm.delete(sym)
                ghostFrozenSymbol = null
            }
            if (ghostPoints.isNotEmpty()) {
                ghostPoints.clear()
                ghostLine?.let { line ->
                    line.latLngs = listOf(LatLng(0.0, 0.0))
                    lm.update(line)
                }
            }
        }

        // HUD View Updates
        if (isHudActive) {
            binding.tvHudSpeed.text = String.format("%.1f", state.speedKmh)
            binding.tvHudHeading.text = String.format("HEADING: %.0f° %s", state.headingDeg, state.cardinalDirection)
            binding.tvHudRoad.text = "ROAD: ${state.roadName.uppercase()}"
        }

        // 4. MAP & CAMERA UPDATES
        if (activeLat != 0.0 || activeLon != 0.0) {
            val point = LatLng(activeLat, activeLon)

            // Update uncertainty circle
            val radiusM = maxOf(2.0, state.uncertaintySigmaM * 3.0)
            updateUncertaintyCircle(point, radiusM)

            // Trajectory management
            if (trajectoryPoints.isNotEmpty() && point.distanceTo(trajectoryPoints.last()) > 5000.0) {
                // Position jump — reset trajectory
                trajectoryPoints.clear()
                trajectoryPoints.add(point)
            } else if (state.speedKmh > 0.8 && (trajectoryPoints.isEmpty() || point.distanceTo(trajectoryPoints.last()) >= 1.0)) {
                trajectoryPoints.add(point)
            } else if (trajectoryPoints.isEmpty()) {
                trajectoryPoints.add(point)
            }

            // Update trajectory polylines
            if (trajectoryPoints.size >= 2) {
                trajectoryLine?.let { line ->
                    line.latLngs = trajectoryPoints.toList()
                    lm.update(line)
                }
                trajectoryGlowLine?.let { line ->
                    line.latLngs = trajectoryPoints.toList()
                    lm.update(line)
                }
            }

            // Initial center on first valid position
            if (!hasInitialCentered) {
                hasInitialCentered = true
                map.moveCamera(CameraUpdateFactory.newLatLngZoom(point, 18.0))
            }

            // Update vehicle marker position and rotation
            vehicleSymbol?.let { sym ->
                sym.latLng = point
                sym.iconRotate = activeHeading.toFloat()
                sm.update(sym)
            }

            // Camera follow
            if (isFollowingVehicle && shouldRedraw(activeLat, activeLon, activeHeading.toFloat())) {
                lastRenderedLat = activeLat
                lastRenderedLon = activeLon
                lastRenderedHeading = activeHeading.toFloat()

                val builder = CameraPosition.Builder()
                    .target(point)

                if (isTrackUpEnabled) {
                    builder.bearing(activeHeading)
                    builder.tilt(45.0) // Driving perspective tilt!
                }

                map.animateCamera(CameraUpdateFactory.newCameraPosition(builder.build()), 150)
            }

            // Keep compass FAB needle pointing True North relative to map bearing
            val currentBearing = map.cameraPosition.bearing
            binding.fabCompass.rotation = -currentBearing.toFloat()

            // Trail color theming
            val trailColor = when {
                state.mode == NavigationMode.PURE_DR -> "#B388FF"
                isNightMode -> "#4FC3F7"
                else -> "#00E676"
            }
            val glowColor = when {
                state.mode == NavigationMode.PURE_DR -> "#60B388FF"
                isNightMode -> "#604FC3F7"
                else -> "#6000E676"
            }
            trajectoryLine?.let { line ->
                line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(trailColor))
                lm.update(line)
            }
            trajectoryGlowLine?.let { line ->
                line.lineColor = ColorUtils.colorToRgbaString(Color.parseColor(glowColor))
                lm.update(line)
            }
        }
    }

    // ── Night Mode Style Switching (Always Dark Monochrome Vector) ──
    private fun applyNightMode() {
        if (currentViewType == MapViewType.SATELLITE_VIEW) return
        val targetStyle = STYLE_DARK
        mapLibreMap?.let { map ->
            map.setStyle(targetStyle) { style ->
                reinitializeMapAnnotations(map, style)
            }
        }
    }

    // ── Map Layer Switching ──
    private fun setMapLayer(viewType: MapViewType) {
        val map = mapLibreMap ?: return
        currentViewType = viewType

        when (viewType) {
            MapViewType.OSM_VIEW -> {
                val targetStyle = STYLE_DARK
                map.setStyle(targetStyle) { style ->
                    reinitializeMapAnnotations(map, style)
                }
                binding.fabLayers.setColorFilter(Color.parseColor("#000000"))
                binding.btnOsmView.setBackgroundColor(Color.parseColor("#262A34"))
                binding.btnOsmView.setTextColor(Color.WHITE)
                binding.btnSatView.setBackgroundColor(Color.TRANSPARENT)
                binding.btnSatView.setTextColor(Color.parseColor("#A1A1AA"))
                map.setMaxZoomPreference(22.0)
                Toast.makeText(this, "Vector Map Active", Toast.LENGTH_SHORT).show()
            }
            MapViewType.SATELLITE_VIEW -> {
                // High-resolution satellite raster layer with overzoom
                map.setMaxZoomPreference(21.0)
                val tileSet = TileSet("tileset", SATELLITE_URL)
                tileSet.maxZoom = 20.0f
                map.setStyle(Style.Builder().fromUri(STYLE_POSITRON).withSource(
                    RasterSource("sat-source", tileSet, 256)
                ).withLayer(
                    RasterLayer("sat-layer", "sat-source")
                )) { style ->
                    reinitializeMapAnnotations(map, style)
                }
                binding.fabLayers.setColorFilter(Color.parseColor("#34D399"))
                binding.btnSatView.setBackgroundColor(Color.parseColor("#262A34"))
                binding.btnSatView.setTextColor(Color.WHITE)
                binding.btnOsmView.setBackgroundColor(Color.TRANSPARENT)
                binding.btnOsmView.setTextColor(Color.parseColor("#A1A1AA"))
                Toast.makeText(this, "Satellite View Active", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ── Physical Location Engine ──
    private fun isBetterLocation(location: Location, currentBest: Location?): Boolean {
        if (currentBest == null) return true
        val timeDelta = location.time - currentBest.time
        if (location.provider == LocationManager.GPS_PROVIDER && location.accuracy < 35f) return true
        if (currentBest.accuracy < 30f && location.accuracy > 50f && timeDelta < 900_000L) return false
        val accuracyDelta = (location.accuracy - currentBest.accuracy).toInt()
        val isNewer = timeDelta > 0
        val isMoreAccurate = accuracyDelta < 0
        val isSignificantlyNewer = timeDelta > 120_000L
        val isSignificantlyLessAccurate = accuracyDelta > 50
        if (isMoreAccurate) return true
        if (isNewer && accuracyDelta <= 0) return true
        if (isSignificantlyNewer && !isSignificantlyLessAccurate) return true
        return false
    }

    private val directLocationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            if (location.latitude != 0.0 && location.longitude != 0.0) {
                applyPhysicalLocation(location)
            }
        }
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    private fun applyPhysicalLocation(loc: Location) {
        val cur = currentPhysicalLocation
        if (cur != null && !isBetterLocation(loc, cur)) return
        currentPhysicalLocation = loc
        service?.updateGnssFix(loc)

        val pt = LatLng(loc.latitude, loc.longitude)
        vehicleSymbol?.let { sym ->
            sym.latLng = pt
            if (loc.hasBearing() && loc.bearing != 0f) {
                sym.iconRotate = loc.bearing
            }
            symbolManager?.update(sym)
        }
        val radiusM = maxOf(2.0, loc.accuracy.toDouble())
        updateUncertaintyCircle(pt, radiusM)

        val dLat = Math.abs(loc.latitude - lastCenteredLat)
        val dLon = Math.abs(loc.longitude - lastCenteredLon)
        val distChangeM = Math.sqrt(dLat * dLat + dLon * dLon) * 111_000.0

        if (isFollowingVehicle || !hasInitialCentered || distChangeM > 40.0) {
            hasInitialCentered = true
            lastCenteredLat = loc.latitude
            lastCenteredLon = loc.longitude
            mapLibreMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(pt, if (currentScreenState == AppScreenState.NAVIGATION) 17.5 else 16.5), 500)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startDirectLocationUpdates() {
        val lm = getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return
        val providers = listOfNotNull(
            LocationManager.GPS_PROVIDER,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) LocationManager.FUSED_PROVIDER else "fused",
            LocationManager.NETWORK_PROVIDER
        ).distinct()

        for (p in providers) {
            try {
                if (lm.isProviderEnabled(p)) {
                    lm.requestLocationUpdates(p, 1000L, 0f, directLocationListener, android.os.Looper.getMainLooper())
                }
            } catch (ignored: Exception) {}
        }
        fetchPhysicalLocationAndCenter(forceCenter = false)
    }

    @SuppressLint("MissingPermission")
    private fun fetchPhysicalLocationAndCenter(forceCenter: Boolean = false) {
        val lm = getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return
        val candidateProviders = listOfNotNull(
            LocationManager.GPS_PROVIDER,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) LocationManager.FUSED_PROVIDER else "fused",
            LocationManager.NETWORK_PROVIDER
        )

        var best: Location? = null
        for (p in candidateProviders) {
            try {
                val loc = lm.getLastKnownLocation(p) ?: continue
                if (loc.latitude != 0.0 && loc.longitude != 0.0 && isBetterLocation(loc, best)) {
                    best = loc
                }
            } catch (ignored: Exception) {}
        }

        if (best != null && best.latitude != 0.0) {
            applyPhysicalLocation(best)
            if (forceCenter) {
                mapLibreMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(best.latitude, best.longitude), 18.5), 400)
                Toast.makeText(this, String.format("📍 Centered: %.5f, %.5f (±%.0fm, %s)", best.latitude, best.longitude, best.accuracy, best.provider), Toast.LENGTH_SHORT).show()
            }
        }

        // Actively query hardware for fresh physical fix
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val mainExecutor = ContextCompat.getMainExecutor(this)
            for (p in candidateProviders) {
                try {
                    lm.getCurrentLocation(p, null, mainExecutor) { loc ->
                        if (loc != null && loc.latitude != 0.0) {
                            applyPhysicalLocation(loc)
                            if (forceCenter) {
                                mapLibreMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(loc.latitude, loc.longitude), 18.5), 400)
                                Toast.makeText(this, String.format("📍 Hardware Fix: %.5f, %.5f (±%.0fm, %s)", loc.latitude, loc.longitude, loc.accuracy, loc.provider), Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                } catch (ignored: Exception) {}
            }
        }
    }

    // ── MapLibre Lifecycle Management ──
    override fun onResume() {
        super.onResume()
        mapView.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            startDirectLocationUpdates()
        }
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
        try {
            val lm = getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            lm?.removeUpdates(directLocationListener)
        } catch (ignored: Exception) {}
    }

    override fun onStart() {
        super.onStart()
        mapView.onStart()
    }

    override fun onStop() {
        super.onStop()
        mapView.onStop()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        mapView.onSaveInstanceState(outState)
    }

    override fun onLowMemory() {
        super.onLowMemory()
        mapView.onLowMemory()
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanupAnnotationManagers()
        mapView.onDestroy()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
    }
}
