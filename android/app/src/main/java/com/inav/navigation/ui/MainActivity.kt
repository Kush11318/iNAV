package com.inav.navigation.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.BlurMaskFilter
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.inav.navigation.R
import com.inav.navigation.databinding.ActivityMainBinding
import com.inav.navigation.model.NavigationMode
import com.inav.navigation.model.NavigationState
import com.inav.navigation.service.DeadReckoningService
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polygon
import org.osmdroid.views.overlay.Polyline
import org.osmdroid.views.overlay.gestures.RotationGestureOverlay

import android.annotation.SuppressLint
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.view.WindowManager
import android.widget.EditText
import android.widget.LinearLayout
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.inav.navigation.imagery.CesiumIonImageryProvider
import com.inav.navigation.routing.RouteManager
import com.inav.navigation.routing.RoutePlan
import com.inav.navigation.routing.PlaceSuggestion

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var service: DeadReckoningService? = null
    private var isBound = false

    private lateinit var mapView: MapView
    private var vehicleMarker: org.osmdroid.views.overlay.Marker? = null
    private val trajectoryPolyline = Polyline()
    private val ghostPolyline = Polyline()
    private val routePolyline = Polyline()
    private var startMarker: org.osmdroid.views.overlay.Marker? = null
    private var destinationMarker: org.osmdroid.views.overlay.Marker? = null
    private var googleMapsFrozenMarker: org.osmdroid.views.overlay.Marker? = null
    private var outageFrozenGeoPoint: GeoPoint? = null
    private lateinit var uncertaintyPolygon: Polygon
    // Glow underlay polyline for premium trail effect (drawn behind main trail)
    private val trajectoryGlowPolyline = Polyline()

    private var isOutageActive = false
    private var isHudActive = false
    private var isTrackUpEnabled = false
    private var isPedestrianMode = false
    private var isFollowingVehicle = true
    private var hasInitialCentered = false
    private var lastCenteredLat = 0.0
    private var lastCenteredLon = 0.0
    private var isUserTouchingMap = false
    private var currentPhysicalLocation: Location? = null

    // ── GEV-Inspired: Smooth Heading Slew (cockpitMath pattern) ──
    // Rate-limited heading changes prevent visual jumps
    private var displayHeading: Float = 0f
    private var lastUpdateTimeMs: Long = 0L
    /** Max heading rotation per second (degrees). GEV uses 60°/s for fast, 12°/s for slow. */
    private val HEADING_SLEW_RATE_DPS = 90f

    // ── GEV-Inspired: Render Governor ──
    // Only invalidate the map when position or heading actually changes
    private var lastRenderedLat = 0.0
    private var lastRenderedLon = 0.0
    private var lastRenderedHeading = 0f
    /** Minimum position change (meters) to trigger a map redraw */
    private val RENDER_POSITION_THRESHOLD_M = 0.5
    /** Minimum heading change (degrees) to trigger a map redraw */
    private val RENDER_HEADING_THRESHOLD_DEG = 0.5f

    // ── GEV-Inspired: Auto Night Mode ──
    // Switches map tiles and marker icon based on ambient light sensor
    private var isNightMode = false
    private var lastNightModeSwitch = 0L
    /** Lux threshold below which night mode activates */
    private val NIGHT_MODE_LUX_THRESHOLD = 50f
    /** Minimum time between night mode switches to prevent flicker (ms) */
    private val NIGHT_MODE_DEBOUNCE_MS = 5000L

    // ── Two Dedicated Map Views: OSM View and Satellite View ──
    enum class MapViewType {
        OSM_VIEW,
        SATELLITE_VIEW
    }
    private var currentViewType = MapViewType.OSM_VIEW

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

        // High-Performance Tile & RAM Cache (instant smooth zoom without checkerboards)
        val config = Configuration.getInstance()
        config.load(this, getSharedPreferences("osmdroid", Context.MODE_PRIVATE))
        config.userAgentValue = "iNAV-DeadReckoning/1.0 (contact@inav.app)"
        config.cacheMapTileCount = 240.toShort()        // 240 tiles in RAM (instant pinch zoom without re-downloading)
        config.cacheMapTileOvershoot = 80.toShort()     // Preload 80 surrounding tiles around visible viewport
        config.tileDownloadThreads = 8.toShort()        // 8 parallel HTTP download threads
        config.tileDownloadMaxQueueSize = 80.toShort()
        config.tileFileSystemThreads = 4.toShort()
        config.tileFileSystemCacheMaxBytes = 800L * 1024 * 1024 // 800MB persistent disk cache
        config.tileFileSystemCacheTrimBytes = 650L * 1024 * 1024
        config.expirationExtendedDuration = 1000L * 60 * 60 * 24 * 60 // Cache for 60 days
        config.animationSpeedDefault = 120              // Snappy 120ms smooth zoom transitions
        config.animationSpeedShort = 80

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initMap()
        setupListeners()
        checkPermissionsAndStart()
    }

    private fun initMap() {
        val fastTileSource = org.osmdroid.tileprovider.tilesource.XYTileSource(
            "FastOSM",
            0,
            19,
            256,
            ".png",
            arrayOf(
                "https://a.tile.openstreetmap.org/",
                "https://b.tile.openstreetmap.org/",
                "https://c.tile.openstreetmap.org/"
            ),
            "© OpenStreetMap contributors",
            org.osmdroid.tileprovider.tilesource.TileSourcePolicy(8, 0)
        )

        mapView = binding.mapView
        mapView.setDestroyMode(false)
        mapView.setTileSource(fastTileSource)
        mapView.setMultiTouchControls(true)
        mapView.isTilesScaledToDpi = true
        mapView.setTilesScaleFactor(1.1f)
        mapView.setZoomRounding(false) // Smooth continuous pinch zoom
        mapView.minZoomLevel = 4.0
        mapView.maxZoomLevel = 20.0
        mapView.controller.setZoom(18.0)

        // Enable Two-Finger Map Rotation (Google Maps Style)
        val rotationOverlay = RotationGestureOverlay(mapView).apply {
            isEnabled = true
        }
        mapView.overlays.add(rotationOverlay)

        // Dynamic 3-Sigma Covariance Uncertainty Circle (Google Maps Soft Blue Disc)
        uncertaintyPolygon = Polygon(mapView).apply {
            fillPaint.color = Color.parseColor("#204285F4") // Translucent Soft Blue
            outlinePaint.color = Color.parseColor("#404285F4") // Subtle Outline
            outlinePaint.strokeWidth = 1.5f
        }
        mapView.overlays.add(uncertaintyPolygon)

        // Planned Route Polyline (Google Maps Classic Vibrant Blue)
        routePolyline.outlinePaint.color = Color.parseColor("#1A73E8")
        routePolyline.outlinePaint.strokeWidth = 12f
        routePolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        routePolyline.outlinePaint.strokeJoin = Paint.Join.ROUND
        routePolyline.outlinePaint.isAntiAlias = true
        mapView.overlays.add(routePolyline)

        // ── GEV-Inspired: Glowing Trajectory Trail ──
        // Soft glow underlay drawn BEHIND the main crisp trail (GEV trailRenderer pattern)
        trajectoryGlowPolyline.outlinePaint.apply {
            color = Color.parseColor("#4000E676")
            strokeWidth = 22f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
            maskFilter = BlurMaskFilter(12f, BlurMaskFilter.Blur.NORMAL)
        }
        mapView.overlays.add(trajectoryGlowPolyline)

        // Trajectory Polyline (Active Road-Fused Trail — crisp core on top of glow)
        trajectoryPolyline.outlinePaint.apply {
            color = Color.parseColor("#00E676")
            strokeWidth = 6f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
        }
        mapView.overlays.add(trajectoryPolyline)

        // The Grey Ghost Polyline (Naive strapdown divergence during outages)
        ghostPolyline.outlinePaint.color = Color.parseColor("#9E9E9E")
        ghostPolyline.outlinePaint.strokeWidth = 7f
        ghostPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        ghostPolyline.outlinePaint.isAntiAlias = true
        ghostPolyline.outlinePaint.pathEffect = android.graphics.DashPathEffect(floatArrayOf(25f, 15f), 0f)
        mapView.overlays.add(ghostPolyline)

        // Start Location Pin (Green Google Maps Pin)
        startMarker = org.osmdroid.views.overlay.Marker(mapView).apply {
            setAnchor(org.osmdroid.views.overlay.Marker.ANCHOR_CENTER, org.osmdroid.views.overlay.Marker.ANCHOR_BOTTOM)
            title = "START: Your Location"
            snippet = "Route Departure Point"
            icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_start_pin)
        }

        // Destination End Location Pin (Red Google Maps Pin)
        destinationMarker = org.osmdroid.views.overlay.Marker(mapView).apply {
            setAnchor(org.osmdroid.views.overlay.Marker.ANCHOR_CENTER, org.osmdroid.views.overlay.Marker.ANCHOR_BOTTOM)
            title = "DESTINATION"
            snippet = "Route Arrival Point"
            icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_dest_pin)
        }

        // Google Maps Frozen Ghost Marker (Demonstrates standard app freezing during tunnel outage)
        googleMapsFrozenMarker = org.osmdroid.views.overlay.Marker(mapView).apply {
            setAnchor(org.osmdroid.views.overlay.Marker.ANCHOR_CENTER, org.osmdroid.views.overlay.Marker.ANCHOR_CENTER)
            title = "Google Maps (Signal Lost - Frozen)"
            snippet = "Standard GPS navigation halts without Dead Reckoning"
            icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_gmaps_frozen)
        }

        // Vehicle Direction Marker with Google Maps Circular Dot
        vehicleMarker = org.osmdroid.views.overlay.Marker(mapView).apply {
            setAnchor(org.osmdroid.views.overlay.Marker.ANCHOR_CENTER, org.osmdroid.views.overlay.Marker.ANCHOR_CENTER)
            title = "Vehicle (iNAV Dead Reckoning)"
            icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_nav_arrow)
            setFlat(true)
        }
        mapView.overlays.add(vehicleMarker)

        // Map Long Press to set exact house location or route destination
        val mapEventsReceiver = object : org.osmdroid.events.MapEventsReceiver {
            override fun singleTapConfirmedHelper(p: GeoPoint): Boolean {
                if (binding.containerSearchSuggestions.visibility == View.VISIBLE) {
                    binding.containerSearchSuggestions.visibility = View.GONE
                    hideKeyboard()
                    return true
                }
                hideKeyboard()
                return false
            }

            override fun longPressHelper(p: GeoPoint): Boolean {
                hideKeyboard()
                MaterialAlertDialogBuilder(this@MainActivity)
                    .setTitle("📍 Pin Location Options")
                    .setMessage(String.format("Coordinates: %.5f, %.5f\n\nChoose an action for this point:", p.latitude, p.longitude))
                    .setPositiveButton("📍 Set as My Exact Location") { _, _ ->
                        setManualLocation(p.latitude, p.longitude)
                    }
                    .setNegativeButton("🏁 Navigate Here") { _, _ ->
                        startNavigationTo(
                            p.latitude,
                            p.longitude,
                            String.format("Pinned Destination (%.4f, %.4f)", p.latitude, p.longitude)
                        )
                    }
                    .setNeutralButton("Cancel", null)
                    .show()
                return true
            }
        }
        mapView.overlays.add(0, org.osmdroid.views.overlay.MapEventsOverlay(mapEventsReceiver))
    }

    private fun setManualLocation(lat: Double, lon: Double) {
        val pt = GeoPoint(lat, lon)
        vehicleMarker?.position = pt
        val manualLoc = Location("manual").apply {
            latitude = lat
            longitude = lon
            accuracy = 2.5f
            time = System.currentTimeMillis()
        }
        currentPhysicalLocation = manualLoc
        uncertaintyPolygon.points = Polygon.pointsAsCircle(pt, 3.0)
        hasInitialCentered = true
        lastCenteredLat = lat
        lastCenteredLon = lon
        mapView.controller.animateTo(pt)
        mapView.controller.setZoom(19.0)
        mapView.invalidate()
        service?.setManualLocation(lat, lon)
        Toast.makeText(this, String.format("📍 Set exact location: %.5f, %.5f", lat, lon), Toast.LENGTH_LONG).show()
    }

    private fun setupListeners() {
        val getRouteMgr: () -> RouteManager = { service?.routeManager ?: RouteManager() }
        val getCurLat = { service?.navState?.value?.latitude?.takeIf { it != 0.0 } ?: (vehicleMarker?.position?.latitude ?: 0.0) }
        val getCurLon = { service?.navState?.value?.longitude?.takeIf { it != 0.0 } ?: (vehicleMarker?.position?.longitude ?: 0.0) }

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

        // Tap Search Icon to Search or Show Presets
        binding.btnSearchIcon.setOnClickListener {
            executeSearch()
        }

        // Tap or Focus Search Input to show quick presets if empty
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

        // Destination Search Input (Auto-Geocoding & Suggestions)
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

        // Cancel / End Route Button
        binding.btnCancelRoute.setOnClickListener {
            cancelNavigation()
        }

        // Detect user touch/pan/rotation on map to immediately unlock camera & orientation follow
        mapView.setOnTouchListener { _, event ->
            val action = event.actionMasked
            when (action) {
                android.view.MotionEvent.ACTION_DOWN,
                android.view.MotionEvent.ACTION_POINTER_DOWN -> {
                    isUserTouchingMap = true
                    if (isFollowingVehicle || isTrackUpEnabled) {
                        isFollowingVehicle = false
                        isTrackUpEnabled = false
                        binding.fabMyLocation.setColorFilter(Color.parseColor("#94A3B8"))
                    }
                }
                android.view.MotionEvent.ACTION_UP,
                android.view.MotionEvent.ACTION_CANCEL -> {
                    isUserTouchingMap = false
                    mapView.invalidate()
                }
            }
            false
        }

        // Map View Switcher: Segmented Tabs in Navigation Card
        binding.btnOsmView.setOnClickListener {
            setMapLayer(MapViewType.OSM_VIEW)
        }
        binding.btnSatView.setOnClickListener {
            setMapLayer(MapViewType.SATELLITE_VIEW)
        }

        // Quick Fetch GPS Button (Forces immediate hardware query and centers with high zoom)
        binding.btnFetchGpsNow.setOnClickListener {
            isFollowingVehicle = true
            binding.fabMyLocation.setColorFilter(Color.parseColor("#00E676"))
            fetchPhysicalLocationAndCenter(forceCenter = true)
        }

        // My Location Button (Tap to Re-center & resume Follow Mode with live GNSS hardware fetch)
        binding.fabMyLocation.setOnClickListener {
            isFollowingVehicle = true
            binding.fabMyLocation.setColorFilter(Color.parseColor("#00E676"))
            fetchPhysicalLocationAndCenter(forceCenter = true)
        }

        // Map Layers: 1-Tap Toggle between OSM View and Satellite View
        binding.fabLayers.setOnClickListener {
            if (currentViewType == MapViewType.OSM_VIEW) {
                setMapLayer(MapViewType.SATELLITE_VIEW)
            } else {
                setMapLayer(MapViewType.OSM_VIEW)
            }
        }

        // Compass Button (Tap to re-align North)
        binding.fabCompass.setOnClickListener {
            isTrackUpEnabled = false
            mapView.mapOrientation = 0f
            binding.fabCompass.rotation = 0f
            mapView.invalidate()
            Toast.makeText(this, "Compass: Reset to North-Up", Toast.LENGTH_SHORT).show()
        }

        // Track-Up Bearing Follow Toggle
        binding.fabTrackUp.setOnClickListener {
            isTrackUpEnabled = !isTrackUpEnabled
            if (isTrackUpEnabled) {
                isFollowingVehicle = true
                binding.fabMyLocation.setColorFilter(Color.parseColor("#00E676"))
                Toast.makeText(this, "Track-Up Follow Mode (Driving Perspective)", Toast.LENGTH_SHORT).show()
            } else {
                mapView.mapOrientation = 0f
                mapView.invalidate()
                Toast.makeText(this, "Fixed North Mode", Toast.LENGTH_SHORT).show()
            }
        }

        // Outage Simulator Toggle
        binding.btnToggleOutage.setOnClickListener {
            isOutageActive = !isOutageActive
            service?.setSimulateOutage(isOutageActive)

            if (isOutageActive) {
                binding.btnToggleOutage.text = "RESUME GPS"
                binding.btnToggleOutage.setBackgroundColor(Color.parseColor("#2E7D32"))
                binding.outageBanner.visibility = View.VISIBLE
            } else {
                binding.btnToggleOutage.text = "BLACKOUT"
                binding.btnToggleOutage.setBackgroundColor(Color.parseColor("#D50000"))
                binding.outageBanner.visibility = View.GONE
            }
        }

        // Expandable Live Sensors Cockpit Toggle
        binding.headerToggleSensors.setOnClickListener {
            val isVisible = binding.containerSensorsDetail.visibility == View.VISIBLE
            binding.containerSensorsDetail.visibility = if (isVisible) View.GONE else View.VISIBLE
            binding.tvSensorsToggleLabel.text = if (isVisible) {
                "▼ ALL SENSORS LIVE STREAM · TAP TO INSPECT"
            } else {
                "▲ ALL SENSORS LIVE STREAM · TAP TO MINIMIZE"
            }
        }

        binding.hudContainer.setOnClickListener {
            isHudActive = false
            binding.hudContainer.visibility = View.GONE
        }
    }

    private fun displaySearchSuggestions(results: List<com.inav.navigation.routing.PlaceSuggestion>) {
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
                onSuccess = { plan ->
                    renderRoutePlan(plan)
                },
                onError = { err ->
                    Toast.makeText(this@MainActivity, "Routing notice: $err", Toast.LENGTH_SHORT).show()
                }
            )
        } else {
            // Direct RouteManager request if service binding is still finalizing
            val curLat = vehicleMarker?.position?.latitude ?: 0.0
            val curLon = vehicleMarker?.position?.longitude ?: 0.0
            RouteManager().requestRoute(
                startLat = curLat,
                startLon = curLon,
                destLat = lat,
                destLon = lon,
                destName = name,
                scope = lifecycleScope,
                onSuccess = { plan: RoutePlan ->
                    renderRoutePlan(plan)
                },
                onError = { err: String ->
                    Toast.makeText(this@MainActivity, "Routing notice: $err", Toast.LENGTH_SHORT).show()
                }
            )
        }
    }

    private fun renderRoutePlan(plan: com.inav.navigation.routing.RoutePlan) {
        val fullPoints = mutableListOf<GeoPoint>()
        val startGeo = if (plan.startLat != 0.0 && plan.startLon != 0.0) {
            GeoPoint(plan.startLat, plan.startLon)
        } else {
            val vPt = vehicleMarker?.position ?: GeoPoint(0.0, 0.0)
            GeoPoint(vPt.latitude, vPt.longitude)
        }
        val destGeo = GeoPoint(plan.destLat, plan.destLon)

        if (plan.points.isNotEmpty()) {
            // Prepend start point if needed to guarantee line connects directly from vehicle marker
            if (plan.points.first().distanceToAsDouble(startGeo) > 0.5) {
                fullPoints.add(startGeo)
            }
            fullPoints.addAll(plan.points)
            // Append dest point if needed to guarantee line connects directly to red pin tip
            if (plan.points.last().distanceToAsDouble(destGeo) > 0.5) {
                fullPoints.add(destGeo)
            }
        } else {
            fullPoints.add(startGeo)
            fullPoints.add(destGeo)
        }

        routePolyline.setPoints(fullPoints)

        // Remove redundant green start pin so user only sees the sleek circular vehicle marker
        if (startMarker != null && mapView.overlays.contains(startMarker)) {
            mapView.overlays.remove(startMarker)
        }

        // RED END PIN (Destination Point): Tip touches the exact end vertex of the blue line
        val destPt = fullPoints.last()
        destinationMarker?.position = destPt
        destinationMarker?.title = "END: ${plan.destinationName}"
        destinationMarker?.snippet = String.format("Distance: %.1f km", plan.totalDistanceM / 1000.0)
        if (destinationMarker != null && !mapView.overlays.contains(destinationMarker)) {
            mapView.overlays.add(destinationMarker)
        }

        // 3. UI Header Updates
        binding.panelActiveRoute.visibility = View.VISIBLE
        binding.panelRoutePicker.visibility = View.GONE
        binding.tvActiveStartTitle.text = String.format("START: Origin (%.4f, %.4f)", startGeo.latitude, startGeo.longitude)
        binding.tvActiveDestTitle.text = "END: ${plan.destinationName}"
        binding.tvActiveRouteMetrics.text = String.format(
            "Remaining: %.1f km · ETA: %d min · OSRM API",
            plan.totalDistanceM / 1000.0,
            (plan.totalDurationS / 60.0).toInt().coerceAtLeast(1)
        )

        // 4. Zoom map to clearly fit BOTH Start Point and End Point
        if (fullPoints.size >= 2) {
            try {
                val box = org.osmdroid.util.BoundingBox.fromGeoPoints(fullPoints)
                mapView.zoomToBoundingBox(box, true, 120)
            } catch (e: Exception) {
                mapView.controller.setCenter(startGeo)
                mapView.controller.setZoom(16.0)
            }
        } else {
            mapView.controller.setCenter(startGeo)
            mapView.controller.setZoom(16.0)
        }

        mapView.invalidate()
        Toast.makeText(
            this@MainActivity,
            String.format("Route loaded: %.1f km to %s", plan.totalDistanceM / 1000.0, plan.destinationName),
            Toast.LENGTH_SHORT
        ).show()
    }

    private fun cancelNavigation() {
        service?.clearRoute()
        routePolyline.setPoints(emptyList())
        if (destinationMarker != null) {
            mapView.overlays.remove(destinationMarker)
        }
        binding.panelActiveRoute.visibility = View.GONE
        binding.panelRoutePicker.visibility = View.VISIBLE
        binding.etSearchDestination.text.clear()
        binding.containerSearchSuggestions.visibility = View.GONE
        binding.containerSearchSuggestions.removeAllViews()
        mapView.invalidate()
        Toast.makeText(this, "Route ended", Toast.LENGTH_SHORT).show()
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

    // ── GEV cockpitMath.slewHeading: rate-limited shortest-arc heading interpolation ──
    private fun slewHeading(current: Float, target: Float, maxStepDeg: Float): Float {
        val from = ((current % 360f) + 360f) % 360f
        val to = ((target % 360f) + 360f) % 360f
        if (maxStepDeg <= 0f) return from
        val delta = ((to - from + 540f) % 360f) - 180f
        val step = delta.coerceIn(-maxStepDeg, maxStepDeg)
        return ((from + step) % 360f + 360f) % 360f
    }

    // ── GEV renderGovernor: only redraw when something actually changed ──
    private fun shouldRedraw(lat: Double, lon: Double, heading: Float): Boolean {
        val dLat = lat - lastRenderedLat
        val dLon = lon - lastRenderedLon
        // Quick distance approximation in meters (1° lat ≈ 111km)
        val distM = Math.sqrt(dLat * dLat + dLon * dLon) * 111_000.0
        val dHeading = Math.abs(((heading - lastRenderedHeading + 540f) % 360f) - 180f)
        return distM > RENDER_POSITION_THRESHOLD_M || dHeading > RENDER_HEADING_THRESHOLD_DEG
    }

    private fun updateUi(state: NavigationState) {
        // When stationary or walking slowly (< 2 km/h), NEVER snap to road!
        // Show the user's true physical building/room location on map & satellite imagery
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

        // ── Smooth Heading Slew (GEV cockpitMath pattern) ──
        // Rate-limit heading changes to prevent visual snapping
        val now = System.currentTimeMillis()
        val dt = if (lastUpdateTimeMs > 0) (now - lastUpdateTimeMs).coerceAtMost(200L) else 16L
        lastUpdateTimeMs = now
        val maxStep = HEADING_SLEW_RATE_DPS * (dt / 1000f)
        displayHeading = slewHeading(displayHeading, rawHeading.toFloat(), maxStep)
        val activeHeading = displayHeading.toDouble()

        // ── Auto Night Mode (GEV sensor-style switching) ──
        val shouldBeNight = state.lightLux < NIGHT_MODE_LUX_THRESHOLD || state.isTunnelLighting
        if (shouldBeNight != isNightMode && (now - lastNightModeSwitch) > NIGHT_MODE_DEBOUNCE_MS) {
            isNightMode = shouldBeNight
            lastNightModeSwitch = now
            applyNightMode()
        }

        // 1. TOP NAVIGATION STATUS BAR & TURN GUIDANCE
        if (state.hasActiveRoute) {
            // Turn-by-Turn Google Maps Style Active Guidance
            binding.tvTopRoadIcon.text = state.nextManeuverIcon
            binding.tvTopRoadName.text = if (state.isArrivedAtDestination) "🏁 Arrived at Destination!" else state.nextManeuverText
            binding.tvTopSubStatus.text = if (state.mode == NavigationMode.PURE_DR) {
                "🚨 GPS LOST (TUNNEL) · iNAV DEAD RECKONING ACTIVE · %.1f km left".format(state.remainingDistanceM / 1000.0)
            } else {
                "In %.0f m · Remaining: %.1f km (ETA: %d min)".format(
                    state.distanceToManeuverM,
                    state.remainingDistanceM / 1000.0,
                    state.etaMinutes
                )
            }

            // Active Route Panel
            binding.panelActiveRoute.visibility = View.VISIBLE
            binding.panelRoutePicker.visibility = View.GONE
            binding.tvActiveDestTitle.text = "To: ${state.destinationName}"
            binding.tvActiveRouteMetrics.text = String.format(
                "Remaining: %.1f km · In %.0f m: %s",
                state.remainingDistanceM / 1000.0,
                state.distanceToManeuverM,
                state.nextManeuverText
            )

            // Ensure route polyline is displayed
            if (routePolyline.actualPoints.isEmpty()) {
                val activePlan = service?.getActiveRoute()
                if (activePlan != null && activePlan.points.isNotEmpty()) {
                    routePolyline.setPoints(activePlan.points)
                    destinationMarker?.position = GeoPoint(activePlan.destLat, activePlan.destLon)
                    destinationMarker?.title = activePlan.destinationName
                    if (destinationMarker != null && !mapView.overlays.contains(destinationMarker)) {
                        mapView.overlays.add(destinationMarker)
                    }
                }
            }
        } else {
            binding.tvTopRoadIcon.text = "📍"
            binding.tvTopRoadName.text = if (state.roadName.isNotBlank()) state.roadName else "Navigating Road"
            binding.tvTopSubStatus.text = if (state.mode == NavigationMode.PURE_DR) {
                "GNSS Outage · Pure IMU Dead Reckoning Active"
            } else {
                "15-State ES-EKF · ${state.satellitesUsed}/${state.satellitesInView} Sats (±%.1fm)".format(state.gnssAccuracyM)
            }
            binding.panelActiveRoute.visibility = View.GONE
            binding.panelRoutePicker.visibility = View.VISIBLE

            // Continuously update START origin label in search picker
            if (activeLat != 0.0 && activeLon != 0.0) {
                val locStr = if (state.roadName.isNotBlank()) "${state.roadName} (%.4f, %.4f)".format(activeLat, activeLon) else "Current GPS Fix (%.4f, %.4f)".format(activeLat, activeLon)
                binding.tvPickerStartLabel.text = locStr
            }
            // Remove startMarker from map so no duplicate pin sits on top of vehicle marker
            if (startMarker != null && mapView.overlays.contains(startMarker)) {
                mapView.overlays.remove(startMarker)
            }
        }

        // Auto Day / Dark Tunnel Theme
        if (state.isTunnelLighting) {
            binding.gmapsNavHeader.setBackgroundColor(Color.parseColor("#311B92")) // Tunnel Dark Purple
        } else {
            binding.gmapsNavHeader.setBackgroundColor(Color.parseColor("#1B5E20")) // Clean Nav Green
        }

        // 2. PRIMARY NAVIGATION HUD (Bottom Card)
        binding.tvSpeedValue.text = String.format("%.1f", state.speedKmh)
        binding.tvHeadingValue.text = String.format("%.0f° %s", state.headingDeg, state.cardinalDirection)
        binding.tvRoadEvent.text = when (state.roadAnomalyType) {
            1 -> "Road: ${state.roadName} · ⚠️ Pothole (Crater Drop)"
            2 -> "Road: ${state.roadName} · ⚠️ Speed Breaker (Bump)"
            else -> "Road: ${state.roadName}"
        }
        binding.tvUncertainty.text = String.format("σ: ±%.2f m", state.uncertaintySigmaM)

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

        // 3D Attitude Indicator (MTN decoupling)
        binding.attitudeIndicator.pitchDeg = state.pitchDeg.toFloat()
        binding.attitudeIndicator.rollDeg = state.rollDeg.toFloat()

        // Road Snapped Badge
        if (state.isRoadSnapped) {
            binding.tvRoadSnappedBadge.visibility = View.VISIBLE
            binding.tvRoadSnappedBadge.text = "ROAD LOCKED"
            binding.tvRoadSnappedBadge.setBackgroundColor(Color.parseColor("#1B5E20"))
        } else {
            binding.tvRoadSnappedBadge.visibility = View.VISIBLE
            binding.tvRoadSnappedBadge.text = "FREE DR"
            binding.tvRoadSnappedBadge.setBackgroundColor(Color.parseColor("#455A64"))
        }

        // Mode Status Badge
        when (state.mode) {
            NavigationMode.AIDED -> {
                binding.tvModeBadge.text = "AIDED (GNSS)"
                binding.tvModeBadge.setBackgroundColor(Color.parseColor("#2E7D32"))
            }
            NavigationMode.DEGRADED -> {
                binding.tvModeBadge.text = "DEGRADED"
                binding.tvModeBadge.setBackgroundColor(Color.parseColor("#F57F17"))
            }
            NavigationMode.PURE_DR -> {
                binding.tvModeBadge.text = "DEAD RECKONING"
                binding.tvModeBadge.setBackgroundColor(Color.parseColor("#7B1FA2"))
            }
        }

        // Outage Banner & Ghost Trace live drift metrics
        if (state.isOutageSimulated) {
            binding.outageBanner.visibility = View.VISIBLE
            binding.tvOutageTimer.text = if (state.outageDistanceTravelledM > 2.0) {
                String.format(
                    "⚠️ TUNNEL GPS OUTAGE: %.1fs | Drift: %.1fm (%.1f%%) over %.0fm",
                    state.outageDurationSec,
                    state.driftDistanceM,
                    state.driftPercentage,
                    state.outageDistanceTravelledM
                )
            } else {
                String.format("⚠️ TUNNEL GPS OUTAGE: %.1fs | Pure Dead Reckoning Active", state.outageDurationSec)
            }

            binding.tvGmapsVsStatus.text = "GOOGLE MAPS: ❌ SIGNAL LOST (FROZEN)"
            binding.tvInavVsStatus.text = "iNAV: ✅ DEAD RECKONING ACTIVE"

            // Freeze Google Maps Ghost Marker at outage origin
            if (outageFrozenGeoPoint == null && (activeLat != 0.0 || activeLon != 0.0)) {
                outageFrozenGeoPoint = GeoPoint(activeLat, activeLon)
                googleMapsFrozenMarker?.position = outageFrozenGeoPoint
                if (googleMapsFrozenMarker != null && !mapView.overlays.contains(googleMapsFrozenMarker)) {
                    mapView.overlays.add(googleMapsFrozenMarker)
                }
            }

            if (state.naiveLatitude != 0.0 && state.naiveLongitude != 0.0) {
                val ghostPoint = GeoPoint(state.naiveLatitude, state.naiveLongitude)
                val ghostPts = ghostPolyline.actualPoints
                if (ghostPts.isEmpty() || ghostPts.last().distanceToAsDouble(ghostPoint) >= 1.0) {
                    ghostPolyline.addPoint(ghostPoint)
                }
            }
        } else {
            binding.outageBanner.visibility = View.GONE
            outageFrozenGeoPoint = null
            if (googleMapsFrozenMarker != null) {
                mapView.overlays.remove(googleMapsFrozenMarker)
            }
            if (ghostPolyline.actualPoints.isNotEmpty()) {
                ghostPolyline.setPoints(emptyList())
            }
        }

        // HUD View Updates
        if (isHudActive) {
            binding.tvHudSpeed.text = String.format("%.1f", state.speedKmh)
            binding.tvHudHeading.text = String.format("HEADING: %.0f° %s", state.headingDeg, state.cardinalDirection)
            binding.tvHudRoad.text = "ROAD: ${state.roadName.uppercase()}"
        }

        // 4. MAP & BEARING AUTO-ROTATION (Track-Up Google Maps Mode)
        if (activeLat != 0.0 || activeLon != 0.0) {
            val point = GeoPoint(activeLat, activeLon)

            // Dynamic 3σ Covariance Uncertainty Circle (only recompute when user is not pinching/zooming)
            if (!isUserTouchingMap) {
                val radiusM = maxOf(2.0, state.uncertaintySigmaM * 3.0)
                uncertaintyPolygon.points = Polygon.pointsAsCircle(point, radiusM)
            }

            val pts = trajectoryPolyline.actualPoints
            if (pts.isNotEmpty() && pts.last().distanceToAsDouble(point) > 5000.0) {
                trajectoryPolyline.setPoints(listOf(point))
                trajectoryGlowPolyline.setPoints(listOf(point))
                ghostPolyline.setPoints(emptyList())
                mapView.controller.setCenter(point)
                mapView.controller.setZoom(18.5)
            } else if (state.speedKmh > 0.8 && (pts.isEmpty() || pts.last().distanceToAsDouble(point) >= 1.0)) {
                trajectoryPolyline.addPoint(point)
                trajectoryGlowPolyline.addPoint(point)
            } else if (pts.isEmpty()) {
                trajectoryPolyline.addPoint(point)
                trajectoryGlowPolyline.addPoint(point)
            }

            // Initial one-time camera center on first valid position
            if (!hasInitialCentered && (activeLat != 0.0 || activeLon != 0.0)) {
                hasInitialCentered = true
                mapView.controller.setCenter(point)
                mapView.controller.setZoom(18.0)
            }

            // Update Vehicle Cursor Position
            vehicleMarker?.position = point

            if (isFollowingVehicle) {
                mapView.controller.setCenter(point)
                if (isTrackUpEnabled) {
                    // In Track-Up Mode: Map rotates so vehicle is always facing UP
                    mapView.mapOrientation = -activeHeading.toFloat()
                    vehicleMarker?.rotation = 0f
                } else {
                    // Follow Mode: Arrow points in the smooth-slewed heading direction
                    vehicleMarker?.rotation = activeHeading.toFloat()
                }
            } else {
                // Free Pan / Two-Finger Rotate Mode:
                // Finger gestures rotate the MAP; arrow direction reflects smooth-slewed heading
                vehicleMarker?.rotation = activeHeading.toFloat()
            }

            // Keep compass FAB needle pointing True North relative to map rotation
            binding.fabCompass.rotation = -mapView.mapOrientation

            // ── GEV Trail Color Theming: mode-adaptive trail + glow colors ──
            if (state.mode == NavigationMode.PURE_DR) {
                trajectoryPolyline.outlinePaint.color = Color.parseColor("#B388FF")
                trajectoryGlowPolyline.outlinePaint.color = Color.parseColor("#40B388FF")
            } else {
                trajectoryPolyline.outlinePaint.color = if (isNightMode) Color.parseColor("#4FC3F7") else Color.parseColor("#00E676")
                trajectoryGlowPolyline.outlinePaint.color = if (isNightMode) Color.parseColor("#404FC3F7") else Color.parseColor("#4000E676")
            }

            // ── Render Governor: only invalidate when position or heading actually changed ──
            if (!isUserTouchingMap && shouldRedraw(activeLat, activeLon, activeHeading.toFloat())) {
                lastRenderedLat = activeLat
                lastRenderedLon = activeLon
                lastRenderedHeading = activeHeading.toFloat()
                mapView.invalidate()
            }
        }
    }

    // ── GEV-Inspired: Night Mode Tile + Theme Switching ──
    private fun applyNightMode() {
        if (currentViewType != MapViewType.OSM_VIEW) {
            // Keep satellite imagery active without overwriting with OSM street tiles
            return
        }
        if (isNightMode) {
            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_nav_arrow_night)
            binding.gmapsNavHeader.setBackgroundColor(Color.parseColor("#0D1117"))
        } else {
            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_nav_arrow)
            binding.gmapsNavHeader.setBackgroundColor(Color.parseColor("#1B5E20"))
        }
        mapView.invalidate()
    }

    // ── Dedicated Map Views: Fast OSM and Esri Photorealistic Satellite ──
    private fun setMapLayer(viewType: MapViewType) {
        currentViewType = viewType

        when (viewType) {
            MapViewType.OSM_VIEW -> {
                mapView.tileProvider.clearTileCache()
                val osmSource = org.osmdroid.tileprovider.tilesource.XYTileSource(
                    "FastOSM",
                    0, 19, 256, ".png",
                    arrayOf(
                        "https://a.tile.openstreetmap.org/",
                        "https://b.tile.openstreetmap.org/",
                        "https://c.tile.openstreetmap.org/"
                    ),
                    "© OpenStreetMap contributors",
                    org.osmdroid.tileprovider.tilesource.TileSourcePolicy(8, 0)
                )
                mapView.setTileSource(osmSource)
                mapView.invalidate()

                binding.fabLayers.setColorFilter(Color.parseColor("#F59E0B"))
                binding.btnOsmView.setBackgroundColor(Color.parseColor("#1E40AF"))
                binding.btnOsmView.setTextColor(Color.WHITE)
                binding.btnSatView.setBackgroundColor(Color.TRANSPARENT)
                binding.btnSatView.setTextColor(Color.parseColor("#94A3B8"))
                Toast.makeText(this, "🗺️ OSM View Active", Toast.LENGTH_SHORT).show()
            }
            MapViewType.SATELLITE_VIEW -> {
                mapView.tileProvider.clearTileCache()
                mapView.setTileSource(CesiumIonImageryProvider.createEsriSatelliteTileSource())
                mapView.invalidate()

                binding.fabLayers.setColorFilter(Color.parseColor("#38BDF8"))
                binding.btnSatView.setBackgroundColor(Color.parseColor("#D97706"))
                binding.btnSatView.setTextColor(Color.WHITE)
                binding.btnOsmView.setBackgroundColor(Color.TRANSPARENT)
                binding.btnOsmView.setTextColor(Color.parseColor("#94A3B8"))
                Toast.makeText(this, "🛰️ Satellite View Active", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ── Direct Physical Location Engine ──
    private fun isBetterLocation(location: Location, currentBest: Location?): Boolean {
        if (currentBest == null) return true
        val timeDelta = location.time - currentBest.time
        val isSignificantlyNewer = timeDelta > 120_000L
        val isSignificantlyOlder = timeDelta < -120_000L
        val isNewer = timeDelta > 0

        // If candidate is accurate GPS (<35m), always prefer over coarse network
        if (location.provider == LocationManager.GPS_PROVIDER && location.accuracy < 35f) {
            return true
        }

        // Never let coarse network/fused (>50m) override accurate GPS/manual (<30m) unless GPS is >15 mins old
        if (currentBest.accuracy < 30f && location.accuracy > 50f && timeDelta < 900_000L) {
            return false
        }

        val accuracyDelta = (location.accuracy - currentBest.accuracy).toInt()
        val isLessAccurate = accuracyDelta > 0
        val isMoreAccurate = accuracyDelta < 0
        val isSignificantlyLessAccurate = accuracyDelta > 50

        if (isMoreAccurate) return true
        if (isNewer && !isLessAccurate) return true
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
        if (cur != null && !isBetterLocation(loc, cur)) {
            return
        }
        currentPhysicalLocation = loc

        val pt = GeoPoint(loc.latitude, loc.longitude)
        vehicleMarker?.position = pt
        val radiusM = maxOf(2.0, loc.accuracy.toDouble())
        uncertaintyPolygon.points = Polygon.pointsAsCircle(pt, radiusM)

        val dLat = Math.abs(loc.latitude - lastCenteredLat)
        val dLon = Math.abs(loc.longitude - lastCenteredLon)
        val distChangeM = Math.sqrt(dLat * dLat + dLon * dLon) * 111_000.0

        if (isFollowingVehicle || !hasInitialCentered || distChangeM > 50.0) {
            hasInitialCentered = true
            lastCenteredLat = loc.latitude
            lastCenteredLon = loc.longitude
            mapView.controller.animateTo(pt)
            if (mapView.zoomLevelDouble < 17.0) {
                mapView.controller.setZoom(18.0)
            }
        }
        mapView.invalidate()
        service?.updateGnssFix(loc)
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
                mapView.controller.animateTo(GeoPoint(best.latitude, best.longitude))
                mapView.controller.setZoom(18.5)
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
                                mapView.controller.animateTo(GeoPoint(loc.latitude, loc.longitude))
                                mapView.controller.setZoom(18.5)
                                Toast.makeText(this, String.format("📍 Hardware Fix: %.5f, %.5f (±%.0fm, %s)", loc.latitude, loc.longitude, loc.accuracy, loc.provider), Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                } catch (ignored: Exception) {}
            }
        }
    }

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

    override fun onDestroy() {
        super.onDestroy()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
    }
}
