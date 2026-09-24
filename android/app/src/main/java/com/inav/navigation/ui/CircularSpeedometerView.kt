package com.inav.navigation.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator

/**
 * Modern Minimalist Circular Speedometer matching Apple/Google Maps white & black theme.
 * Features:
 * - Pure white circular card with soft drop shadow matching right-side floating action buttons
 * - Minimalist hairline slate track ring
 * - Sleek jet-black progress arc that fills smoothly with speed
 * - Bold deep-black speed numeral in center
 * - Clean dark-slate "KM/H" / "MPH" unit subtitle
 * - Clean tap interaction to toggle units
 */
class CircularSpeedometerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var currentSpeed: Float = 0f
    private var targetSpeed: Float = 0f
    private var maxSpeed: Float = 140f
    private var isMph: Boolean = false // Default to KM/H

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        setShadowLayer(dpToPx(4f), 0f, dpToPx(2f), Color.parseColor("#33000000"))
    }

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.parseColor("#F1F5F9") // Ultra clean subtle track
        strokeWidth = dpToPx(3f)
        strokeCap = Paint.Cap.ROUND
    }

    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.parseColor("#18181B") // Jet black progress arc matching black icons
        strokeWidth = dpToPx(3.2f)
        strokeCap = Paint.Cap.ROUND
    }

    private val speedTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#09090B") // Deep solid black
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val unitTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#71717A") // Slate gray subtitle
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        letterSpacing = 0.08f
    }

    private val arcRect = RectF()
    private var animator: ValueAnimator? = null

    init {
        setLayerType(LAYER_TYPE_SOFTWARE, null) // Required for crisp shadow layer
        setOnClickListener {
            isMph = !isMph
            invalidate()
        }
    }

    fun setSpeed(speedKmh: Double) {
        val raw = speedKmh.toFloat().coerceAtLeast(0f)
        val displayVal = if (isMph) raw * 0.621371f else raw
        if (kotlin.math.abs(displayVal - targetSpeed) > 0.3f) {
            targetSpeed = displayVal
            animator?.cancel()
            animator = ValueAnimator.ofFloat(currentSpeed, targetSpeed).apply {
                duration = 220
                interpolator = DecelerateInterpolator()
                addUpdateListener {
                    currentSpeed = it.animatedValue as Float
                    invalidate()
                }
                start()
            }
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2f
        val cy = h / 2f
        val padding = dpToPx(4f)
        val outerRadius = (minOf(w, h) / 2f) - padding

        if (outerRadius <= 0) return

        // 1. Pure white circular disc background with drop shadow
        canvas.drawCircle(cx, cy, outerRadius, bgPaint)

        // 2. Dynamic stroke width based on size
        val strokeW = (outerRadius * 0.13f).coerceIn(dpToPx(2.5f), dpToPx(5f))
        trackPaint.strokeWidth = strokeW
        arcPaint.strokeWidth = strokeW * 1.15f

        // Track & Progress Arc geometry
        val arcRadius = outerRadius - (strokeW * 1.1f)
        arcRect.set(cx - arcRadius, cy - arcRadius, cx + arcRadius, cy + arcRadius)

        // 360-degree full track
        canvas.drawArc(arcRect, 0f, 360f, false, trackPaint)

        // Active speed progress arc (starts from top 270° and sweeps clockwise)
        val progressFraction = (currentSpeed / maxSpeed).coerceIn(0f, 1f)
        if (progressFraction > 0.005f) {
            val activeSweep = (progressFraction * 360f).coerceAtMost(359f)
            canvas.drawArc(arcRect, 270f, activeSweep, false, arcPaint)
        }

        // 3. Center Typography: Speed & Unit
        val speedInt = currentSpeed.toInt()
        val speedStr = speedInt.toString()
        val unitStr = if (isMph) "MPH" else "KM/H"

        // Scale font sizes dynamically relative to view diameter
        speedTextPaint.textSize = outerRadius * 0.78f
        unitTextPaint.textSize = outerRadius * 0.28f

        // Center typography calculation
        val speedBounds = Rect()
        speedTextPaint.getTextBounds(speedStr, 0, speedStr.length, speedBounds)
        val textCenterOffset = (speedBounds.height() / 2f) - dpToPx(1f)

        val textY = cy + textCenterOffset - dpToPx(3.5f)
        canvas.drawText(speedStr, cx, textY, speedTextPaint)

        val unitY = textY + (outerRadius * 0.40f)
        canvas.drawText(unitStr, cx, unitY, unitTextPaint)
    }

    private fun dpToPx(dp: Float): Float = dp * resources.displayMetrics.density
}
