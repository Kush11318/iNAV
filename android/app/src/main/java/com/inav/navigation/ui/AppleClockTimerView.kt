package com.inav.navigation.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import kotlin.math.cos
import kotlin.math.sin

/**
 * AppleClockTimerView
 * Authentic Apple Clock circular dial:
 * - 60 fine radial tick marks with highlighted cardinal intervals
 * - Background track ring
 * - Vibrant animated sweep arc (Orange #FF9500 / Emerald #00E676 on finish)
 * - Transitions.dev Number Pop-in center countdown display
 */
class AppleClockTimerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {

    private val dialCanvasView: DialCanvasView
    private val popInTextView: NumberPopInTextView
    private val sublabelTextView: TextView

    private var currentProgress: Float = 0f
    private var progressAnimator: ValueAnimator? = null

    init {
        // Dial Canvas
        dialCanvasView = DialCanvasView(context)
        addView(
            dialCanvasView,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        )

        // Center Content Container
        val centerContainer = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
        }

        popInTextView = NumberPopInTextView(context).apply {
            setTextSizeSp(52f)
            setTextColorInt(Color.WHITE)
        }
        centerContainer.addView(popInTextView)

        sublabelTextView = TextView(context).apply {
            text = "REMAINING"
            textSize = 11f
            setTextColor(Color.parseColor("#8E99A8"))
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            letterSpacing = 0.18f
            gravity = Gravity.CENTER
            setPadding(0, dpToPx(2f).toInt(), 0, 0)
        }
        centerContainer.addView(sublabelTextView)

        val centerParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
            gravity = Gravity.CENTER
        }
        addView(centerContainer, centerParams)
    }

    private var lastSecondsRemaining: Int = -1

    fun setProgress(progress: Float, animated: Boolean = false) {
        val target = progress.coerceIn(0f, 1f)
        currentProgress = target
        dialCanvasView.progress = target
        dialCanvasView.invalidate()
    }

    fun setTimeRemaining(seconds: Int) {
        if (seconds == lastSecondsRemaining) return
        lastSecondsRemaining = seconds
        val text = "${seconds}s"
        popInTextView.animateText(text)
    }

    fun setStatusText(text: String) {
        sublabelTextView.text = text.uppercase()
    }

    fun setCompletedTheme() {
        dialCanvasView.setArcColor(Color.parseColor("#00E676"))
        popInTextView.setTextColorInt(Color.parseColor("#00E676"))
        sublabelTextView.setTextColor(Color.parseColor("#00E676"))
        sublabelTextView.text = "CALIBRATED"
    }

    fun setMotionWarningTheme() {
        dialCanvasView.setArcColor(Color.parseColor("#FF9800"))
        popInTextView.setTextColorInt(Color.parseColor("#FFB74D"))
        sublabelTextView.setTextColor(Color.parseColor("#FF9800"))
        sublabelTextView.text = "PAUSED"
    }

    fun setNormalTheme() {
        dialCanvasView.setArcColor(Color.parseColor("#FF9500"))
        popInTextView.setTextColorInt(Color.WHITE)
        sublabelTextView.setTextColor(Color.parseColor("#8E99A8"))
        sublabelTextView.text = "REMAINING"
    }

    private fun dpToPx(dp: Float): Float {
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            dp,
            resources.displayMetrics
        )
    }

    private inner class DialCanvasView(context: Context) : View(context) {
        var progress: Float = 0f

        private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
        }

        private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dpToPx(6f)
            color = Color.parseColor("#171B24")
        }

        private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dpToPx(7f)
            strokeCap = Paint.Cap.ROUND
            color = Color.parseColor("#FF9500") // Apple Amber
        }

        private val arcRect = RectF()

        fun setArcColor(colorInt: Int) {
            arcPaint.color = colorInt
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            val cx = w / 2f
            val cy = h / 2f
            val outerRadius = (Math.min(w, h) / 2f) - dpToPx(14f)
            val trackRadius = outerRadius - dpToPx(12f)

            if (outerRadius <= 0) return

            // 1. Draw 60 Apple Clock Ticks
            val totalTicks = 60
            val tickLengthMinor = dpToPx(6f)
            val tickLengthMajor = dpToPx(12f)

            for (i in 0 until totalTicks) {
                val angleDeg = i * (360f / totalTicks) - 90f
                val angleRad = Math.toRadians(angleDeg.toDouble())

                val isMajor = (i % 5 == 0)
                val tickLength = if (isMajor) tickLengthMajor else tickLengthMinor

                tickPaint.strokeWidth = if (isMajor) dpToPx(2.2f) else dpToPx(1.2f)
                tickPaint.color = if (isMajor) Color.parseColor("#3B4354") else Color.parseColor("#1F2430")

                val startX = cx + (outerRadius - tickLength) * cos(angleRad).toFloat()
                val startY = cy + (outerRadius - tickLength) * sin(angleRad).toFloat()
                val endX = cx + outerRadius * cos(angleRad).toFloat()
                val endY = cy + outerRadius * sin(angleRad).toFloat()

                canvas.drawLine(startX, startY, endX, endY, tickPaint)
            }

            // 2. Draw Track Ring
            arcRect.set(
                cx - trackRadius,
                cy - trackRadius,
                cx + trackRadius,
                cy + trackRadius
            )
            canvas.drawCircle(cx, cy, trackRadius, trackPaint)

            // 3. Draw Active Progress Arc
            if (progress > 0.001f) {
                val sweepAngle = progress * 360f
                canvas.drawArc(arcRect, -90f, sweepAngle, false, arcPaint)
            }
        }
    }
}
