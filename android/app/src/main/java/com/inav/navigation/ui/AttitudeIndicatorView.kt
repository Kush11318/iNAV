package com.inav.navigation.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * 3D Artificial Horizon / Gimbal Visualizer
 * Visualizes the Motion Transformation Network (MTN) attitude alignment:
 * decuples phone orientation from the vehicle frame in real time.
 */
class AttitudeIndicatorView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var pitchDeg: Float = 0f
        set(value) {
            field = value.coerceIn(-60f, 60f)
            invalidate()
        }

    var rollDeg: Float = 0f
        set(value) {
            field = value
            invalidate()
        }

    // Paints
    private val skyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#0C2340") // Deep Aerospace Navy
        style = Paint.Style.FILL
    }

    private val groundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#261C14") // Earth Brown
        style = Paint.Style.FILL
    }

    private val horizonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF") // Neon Cyan Horizon
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    private val ladderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#80FFFFFF")
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }

    private val reticlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFD600") // Aircraft Gold
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }

    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    private val bgBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1E293B")
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 22f
        typeface = Typeface.MONOSPACE
        textAlign = Paint.Align.CENTER
    }

    private val clipPath = Path()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2f
        val cy = h / 2f
        val radius = (Math.min(w, h) / 2f) - 6f

        // Rounded circular bezel
        clipPath.reset()
        clipPath.addCircle(cx, cy, radius, Path.Direction.CW)

        canvas.drawCircle(cx, cy, radius, bgBorderPaint)

        canvas.save()
        canvas.clipPath(clipPath)

        // Rotate and translate for Pitch and Roll
        val pixelsPerDegree = radius / 40f
        val pitchOffset = pitchDeg * pixelsPerDegree

        canvas.save()
        canvas.rotate(-rollDeg, cx, cy)
        canvas.translate(0f, pitchOffset)

        // Draw Sky and Ground
        val huge = radius * 4f
        canvas.drawRect(cx - huge, cy - huge, cx + huge, cy, skyPaint)
        canvas.drawRect(cx - huge, cy, cx + huge, cy + huge, groundPaint)
        canvas.drawLine(cx - huge, cy, cx + huge, cy, horizonPaint)

        // Draw Pitch Ladder (+10, -10, +20, -20)
        for (p in listOf(-20, -10, 10, 20)) {
            val y = cy - (p * pixelsPerDegree)
            val barHalfWidth = if (p % 20 == 0) radius * 0.35f else radius * 0.20f
            canvas.drawLine(cx - barHalfWidth, y, cx + barHalfWidth, y, ladderPaint)
        }

        canvas.restore() // Undo rotation and pitch

        // Fixed Aircraft Crosshair / Reticle in Center
        val reticleRadius = 14f
        canvas.drawCircle(cx, cy, 3f, reticlePaint)
        // Left wing
        canvas.drawLine(cx - radius * 0.45f, cy, cx - reticleRadius, cy, reticlePaint)
        canvas.drawLine(cx - reticleRadius, cy, cx - reticleRadius, cy + 8f, reticlePaint)
        // Right wing
        canvas.drawLine(cx + reticleRadius, cy, cx + radius * 0.45f, cy, reticlePaint)
        canvas.drawLine(cx + reticleRadius, cy, cx + reticleRadius, cy + 8f, reticlePaint)

        // Text display: Pitch and Roll
        canvas.drawText(
            String.format("P:%.0f° R:%.0f°", pitchDeg, rollDeg),
            cx,
            cy + radius * 0.75f,
            textPaint
        )

        canvas.restore()

        // Outer glowing bezel ring
        canvas.drawCircle(cx, cy, radius, borderPaint)
    }
}
