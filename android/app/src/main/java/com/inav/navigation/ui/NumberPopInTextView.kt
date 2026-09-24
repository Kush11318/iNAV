package com.inav.navigation.ui

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.animation.OvershootInterpolator
import android.widget.LinearLayout
import android.widget.TextView

/**
 * NumberPopInTextView
 * Implements Transitions.dev Number Pop-in animation:
 * - Translation Y (+8dp to 0)
 * - Alpha fade (0 to 1)
 * - Overshoot spring curve (cubic-bezier(0.34, 1.45, 0.64, 1))
 * - Staggered delay per digit (index * 70ms)
 */
class NumberPopInTextView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {

    private var currentText: String = ""
    private var textSizeSp: Float = 48f
    private var textColor: Int = Color.WHITE
    private val overshootInterpolator = OvershootInterpolator(1.45f)

    init {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER
    }

    fun setTextSizeSp(sp: Float) {
        textSizeSp = sp
    }

    fun setTextColorInt(color: Int) {
        textColor = color
        for (i in 0 until childCount) {
            (getChildAt(i) as? TextView)?.setTextColor(color)
        }
    }

    fun animateText(newText: String) {
        if (newText == currentText && childCount == newText.length) return

        val oldText = currentText
        currentText = newText

        // Ensure proper number of child TextViews
        while (childCount < newText.length) {
            val tv = TextView(context).apply {
                setTextSize(TypedValue.COMPLEX_UNIT_SP, textSizeSp)
                setTextColor(textColor)
                typeface = Typeface.create("sans-serif-black", Typeface.BOLD)
                gravity = Gravity.CENTER
                includeFontPadding = false
            }
            addView(tv)
        }
        while (childCount > newText.length) {
            removeViewAt(childCount - 1)
        }

        val offsetPx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            8f,
            resources.displayMetrics
        )

        for (i in newText.indices) {
            val tv = getChildAt(i) as TextView
            val newChar = newText[i].toString()
            val oldChar = if (i < oldText.length) oldText[i].toString() else null

            val changed = oldChar != newChar || tv.text.toString() != newChar
            tv.text = newChar

            if (changed) {
                tv.translationY = offsetPx
                tv.alpha = 0.2f
                tv.scaleX = 0.88f
                tv.scaleY = 0.88f

                tv.animate()
                    .translationY(0f)
                    .alpha(1f)
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(340)
                    .setStartDelay((i * 70L))
                    .setInterpolator(overshootInterpolator)
                    .start()
            }
        }
    }
}
