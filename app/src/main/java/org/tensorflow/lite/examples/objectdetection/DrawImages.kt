package org.tensorflow.lite.examples.objectdetection


import android.graphics.*
import org.tensorflow.lite.examples.objectdetection.SegmentationResult

fun drawDetections(results: List<SegmentationResult>, width: Int, height: Int): Bitmap {
    val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(outputBitmap)

    results.forEach { result ->
        // Draw bounding box
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f
        canvas.drawRect(
            result.box.x1 * width,
            result.box.y1 * height,
            result.box.x2 * width,
            result.box.y2 * height,
            paint
        )

        // Draw mask
        val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val maskCanvas = Canvas(maskBitmap)
        val maskPaint = Paint()
        maskPaint.color = Color.RED
        maskPaint.alpha = 128 // Semi-transparent

        val maskHeight = result.mask.size
        val maskWidth = if (maskHeight > 0) result.mask[0].size else 0

        for (y in 0 until maskHeight) {
            for (x in 0 until maskWidth) {
                if (result.mask[y][x] > 0.5f) {
                    val canvasX = x.toFloat() * width / maskWidth
                    val canvasY = y.toFloat() * height / maskHeight
                    maskCanvas.drawPoint(canvasX, canvasY, maskPaint)
                }
            }
        }

        canvas.drawBitmap(maskBitmap, 0f, 0f, null)

        // Draw label
        val labelPaint = Paint()
        labelPaint.color = Color.WHITE
        labelPaint.textSize = 18f
        canvas.drawText(
            "${result.box.clsName} (${String.format("%.2f", result.box.cnf)})",
            result.box.x1 * width,
            result.box.y1 * height - 5,
            labelPaint
        )
    }

    return outputBitmap
}