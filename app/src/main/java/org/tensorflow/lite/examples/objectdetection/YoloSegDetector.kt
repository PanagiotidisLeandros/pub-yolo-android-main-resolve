package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.examples.objectdetection.detectors.DetectionResult
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector
import org.tensorflow.lite.examples.objectdetection.detectors.Category
import org.tensorflow.lite.support.image.TensorImage

class YoloSegDetector(
    private val context: Context,
    private val confidenceThreshold: Float,
    private val maxResults: Int
) : ObjectDetector {
    private var segResults: List<SegmentationResult>? = null
    private var inferenceTime: Long = 0
    private val drawImages = DrawImages(context)
    private val instanceSegmentation = InstanceSegmentation(
        context,
        "yolo11n-seg_float16.tflite",
        null,
        object : InstanceSegmentation.InstanceSegmentationListener {
            override fun onError(error: String) {
                android.util.Log.e("YoloSegDetector", "Segmentation error: $error")
                segResults = null
            }
            override fun onEmpty() {
                segResults = emptyList()
            }
            override fun onDetect(
                interfaceTime: Long,
                results: List<SegmentationResult>,
                preProcessTime: Long,
                postProcessTime: Long
            ) {
                segResults = results
                inferenceTime = interfaceTime + preProcessTime + postProcessTime
            }
        },
        message = { msg -> android.util.Log.d("YoloSegDetector", msg) }
    )

    override fun detect(image: TensorImage, imageRotation: Int): DetectionResult {
        val bitmap = image.bitmap
        val imgH = bitmap.height
        val imgW = bitmap.width

        android.util.Log.d("PreviewDebug","Camera preview ${bitmap.width}x${bitmap.height}, rotation=$imageRotation")

        // Create an empty transparent bitmap matching INPUT dimensions
        val emptyBitmap = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888)

        try {
            instanceSegmentation.invoke(bitmap)
        } catch (e: Exception) {
            android.util.Log.e("YoloSegDetector", "Error during segmentation", e)
            return DetectionResult(emptyBitmap, emptyList()).apply {
                info = 0
            }
        }

        val rawResults = segResults ?: emptyList()
        segResults = null

        val maxResultsToUse = if (maxResults > 0) maxResults else rawResults.size

        val filteredResults = rawResults
            .filter { it.box.cnf >= confidenceThreshold }
            .sortedByDescending { it.box.cnf }
            .let { results ->
                if (maxResultsToUse in 1..results.size) {
                    results.take(maxResultsToUse)
                } else {
                    results
                }
            }

        if (filteredResults.isEmpty()) {
            android.util.Log.d("YoloSegDetector", "No segmentation results found after filtering")
            return DetectionResult(emptyBitmap, emptyList()).apply {
                info = inferenceTime
            }
        }

        // Draw overlay at the SAME dimensions as the input bitmap
        val overlay = try {
            drawImages.invoke(filteredResults)
        } catch (e: Exception) {
            android.util.Log.e("YoloSegDetector", "Error drawing overlay", e)
            emptyBitmap
        }

        val detections = filteredResults.map { segResult ->
            val box = segResult.box
            val rect = RectF(
                box.x1 * imgW,
                box.y1 * imgH,
                box.x2 * imgW,
                box.y2 * imgH
            )
            ObjectDetection(
                rect,
                Category(
                    box.clsName,
                    box.cnf
                )
            )
        }

        android.util.Log.d("YoloSegDetector", "Overlay bitmap: ${overlay.width}x${overlay.height}")

        return DetectionResult(overlay, detections).apply {
            info = inferenceTime
        }
    }
}
