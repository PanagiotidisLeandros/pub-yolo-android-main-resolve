package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.examples.objectdetection.detectors.DetectionResult
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector
import org.tensorflow.lite.examples.objectdetection.detectors.Category
import org.tensorflow.lite.support.image.TensorImage

class YoloSegDetector(private val context: Context) : ObjectDetector {
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

        android.util.Log.d("YoloSegDetector", "Input bitmap: ${imgW}x${imgH}, rotation: $imageRotation")

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

        val results = segResults ?: emptyList()
        segResults = null

        if (results.isEmpty()) {
            android.util.Log.d("YoloSegDetector", "No segmentation results found")
            return DetectionResult(emptyBitmap, emptyList()).apply {
                info = inferenceTime
            }
        }

        // Draw overlay at the SAME dimensions as the input bitmap
        val overlay = try {
            drawImages.invoke(results)
        } catch (e: Exception) {
            android.util.Log.e("YoloSegDetector", "Error drawing overlay", e)
            emptyBitmap
        }

        android.util.Log.d("YoloSegDetector", "Overlay bitmap: ${overlay.width}x${overlay.height}")

        return DetectionResult(overlay, emptyList()).apply {
            info = inferenceTime
        }
    }
}