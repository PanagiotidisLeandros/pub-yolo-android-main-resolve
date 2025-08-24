
package org.tensorflow.lite.examples.objectdetection


import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.examples.objectdetection.InstanceSegmentation
import org.tensorflow.lite.examples.objectdetection.detectors.Category
import org.tensorflow.lite.examples.objectdetection.detectors.DetectionResult
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector
import org.tensorflow.lite.support.image.TensorImage

class YoloSegDetector(
    private val context: Context
) : ObjectDetector {

    private val instanceSegmentation: InstanceSegmentation

    init {
        instanceSegmentation = InstanceSegmentation(
            context,
            "yolov11n-seg_float16.tflite",
            null,
            object : InstanceSegmentation.InstanceSegmentationListener {
                override fun onDetect(
                    interfaceTime: Long,
                    results: List<SegmentationResult>,
                    preProcessTime: Long,
                    postProcessTime: Long
                ) {
                    segResults = results
                }

                override fun onError(error: String) {
                    // Handle error
                }

                override fun onEmpty() {
                    // Handle empty results
                }
            },
            { message -> /* Log or handle message */ }
        )
    }

    private var segResults: List<SegmentationResult>? = null

    override fun detect(image: TensorImage, imageRotation: Int): DetectionResult {
        val bitmap = image.bitmap
        instanceSegmentation.invoke(bitmap)

        val results = segResults ?: emptyList()
        segResults = null

        // Convert SegmentationResult to ObjectDetection
        val detections = results.map { result ->
            ObjectDetection(
                boundingBox = RectF(
                    result.box.x1 * bitmap.width,
                    result.box.y1 * bitmap.height,
                    result.box.x2 * bitmap.width,
                    result.box.y2 * bitmap.height
                ),
                category = Category(result.box.clsName, result.box.cnf)
            )
        }

        // Use the drawDetections function for visualization
        val outputBitmap = drawDetections(results, bitmap.width, bitmap.height)

        return DetectionResult(outputBitmap, detections)
    }
}