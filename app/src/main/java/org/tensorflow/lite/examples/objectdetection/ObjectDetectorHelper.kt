package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.examples.objectdetection.detectors.DetectionResult
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector
import org.tensorflow.lite.examples.objectdetection.detectors.TaskVisionDetector
import org.tensorflow.lite.examples.objectdetection.detectors.YoloDetector
import org.tensorflow.lite.gpu.CompatibilityList
//import org.tensorflow.lite.examples.objectdetection.detectors.YoloSegDetector
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions

class ObjectDetectorHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    fun setupObjectDetector() {
        try {
            val baseOptionsBuilder = BaseOptions.builder()
                .setNumThreads(numThreads)

            when (currentDelegate) {
                DELEGATE_CPU -> {
                    // Default
                }
                DELEGATE_GPU -> {
                    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                        baseOptionsBuilder.useGpu()
                    } else {
                        objectDetectorListener?.onError("GPU is not supported on this device")
                    }
                }
                DELEGATE_NNAPI -> {
                    baseOptionsBuilder.useNnapi()
                }
            }

            val optionsBuilder = ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)


            objectDetector = when (currentModel) {
                MODEL_YOLO -> YoloDetector(
                    confidenceThreshold = threshold,
                    iouThreshold = 0.3f,
                    numThreads = numThreads,
                    maxResults = maxResults,
                    currentDelegate = currentDelegate,
                    currentModel = currentModel,
                    context = context
                )
                MODEL_YOLO_SEG -> YoloSegDetector(context)
                else -> TaskVisionDetector(
                    options = optionsBuilder.build(),
                    currentModel = currentModel,
                    context = context
                )
            }
        } catch (e: Exception) {
            objectDetectorListener?.onError(e.toString())
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        try {
            // Apply rotation
            val imageProcessorBuilder = ImageProcessor.Builder()
            imageProcessorBuilder.add(Rot90Op(-imageRotation / 90))

            val imageProcessor = imageProcessorBuilder.build()
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

            var inferenceTime = SystemClock.uptimeMillis()
            val results = objectDetector?.detect(tensorImage, imageRotation)
            inferenceTime = SystemClock.uptimeMillis() - inferenceTime

            if (results != null) {
                objectDetectorListener?.onResults(results, inferenceTime)
            } else {
                android.util.Log.e("ObjectDetectorHelper", "Detection returned null results")
            }
        } catch (e: Exception) {
            android.util.Log.e("ObjectDetectorHelper", "Error during detection", e)
            objectDetectorListener?.onError("Detection failed: ${e.message}")
        }
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: DetectionResult,
            inferenceTime: Long
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
        const val MODEL_YOLO = 4
        const val MODEL_YOLO_SEG = 5
    }
}
