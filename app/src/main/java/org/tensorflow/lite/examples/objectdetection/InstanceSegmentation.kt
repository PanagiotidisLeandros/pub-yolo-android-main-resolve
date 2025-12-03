package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.objectdetection.ImageUtils.clone
import org.tensorflow.lite.examples.objectdetection.ImageUtils.scaleMask
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.examples.objectdetection.MetaData.extractNamesFromLabelFile
import org.tensorflow.lite.examples.objectdetection.MetaData.extractNamesFromMetadata
import java.nio.ByteBuffer
import kotlin.math.min
import kotlin.math.roundToInt

class InstanceSegmentation(
    context: Context,
    modelPath: String,
    labelPath: String?,
    private val instanceSegmentationListener: InstanceSegmentationListener,
    private val message: (String) -> Unit
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private var xPoints = 0
    private var yPoints = 0
    private var masksNum = 0
    private var isMaskChannelsFirst = false

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        val options = Interpreter.Options()
        options.setNumThreads(4)

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        labels.addAll(extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                message("Model not contains metadata, provide LABELS_PATH in Constants.kt")
                labels.addAll(MetaData.TEMP_CLASSES)
            } else {
                labels.addAll(extractNamesFromLabelFile(context, labelPath))
            }
        }

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape0 = interpreter.getOutputTensor(0)?.shape()
        val outputShape1 = interpreter.getOutputTensor(1)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape0 != null) {
            numChannel = outputShape0[1]
            numElements = outputShape0[2]
        }

        if (outputShape1 != null) {
            if (outputShape1[1] == 32) {
                masksNum = outputShape1[1]
                yPoints = outputShape1[2]
                xPoints = outputShape1[3]
                isMaskChannelsFirst = true
            } else {
                yPoints = outputShape1[1]
                xPoints = outputShape1[2]
                masksNum = outputShape1[3]
            }
        }
    }

    fun close() {
        interpreter.close()
    }

    fun invoke(frame: Bitmap) {
        if (tensorWidth == 0 || tensorHeight == 0
            || numChannel == 0 || numElements == 0
            || xPoints == 0 || yPoints == 0 || masksNum == 0) {
            instanceSegmentationListener.onError("Interpreter not initialized properly")
            return
        }

        var preProcessTime = SystemClock.uptimeMillis()

        val preProcessResult = preProcess(frame)
        val imageBuffer = preProcessResult.buffer


        val coordinatesBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1 , numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        )

        val maskProtoBuffer = TensorBuffer.createFixedSize(
            if (isMaskChannelsFirst) {
                intArrayOf(1, masksNum, yPoints, xPoints)
            } else {
                intArrayOf(1, yPoints, xPoints, masksNum)
            },
            OUTPUT_IMAGE_TYPE
        )

        val outputBuffer = mapOf<Int, Any>(
            0 to coordinatesBuffer.buffer.rewind(),
            1 to maskProtoBuffer.buffer.rewind()
        )

        preProcessTime = SystemClock.uptimeMillis() - preProcessTime

        var interfaceTime = SystemClock.uptimeMillis()

        interpreter.runForMultipleInputsOutputs(imageBuffer, outputBuffer)

        interfaceTime = SystemClock.uptimeMillis() - interfaceTime

        var postProcessTime = SystemClock.uptimeMillis()

        val bestBoxes = bestBox(coordinatesBuffer.floatArray) ?: run {
            instanceSegmentationListener.onEmpty()
            return
        }

        val maskProto = reshapeMaskOutput(maskProtoBuffer.floatArray)

        val segmentationResults = bestBoxes.map { box ->
            val adjustedBox = adjustBoxToOriginal(box, frame.width, frame.height, preProcessResult.letterboxInfo)
            SegmentationResult(
                box = adjustedBox,
                mask = getFinalMask(frame.width, frame.height, box, maskProto, preProcessResult.letterboxInfo)
            )
        }

        postProcessTime = SystemClock.uptimeMillis() - postProcessTime

        instanceSegmentationListener.onDetect(
            preProcessTime = preProcessTime,
            interfaceTime = interfaceTime,
            postProcessTime = postProcessTime,
            results = segmentationResults
        )
    }

    private fun getFinalMask(
        width: Int,
        height: Int,
        output0: Output0,
        output1: List<Array<FloatArray>>,
        letterboxInfo: LetterboxInfo
    ): Array<FloatArray> {
        val output1Copy = output1.clone()
        val relX1 = output0.x1 * xPoints
        val relY1 = output0.y1 * yPoints
        val relX2 = output0.x2 * xPoints
        val relY2 = output0.y2 * yPoints

        val zero: Array<FloatArray> = Array(yPoints) { FloatArray(xPoints) { 0F } }
        for ((index, proto) in output1Copy.withIndex()) {
            for (y in 0 until yPoints) {
                for (x in 0 until xPoints) {
                    proto[y][x] *= output0.maskWeight[index]
                    if (x + 1  > relX1 && x + 1 < relX2 && y + 1 > relY1 && y + 1 < relY2) {
                        zero[y][x] += proto[y][x]
                    }
                }
            }
        }

        val protoScaleX = xPoints.toFloat() / tensorWidth
        val protoScaleY = yPoints.toFloat() / tensorHeight
        val padXProto = (letterboxInfo.padX * protoScaleX).roundToInt()
        val padYProto = (letterboxInfo.padY * protoScaleY).roundToInt()
        val contentWidth = xPoints - padXProto * 2
        val contentHeight = yPoints - padYProto * 2

        val cropped = Array(contentHeight) { y ->
            FloatArray(contentWidth) { x ->
                zero[y + padYProto][x + padXProto]
            }
        }

        return cropped.scaleMask(width , height)
    }


    private fun reshapeMaskOutput(floatArray: FloatArray): List<Array<FloatArray>> {
        return if (isMaskChannelsFirst) {
            List(masksNum) { mask ->
                Array(yPoints) { y ->
                    FloatArray(xPoints) { x ->
                        floatArray[(mask * yPoints + y) * xPoints + x]
                    }
                }
            }
        } else {
            List(masksNum) { mask ->
                Array(yPoints) { y ->
                    FloatArray(xPoints) { x ->
                        floatArray[(y * xPoints + x) * masksNum + mask]
                    }
                }
            }
        }
    }

    private fun bestBox(array: FloatArray) : List<Output0>? {

        val output0List = mutableListOf<Output0>()

        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var currentInd = 4
            var arrayIdx = c + numElements * currentInd

            while (currentInd < (numChannel - masksNum)){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = currentInd - 4
                }
                currentInd++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                val maskWeight = mutableListOf<Float>()
                while (currentInd < numChannel){
                    maskWeight.add(array[arrayIdx])
                    currentInd++
                    arrayIdx += numElements
                }

                output0List.add(
                    Output0(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName,
                        maskWeight = maskWeight
                    )
                )
            }
        }

        if (output0List.isEmpty()) return null

        return applyNMS(output0List)
    }

    private fun applyNMS(output0List: List<Output0>) : MutableList<Output0> {
        val sortedBoxes = output0List.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<Output0>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: Output0, box2: Output0): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    private fun preProcess(frame: Bitmap): PreProcessResult {
        val scale = min(
            tensorWidth.toFloat() / frame.width.toFloat(),
            tensorHeight.toFloat() / frame.height.toFloat()
        )

        val resizedWidth = (frame.width * scale).roundToInt()
        val resizedHeight = (frame.height * scale).roundToInt()

        val padX = (tensorWidth - resizedWidth) / 2f
        val padY = (tensorHeight - resizedHeight) / 2f

        val resizedBitmap = Bitmap.createScaledBitmap(frame, resizedWidth, resizedHeight, false)
        val paddedBitmap = Bitmap.createBitmap(tensorWidth, tensorHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawBitmap(resizedBitmap, padX, padY, null)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(paddedBitmap)
        val processedImage = imageProcessor.process(tensorImage)

        return PreProcessResult(
            buffer = arrayOf(processedImage.buffer),
            letterboxInfo = LetterboxInfo(
                scale = scale,
                padX = padX,
                padY = padY,
                inputWidth = tensorWidth,
                inputHeight = tensorHeight
            )
        )
    }

    private fun adjustBoxToOriginal(
        box: Output0,
        originalWidth: Int,
        originalHeight: Int,
        letterboxInfo: LetterboxInfo
    ): Output0 {
        fun adjustX(value: Float): Float {
            val pixel = value * letterboxInfo.inputWidth - letterboxInfo.padX
            return (pixel / letterboxInfo.scale / originalWidth).coerceIn(0f, 1f)
        }

        fun adjustY(value: Float): Float {
            val pixel = value * letterboxInfo.inputHeight - letterboxInfo.padY
            return (pixel / letterboxInfo.scale / originalHeight).coerceIn(0f, 1f)
        }

        val newX1 = adjustX(box.x1)
        val newX2 = adjustX(box.x2)
        val newY1 = adjustY(box.y1)
        val newY2 = adjustY(box.y2)

        val newW = newX2 - newX1
        val newH = newY2 - newY1
        val newCx = newX1 + newW / 2f
        val newCy = newY1 + newH / 2f

        return box.copy(
            x1 = newX1,
            y1 = newY1,
            x2 = newX2,
            y2 = newY2,
            w = newW,
            h = newH,
            cx = newCx,
            cy = newCy
        )
    }

    private data class PreProcessResult(
        val buffer: Array<ByteBuffer>,
        val letterboxInfo: LetterboxInfo
    )

    private data class LetterboxInfo(
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val inputWidth: Int,
        val inputHeight: Int
    )


    interface InstanceSegmentationListener {
        fun onError(error: String)
        fun onEmpty()
        fun onDetect(
            interfaceTime: Long,
            results: List<SegmentationResult>,
            preProcessTime: Long,
            postProcessTime: Long
        )
    }


    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}
