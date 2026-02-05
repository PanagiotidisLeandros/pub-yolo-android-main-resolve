/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 * ...
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection
//additional imports for accelerometer sensor
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.MotionEvent
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import kotlin.math.tan
import android.widget.Toast
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.SpeechRecognizer
import androidx.core.content.ContentProviderCompat.requireContext


class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs), SensorEventListener  {                        //added ,SensorEventListener needed to use accelerometer to calculate distance from abject

    private var results: List<ObjectDetection> = LinkedList()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var scaleFactor: Float = 1f
    private var bounds = Rect()
    private var isTrackingFinger = false//
    private var initialTouchX: Float = 0.0f//
    private var initialTouchY: Float = 0.0f//
    private var textToSpeech: TextToSpeech? = null//
    private var HaveSpoken = false//
    var previousDetectedClass: String? = null//
    val vibrationPattern = longArrayOf(0, 100)//
    val vibrationEffect = VibrationEffect.createWaveform(vibrationPattern, -1)//


    private lateinit var sensorManager: SensorManager //
    private var currentDistance: Double? = null //
    private var crosshairPaint = Paint()//!




    init {
        initPaints()
        setUpSensorStuff()
        initTextToSpeech()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
        setUpSensorStuff() //
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        crosshairPaint.color = Color.RED//!
        crosshairPaint.strokeWidth = 4f//!
        crosshairPaint.style = Paint.Style.STROKE//!
    }

    private fun setUpSensorStuff() {


        // Create the sensor manager
        sensorManager = context?.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (rotationVectorSensor == null) {
            //android.util.Log.e("OverlayView", "Rotation vector sensor NOT available on this device!")
            return
        } else {
            //android.util.Log.d("OverlayView", "Rotation vector sensor found, registering listener...")
        }

        // Specify the sensor you want to listen to
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.also { rotationVectorSensor  ->
            sensorManager.registerListener(
                this,
                rotationVectorSensor,
                SensorManager.SENSOR_DELAY_NORMAL
            )
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            //android.util.Log.d("OverlayView", "Sensor event values: ${event.values.joinToString()}")
            val rotationMatrix = FloatArray(9)
            val orientationAngles = FloatArray(3)

            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientationAngles)

            val azimuth = Math.toDegrees(orientationAngles[0].toDouble()) // Yaw
            val pitch = Math.toDegrees(orientationAngles[1].toDouble())   // Forward/back tilt
            val roll = Math.toDegrees(orientationAngles[2].toDouble())    // Left/right tilt

            val cameraHeight = 1.5

            val absPitch = kotlin.math.abs(pitch)
            val absRoll = kotlin.math.abs(roll)

            val isPortrait = absRoll < 45
            val isLandscape = absRoll >= 45

            var zenithAngleDeg: Double? = null

            if (isPortrait) {
                if (absPitch in 1.0..85.0) {
                    zenithAngleDeg = absPitch
                }
            } else if (isLandscape) {
                if (absRoll in 1.0..85.0) {
                    zenithAngleDeg = absRoll
                }
            }

            if (zenithAngleDeg != null) {
                //android.util.Log.d("OverlayView", "Calculated distance: ${currentDistance}")
                val zenithAngleRad = Math.toRadians(zenithAngleDeg)
                val distance = cameraHeight * kotlin.math.tan(zenithAngleRad)
                currentDistance = distance;
                postInvalidate()


//            square.text = """
//                Orientation: ${if (isPortrait) "Portrait" else "Landscape"}
//                Angle: ${String.format("%.1f", zenithAngleDeg)}°
//                Distance: ${String.format("%.2f", distance)} m
//            """.trimIndent()
            } else {
                //android.util.Log.d("OverlayView", "Zenith angle is null or invalid")
//            square.text = "Hold phone at a downward angle to measure distance"
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        return
    }

    fun releaseSensor() {
        sensorManager.unregisterListener(this)
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        //android.util.Log.d("OverlayView", "Current distance: $currentDistance")

        for (result in results) {
            val boundingBox = result.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                result.category.label + " " +
                        String.format("%.2f", result.category.confidence) +
                        (currentDistance?.let { " | ${String.format("%.1f", it)} meters" } ?: "")


            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(
        detectionResults: List<ObjectDetection>,
        imageHeight: Int,
        imageWidth: Int
    ) {
        results = detectionResults


        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isTrackingFinger = true
                initialTouchX = event.x
                initialTouchY = event.y
            }



            MotionEvent.ACTION_MOVE -> {
                if (isTrackingFinger) {
                    val touchX = event.x
                    val touchY = event.y

                    var objectDetectedInsideBox = false
                    for (detection in results) {
                        val (scaledLeft, scaledTop, scaledRight, scaledBottom) =
                            calculateScaledBoundingBox(detection.boundingBox)
                        if (isTouchWithinBoundingBox(
                                scaledLeft,
                                scaledTop,
                                scaledRight,
                                scaledBottom,
                                touchX,
                                touchY
                            )
                        ) {
                            objectDetectedInsideBox = true
                            val currentDetectedClass = detection.category.label

                            triggerVibration()

                            // Check if class changed and text hasn't been spoken yet
                            if (previousDetectedClass != currentDetectedClass && !HaveSpoken) {
                                speakText(currentDetectedClass, currentDistance)
                                HaveSpoken = true

                                triggerVibration()

                            }

                            triggerVibration()
                            break
                        }
                    }

                    // Reset HaveSpoken if finger moves outside all bounding boxes
                    if (!objectDetectedInsideBox) {
                        HaveSpoken = false
                    }
                }
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isTrackingFinger = false
                HaveSpoken = false
                previousDetectedClass = null
            }
        }
        return true
    }


    private fun calculateScaledBoundingBox(boundingBox: RectF): List<Float> {
        val top = boundingBox.top * scaleFactor
        val bottom = boundingBox.bottom * scaleFactor
        val left = boundingBox.left * scaleFactor
        val right = boundingBox.right * scaleFactor
        return listOf(left, top, right, bottom)
    }

    private fun isTouchWithinBoundingBox(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        touchX: Float,
        touchY: Float
    ): Boolean {
        return touchX >= left && touchX <= right && touchY >= top && touchY <= bottom
    }
    private fun triggerVibration() {

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager =
                context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            val vibrator = vibratorManager.getDefaultVibrator()
            vibrator.vibrate(vibrationEffect)
        } else{
            val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            vibrator.vibrate(vibrationEffect)
        }
    }


    private fun initTextToSpeech() {
        textToSpeech = TextToSpeech(context!!, object : TextToSpeech.OnInitListener {
            override fun onInit(status: Int) {
                if (status == TextToSpeech.SUCCESS) {
                } else {
                    Toast.makeText(context, "Text-to-Speech failed to initialize",
                        Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        textToSpeech?.shutdown()
    }

    private fun speakText(label: String, distance: Double?) {
        val translations = coco_translations.CocoTranslations.translations
        val key = label.toLowerCase()

        val greekTranslation = translations[key] ?: label // Fallback to English label
        val distanceString = distance?.let { String.format("%.1f", it) } ?: ""

        val message = if (distance != null) {
            "$greekTranslation απόσταση $distanceString μέτρα" // "at a distance of X meters" in Greek
        } else {
            greekTranslation
        }

        if (textToSpeech?.speak(message, TextToSpeech.QUEUE_FLUSH, null) == TextToSpeech.ERROR) {
            Toast.makeText(context, "Text-to-Speech failed to speak", Toast.LENGTH_SHORT).show()
        }
    }

}