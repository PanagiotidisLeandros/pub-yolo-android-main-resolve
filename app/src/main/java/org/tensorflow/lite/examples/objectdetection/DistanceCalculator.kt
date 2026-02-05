package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

class DistanceCalculator(context: Context) : SensorEventListener {

    private val sensorManager: SensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private var angleListener: ((Float) -> Unit)? = null

    fun setAngleListener(listener: (Float) -> Unit) {
        this.angleListener = listener
    }

    fun registerSensorListener() {
        val rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) ?: return
        sensorManager.registerListener(
            this,
            rotationVectorSensor,
            SensorManager.SENSOR_DELAY_NORMAL
        )
    }

    fun unregisterSensorListener() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            val rotationMatrix = FloatArray(9)
            val orientationAngles = FloatArray(3)

            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientationAngles)

            val pitch = Math.toDegrees(orientationAngles[1].toDouble())
            val roll = Math.toDegrees(orientationAngles[2].toDouble())

            val absPitch = kotlin.math.abs(pitch)
            val absRoll = kotlin.math.abs(roll)

            val isPortrait = absRoll < 45
            val isLandscape = absRoll >= 45

            var zenithAngleDeg: Double? = null

            if (isPortrait) {
                // Phase 2 (Step 1): Expand sensor range from 1.0..85.0 to 0.1..89.9
                if (absPitch in 0.1..89.9) {
                    zenithAngleDeg = absPitch
                }
            } else if (isLandscape) {
                // Phase 2 (Step 1): Expand sensor range from 1.0..85.0 to 0.1..89.9
                if (absRoll in 0.1..89.9) {
                    zenithAngleDeg = absRoll
                }
            }

            if (zenithAngleDeg != null) {
                val zenithAngleRad = Math.toRadians(zenithAngleDeg)
                angleListener?.invoke(zenithAngleRad.toFloat())
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No implementation needed
    }
}