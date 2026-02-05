package org.tensorflow.lite.examples.objectdetection

import kotlin.math.abs
import kotlin.math.tan

class DistanceEstimator(
    private var cameraHeight: Double = 1.5, // Meters
    private var verticalFov: Double = 60.0  // Default VFOV in degrees
) {


    private val jumpCounters = mutableMapOf<String, Int>()

    companion object {
        private const val MAX_DISTANCE_JUMP = 3.0
        private const val JUMP_STABILITY_THRESHOLD = 3
    }

    fun setVerticalFov(fov: Double) {
        this.verticalFov = fov
    }

    fun setCameraHeight(height: Double) {
        this.cameraHeight = height
    }

    /**
     * Clears history of jump counters (e.g. when camera is cleared).
     */
    fun clearHistory() {
        jumpCounters.clear()
    }

    /**
     * Calculates the distance to an object with Outlier Rejection.
     *
     * @param bottomPixel The Y-coordinate of the bottom of the bounding box.
     * @param frameHeight The total height of the original image frame.
     * @param deviceAngleRad The tilt of the device in radians (Zenith angle).
     * @param previousDistance The last valid distance for this object.
     * @param objectId A unique identifier (e.g., category label) to track jump stability.
     * @return The estimated distance in meters, or null if calculation is invalid.
     */
    fun calculateDistance(
        bottomPixel: Float,
        frameHeight: Int,
        deviceAngleRad: Float,
        previousDistance: Double? = null,
        objectId: String? = null
    ): Double? {
        val pixelOffsetFromCenter = bottomPixel - (frameHeight / 2.0)
        val angularOffsetDeg = (pixelOffsetFromCenter / frameHeight) * verticalFov
        val angularOffsetRad = Math.toRadians(angularOffsetDeg)

        val totalAngleRad = deviceAngleRad + angularOffsetRad
        val minAngleRad = Math.toRadians(0.5)

        if (totalAngleRad < minAngleRad || totalAngleRad >= Math.PI / 2) {
            return null
        }

        val distance = cameraHeight / tan(totalAngleRad)
        val calculatedDistance = if (distance > 0) distance else null

        // Outlier Rejection / Jump Filtering logic
        if (calculatedDistance != null && previousDistance != null && objectId != null) {
            val diff = abs(calculatedDistance - previousDistance)
            if (diff > MAX_DISTANCE_JUMP) {
                val currentJumpCount = jumpCounters.getOrDefault(objectId, 0) + 1
                jumpCounters[objectId] = currentJumpCount
                
                // If the "jump" persists, we eventually accept it (it might be a real move)
                return if (currentJumpCount >= JUMP_STABILITY_THRESHOLD) {
                    jumpCounters[objectId] = 0
                    calculatedDistance
                } else {
                    previousDistance
                }
            } else {
                jumpCounters[objectId] = 0 // Reset if measurements are stable
            }
        }

        return calculatedDistance
    }
}
