package me.jacobrr

import weka.core.Instances

class ModifiedCompanion(instances: Instances) {
    private val learningCompanion = LearningCompanion("N", "K")

    init {
        learningCompanion.trainClassifiers(instances)
    }

    fun calculateDistance(baseDifference: Double, index: Int): Double {
        val kappa = learningCompanion.weights[index]!!
        if (kappa < 0.5) {
            return baseDifference
        }
        val normalized = kappa * baseDifference
        return normalized
    }
}