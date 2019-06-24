package me.jacobrr

import weka.core.Instances

class ModifiedCompanion(instances: Instances) {
    private val learningCompanion = LearningCompanion("N", "K")

    init {
        learningCompanion.trainClassifiers(instances)
    }

    fun calculateDistance(baseDifference: Double, index: Int): Double {
        val weight = learningCompanion.weights[index]!!
        if (weight < 0.5) {
            return 0.0
        }
        val normalized = (1 - weight) * baseDifference
        return normalized
    }
}