package me.jacobrr

import weka.core.Instances

class ModifiedCompanion(
    instances: Instances, private val option: ModifiedOption, private val multiply: MultiplyOption,
    weight: String
) {
    private val learningCompanion = LearningCompanion("N", weight, weight)

    init {
        learningCompanion.trainClassifiers(instances)
    }

    fun calculateDistance(baseDifference: Double, index: Int): Double {
        val weight = learningCompanion.weights[index]!!

        if (weight < 0.5) {
            when (option) {
                ModifiedOption.BASE -> Unit
                ModifiedOption.DISCARD_LOW -> return 0.0
                ModifiedOption.MAX_LOW -> return 1.0
                ModifiedOption.BASE_LOW -> return baseDifference
            }
        }

        val normalized = when (multiply) {
            MultiplyOption.NORMAL -> weight * baseDifference
            MultiplyOption.ONE_MINUS -> (1 - weight) * baseDifference
        }
        return normalized
    }
}
