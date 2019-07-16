package me.jacobrr

import weka.core.Instances

class ModifiedCompanion(
    instances: Instances, private val option: ModifiedOption, private val multiply: MultiplyOption,
    weighStyle: String
) {
    private val learningCompanion = LearningCompanion("N", weighStyle)

    init {
        learningCompanion.trainClassifiers(instances)
    }

    fun calculateDistance(baseDifference: Double, index: Int): Double {
        val weight = learningCompanion.weights[index]!!

        if (weight < 0.5) {
            when (option) {
                ModifiedOption.BASE -> println("Base case")
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

enum class ModifiedOption(val s: String) {
    BASE("B"),
    DISCARD_LOW("D"),
    MAX_LOW("M"),
    BASE_LOW("L")
}

enum class MultiplyOption(val s: String) {
    NORMAL("N"),
    ONE_MINUS("I")
}
