package me.jacobrr

import weka.core.Instances
import weka.core.Option
import weka.core.Utils
import java.io.Serializable
import java.util.*

class ModifiedCompanion(
    private var option: ModifiedOption = ModifiedOption.BASE,
    private var multiply: MultiplyOption = MultiplyOption.NORMAL,
    private var weight: String = "A"
): Serializable {

    private lateinit var learningCompanion: LearningCompanion

    fun createLearningCompanion(instances: Instances) {
        learningCompanion = LearningCompanion("N", weight, weight)
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

    fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }

    fun listOptions(options: Enumeration<Option>): Enumeration<Option> {
        // TODO: Fix options not appearing in the Weka UI
        val result = options.toList().toMutableList()
        result.add(
            Option(
                "Which weight is going to be used. Options are K for kappa, A for Auc and N for a " +
                        "uniform weight. Defaults to N", "w", 1, "-w <weight>"
            )
        )
        result.add(
            Option(
                "Which option of kappa is going to be used. Options are B for Base, D for discard when the " +
                        "weight is low, M for mac when the weight is low and L to not multiply for the weight when is " +
                        "low. Defaults to B", "o", 1, "-o <option>"
            )
        )
        result.add(
            Option(
                "Which weight type is going to be used. Options are N for normal, I for 1 - weight. Defaults " +
                        "to N", "t", 1, "-t <weightType>"
            )
        )

        return result.toEnumeration()
    }

    fun getOptions(original: Array<String>): Array<String> {
        val result = original.toMutableList()
        result.add("-w")
        result.add(weight)
        result.add("-o")
        result.add(option.s)
        result.add("-t")
        result.add(multiply.s)
        return result.toTypedArray()
    }

    fun setOptions(options: Array<out String>?) {

        val sWeight = Utils.getOption('w', options)
        if (sWeight.isNotEmpty()) {
            weight = sWeight
        }
        val mOption = Utils.getOption('o', options)
        option = when (mOption) {
            "D" -> ModifiedOption.DISCARD_LOW
            "M" -> ModifiedOption.MAX_LOW
            "L" -> ModifiedOption.BASE_LOW
            else -> ModifiedOption.BASE
        }

        val type = Utils.getOption('t', options)
        multiply = when (type) {
            "I" -> MultiplyOption.ONE_MINUS
            else -> MultiplyOption.NORMAL
        }
    }
}
