package weka.core

import me.jacobrr.LearningCompanion
import me.jacobrr.toEnumeration
import java.util.*

open class LearningBasedDissimilarity : BaseCategoricalDistance() {

    protected lateinit var learningCompanion: LearningCompanion
    protected var strategy = "A"
    protected var weightStyle = "N"

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        learningCompanion = LearningCompanion(strategy, weightStyle)
        learningCompanion.trainClassifiers(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        if (learningCompanion.weights[index] == 0.0)
            return 0.0
        return learningCompanion.weights[index]!! * learningCompanion.similarityMatrices[index]!![val1]!![val2]!!
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }

    override fun globalInfo(): String {
        return "This is the learning based measure, designed for categorical data"
    }

    override fun listOptions(): Enumeration<Option> {
        // TODO: Fix options not appearing in the Weka UI
        val result = super.listOptions().toList().toMutableList()
        result.add(
            Option(
                "The strategy to be used. Options are A, B, C, D, E or N for none. Defaults to A",
                "S", 1, "-S <strategy>"
            )
        )
        result.add(
            Option(
                "Which weight is going to be used. Options are K for kappa, A for Auc and N for a " +
                        "uniform weight. Defaults to N", "w", 1, "-w <weight>"
            )
        )
        return result.toEnumeration()
    }

    override fun setOptions(options: Array<out String>?) {
        super.setOptions(options)
        val strat = Utils.getOption('S', options)
        if (strat.isNotEmpty()) {
            strategy = strat
        }
        val weight = Utils.getOption('w', options)
        if (weight.isNotEmpty()) {
            weightStyle = weight
        }
    }

    override fun getOptions(): Array<String> {
        val result = super.getOptions().toMutableList()
        result.add("-S")
        result.add(strategy)
        result.add("-w")
        result.add(weightStyle)
        return result.toTypedArray()
    }


}
