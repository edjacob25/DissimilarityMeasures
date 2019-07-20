package weka.core

import me.jacobrr.LearningCompanion
import me.jacobrr.ModifiedOption
import me.jacobrr.MultiplyOption
import me.jacobrr.toEnumeration
import java.lang.Exception
import java.util.*

open class LearningBasedDissimilarity : BaseCategoricalDistance() {

    private lateinit var learningCompanion: LearningCompanion
    private var strategy = "A"
    private var multiplyWeight = "N"
    private var decideWeight = "A"
    private var symmetric = false
    private var option: ModifiedOption = ModifiedOption.BASE
    private var multiplyOption: MultiplyOption = MultiplyOption.NORMAL

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        learningCompanion = LearningCompanion(strategy, multiplyWeight, decideWeight, symmetric)
        learningCompanion.trainClassifiers(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val weight = learningCompanion.weights[index]!!

        if (weight == 0.0)
            return 0.0

        val baseDifference = learningCompanion.similarityMatrices[index]!![val1]!![val2]!!
        if (weight < 0.5) {
            when (option) {
                ModifiedOption.BASE -> println("Base case")
                ModifiedOption.DISCARD_LOW -> return 0.0
                ModifiedOption.MAX_LOW -> return 1.0
                ModifiedOption.BASE_LOW -> return if (val1 == val2){
                    0.0
                }                                                                                                                                                                            else{
                    1.0
                }
            }
        }

        val normalized = when (multiplyOption) {
            MultiplyOption.NORMAL -> weight * baseDifference
            MultiplyOption.ONE_MINUS -> (1 - weight) * baseDifference
        }

        return normalized
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
                "Which weight is going to be used to decide. Options are K for kappa and A for Auc." +
                        "Defaults to A, and if -W is set, it uses that", "w", 1, "-w <weight>"
            )
        )

        result.add(
            Option(
                "Which weight is going to be used to multiply. Options are K for kappa, A for Auc and N for a " +
                        "uniform weight. Defaults to N", "W", 1, "-W <weight>"
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

        result.add(
            Option(
                "Whether to make the similarity matrix symmetric", "s", 0, "-s"
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
        val mWeight = Utils.getOption('W', options)
        if (mWeight.isNotEmpty()) {
            multiplyWeight = mWeight
        }

        val dWeight = Utils.getOption('w', options)
        if (dWeight.isNotEmpty()) {
            decideWeight = dWeight
        }
        else if (mWeight.isNotEmpty()){
            decideWeight = mWeight
        }

        val mOption = Utils.getOption('o', options)
        option = when (mOption) {
            "D" -> ModifiedOption.DISCARD_LOW
            "M" -> ModifiedOption.MAX_LOW
            "L" -> ModifiedOption.BASE_LOW
            else -> ModifiedOption.BASE
        }

        val type = Utils.getOption('t', options)
        multiplyOption = when (type) {
            "I" -> MultiplyOption.ONE_MINUS
            else -> MultiplyOption.NORMAL
        }

        val symmetricFlag = Utils.getFlag('s', options)
        symmetric = symmetricFlag
    }

    override fun getOptions(): Array<String> {
        val result = super.getOptions().toMutableList()
        result.add("-S")
        result.add(strategy)
        result.add("-w")
        result.add(multiplyWeight)
        result.add("-o")
        result.add(option.s)
        result.add("-t")
        result.add(multiplyOption.s)
        return result.toTypedArray()
    }
}
