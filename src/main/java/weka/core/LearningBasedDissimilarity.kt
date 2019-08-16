package weka.core

import me.jacobrr.*
import java.util.*

open class LearningBasedDissimilarity : BaseCategoricalDistance() {

    private lateinit var learningCompanion: LearningCompanion
    private var strategy = "A"
    private var multiplyWeight = "N"
    private var decideWeight = "A"
    private var symmetric = false
    private var normalizeDissimilarity = false
    private var option: ModifiedOption = ModifiedOption.BASE
    private var multiplyOption: MultiplyOption = MultiplyOption.NORMAL
    private var aucOption: AUCOption = AUCOption.NORMAL

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        learningCompanion = LearningCompanion(strategy, multiplyWeight, decideWeight, symmetric, normalizeDissimilarity,
            aucOption)
        learningCompanion.trainClassifiers(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val weight = learningCompanion.weights[index]!!

        if (weight == 0.0)
            return 0.0

        val baseDifference = learningCompanion.dissimilarityMatrices[index]!![val1]!![val2]!!
        if (weight < 0.5) {
            when (option) {
                ModifiedOption.BASE -> Unit
                ModifiedOption.DISCARD_LOW -> return 0.0
                ModifiedOption.MAX_LOW -> return 1.0
                ModifiedOption.BASE_LOW -> return if (val1 == val2) {
                    0.0
                } else {
                    1.0
                }
            }
        }

        val normalized = when (multiplyOption) {
            MultiplyOption.NORMAL -> weight * baseDifference
            MultiplyOption.ONE_MINUS -> (1 - weight) * baseDifference
            else -> baseDifference
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
                "Which AUC type is going to be used. Options are N for normal, S Sor second and W for weka. " +
                        "Defaults to N", "a", 1, "-a <weightType>"
            )
        )

        result.add(
            Option(
                "Whether to make the similarity matrix symmetric", "s", 0, "-s"
            )
        )

        result.add(
            Option(
                "Whether to make the dissimilarity matrix normalized", "n", 0, "-n"
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
        } else if (mWeight.isNotEmpty()) {
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
            "N" -> MultiplyOption.NORMAL
            else -> MultiplyOption.NO_MULTIPLY
        }

        val auc = Utils.getOption('a', options)
        aucOption = when (auc) {
            "S" -> AUCOption.SECOND
            "W" -> AUCOption.WEKA
            else -> AUCOption.NORMAL
        }

        val symmetricFlag = Utils.getFlag('s', options)
        symmetric = symmetricFlag


        val normalizedFlag = Utils.getFlag('n', options)
        normalizeDissimilarity = normalizedFlag
    }

    override fun getOptions(): Array<String> {
        val result = super.getOptions().toMutableList()
        result.add("-S")
        result.add(strategy)
        result.add("-W")
        result.add(multiplyWeight)
        result.add("-w")
        result.add(decideWeight)
        result.add("-o")
        result.add(option.s)
        result.add("-t")
        result.add(multiplyOption.s)
        result.add("-a")
        result.add(aucOption.s)
        return result.toTypedArray()
    }
}
