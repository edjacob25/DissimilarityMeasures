package weka.core

import me.jacobrr.LearningCompanion
import weka.core.neighboursearch.PerformanceStats

open class LinModified_KappaMax : BaseCategoricalDistance() {
    protected lateinit var learningCompanion: LearningCompanion
    var instancesWeigth = 0.0

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        learningCompanion = LearningCompanion("N", "K")
        learningCompanion.trainClassifiers(instances)

        for (weight in learningCompanion.weights){
            println("Attribute ${weight.key} has weight ${weight.value}")
        }
    }

    override fun distance(first: Instance?, second: Instance?, cutOffValue: Double, stats: PerformanceStats?): Double {
        // This calculates the Lin weight for each pair of instances
        var result = 0.0
        for (attribute in this.instances.enumerateAttributes()) {
            val index = attribute.index()
            val log1 = Math.log(probabilityA(index, first!!.stringValue(index)))
            val log2 = Math.log(probabilityA(index, second!!.stringValue(index)))
            result += log1 + log2
        }
        instancesWeigth = 1.0 / result

        // Returns distance as normal
        return super.distance(first, second, cutOffValue, stats)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val lowerLimit = if (val1 == val2) {
            - 2 * Math.log(instances.size.toDouble())
        } else {
            - 2 * Math.log(instances.size.toDouble() / 2)
        }

        val lin = if (val1 == val2) {
            2 * Math.log(probabilityA(index, val1))
        } else {
            2 * Math.log(probabilityA(index, val1) + probabilityA(index, val2))
        }
        val kappa = learningCompanion.weights[index]!!
        if (kappa < 0.5) {
            return lowerLimit
        }
        val normalizedLin = kappa * lin
        return normalizedLin
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + (diff * instancesWeigth)
    }


    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}