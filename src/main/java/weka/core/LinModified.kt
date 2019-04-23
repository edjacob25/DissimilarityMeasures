package weka.core

import weka.core.neighboursearch.PerformanceStats

open class LinModified : LearningBasedDissimilarity() {
    lateinit var activeInstance1: Instance
    lateinit var activeInstance2: Instance

    override fun setInstances(insts: Instances?) {
        weightStyle = "K"
        strategy = "N"
        super.setInstances(insts)
    }

    override fun distance(first: Instance?, second: Instance?, cutOffValue: Double, stats: PerformanceStats?): Double {
        activeInstance1 = first!!
        activeInstance2 = second!!
        return super.distance(first, second, cutOffValue, stats)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2) {
            weights[index]!! *  (-1 - (2 * Math.log(probabilityA(index, val1))))
        } else {
            weights[index]!! *  (-1 - (2 * Math.log(probabilityA(index, val1) + probabilityA(index, val2))))
        }

    }


    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}