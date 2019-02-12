package weka.core

import weka.core.neighboursearch.PerformanceStats
import kotlin.math.log

class Lin : BaseCategoricalDistance() {
    lateinit var activeInstance1: Instance
    lateinit var activeInstance2: Instance

    override fun distance(first: Instance?, second: Instance?, cutOffValue: Double, stats: PerformanceStats?): Double {
        activeInstance1 = first!!
        activeInstance2 = second!!
        return super.distance(first, second, cutOffValue, stats)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2){
            -1 - (2 * Math.log(probabilityA(index, val1)))
        } else {
            -1 - (2 * Math.log(probabilityA(index, val1) + probabilityA(index, val2)))
        }

    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        var result = 0.0
        for (attribute in this.instances.enumerateAttributes()){
            val index = attribute.index()
            val log1 = Math.log(probabilityA(index, activeInstance1.stringValue(index)))
            val log2 = Math.log(probabilityA(index, activeInstance2.stringValue(index)))
            result += log1 + log2

        }
        return currDist + ((1.0/result) * diff)
    }

    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}