package weka.core

import weka.core.neighboursearch.PerformanceStats

class Lin : BaseCategoricalDistance() {
    var instancesWeigth = 0.0
    override fun distance(first: Instance?, second: Instance?, cutOffValue: Double, stats: PerformanceStats?): Double {
        var result = 0.0
        for (attribute in this.instances.enumerateAttributes()) {
            val index = attribute.index()
            val log1 = Math.log(probabilityA(index, first!!.stringValue(index)))
            val log2 = Math.log(probabilityA(index, second!!.stringValue(index)))
            result += log1 + log2
        }
        instancesWeigth = 1.0 / result

        val distance = super.distance(first, second, cutOffValue, stats)
        return 1 - distance
    }

    override fun clean() {
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val result = if (val1 == val2) {
            val prob = probabilityA(index, val1)
            2 * Math.log(prob)
        } else {
            2 * (Math.log(probabilityA(index, val1) + probabilityA(index, val2)))
        }
        return result
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + (instancesWeigth * diff)
    }

    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}