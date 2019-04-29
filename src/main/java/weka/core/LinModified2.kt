package weka.core

import weka.core.neighboursearch.PerformanceStats

class LinModified2 : LinModified() {
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
        println(distance)
        return 1.0 - distance
    }
    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + (instancesWeigth * diff)
    }
}