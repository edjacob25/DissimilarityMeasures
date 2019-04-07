package weka.core

import kotlin.math.log

class Gambaryan : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 != val2) {
            1.0
        } else {
            val prob = probabilityA(index, val1)
            val calc = -((prob * log(prob, 2.0)) + (1 - prob * log(1 - prob, 2.0)))
            return 1 - calc
        }
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        var totalAttributes = 0
        for (i in 0 until this.m_Data.numAttributes()) {
            totalAttributes += this.m_Data.attribute(i).numValues()
        }
        return currDist + ((1.0 / totalAttributes) * diff)
    }

    override fun globalInfo(): String {
        return "This is the Gambaryan measure, designed for categorical data"
    }
}