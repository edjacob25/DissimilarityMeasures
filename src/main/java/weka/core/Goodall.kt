package weka.core

open class Goodall : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 != val2) {
            1.0
        } else {
            val prob = probabilityA(index, val1)
            return getSummatoryOfProbs(prob, index)
        }
    }


    fun getSummatoryOfProbs(baseProbability: Double, index: Int): Double {
        var result = 0.0
        for (value in m_Data.attribute(index).enumerateValues()) {
            if (probabilityA(index, value as String) <= baseProbability) {
                result += probabilityB(index, value)
            }
        }
        return result
    }


    override fun globalInfo(): String {
        return "This is the Goodall measure, designed for categorical data"
    }
}