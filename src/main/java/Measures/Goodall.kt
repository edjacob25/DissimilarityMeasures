package Measures
import BaseCategoricalDistance

class Goodall : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 != val2){
            0.0
        } else{
            val prob = this.m_Data.probabilityA(index, val1)
            return 1 - getSummatoryOfProbs(prob, index)
        }
    }


    fun getSummatoryOfProbs(baseProbability: Double, index: Int): Double {
        var result = 0.0
        for (value in m_Data.attribute(index).enumerateValues()) {
            if (m_Data.probabilityA(index, value as String) <= baseProbability){
                result += m_Data.probabilityB(index, value)
            }
        }
        return result
    }


    override fun globalInfo(): String {
        return "This is the Goodall measure, designed for categorical data"
    }
}