package Measures
import BaseCategoricalDistance

class Lin : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2){
            2 * Math.log(m_Data.probabilityA(index, val1))
        } else {
            2 * Math.log(m_Data.probabilityA(index, val1) + m_Data.probabilityA(index, val2))
        }

    }

    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}