package Measures
import BaseCategoricalDistance
import kotlin.math.log

class Gambaryan : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 != val2){
            0.0
        } else{
            val prob = this.m_Data.probabilityA(index, val1)
            return  -((prob * log(prob, 2.0)) + (1 - prob * log(1 - prob, 2.0)))
        }
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        var totalAttributes = 0
        for (i in 0 until this.m_Data.numAttributes()) {
            totalAttributes += this.m_Data.attribute(i).numValues()
        }
        return currDist + ((1/totalAttributes) * diff)
    }

    override fun globalInfo(): String {
        return "This is the Gambaryan measure, designed for categorical data"
    }
}