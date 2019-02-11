package weka.core
import weka.core.BaseCategoricalDistance

class Lin : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2){
            1 - (2 * Math.log(probabilityA(index, val1)))
        } else {
            1 - (2 * Math.log(probabilityA(index, val1) + probabilityA(index, val2)))
        }

    }

    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}