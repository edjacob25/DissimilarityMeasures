package weka.core
import weka.core.BaseCategoricalDistance

class Eskin : BaseCategoricalDistance() {


    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2){
            1.0
        } else {
            val attributes = this.m_Data.attributeStats(index).distinctCount
            (attributes / (attributes +2)).toDouble()
        }
    }

    override fun globalInfo(): String {
        return "This is the Eskin measure, designed for categorical data"
    }

}