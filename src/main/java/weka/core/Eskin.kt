package weka.core

open class Eskin : BaseCategoricalDistance() {

    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2) {
            0.0
        } else {
            val attributes = this.m_Data.attributeStats(index).distinctCount.toDouble()
            1.0 - ((attributes * attributes) / ((attributes * attributes) + 2))
        }
    }

    override fun globalInfo(): String {
        return "This is the Eskin measure, designed for categorical data"
    }

}