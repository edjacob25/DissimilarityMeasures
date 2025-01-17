package weka.core

open class OccurenceFrequency : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2) {
            0.0
        } else {

            val freqA = frequencies.getFrequency(m_Data.attribute(index).name(), val1).toDouble()
            val freqB = frequencies.getFrequency(m_Data.attribute(index).name(), val2).toDouble()
            val items = this.m_Data.numInstances()
            return 1 - (1 / (1 + (Math.log10(items / freqA) * Math.log10(items / freqB))))
        }
    }

    override fun globalInfo(): String {
        return "This is the OF measure, designed for categorical data"
    }
}