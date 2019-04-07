package weka.core

class InverseOccurenceFrequency : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2) {
            0.0
        } else {

            val freqA = frequencies.getFrequency(m_Data.attribute(index).name(), val1).toDouble()
            val freqB = frequencies.getFrequency(m_Data.attribute(index).name(), val2).toDouble()
            return 1 - (1 / (1 + (Math.log10(freqA) * Math.log10(freqB))))
        }
    }

    override fun globalInfo(): String {
        return "This is the IOF measure, designed for categorical data"
    }

}