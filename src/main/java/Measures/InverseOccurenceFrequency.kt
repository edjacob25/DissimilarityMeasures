package Measures
import BaseCategoricalDistance

class InverseOccurenceFrequency : BaseCategoricalDistance() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2){
            1.0
        } else {

            val freqA = this.m_Data.frequency(index, val1).toDouble()
            val freqB = this.m_Data.frequency(index, val2).toDouble()
            return (1/ (1 + (Math.log10(freqA) * Math.log10(freqB))))
        }
    }

    override fun globalInfo(): String {
        return "This is the IOF measure, designed for categorical data"
    }

}