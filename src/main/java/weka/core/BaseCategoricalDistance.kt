package weka.core

import me.jacobrr.FrequencyCompanion
import weka.core.neighboursearch.PerformanceStats
import kotlin.math.abs

abstract class BaseCategoricalDistance : NormalizableDistance {
    constructor() : super()
    constructor(data: Instances?) : super(data)

    lateinit var frequencies: FrequencyCompanion
    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        println("Setting instances")
        frequencies = FrequencyCompanion(insts!!)
    }

    override fun distance(first: Instance?, second: Instance?, cutOffValue: Double, stats: PerformanceStats?): Double {
        var distance = 0.0
        var firstI: Int
        var secondI: Int
        val firstNumValues = first!!.numValues()
        val secondNumValues = second!!.numValues()
        val numAttributes = m_Data.numAttributes()
        val classIndex = m_Data.classIndex()

        validate()

        var p1 = 0
        var p2 = 0
        while (p1 < firstNumValues || p2 < secondNumValues) {
            firstI = if (p1 >= firstNumValues) {
                numAttributes
            } else {
                first.index(p1)
            }

            secondI = if (p2 >= secondNumValues) {
                numAttributes
            } else {
                second.index(p2)
            }

            if (firstI == classIndex) {
                p1++
                continue
            }
            if (firstI < numAttributes && !m_ActiveIndices[firstI]) {
                p1++
                continue
            }

            if (secondI == classIndex) {
                p2++
                continue
            }
            if (secondI < numAttributes && !m_ActiveIndices[secondI]) {
                p2++
                continue
            }
            val diff: Double

            if (!first.attribute(firstI).isNominal) {
                diff = difference(firstI, first.valueSparse(p1), second.valueSparse(p2))
                distance = updateDistance(distance, diff)
                p1++
                p2++
                continue
            }

            val value1 = first.stringValue(firstI)
            val value2 = second.stringValue(secondI)
            if (value1 == "?" || value2 == "?") {
                diff = 1.0
                p1++
                p2++
            } else {
                when {
                    firstI == secondI -> {
                        diff = difference(firstI, value1, value2)
                        p1++
                        p2++
                    }
                    firstI > secondI -> {
                        diff = difference(secondI, "", value2)
                        p2++
                    }
                    else -> {
                        diff = difference(firstI, value1, "")
                        p1++
                    }
                }
            }
            stats?.incrCoordCount()
            distance = updateDistance(distance, diff)
        }

        return distance
    }

    abstract fun difference(index: Int, val1: String, val2: String): Double

    override fun difference(index: Int, val1: Double, val2: Double): Double {
        when (m_Data.attribute(index).type()) {
            Attribute.NUMERIC -> if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2)) {
                if (Utils.isMissingValue(val1) && Utils.isMissingValue(val2)) {
                    return 1.0
                } else {
                    val diff = if (Utils.isMissingValue(val2)) {
                        norm(val1, index)
                    } else {
                        norm(val2, index)
                    }
                    return diff
                }
            } else {
                val diff = abs(norm(val1, index) - norm(val2, index))
                return diff
            }

            else -> return 0.0
        }
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        val proportion = 1.0 / this.m_Data.numAttributes()
        val difference = proportion * diff
        return currDist + difference
    }

    override fun getRevision(): String {
        return RevisionUtils.extract("Revision: 1")
    }


    fun probabilityA(index: Int, value: String): Double {
        val freq = frequencies.getFrequency(m_Data.attribute(index).name(), value)
        val numInstances = frequencies.originalNumOfInstances.toDouble()
        return freq / numInstances
    }

    fun probabilityB(index: Int, value: String): Double {
        val freq = frequencies.getFrequency(m_Data.attribute(index).name(), value)
        val numberInstances = frequencies.originalNumOfInstances.toDouble()
        return (freq * (freq - 1)) / (numberInstances * (numberInstances - 1))
    }
}