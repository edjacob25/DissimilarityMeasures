package weka.core

import weka.core.neighboursearch.PerformanceStats

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

            when {
                firstI == secondI -> {
                    diff = difference(firstI, first.stringValue(first.valueSparse(p1).toInt()), second.stringValue(second.valueSparse(p2).toInt()))
                    p1++
                    p2++
                }
                firstI > secondI -> {
                    diff = difference(secondI,"", second.stringValue(second.valueSparse(p2).toInt()))
                    p2++
                }
                else -> {
                    diff = difference(firstI, first.stringValue(first.valueSparse(p1).toInt()), "")
                    p1++
                }
            }
            stats?.incrCoordCount()
            println("Distance so far: $distance --- Diff calculated: $diff")
            distance = updateDistance(distance, diff)
            if (distance > cutOffValue) {
                return java.lang.Double.POSITIVE_INFINITY
            }
        }

        return distance
    }

    abstract fun difference(index: Int, val1: String, val2: String): Double

    override fun difference(index: Int, val1: Double, val2: Double): Double {
        when (m_Data.attribute(index).type()) {
            Attribute.NOMINAL -> return if (Utils.isMissingValue(val1) || Utils.isMissingValue(
                    val2
                )
                || val1.toInt() != val2.toInt()
            ) {
                1.0
            } else {
                0.0
            }

            Attribute.NUMERIC -> if (Utils.isMissingValue(val1) || Utils.isMissingValue(
                    val2
                )
            ) {
                if (Utils.isMissingValue(val1) && Utils.isMissingValue(val2)) {
                    return if (!m_DontNormalize) {
                        1.0
                    } else {
                        m_Ranges[index][R_WIDTH]
                    }
                } else {
                    var diff: Double
                    diff = if (Utils.isMissingValue(val2)) {
                        if (!m_DontNormalize) norm(val1, index) else val1
                    } else {
                        if (!m_DontNormalize) norm(val2, index) else val2
                    }
                    if (!m_DontNormalize && diff < 0.5) {
                        diff = 1.0 - diff
                    } else if (m_DontNormalize) {
                        return if (m_Ranges[index][R_MAX] - diff > diff - m_Ranges[index][R_MIN]) {
                            m_Ranges[index][R_MAX] - diff
                        } else {
                            diff - m_Ranges[index][R_MIN]
                        }
                    }
                    return diff
                }
            } else {
                return if (!m_DontNormalize) norm(val1, index) - norm(val2, index) else val1 - val2
            }

            else -> return 0.0
        }
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + ((1/this.m_Data.numAttributes()) * diff)
    }

    override fun getRevision(): String {
        return RevisionUtils.extract("Revision: 1")
    }


    fun probabilityA(index: Int, value: String): Double {
        return frequencies.getFrequency(m_Data.attribute(index).name(), value) / m_Data.numInstances().toDouble()
    }

    fun probabilityB(index: Int, value: String): Double {
        val freq = frequencies.getFrequency(m_Data.attribute(index).name(), value)
        val numberInstances = m_Data.numInstances().toDouble()
        return (freq * (freq -1)) / (numberInstances * (numberInstances - 1))
    }
}