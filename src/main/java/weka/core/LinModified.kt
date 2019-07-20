package weka.core

import me.jacobrr.ModifiedCompanion
import java.util.*
import kotlin.math.ln

open class LinModified : BaseCategoricalDistance() {
    protected lateinit var modifiedCompanion: ModifiedCompanion

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        modifiedCompanion = ModifiedCompanion(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val lowerLimit = if (val1 == val2) {
            2 * ln(frequencies.originalNumOfInstances.toDouble())
        } else {
            2 * ln(frequencies.originalNumOfInstances.toDouble() / 2)
        }

        val lin = if (val1 == val2) {
            2 * ln(probabilityA(index, val1))
        } else {
            2 * ln(probabilityA(index, val1) + probabilityA(index, val2))
        }
        val normalizedLin = (lin + lowerLimit) / lowerLimit

        return modifiedCompanion.calculateDistance(normalizedLin, index)
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }


    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }

    override fun listOptions(): Enumeration<Option> {
        return modifiedCompanion.listOptions(super.listOptions())
    }

    override fun setOptions(options: Array<out String>?) {
        super.setOptions(options)
        modifiedCompanion.setOptions(options)
    }

    override fun getOptions(): Array<String> {
        return modifiedCompanion.getOptions(super.getOptions())
    }
}