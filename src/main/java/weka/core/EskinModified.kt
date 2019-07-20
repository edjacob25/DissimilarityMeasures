package weka.core

import me.jacobrr.ModifiedCompanion
import java.util.*

class EskinModified : Eskin() {
    private lateinit var modifiedCompanion: ModifiedCompanion

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        modifiedCompanion = ModifiedCompanion(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val baseDifference = super.difference(index, val1, val2)
        return modifiedCompanion.calculateDistance(baseDifference, index)
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