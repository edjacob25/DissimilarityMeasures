package weka.core

import me.jacobrr.ModifiedCompanion

class IOFModified : InverseOccurenceFrequency() {
    protected lateinit var modifiedCompanion: ModifiedCompanion

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        modifiedCompanion = ModifiedCompanion(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val baseDifference = super.difference(index, val1, val2)
        return modifiedCompanion.calculateDistance(baseDifference, index)
    }
}