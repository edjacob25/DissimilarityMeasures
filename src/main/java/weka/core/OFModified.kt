package weka.core

import me.jacobrr.LearningCompanion

class OFModified : OccurenceFrequency() {
    protected lateinit var learningCompanion: LearningCompanion

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        learningCompanion = LearningCompanion("N", "K")
        learningCompanion.trainClassifiers(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val baseDifference = super.difference(index, val1, val2)
        val kappa = learningCompanion.weights[index]!!
        if (kappa < 0.5) {
            return 1.0
        }
        val normalized = (1 - kappa) * baseDifference
        return normalized
    }
}