package weka.core

open class LinModified : BaseCategoricalDistance() {
    protected lateinit var learningCompanion: LearningCompanion

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        learningCompanion = LearningCompanion("N", "K")
        learningCompanion.trainClassifiers(instances)

        for (weight in learningCompanion.weights){
            println("Attribute ${weight.key} has weight ${weight.value}")
        }
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        return if (val1 == val2) {
            learningCompanion.weights[index]!! *  (-1 - (2 * Math.log(probabilityA(index, val1))))
        } else {
            learningCompanion.weights[index]!! *  (-1 - (2 * Math.log(probabilityA(index, val1) + probabilityA(index, val2))))
        }
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }


    override fun globalInfo(): String {
        return "This is the Lin measure, designed for categorical data"
    }
}