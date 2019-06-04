package weka.core

open class LinModified2 : LinModified() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return learningCompanion.weights[index]!! * super.difference(index, val1, val2)
    }
}