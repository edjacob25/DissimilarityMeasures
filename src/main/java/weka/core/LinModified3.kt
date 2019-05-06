package weka.core

class LinModified3 : LinModified() {
    override fun difference(index: Int, val1: String, val2: String): Double {
        return super.difference(index, val1, val2) / instances.numAttributes()
    }
}