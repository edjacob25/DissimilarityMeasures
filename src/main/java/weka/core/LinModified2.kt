package weka.core

class LinModified2 : LinModified() {
    override fun updateDistance(currDist: Double, diff: Double): Double {
        var result = 0.0
        for (attribute in this.instances.enumerateAttributes()) {
            val index = attribute.index()
            val log1 = Math.log(probabilityA(index, activeInstance1.stringValue(index)))
            val log2 = Math.log(probabilityA(index, activeInstance2.stringValue(index)))
            result += log1 + log2

        }
        return currDist + ((1.0 / result) * diff)
    }
}