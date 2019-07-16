package weka.clusterers

import weka.core.Capabilities
import weka.core.DistanceFunction
import weka.core.Instance

class CategoricalKMeans : SimpleKMeans() {

    override fun getCapabilities(): Capabilities {
        val result = super.getCapabilities()
        result.disableAll()

        result.enable(Capabilities.Capability.NO_CLASS)
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES)
        result.enable(Capabilities.Capability.MISSING_VALUES)
        return result
    }

    override fun setDistanceFunction(df: DistanceFunction?) {
        m_DistanceFunction = df
    }

    override fun clusterInstance(instance: Instance?): Int {
        return clusterProcessedInstance(instance!!, true)
    }

    private fun clusterProcessedInstance(instance: Instance, useFastDistCalc: Boolean): Int {
        var minDist = Integer.MAX_VALUE.toDouble()
        var bestCluster = 0
        for (i in 0 until m_NumClusters) {
            val dist: Double = if (useFastDistCalc) {
                m_DistanceFunction.distance(instance, m_ClusterCentroids.instance(i), minDist)
            } else {
                m_DistanceFunction.distance(instance, m_ClusterCentroids.instance(i))
            }
            if (dist < minDist) {
                minDist = dist
                bestCluster = i
            }
        }
        return bestCluster
    }
}

