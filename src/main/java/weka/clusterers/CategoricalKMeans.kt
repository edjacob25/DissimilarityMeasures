package weka.clusterers

import weka.core.*

class CategoricalKMeans : SimpleKMeans() {

    override fun getCapabilities(): Capabilities {
        val result = super.getCapabilities()
        result.disableAll()

        result.enable(Capabilities.Capability.NO_CLASS)
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES)
        return result
    }

    override fun setDistanceFunction(df: DistanceFunction?) {
        m_DistanceFunction = df
    }
}

