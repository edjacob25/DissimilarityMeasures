package weka.core

import weka.clusterers.SimpleKMeans
import weka.core.*

class KMeans : SimpleKMeans() {
    override fun getOptions(): Array<String> {
        return super.getOptions()
    }

    override fun buildClusterer(data: Instances?) {
        super.buildClusterer(data)
    }

    override fun getCapabilities(): Capabilities {
        val result = super.getCapabilities()
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES)
        result.enable(Capabilities.Capability.NOMINAL_CLASS)
        return result
    }

    override fun clusterInstance(instance: Instance?): Int {
        return super.clusterInstance(instance)
    }

    override fun distributionForInstance(instance: Instance?): DoubleArray {
        return super.distributionForInstance(instance)
    }



}

