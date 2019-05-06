package tests

import org.junit.Assert
import org.junit.Test
import weka.core.Lin
import weka.core.LinModified
import weka.core.LinModified2
import weka.core.LinModified3

class LinTests {
    @Test
    fun LinTest() {
        val instances = createDataset()
        val measure = Lin()
        measure.instances = instances

        for (instance in instances) {
            for (instance2 in instances) {
                val distance = measure.distance(instance, instance2)
                val geq0 = distance >= 0
                println("Distance between $instance and $instance2 = $distance")
                Assert.assertTrue(geq0)
            }
            println("--------------")
        }
    }

    @Test
    fun LinModifiedTest() {
        val instances = createDataset()
        val measure = LinModified()
        measure.instances = instances

        for (instance in instances) {
            for (instance2 in instances) {
                val distance = measure.distance(instance, instance2)
                val geq0 = distance >= 0
                println("Distance between $instance and $instance2 = $distance")
                Assert.assertTrue(geq0)
            }
            println("--------------")
        }
    }

    @Test
    fun LinModified2Test() {
        val instances = createDataset()
        val measure = LinModified2()
        measure.instances = instances

        for (instance in instances) {
            for (instance2 in instances) {
                val distance = measure.distance(instance, instance2)
                val geq0 = distance >= 0
                println("Distance between $instance and $instance2 = $distance")
                Assert.assertTrue(geq0)
            }
            println("--------------")
        }
    }

    @Test
    fun LinModified3Test() {
        val instances = createDataset()
        val measure = LinModified3()
        measure.instances = instances

        for (instance in instances) {
            for (instance2 in instances) {
                val distance = measure.distance(instance, instance2)
                val geq0 = distance >= 0
                println("Distance between $instance and $instance2 = $distance")
                Assert.assertTrue(geq0)
            }
            println("--------------")
        }
    }

}