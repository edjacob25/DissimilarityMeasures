package tests

import weka.clusterers.CategoricalKMeans
import weka.core.*
import kotlin.test.assertEquals
import org.junit.Test as test

class TestSource() {
    @test fun basicTest() {
        assertEquals(true, true)
    }

    @test fun eskinTest() {
        val instances = createDataset()
        val measure = Eskin()
        measure.instances = instances

        val instance1 = DenseInstance(1.0, doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0))
        val instance2 = DenseInstance(1.0, doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0))
        instance1.setDataset(instances)
        instance2.setDataset(instances)

        val distance = measure.distance(instance1, instance2)
        assertEquals((1 - (4.0 / 6.0)) * 0.2, distance)
    }

    // This is the weather.nominal dataset, created in memory for the purpose of testing in a small, known dataset
    private fun createDataset(): Instances{
        val attributes = arrayListOf(
            Attribute("outlook", listOf("sunny", "overcast", "rainy")),
            Attribute("temperature", listOf("hot", "mild", "cool")),
            Attribute("humidity", listOf("high", "normal")),
            Attribute("windy", listOf("TRUE", "FALSE")),
            Attribute("play", listOf("yes", "no")))
        val instances = Instances("weather.nominal", attributes, 14)

        val instancesList = arrayListOf(
            DenseInstance(1.0, doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0)),
            DenseInstance(1.0, doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0)),
            DenseInstance(1.0, doubleArrayOf(1.0, 0.0, 0.0, 1.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(2.0, 1.0, 0.0, 1.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(2.0, 2.0, 1.0, 1.0, .0)),
            DenseInstance(1.0, doubleArrayOf(2.0, 2.0, 1.0, 0.0, 1.0)),
            DenseInstance(1.0, doubleArrayOf(1.0, 2.0, 1.0, 0.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(0.0, 1.0, 0.0, 1.0, 1.0)),
            DenseInstance(1.0, doubleArrayOf(0.0, 2.0, 1.0, 1.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(2.0, 1.0, 1.0, 1.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(0.0, 1.0, 1.0, 0.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(1.0, 1.0, 0.0, 0.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(1.0, 0.0, 1.0, 1.0, 0.0)),
            DenseInstance(1.0, doubleArrayOf(2.0, 1.0, 0.0, 0.0, 1.0)))
        instancesList.forEach { it.setDataset(instances) }
        instances.addAll(instancesList)

        return instances
    }
}