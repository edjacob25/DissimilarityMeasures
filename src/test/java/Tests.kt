package tests

import org.junit.Assert
import weka.clusterers.CategoricalKMeans
import weka.core.*
import weka.core.converters.ConverterUtils
import weka.filters.Filter
import weka.filters.unsupervised.attribute.AddCluster
import kotlin.test.assertEquals
import org.junit.Test as test

class TestSource {
    @test
    fun basicTest() {
        assertEquals(true, true)
    }

    @test
    fun eskinTest() {
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

    @test
    fun goodallTest() {
        val instances = createDataset()
        val measure = Goodall()
        measure.instances = instances

        val instance1 = DenseInstance(1.0, doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0))
        val instance2 = DenseInstance(1.0, doubleArrayOf(1.0, 1.0, 1.0, 1.0, 0.0))
        instance1.setDataset(instances)
        instance2.setDataset(instances)

        val distance = measure.distance(instance1, instance2)

        val prob1 = (6.0 * (6 - 1)) / (14 * (14 - 1))
        val prob2 = (8.0 * (8 - 1)) / (14 * (14 - 1))
        Assert.assertEquals(0.8 + ((prob1 + prob2) * 0.2), distance, 0.00001)
    }

    @test
    fun LearningSimmilarityTest() {
        val instances = loadDataset("F:\\Datasets\\CleanedDatasets2\\arrhythmia_cleaned.arff")
        val measure = LearningBasedDissimilarity()
        measure.options = arrayOf("-S", "A", "-w", "N")
        measure.instances = instances

        Assert.assertEquals(true, true)
    }

    @test
    fun CategoricalKMeansTest() {
        //java -Xmx8192m  -W 'weka.clusterers.CategoricalKMeans -init 1 -max-candidates 100 -periodic-pruning 10000
        // -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 16 -A "weka.core.LearningBasedDissimilarity -R first-last -S A -w N"
        // -I 500 -num-slots 4 -S 10 -i /mnt/f/Datasets/CleanedDatasets2/arrhythmia_cleaned.arff
        // -o /mnt/f/Datasets/CleanedDatasets2/arrhythmia_cleaned_clustered.arff -I Last
        val command = arrayOf(
            "-W",
            "weka.clusterers.CategoricalKMeans -init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 " +
                    "-t1 -1.25 -t2 -1.0 -N 16 -M -A \"weka.core.LearningBasedDissimilarity -w A -o D -t I -R first-last\" " +
                    "-I 500 -num-slots 4 -S 10",
            "-i",
            "F:\\Datasets\\Categorical\\balloons.arff",
            "-o",
            "F:\\Datasets\\Categorical\\balloons_clustered.arff",
            "-I",
            "Last"
        )
        Filter.runFilter(AddCluster(), command)
    }

    @test
    fun CleanTest() {
        val instances = loadDataset("F:\\Datasets\\Categorical\\balloons.arff")
        val measure = Clean()
        measure.instances = instances


        val kmeans = CategoricalKMeans()
        kmeans.distanceFunction = measure
        kmeans.buildClusterer(instances)
        val instance = instances[0]
        kmeans.clusterInstance(instance)

        Assert.assertEquals(true, true)
    }
}

// This is the weather.nominal dataset, created in memory for the purpose of testing in a small, known dataset
fun createDataset(): Instances {
    val attributes = arrayListOf(
        Attribute("outlook", listOf("sunny", "overcast", "rainy")),
        Attribute("temperature", listOf("hot", "mild", "cool")),
        Attribute("humidity", listOf("high", "normal")),
        Attribute("windy", listOf("TRUE", "FALSE")),
        Attribute("play", listOf("yes", "no"))
    )
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
        DenseInstance(1.0, doubleArrayOf(2.0, 1.0, 0.0, 0.0, 1.0))
    )
    instancesList.forEach { it.setDataset(instances) }
    instances.addAll(instancesList)

    return instances
}

private fun loadDataset(route: String): Instances {
    val datasource = ConverterUtils.DataSource(route)
    return datasource.dataSet
}