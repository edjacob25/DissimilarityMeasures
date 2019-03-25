package weka.core

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.BayesNet
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.trees.RandomForest
import java.util.*
import kotlin.collections.HashMap

class LearningBasedDissimilarity : BaseCategoricalDistance() {
    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        trainClassifiers(insts)
    }

    private lateinit var weights: HashMap<Int, Double>
    private lateinit var similarityMatrices: MutableMap<Int, MutableMap<String, MutableMap<String, Double>>>

    private fun trainClassifiers(insts: Instances?) {
        val instances = Instances(insts)

        weights = HashMap()
        similarityMatrices = HashMap()
        for (attribute in instances.enumerateAttributes()) {
            val classifiers = initializeClassifiers()

            val stats = instances.attributeStats(attribute.index())

            if (attribute.numValues() > 50 || !stats.nominalCounts.all { it > 0 }) {
                weights[attribute.index()] = 0.0
                continue
            }
            instances.setClass(attribute)

            val results = mutableListOf<Pair<Array<DoubleArray>, Double>>()
            for (classifier in classifiers) {
                results.add(evaluateClassifier(instances, classifier))
            }
            val (confusion, auc) = results.maxBy { it.second }!!
            val similarity = calculateSimilarityMatrix(confusion)
            weights[attribute.index()] = auc
            val attributeIMap = mutableMapOf<String, MutableMap<String, Double>>()
            for (i in 0 until similarity.size){
                val attributeJMap = mutableMapOf<String, Double>()
                for (j in 0 until similarity.size){
                    attributeJMap[attribute.value(j)] = similarity[i][j]
                }
                attributeIMap[attribute.value(i)] = attributeJMap
            }
            similarityMatrices[attribute.index()] = attributeIMap

        }
    }

    private fun initializeClassifiers(): List<Classifier> {
        val classifiers = mutableListOf<Classifier>()
        classifiers.add(RandomForest())
        classifiers.add(NaiveBayes())
        classifiers.add(BayesNet())
        return classifiers
    }

    private fun evaluateClassifier(instances: Instances, classifier: Classifier): Pair<Array<DoubleArray>, Double> {
        val folds = createFolds(instances)
        val size = instances.numDistinctValues(instances.classAttribute())
        val confusionMatrix = Array(size) { DoubleArray(size) }
        var auc = 0.0
        for ((training, testing) in folds) {
            classifier.buildClassifier(training)
            val eval = Evaluation(instances)
            eval.evaluateModel(classifier, testing)
            auc += eval.weightedAreaUnderROC()
            val confusion = eval.confusionMatrix()
            for (i in 0 until size) {
                for (j in 0 until size) {
                    confusionMatrix[i][j] += confusion[i][j]
                }
            }
        }

        return Pair(confusionMatrix, auc)
    }

    private fun createFolds(insts: Instances?, folds: Int = 10): List<Pair<Instances, Instances>> {
        val seed = 1L
        val rand = Random(seed)   // create seeded number generator
        val randData = Instances(insts)   // create copy of original data
        randData.randomize(rand)         // randomize data with number generator
        randData.stratify(folds)
        val sets = mutableListOf<Pair<Instances, Instances>>()
        for (i in 0 until folds) {
            val training = randData.trainCV(folds, i, rand)
            val test = randData.testCV(folds, i)
            sets.add(Pair(training, test))
        }
        return sets

    }

    private fun calculateSimilarityMatrix(confusionMatrix: Array<DoubleArray>): Array<DoubleArray> {
        val size = confusionMatrix.size
        val result = mutableListOf<DoubleArray>()
        for (row in confusionMatrix) {
            val sum = row.sum()
            val newRow = DoubleArray(size)
            var i = 0
            for (value in row) {
                newRow[i] = row[i] / sum
                i += 1
            }
            result.add(newRow)
        }
        return result.toTypedArray()
    }

    fun computeMulticlassAUC(confusionMatrix: Array<DoubleArray>): Double {
        var sum = 0.0
        var count = 0
        for (i in 0 until confusionMatrix.size) {
            for (j in i + 1 until confusionMatrix.size) {
                val tp = confusionMatrix[i][i]
                val fp = confusionMatrix[j][i]
                val fn = confusionMatrix[i][j]
                val tn = confusionMatrix[j][j]
                sum += tp + fp + fn + tn
                count = 0
            }
        }
        return sum / count
    }

    override fun listOptions(): Enumeration<Option> {
        val result = super.listOptions().toList().toMutableList()
        result.add(Option("The classifier to be used", "C", 1, "-C"))
        return result.toEnumeration()
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        return weights[index]!! * similarityMatrices[index]!![val1]!![val2]!!
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }

    override fun globalInfo(): String {
        return "This is the learning based measure, designed for categorical data"
    }
}

fun <T> List<T>.toEnumeration(): Enumeration<T> {
    return object : Enumeration<T> {
        var count = 0

        override fun hasMoreElements(): Boolean {
            return this.count < size
        }

        override fun nextElement(): T {
            if (this.count < size) {
                return get(this.count++)
            }
            throw NoSuchElementException("List enumeration asked for more elements than present")
        }
    }
}