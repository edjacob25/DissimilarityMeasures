package weka.core

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.BayesNet
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.SimpleLogistic
import weka.classifiers.lazy.IBk
import weka.classifiers.lazy.KStar
import weka.classifiers.meta.Bagging
import weka.classifiers.trees.RandomForest
import java.util.*
import kotlin.collections.HashMap

open class Clean : BaseCategoricalDistance() {

    private lateinit var weights: MutableMap<Int, Double>
    private lateinit var dissimilarityMatrices: MutableMap<Int, MutableMap<String, MutableMap<String, Double>>>

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        trainClassifiers(instances)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        if (weights[index] == 0.0)
            return 0.0

        return (1 - weights[index]!!) * dissimilarityMatrices[index]!![val1]!![val2]!!
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }

    override fun globalInfo(): String {
        return "This is the learning based measure, designed for categorical data"
    }

    private fun trainClassifiers(insts: Instances?) {
        val instances = Instances(insts)
        weights = HashMap()
        dissimilarityMatrices = HashMap()
        for (attribute in instances.enumerateAttributes()) {
            val isLessThan1000 = instances.numInstances() < 1000
            val classifiers = initializeClassifiers(isLessThan1000)

            val stats = instances.attributeStats(attribute.index())

            if (attribute.numValues() > 50 || !stats.nominalCounts.all { it > 0 } || stats.missingCount > stats.totalCount / 2) {
                weights[attribute.index()] = 0.0
                println("Attribute ${attribute.name()} has a weight 0, meaning it won't be taken on account when calculating the dissimilarity")
                continue
            }
            instances.setClass(attribute)

            val results = mutableListOf<ClassifierResult>()
            for (classifier in classifiers) {
                results.add(evaluateClassifier(instances, classifier))
            }

            val (confusion, auc, kappa, name) = results.maxBy { it.auc }!!
            println("The chosen classifier is $name")
            val size = confusion.size
            val simmilarity = normalizeMatrix(confusion)
            printMatrix(confusion, "confusion")
            printMatrix(simmilarity, "simmilarity")
            for (i in 0 until size) {
                simmilarity[i][i] = 1.0
            }
            printMatrix(simmilarity, "simmilarity when diagonal set to 1")
            weights[attribute.index()] = kappa

            var dissimilarity = Array(size) {DoubleArray(size)}
            for (i in 0 until size) {
                for (j in 0 until size) {
                    dissimilarity[i][j] = 1 - simmilarity[i][j]
                }
            }
            printMatrix(dissimilarity, "dissimilarity")
            dissimilarity = normalizeMatrix(dissimilarity)
            printMatrix(dissimilarity, "dissimilarity normalized")
            val attributeIMap = mutableMapOf<String, MutableMap<String, Double>>()
            for (i in 0 until dissimilarity.size) {
                val attributeJMap = mutableMapOf<String, Double>()
                print(attribute.value(i))
                for (j in 0 until dissimilarity.size) {
                    attributeJMap[attribute.value(j)] = dissimilarity[i][j]
                    print("|${dissimilarity[i][j]}|")
                }
                print("\n")
                attributeIMap[attribute.value(i)] = attributeJMap
            }
            dissimilarityMatrices[attribute.index()] = attributeIMap
            println("Attribute ${attribute.name()} has a weight $kappa")
        }
    }

    // TODO: Add code to detect SVM from packages and flag to activate it
    private fun initializeClassifiers(lessThan1000instances: Boolean = false): List<Classifier> {
        val classifiers = mutableListOf<Classifier>()
        classifiers.add(RandomForest())
        classifiers.add(NaiveBayes())
        classifiers.add(BayesNet())
        classifiers.add(Bagging())
        classifiers.add(SimpleLogistic())
        // TODO: Add an option to not use KStar as is very heavy
        classifiers.add(KStar())
        if (lessThan1000instances) {
            classifiers.add(IBk())
        }
        return classifiers
    }

    private fun evaluateClassifier(instances: Instances, classifier: Classifier): ClassifierResult {
        val folds = createFolds(instances)
        val size = instances.numDistinctValues(instances.classAttribute())
        val confusionMatrix = Array(size) { DoubleArray(size) }
        var auc = 0.0
        var kappa = 0.0
        for ((training, testing) in folds) {
            classifier.buildClassifier(training)
            val eval = Evaluation(instances)
            eval.evaluateModel(classifier, testing)
            val confusion = eval.confusionMatrix()
            auc += computeMulticlassAUC(confusion)
            kappa += eval.kappa()
            for (i in 0 until size) {
                for (j in 0 until size) {
                    confusionMatrix[i][j] += confusion[i][j]
                }
            }
        }
        auc /= folds.size
        kappa /= folds.size
        return ClassifierResult(confusionMatrix, auc, kappa, classifier.javaClass.simpleName)
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

    private fun printMatrix(confusionMatrix: Array<DoubleArray>, title: String = "") {
        println(title)
        for (i in 0 until confusionMatrix.size) {
            for (j in 0 until confusionMatrix.size) {
                print("|${confusionMatrix[i][j]}|")
            }
            print("\n")
        }
    }

    /**
     * Normalizes the [matrix] to a value between 0 and 1
     */
    private fun normalizeMatrix(matrix: Array<DoubleArray>): Array<DoubleArray> {
        val size = matrix.size
        val result = mutableListOf<DoubleArray>()
        for (row in matrix) {
            val sum = row.sum()
            val newRow = DoubleArray(size)
            var i = 0
            for (value in row) {
                if (sum > 0.0){
                    newRow[i] = row[i] / sum
                } else {
                    newRow[i] = 0.0
                }

                i += 1
            }
            result.add(newRow)
        }
        return result.toTypedArray()
    }

    private fun computeMulticlassAUC(confusionMatrix: Array<DoubleArray>): Double {
        var sum = 0.0
        var count = 0
        for (i in 0 until confusionMatrix.size) {
            for (j in i + 1 until confusionMatrix.size) {
                val tp = confusionMatrix[i][i]
                val fp = confusionMatrix[j][i]
                val fn = confusionMatrix[i][j]
                val tn = confusionMatrix[j][j]
                val positives = tp + fn
                val negatives = tn + fp
                val tprate = if (positives > 0.0) tp / positives else 1.0
                val fprate = if (negatives > 0.0) tn / negatives else 1.0
                sum += (tprate + fprate) / 2.0
                count += 1
            }
        }
        return sum / count
    }
}

private data class ClassifierResult(
    val confusionMatrix: Array<DoubleArray>,
    val auc: Double,
    val kappa: Double,
    val name: String
)