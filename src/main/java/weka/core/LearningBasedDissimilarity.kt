package weka.core

import me.jacobrr.toEnumeration
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.BayesNet
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.SimpleLogistic
import weka.classifiers.lazy.IBk
import weka.classifiers.meta.Bagging
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
    protected var strategy = "A"

    private fun trainClassifiers(insts: Instances?) {
        val instances = Instances(insts)

        weights = HashMap()
        similarityMatrices = HashMap()
        for (attribute in instances.enumerateAttributes()) {
            val isLessThan1000 = instances.numInstances() < 1000
            val classifiers = initializeClassifiers(isLessThan1000)

            val stats = instances.attributeStats(attribute.index())

            if (attribute.numValues() > 50 || !stats.nominalCounts.all { it > 0 }) {
                weights[attribute.index()] = 0.0
                continue
            }
            instances.setClass(attribute)

            val results = mutableListOf<Triple<Array<DoubleArray>, Double, String>>()
            for (classifier in classifiers) {
                results.add(evaluateClassifier(instances, classifier))
            }
            val (confusion, auc, name) = results.maxBy { it.second }!!
            println("The chosen classifier is $name")
            val similarity = normalizeMatrix(confusion)
            val fixedSimilarity = fixSimilarityMatrix(similarity)
            weights[attribute.index()] = auc
            val attributeIMap = mutableMapOf<String, MutableMap<String, Double>>()
            for (i in 0 until similarity.size) {
                val attributeJMap = mutableMapOf<String, Double>()
                print(attribute.value(i))
                for (j in 0 until similarity.size) {
                    attributeJMap[attribute.value(j)] = 1 - fixedSimilarity[i][j]
                    print("|${fixedSimilarity[i][j]}|")
                }
                print("\n")
                attributeIMap[attribute.value(i)] = attributeJMap
            }
            similarityMatrices[attribute.index()] = attributeIMap
            println("Attribute ${attribute.name()} has a weight $auc}")

        }
    }

    private fun initializeClassifiers(lessThan1000instances: Boolean = false): List<Classifier> {
        val classifiers = mutableListOf<Classifier>()
        classifiers.add(RandomForest())
        classifiers.add(NaiveBayes())
        classifiers.add(BayesNet())
        classifiers.add(Bagging())
        classifiers.add(SimpleLogistic())
        if (lessThan1000instances) {
            classifiers.add(IBk())
        }
        return classifiers
    }

    private fun evaluateClassifier(
        instances: Instances,
        classifier: Classifier
    ): Triple<Array<DoubleArray>, Double, String> {
        val folds = createFolds(instances)
        val size = instances.numDistinctValues(instances.classAttribute())
        val confusionMatrix = Array(size) { DoubleArray(size) }
        var auc = 0.0
        for ((training, testing) in folds) {
            classifier.buildClassifier(training)
            val eval = Evaluation(instances)
            eval.evaluateModel(classifier, testing)
            val confusion = eval.confusionMatrix()
            auc += computeMulticlassAUC(confusion)
            for (i in 0 until size) {
                for (j in 0 until size) {
                    confusionMatrix[i][j] += confusion[i][j]
                }
            }
        }
        auc /= folds.size

        return Triple(confusionMatrix, auc, classifier.javaClass.simpleName)
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

    private fun printConfusionMatrix(confusionMatrix: Array<DoubleArray>) {

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
                newRow[i] = row[i] / sum
                i += 1
            }
            result.add(newRow)
        }
        return result.toTypedArray()
    }

    private fun fixSimilarityMatrix(confusionMatrix: Array<DoubleArray>): Array<DoubleArray> {
        val size = confusionMatrix.size

        if (strategy == "B") {
            for (i in 0 until size) {
                confusionMatrix[i][i] += 2.0
            }
            return normalizeMatrix(confusionMatrix)
        }
        if (strategy == "C") {
            for (i in 0 until size) {
                confusionMatrix[i][i] = 1.0
            }
            return confusionMatrix

        }
        if (strategy == "C") {
            for (i in 0 until size) {
                confusionMatrix[i][i] = confusionMatrix[i][i] + 1
            }
            val result = normalizeMatrix(confusionMatrix)
            for (i in 0 until size) {
                result[i][i] = 1.0
            }
            return result
        }
        if (strategy == "C") {
            for (i in 0 until size) {
                confusionMatrix[i][i] = confusionMatrix[i][i] + 2
            }
            val result = normalizeMatrix(confusionMatrix)
            for (i in 0 until size) {
                result[i][i] = 1.0
            }
            return result
        }

        // Default

        for (i in 0 until size) {
            confusionMatrix[i][i] = confusionMatrix[i][i] + 1
        }

        return normalizeMatrix(confusionMatrix)
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

    override fun difference(index: Int, val1: String, val2: String): Double {
        if (weights[index] == 0.0)
            return 0.0
        return weights[index]!! * similarityMatrices[index]!![val1]!![val2]!!
    }

    override fun updateDistance(currDist: Double, diff: Double): Double {
        return currDist + diff
    }

    override fun globalInfo(): String {
        return "This is the learning based measure, designed for categorical data"
    }

    override fun listOptions(): Enumeration<Option> {
        val result = super.listOptions().toList().toMutableList()
        result.add(Option("The strategy to be used", "S", 1, "-S"))
        return result.toEnumeration()
    }

    override fun setOptions(options: Array<out String>?) {
        super.setOptions(options)
        val str = Utils.getOption('S', options)
        if (str.isNotEmpty()) {
            strategy = str
        }
    }

    override fun getOptions(): Array<String> {
        val result = super.getOptions().toMutableList()
        result.add("-S")
        result.add(strategy)
        return result.toTypedArray()
    }

}
