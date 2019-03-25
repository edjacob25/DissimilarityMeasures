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
            val similarity = calculateSimilarityMatrix(confusion)
            weights[attribute.index()] = auc
            val attributeIMap = mutableMapOf<String, MutableMap<String, Double>>()
            for (i in 0 until similarity.size){
                val attributeJMap = mutableMapOf<String, Double>()
                print(attribute.value(i))
                for (j in 0 until similarity.size){
                    attributeJMap[attribute.value(j)] = 1 - similarity[i][j]
                    print("|${similarity[i][j]}|")
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
        if(lessThan1000instances){
            classifiers.add(IBk())
        }
        return classifiers
    }

    private fun evaluateClassifier(instances: Instances, classifier: Classifier): Triple<Array<DoubleArray>, Double, String> {
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

    private fun printConfusionMatrix(confusionMatrix: Array<DoubleArray>){
        for (i in 0 until confusionMatrix.size){
            for (j in 0 until confusionMatrix.size){
                print("|${confusionMatrix[i][j]}|")
            }
            print("\n")
        }
    }

    /**
     * Normalizes the [confusionMatrix] to a value between 0 and 1
     */
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
