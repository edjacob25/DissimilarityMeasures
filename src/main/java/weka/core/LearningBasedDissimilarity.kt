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
    private lateinit var simmilarityMatrices: HashMap<Int, HashMap<String, HashMap<String, Double>>>

    fun trainClassifiers(insts: Instances?) {
        val instances = Instances(insts)


        weights = HashMap()
        simmilarityMatrices = HashMap(HashMap(HashMap()))
        for (attribute in instances.enumerateAttributes()) {
            val classifiers = initializeClassifiers()

            val stats = instances.attributeStats(attribute.index())

            if (attribute.numValues() > 50 || !stats.nominalCounts.all { it > 0 }) {
                weights[attribute.m_Index] = 0.0
                continue
            }

            instances.setClass(attribute)

            val results = mutableListOf<Pair<Array<DoubleArray>, Double>>()
            for (classifier in classifiers) {
                results.add(evaluateClassifier(instances, classifier))
            }
            val (confusion, auc) = results.maxBy { it.second }!!
            val simmilarity = calculateSimilarityMatrix(confusion)
            weights[attribute.m_Index] = auc

            //simmilarityMatrices[attribute.m_Index] = simmilarity[]

        }
    }

    fun initializeClassifiers(): List<Classifier> {
        val classifiers = mutableListOf<Classifier>()
        classifiers.add(1, RandomForest())
        classifiers.add(2, NaiveBayes())
        classifiers.add(3, BayesNet())
        return classifiers
    }

    fun evaluateClassifier(instances: Instances, classifier: Classifier): Pair<Array<DoubleArray>, Double> {
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
            for (i in 0..size) {
                for (j in 0..size) {
                    confusionMatrix[i][j] += confusion[i][j]
                }
            }
        }

        return Pair(confusionMatrix, auc)
    }

    fun createFolds(insts: Instances?, folds: Int = 10): List<Pair<Instances, Instances>> {
        val seed = 1L
        val rand = Random(seed)   // create seeded number generator
        val randData = Instances(insts)   // create copy of original data
        randData.randomize(rand)         // randomize data with number generator
        randData.stratify(folds)
        val sets = mutableListOf<Pair<Instances, Instances>>()
        for (i in 0..folds) {
            val training = randData.trainCV(folds, i, rand)
            val test = randData.testCV(folds, i)
            sets.add(Pair(training, test))
        }
        return sets

    }

    fun calculateSimilarityMatrix(confusionMatrix: Array<DoubleArray>): Array<DoubleArray> {
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
        for (i in 0..confusionMatrix.size) {
            for (j in i + 1..confusionMatrix.size) {
                val TP = confusionMatrix[i][i]
                val FP = confusionMatrix[j][i]
                val FN = confusionMatrix[i][j]
                val TN = confusionMatrix[j][j]
                sum += TP
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

    override fun getOptions(): Array<String> {
        return super.getOptions()
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        var sum = 0.0
        for (attribute in m_Data.m_Attributes) {
            sum += weights[index]!! * simmilarityMatrices[index]!![val1]!![val2]!!
        }
        return sum / m_Data.m_Attributes.size
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