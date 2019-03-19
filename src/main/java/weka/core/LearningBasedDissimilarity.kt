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

    lateinit var weights : HashMap<Int, Double>
    lateinit var simmilarityMatrices: HashMap<Int, HashMap<String, HashMap<String, Double>>>

    fun trainClassifiers(insts: Instances?){
        weights = HashMap()
        simmilarityMatrices = HashMap(HashMap(HashMap()))
        for (attribute in instances.enumerateAttributes()) {
            val classifiers = initializeClassifiers()

            for (classifier in classifiers){
                classifier.buildClassifier(m_Data)
                val eval = Evaluation(insts)
                eval.evaluateModel(classifier, insts)
                val confusion = eval.confusionMatrix()
                val simmilarity = calculateSimilarityMatrix(confusion)
            }

            weights[attribute.m_Index] = 1.0 / instances.m_Attributes.size
            // TODO: How to set this
            //simmilarityMatrices[attribute.m_Index][][] = simmilarity[]

        }
    }

    fun initializeClassifiers(): List<Classifier>{
        val classifiers = mutableListOf<Classifier>()
        classifiers.add(1, RandomForest())
        classifiers.add(2, NaiveBayes())
        classifiers.add(3, BayesNet())
        return classifiers
    }

    fun calculateSimilarityMatrix(confusionMatrix: Array<DoubleArray>): Array<DoubleArray>{
        val size = confusionMatrix.size
        val result = mutableListOf<DoubleArray>()
        for (row in confusionMatrix) {
            val sum = row.sum()
            val newRow = DoubleArray(size)
            var i = 0
            for (value in row) {
                newRow[i] = row[i]/sum
                i += 1
            }
            result.add(newRow)
        }
        return result.toTypedArray()
    }

    override fun listOptions(): Enumeration<Option> {
        val result = super.listOptions().toList().toMutableList()
        result.add(Option("The classifier to be used","C",1,"-C"))
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