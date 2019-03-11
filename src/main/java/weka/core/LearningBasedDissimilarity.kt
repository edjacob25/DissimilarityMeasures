package weka.core

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
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
        /*val params = Utils.splitOptions("-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1")
        val randomForest = Utils.forName(Classifier::class.java,"weka.classifiers.trees", params) as Classifier
        randomForest.buildClassifier(m_Data)
        randomForest.distributionForInstance()*/
        weights = HashMap()
        simmilarityMatrices = HashMap(HashMap(HashMap()))
        for (attribute in instances.enumerateAttributes()) {
            val randomForest = RandomForest()
            randomForest.buildClassifier(m_Data)
            val eval = Evaluation(insts)
            eval.evaluateModel(randomForest, insts)
            val confusion = eval.confusionMatrix()
            val simmilarity = calculateSimilarityMatrix(confusion)
            weights[attribute.m_Index] = 1.0
            // TODO: How to set this
            //simmilarityMatrices[attribute.m_Index][][] = simmilarity[]

        }
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