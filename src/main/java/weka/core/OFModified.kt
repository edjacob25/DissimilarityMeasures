package weka.core

import me.jacobrr.ModifiedCompanion
import me.jacobrr.ModifiedOption
import me.jacobrr.MultiplyOption
import me.jacobrr.toEnumeration
import java.util.*

class OFModified : OccurenceFrequency() {
    protected lateinit var modifiedCompanion: ModifiedCompanion
    private lateinit var weightStyle: String
    private lateinit var option: ModifiedOption
    private lateinit var multiplyOption: MultiplyOption

    override fun setInstances(insts: Instances?) {
        super.setInstances(insts)
        modifiedCompanion = ModifiedCompanion(instances, option, multiplyOption, weightStyle)
    }

    override fun difference(index: Int, val1: String, val2: String): Double {
        val baseDifference = super.difference(index, val1, val2)
        return modifiedCompanion.calculateDistance(baseDifference, index)
    }

    override fun listOptions(): Enumeration<Option> {
        // TODO: Fix options not appearing in the Weka UI
        val result = super.listOptions().toList().toMutableList()
        result.add(
            Option(
                "Which weight is going to be used. Options are K for kappa, A for Auc and N for a " +
                        "uniform weight. Defaults to N", "w", 1, "-w <weight>"
            )
        )
        result.add(
            Option(
                "Which option of kappa is going to be used. Options are B for Base, D for discard when the " +
                        "weight is low, M for mac when the weight is low and L to not multiply for the weight when is " +
                        "low. Defaults to B", "o", 1, "-o <option>"
            )
        )
        result.add(
            Option(
                "Which weight type is going to be used. Options are N for normal, I for 1 - weight. Defaults " +
                        "to N", "t", 1, "-t <weightType>"
            )
        )


        return result.toEnumeration()
    }

    override fun setOptions(options: Array<out String>?) {
        super.setOptions(options)
        val weight = Utils.getOption('w', options)
        if (weight.isNotEmpty()) {
            weightStyle = weight
        }
        val mOption = Utils.getOption('o', options)
        option = when (mOption) {
            "D" -> ModifiedOption.DISCARD_LOW
            "M" -> ModifiedOption.MAX_LOW
            "L" -> ModifiedOption.BASE_LOW
            else -> ModifiedOption.BASE
        }

        val type = Utils.getOption('t', options)
        multiplyOption = when (type) {
            "I" -> MultiplyOption.ONE_MINUS
            else -> MultiplyOption.NORMAL
        }
    }

    override fun getOptions(): Array<String> {
        val result = super.getOptions().toMutableList()
        result.add("-w")
        result.add(weightStyle)
        result.add("-o")
        result.add(option.s)
        result.add("-t")
        result.add(multiplyOption.s)
        return result.toTypedArray()
    }
}