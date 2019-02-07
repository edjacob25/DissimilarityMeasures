import weka.core.Instances

class FrequencyCompanion constructor(instances: Instances) {
    private val freqs : HashMap<String, HashMap<String, Int>> = createStats(instances)

    private fun createStats(instances: Instances): HashMap<String, HashMap<String, Int>> {
        val map = HashMap<String, HashMap<String, Int>>()

        for (attribute in instances.enumerateAttributes())
        {
            val name = attribute.name()
            val attributesCount = map.getOrDefault(name, HashMap())
            for (instance in instances) {
                val value = instance.stringValue(attribute)
                var count = attributesCount.getOrDefault(value, 0)
                count += 1
                attributesCount[value] = count
            }
            map[name] = attributesCount
        }
        return map
    }

    fun getFrequency(attributeName: String, value: String): Int {
        val attribute  = freqs[attributeName]
        return attribute?.getOrDefault(value, 0) ?: 0
    }
}