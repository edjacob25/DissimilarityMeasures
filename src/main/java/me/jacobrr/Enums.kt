package me.jacobrr

enum class ModifiedOption(val s: String) {
    BASE("B"),
    DISCARD_LOW("D"),
    MAX_LOW("M"),
    BASE_LOW("L")
}

enum class MultiplyOption(val s: String) {
    NORMAL("N"),
    ONE_MINUS("I"),
    NO_MULTIPLY("O")
}

enum class AUCOption(val s: String){
    NORMAL("N"),
    SECOND("S"),
    WEKA("W")
}