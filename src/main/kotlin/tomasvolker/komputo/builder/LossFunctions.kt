package tomasvolker.komputo.builder

import tomasvolker.komputo.TFOperand
import tomasvolker.numeriko.core.dsl.I

interface Metric {

    fun buildOperations(builder: ModelBuilder, output: TFOperand, target: TFOperand): TFOperand

}

fun metric(function: ModelBuilder.(output: TFOperand, target: TFOperand)->TFOperand) =
    object : Metric {
        override fun buildOperations(builder: ModelBuilder, output: TFOperand, target: TFOperand) =
                builder.function(output, target)
    }

val meanSquareError = metric { output, target ->
    reduceMean(square(target - output), I[0, 1])
}

val crossEntropyWithLogits = metric { output, target ->
    reduceMean(softmaxCrossEntropyWithLogits(output, target), I[0])
}

val meanAbsoluteError = metric { output, target -> reduceMean(abs(target - output), I[0, 1]) }
