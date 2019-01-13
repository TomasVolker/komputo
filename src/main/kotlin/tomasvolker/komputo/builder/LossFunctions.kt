package tomasvolker.komputo.builder

import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.dsl.shape
import tomasvolker.numeriko.core.dsl.I

interface LossFunction {

    fun buildOperations(builder: ModelBuilder, output: TFOperand, target: TFOperand): TFOperand

}

fun loss(function: ModelBuilder.(output: TFOperand, target: TFOperand)->TFOperand) =
    object : LossFunction {
        override fun buildOperations(builder: ModelBuilder, output: TFOperand, target: TFOperand) =
                builder.function(output, target)
    }

val meanSquareError = loss { output, target ->
    reduceMean(square(target - output), I[0, 1])
}

val crossEntropyWithLogits = loss { output, target ->
    reduceMean(softmaxCrossEntropyWithLogits(output, target), I[0])
}

val meanAbsoluteError = loss { output, target -> reduceMean(abs(target - output), I[0, 1]) }
