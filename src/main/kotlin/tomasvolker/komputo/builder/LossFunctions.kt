package tomasvolker.komputo.builder

import org.tensorflow.op.Ops
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfNumber
import tomasvolker.komputo.dsl.constant
import tomasvolker.numeriko.core.dsl.I

fun meanSquareError(ops: Ops, output: TFOperand, target: TFOperand): TFOperand {
    return with(ops) {
        reduceMean(
            square(
                sub(
                    target.asOfNumber(), output.asOfNumber()
                )
            ),
            constant(I[0, 1]).asOfNumber()
        )
    }
}

fun crossEntropyWithLogits(ops: Ops, output: TFOperand, target: TFOperand): TFOperand {
    return with(ops) {
        reduceMean(
            softmaxCrossEntropyWithLogits(
                output.asOfNumber(),
                target.asOfNumber()
            ).loss(),
            constant(I[0]).asOfNumber()
        )
    }
}
