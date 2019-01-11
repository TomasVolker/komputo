package tomasvolker.komputo.dsl

import org.tensorflow.Operand
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.asOfNumber
import tomasvolker.komputo.dsl.builder.ModelBuilder
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.operations.reduction.product
import kotlin.math.sqrt

fun ModelBuilder.residual(
    input: Operand<*>,
    inputSize: Int,
    activation: (linearOutput: Operand<*>) -> Operand<*>
): Operand<*> = dense(
    input = input,
    inputSize = inputSize,
    outputSize = inputSize,
    activation = activation
) + input

fun ModelBuilder.dense(
    input: Operand<*>,
    inputSize: Int,
    outputSize: Int,
    activation: (linearOutput: Operand<*>)-> Operand<*> = ::identity
): Operand<*> {

    var result: Operand<*>? = null

    scope("dense_layer") {

        val weightsShape = I[inputSize, outputSize]
        val biasesShape = I[1, outputSize]

        // Xavier
        val elementInputSize = inputSize

        val w = variable(
            name = "weightMatrix",
            shape = weightsShape,
            initialValue = randomNormal(weightsShape, deviation = sqrt(1.0 / elementInputSize))
        )

        val b = variable(
            name = "biasVector",
            shape = biasesShape,
            initialValue = randomNormal(biasesShape, deviation = sqrt(1.0 / elementInputSize))
        )

        result = activation((input matmul w) + b)

    }

    return result ?: error("")
}

enum class ConvPadding {
    SAME,
    VALID
}

fun ModelBuilder.conv2D(
    input: Operand<*>,
    kernelSize: IntArray1D,
    filterCount: Int = 1,
    stride: IntArray1D = I[1, 1],
    padding: ConvPadding = ConvPadding.SAME,
    activation: (linearOutput: Operand<*>)-> Operand<*> = ::identity
): Operand<*> {

    var result: Operand<*>? = null

    val inputShape = input.shape

    scope("conv2d_layer") {

        val filterShape = I[kernelSize[0], kernelSize[1], inputShape.last(), filterCount]

        // Xavier
        val elementInputSize = filterShape[0..2].product()

        val filter = variable(
            name = "filter",
            shape = filterShape,
            initialValue = randomNormal(filterShape, deviation = sqrt(1.0 / elementInputSize)),
            trainable = true
        )

        result = activation(
            ops.conv2D(
                input.asOfNumber(),
                filter.asOfNumber(),
                I[1, stride[0], stride[1], 1].toList().map { it.toLong() },
                when(padding) {
                    ConvPadding.SAME -> "SAME"
                    ConvPadding.VALID -> "VALID"
                }
            )
        )

    }

    return result ?: error("")
}
