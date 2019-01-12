package tomasvolker.komputo.dsl

import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfNumber
import tomasvolker.komputo.builder.ModelBuilder
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.Last
import tomasvolker.numeriko.core.index.rangeTo
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.operations.reduction.product
import kotlin.math.sqrt

fun ModelBuilder.residual(
    input: TFOperand,
    activation: (linearOutput: TFOperand) -> TFOperand
): TFOperand = dense(
    input = input,
    outputSize = input.shape.last(),
    activation = activation
) + input


interface WeightInitializer {

    fun initializeValue(
        builder: ModelBuilder,
        shape: IntArray1D,
        inputSize: Int,
        outputSize: Int
    ): TFOperand

}

object Xavier: WeightInitializer {

    override fun initializeValue(
        builder: ModelBuilder,
        shape: IntArray1D,
        inputSize: Int,
        outputSize: Int
    ): TFOperand {
        return builder.randomNormal(shape, deviation = sqrt(1.0 / inputSize))
    }

}

fun ModelBuilder.dense(
    input: TFOperand,
    outputSize: Int,
    initializer: WeightInitializer = Xavier,
    regularization: ((TFOperand)->TFOperand)? = null,
    activation: ((TFOperand)-> TFOperand)? = null
): TFOperand {

    var result: TFOperand? = null

    val inputSize = input.shape[Last]
    
    scope("dense_layer") {

        val weightsShape = I[inputSize, outputSize]
        val biasesShape = I[1, outputSize]
        
        val w = parameter(
            name = "weightMatrix",
            shape = weightsShape,
            initialValue = initializer.initializeValue(this, weightsShape, inputSize, outputSize)
        )

        regularization?.let {
            regularize(regularization(w))
        }

        val b = parameter(
            name = "biasVector",
            shape = biasesShape,
            initialValue = initializer.initializeValue(this, biasesShape, inputSize, outputSize)
        )

        regularization?.let {
            regularize(regularization(b))
        }

        val linearOutput = (input matmul w) + b

        result = activation?.invoke(linearOutput) ?: linearOutput

    }

    return result ?: error("")
}


fun ModelBuilder.dropout(
    input: TFOperand,
    keepProbability: Double
): TFOperand {

    require(keepProbability in 0.0..1.0) {
        "keepProbability must be between 0 and 1 ($keepProbability)"
    }

    val probability = trainingFactor * (keepProbability - 1.0) + 1.0 // When training

    val mask = floor(randomUniform(input.shape[1..Last]) + probability)

    return input * mask / probability
}

enum class ConvPadding {
    SAME,
    VALID
}

fun ModelBuilder.conv2D(
    input: TFOperand,
    kernelSize: IntArray1D,
    filterCount: Int = 1,
    stride: IntArray1D = I[1, 1],
    padding: ConvPadding = ConvPadding.SAME,
    initializer: WeightInitializer = Xavier,
    activation: (linearOutput: TFOperand)-> TFOperand = ::identity
): TFOperand {

    var result: TFOperand? = null

    val inputShape = input.shape

    scope("conv2d_layer") {

        val filterShape = I[kernelSize[0], kernelSize[1], inputShape.last(), filterCount]

        // Xavier
        val elementInputSize = filterShape[0..2].product()

        val filter = parameter(
            name = "filter",
            shape = filterShape,
            initialValue = initializer.initializeValue(this, filterShape, elementInputSize, elementInputSize)
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
