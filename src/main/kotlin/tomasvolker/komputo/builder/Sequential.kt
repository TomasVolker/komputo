package tomasvolker.komputo.builder

import org.tensorflow.op.Ops
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.dsl.*
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.Last
import tomasvolker.numeriko.core.index.rangeTo
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.factory.toIntArray1D
import tomasvolker.numeriko.core.operations.concatenate
import tomasvolker.numeriko.core.operations.reduction.product

class SequentialBuilder(
    val builder: ModelBuilder,
    val input: TFOperand
) {

    val ops: Ops get() = builder.ops

    private var _lastOutput: TFOperand = input

    val output: TFOperand
        get() = _lastOutput

    val lastShape: IntArray1D get() = output.shape

    fun dense(
        outputSize: Int,
        initializer: WeightInitializer = Xavier,
        regularization: ((TFOperand)->TFOperand)? = null,
        activation: ((TFOperand)-> TFOperand)? = null
    ): TFOperand {

        if (lastShape.rank > 2) error("Dense layer cannot be applied to shape $lastShape")

        _lastOutput = builder.dense(
            output,
            outputSize,
            initializer,
            regularization,
            activation
        )
        return output
    }

    fun dropout(
        keepProbability: Double
    ): TFOperand {
        _lastOutput = builder.dropout(
            output,
            keepProbability
        )
        return output
    }

    fun conv2d(
        kernelSize: IntArray1D,
        strides: IntArray1D = I[1, 1],
        filterCount: Int = 1,
        padding: ConvPadding = ConvPadding.SAME,
        activation: (TFOperand)-> TFOperand = builder::identity
    ): TFOperand {

        require(lastShape.size == 4) {
            "invalid shape for conv2D: $lastShape"
        }

        _lastOutput = builder.conv2D(
            input = output,
            kernelSize = kernelSize,
            filterCount = filterCount,
            stride = strides,
            padding = padding,
            activation = activation
        )
        return output
    }

    fun reshape(vararg sizes: Int): TFOperand =
        reshape(sizes.toIntArray1D())

    fun reshape(shape: IntArray1D): TFOperand {
        _lastOutput = ops.reshape(output, I[-1] concatenate shape)
        return output
    }

    fun flatten(): TFOperand = reshape(lastShape[1..Last].product())

    fun maxPool2D(windowSize: IntArray1D, strides: IntArray1D = I[1, 1]): TFOperand {
        _lastOutput = ops.maxPool(
            output,
            I[1, windowSize[0], windowSize[1], 1].toList().map { it.toLong() },
            I[1, strides[0], strides[1], 1].toList().map { it.toLong() },
            "SAME"
        )
        return output
    }


}
