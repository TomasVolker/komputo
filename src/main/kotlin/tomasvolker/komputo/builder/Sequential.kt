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

    fun layer(function: ModelBuilder.(TFOperand)->TFOperand): TFOperand {
        _lastOutput = builder.function(output)
        return output
    }

    fun residual(function: SequentialBuilder.()->Unit): TFOperand =
        layer { sequential(it, function) + it }


    fun dense(
        outputSize: Int,
        activation: Activation? = null,
        initializer: WeightInitializer = Xavier,
        regularization: ((Ops, TFOperand)->TFOperand)? = null
    ): TFOperand {

        if (lastShape.rank > 2) error("Dense layer cannot be applied to shape $lastShape")

        return layer {
            dense(
                it,
                outputSize,
                activation,
                initializer,
                regularization
            )
        }
    }

    fun dropout(
        keepProbability: Double
    ): TFOperand = layer { dropout(it, keepProbability) }

    fun conv2d(
        kernelSize: IntArray1D,
        strides: IntArray1D = I[1, 1],
        filterCount: Int = 1,
        padding: ConvPadding = ConvPadding.SAME,
        activation: Activation = IDENTITY
    ): TFOperand {

        when(lastShape.size) {
            3 -> reshape(lastShape[1..Last] concatenate 1)
            4 -> {}
            else -> "invalid shape for conv2D: $lastShape"
        }

        return layer {
            conv2D(
                input = it,
                kernelSize = kernelSize,
                filterCount = filterCount,
                stride = strides,
                padding = padding,
                activation = activation
            )
        }
    }

    fun reshape(vararg sizes: Int): TFOperand =
        reshape(sizes.toIntArray1D())

    fun reshape(shape: IntArray1D): TFOperand =
        layer { ops.reshape(it, I[-1] concatenate shape) }

    fun flatten(): TFOperand = reshape(lastShape[1..Last].product())

    fun maxPool2D(windowSize: IntArray1D, strides: IntArray1D = I[1, 1]): TFOperand {
        return layer {
            ops.maxPool(
                it,
                I[1, windowSize[0], windowSize[1], 1].toList().map { it.toLong() },
                I[1, strides[0], strides[1], 1].toList().map { it.toLong() },
                "SAME"
            )
        }
    }


}
