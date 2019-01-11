package tomasvolker.komputo.dsl.builder

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.Last
import tomasvolker.numeriko.core.index.rangeTo
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.factory.toIntArray1D
import tomasvolker.numeriko.core.operations.concatenate
import tomasvolker.numeriko.core.operations.reduction.product
import tomasvolker.tensorflow.dsl.*

class SequentialBuilder(
    val builder: ModelBuilder,
    input: Operand<*>
) {

    val ops: Ops get() = builder.ops

    private var _lastOutput: Operand<*> = input
    val lastOutput: Operand<*>
        get() = _lastOutput

    val lastShape: IntArray1D get() = lastOutput.shape.toIntArray1D()

    fun dense(outputSize: Int, activation: (Operand<Float>)-> Operand<Float> = ops::identity): Operand<*> {

        if (lastShape.rank > 2) error("Dense layer cannot be applied to shape $lastShape")

        _lastOutput = builder.dense(lastOutput as Operand<Float>, lastShape.last(), outputSize, activation)
        return lastOutput
    }

    fun conv2d(
        kernelSize: IntArray1D,
        strides: IntArray1D = I[1, 1],
        filterCount: Int = 1,
        padding: ConvPadding = ConvPadding.SAME,
        activation: (Operand<Float>)-> Operand<Float> = ops::identity
    ): Operand<*> {

        require(lastShape.size == 4) {
            "invalid shape for conv2D: $lastShape"
        }

        _lastOutput = builder.conv2D(
            input = lastOutput as Operand<Float>,
            kernelSize = kernelSize,
            filterCount = filterCount,
            stride = strides,
            padding = padding,
            activation = activation
        )
        return lastOutput
    }

    fun reshape(vararg sizes: Int): Operand<*> =
        reshape(sizes.toIntArray1D())

    fun reshape(shape: IntArray1D): Operand<*> {
        _lastOutput = ops.reshape(lastOutput, I[-1] concatenate shape)
        return lastOutput
    }

    fun flatten(): Operand<*> = reshape(lastShape[1..Last].product())

    fun maxPool2D(windowSize: IntArray1D, strides: IntArray1D = I[1, 1]): Operand<*> {
        _lastOutput = ops.maxPool(
            lastOutput,
            I[1, windowSize[0], windowSize[1], 1].toList().map { it.toLong() },
            I[1, strides[0], strides[1], 1].toList().map { it.toLong() },
            "SAME"
        )
        return lastOutput
    }


}
