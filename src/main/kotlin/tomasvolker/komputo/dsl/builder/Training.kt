package tomasvolker.komputo.dsl.builder

import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.tensorflow.dsl.group
import tomasvolker.tensorflow.dsl.name
import tomasvolker.tensorflow.dsl.shape
import tomasvolker.tensorflow.dsl.toShape

fun <T> ModelBuilder.meanSquareError(output: Operand<T>, target: Operand<T>): Operand<T> =
    ops.reduceMean(square(target - output), constant(I[0, 1]))

interface TrainingAlgorithm {

    fun buildOperation(
        builder: TrainableModelBuilder,
        loss: Operand<Float>,
        variableList: List<Variable<Float>>
    ): Operand<*>

}

abstract class GradientAlgorithm: TrainingAlgorithm {

    override fun buildOperation(
        builder: TrainableModelBuilder,
        loss: Operand<Float>,
        variableList: List<Variable<Float>>
    ): Operand<*> {

        with(builder) {

            var result: List<Operand<*>>? = null

            scope("gradient_descent") {

                val grad = gradients(loss, variableList, name = "gradient")

                result = buildUpdateOperations(builder, variableList, grad)

            }

            return ops.withName("Train").group(result ?: error(""))

        }

    }

    abstract fun buildUpdateOperations(
        builder: TrainableModelBuilder,
        variableList: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<*>>

}

class GradientDescent(val rate: Double): GradientAlgorithm() {

    override fun buildUpdateOperations(
        builder: TrainableModelBuilder,
        variableList: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<*>> {
        with(builder) {
            val rate = ops.constant(rate.toFloat())

            return variableList.mapIndexed { i, variable ->
                ops.applyGradientDescent(variable, rate, gradients.dy(i))
            }
        }
    }

}

class Momentum(
    val rate: Double,
    val momentum: Double = 0.9
): GradientAlgorithm() {

    override fun buildUpdateOperations(
        builder: TrainableModelBuilder,
        variableList: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<*>> {
        with(builder) {
            val rate = ops.constant(rate.toFloat())
            val momentum = ops.constant(momentum.toFloat())

            return variableList.mapIndexed { i, variable ->

                val variableShape = variable.shape.toIntArray1D()

                val accumulator = variable(
                    "${variable.name.substringAfter('/')}_acc",
                    ops.broadcastTo(constant(0f), constant(variableShape)),
                    shape = variableShape,
                    trainable = false
                )

                ops.applyMomentum(variable, accumulator, rate, gradients.dy(i), momentum)
            }
        }
    }


}

class Adagrad(
    val rate: Double = 0.01
): GradientAlgorithm() {

    override fun buildUpdateOperations(
        builder: TrainableModelBuilder,
        variableList: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<*>> {
        with(builder) {
            val rate = ops.constant(rate.toFloat())

            return variableList.mapIndexed { i, variable ->

                val variableShape = variable.shape.toIntArray1D()

                val accumulator = variable(
                    "${variable.name.substringAfter('/')}_acc",
                    ops.broadcastTo(constant(1e-8f), constant(variableShape)),
                    shape = variableShape,
                    trainable = false
                )

                ops.applyAdagrad(variable, accumulator, rate, gradients.dy(i))
            }
        }
    }


}