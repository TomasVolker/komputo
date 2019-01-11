package tomasvolker.komputo.dsl.builder

import org.tensorflow.Operand
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.asOfAny
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.komputo.dsl.group
import tomasvolker.komputo.dsl.localName
import tomasvolker.komputo.dsl.name
import tomasvolker.komputo.dsl.shape

fun ModelBuilder.meanSquareError(output: Operand<*>, target: Operand<*>): Operand<*> =
    reduceMean(square(target - output), constant(I[0, 1]))

interface TrainingAlgorithm {

    fun buildOperation(
        builder: TrainableModelBuilder,
        loss: Operand<*>,
        variableList: List<Variable<*>>
    ): Operand<*>

}

abstract class GradientAlgorithm: TrainingAlgorithm {

    override fun buildOperation(
        builder: TrainableModelBuilder,
        loss: Operand<*>,
        variableList: List<Variable<*>>
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
        variableList: List<Variable<*>>,
        gradients: Gradients
    ): List<Operand<*>>

}

class GradientDescent(val rate: Double): GradientAlgorithm() {

    override fun buildUpdateOperations(
        builder: TrainableModelBuilder,
        variableList: List<Variable<*>>,
        gradients: Gradients
    ): List<Operand<*>> {
        with(builder) {
            val rate = ops.constant(rate.toFloat())

            return variableList.mapIndexed { i, variable ->
                ops.applyGradientDescent(
                    variable.asOfAny(),
                    rate.asOfAny(),
                    gradients.dy(i)
                )
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
        variableList: List<Variable<*>>,
        gradients: Gradients
    ): List<Operand<*>> {
        with(builder) {
            val rate = ops.constant(rate.toFloat())
            val momentum = ops.constant(momentum.toFloat())

            return variableList.mapIndexed { i, variable ->

                val variableShape = variable.shape

                val accumulator = variable(
                    "${variable.localName}_accumulator",
                    broadcastTo(constant(0.0), constant(variableShape)),
                    shape = variableShape,
                    trainable = false
                )

                ops.applyMomentum(
                    variable.asOfAny(),
                    accumulator.asOfAny(),
                    rate.asOfAny(),
                    gradients.dy(i),
                    momentum.asOfAny()
                )
            }
        }
    }


}

class Adagrad(
    val rate: Double = 0.01,
    val epsilon: Double = 1e-8
): GradientAlgorithm() {

    override fun buildUpdateOperations(
        builder: TrainableModelBuilder,
        variableList: List<Variable<*>>,
        gradients: Gradients
    ): List<Operand<*>> {
        with(builder) {
            val rate = constant(rate)

            return variableList.mapIndexed { i, variable ->

                val variableShape = variable.shape

                val accumulator = variable(
                    "${variable.localName}_accumulator",
                    broadcastTo(constant(epsilon), constant(variableShape)),
                    shape = variableShape,
                    trainable = false
                )

                ops.applyAdagrad(
                    variable.asOfAny(),
                    accumulator.asOfAny(),
                    rate.asOfAny(),
                    gradients.dy(i)
                )
            }
        }
    }


}