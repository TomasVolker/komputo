package tomasvolker.komputo.builder

import org.tensorflow.DataType
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.dsl.*
import tomasvolker.numeriko.core.dsl.I

fun ModelBuilder.meanSquareError(output: TFOperand, target: TFOperand): TFOperand =
    reduceMean(square(target - output), constant(I[0, 1]))

data class OptimizerOperations(
    val initialize: TFOperand,
    val optimize: TFOperand
)

interface Optimizer {

    fun buildOperations(
        ops: Ops,
        loss: TFOperand,
        parameterList: List<Variable<*>>
    ): OptimizerOperations

}

abstract class GradientAlgorithm: Optimizer {

    override fun buildOperations(
        ops: Ops,
        loss: TFOperand,
        parameterList: List<Variable<*>>
    ): OptimizerOperations {

        with(ops) {

            var result: List<OptimizerOperations>? = null

            scope("gradient_descent") {

                val grad = gradients(loss, parameterList)

                result = buildUpdateOperations(ops, parameterList, grad)

            }

            return OptimizerOperations(
                initialize = ops.withName("initialize_optimizer").group(result?.map { it.initialize } ?: error("")),
                optimize = ops.withName("optimize").group(result?.map { it.optimize } ?: error(""))
            )
        }

    }

    abstract fun buildUpdateOperations(
        ops: Ops,
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<OptimizerOperations>

}

class GradientDescent(val rate: Double): GradientAlgorithm() {

    override fun buildUpdateOperations(
        ops: Ops,
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<OptimizerOperations> {
        with(ops) {
            val rate = constant(rate.toFloat())

            return parameterList.mapIndexed { i, parameter ->
                OptimizerOperations(
                    initialize = ops.noOperation(),
                    optimize = ops.applyGradientDescent(
                        parameter.asOfAny(),
                        rate.asOfAny(),
                        gradients.dy(i)
                    )
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
        ops: Ops,
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<OptimizerOperations> {
        with(ops) {
            val rate = ops.constant(rate.toFloat())
            val momentum = ops.constant(momentum.toFloat())

            return parameterList.mapIndexed { i, parameter ->

                val parameterShape = parameter.shape


                val accumulator = variable(
                    DataType.FLOAT,
                    "${parameter.localName}_accumulator",
                    shape = parameterShape
                )

                val optimize = ops.applyMomentum(
                    parameter.asOfAny(),
                    accumulator.asOfAny(),
                    rate.asOfAny(),
                    gradients.dy(i),
                    momentum.asOfAny()
                )

                OptimizerOperations(
                    initialize = assign(accumulator, broadcastTo(constant(0.0f), parameterShape)),
                    optimize = optimize
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
        ops: Ops,
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<OptimizerOperations> {
        with(ops) {
            val rate = constant(rate.toFloat())

            return parameterList.mapIndexed { i, parameter ->

                val parameterShape = parameter.shape

                val accumulator = variable(
                    DataType.FLOAT,
                    "${parameter.localName}_accumulator",
                    shape = parameterShape
                )

                val optimize = ops.applyAdagrad(
                    parameter.asOfAny(),
                    accumulator.asOfAny(),
                    rate.asOfAny(),
                    gradients.dy(i)
                )

                OptimizerOperations(
                    initialize = assign(accumulator, broadcastTo(constant(epsilon.toFloat()), parameterShape)),
                    optimize = optimize
                )
            }
        }
    }


}