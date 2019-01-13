package tomasvolker.komputo.builder


import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.dsl.*



interface Optimizer {

    fun buildOperations(
        builder: ModelBuilder,
        loss: TFOperand,
        parameterList: List<Variable<*>>
    ): TFOperand

}

abstract class GradientAlgorithm: Optimizer {

    override fun buildOperations(
        builder: ModelBuilder,
        loss: TFOperand,
        parameterList: List<Variable<*>>
    ): TFOperand {

        with(builder) {

            var result: List<TFOperand>? = null

            scope("gradient") {

                val grad = gradients(loss, parameterList)

                result = buildOptimzeOperations(builder, parameterList, grad)

            }

            scope("optmizer") {
                return group("optimize", result ?: error(""))
            }

        }

    }

    fun buildOptimzeOperations(
        builder: ModelBuilder,
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> = builder.buildOperations(parameterList, gradients)

    abstract fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand>

}

class GradientDescent(val rate: Double): GradientAlgorithm() {

    override fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> {
        val rate = constant(rate)

        return parameterList.mapIndexed { i, parameter ->
            ops.applyGradientDescent(
                parameter.asOfAny(),
                rate.asOfAny(),
                gradients.dy(i)
            )
        }
    }

}

class Momentum(
    val rate: Double,
    val momentum: Double = 0.9
): GradientAlgorithm() {

    override fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> {
        val rate = constant(rate)
        val momentum = constant(momentum)

        return parameterList.mapIndexed { i, parameter ->

            val parameterShape = parameter.shape

            val accumulator = variable(
                "${parameter.localName}_accumulator",
                shape = parameterShape,
                dataType = defaultFloatDataType,
                initialValue = broadcastTo(constant(0.0), constant(parameterShape))
            )

            ops.applyMomentum(
                parameter.asOfAny(),
                accumulator.asOfAny(),
                rate.asOfAny(),
                gradients.dy(i),
                momentum.asOfAny()
            )
        }
    }

}

class Adagrad(
    val rate: Double = 0.01,
    val epsilon: Double = 1e-8
): GradientAlgorithm() {

    override fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> {
        val rate = constant(rate)

        return parameterList.mapIndexed { i, parameter ->

            val parameterShape = parameter.shape

            val accumulator = variable(
                "${parameter.localName}_accumulator",
                shape = parameterShape,
                dataType = defaultFloatDataType,
                initialValue = broadcastTo(constant(epsilon), constant(parameterShape))
            )

            ops.applyAdagrad(
                parameter.asOfAny(),
                accumulator.asOfAny(),
                rate.asOfAny(),
                gradients.dy(i)
            )
        }
    }


}